import os
from collections import defaultdict

import pandas as pd
import torch
import torch.nn.functional as F


def _extract_analysis(model_output):
    if isinstance(model_output, tuple) and model_output and isinstance(model_output[-1], dict):
        return model_output[-1]
    return {}


def _add_weighted(total, count, key, value, weight):
    total[key] += float(value) * weight
    count[key] += weight


def _orthogonality_batch_stats(features):
    t_feat = F.normalize(features["text"], p=2, dim=1)
    a_feat = F.normalize(features["audio"], p=2, dim=1)
    v_feat = F.normalize(features["vision"], p=2, dim=1)

    pairs = {
        "text_audio": (t_feat * a_feat).sum(dim=1),
        "text_vision": (t_feat * v_feat).sum(dim=1),
        "audio_vision": (a_feat * v_feat).sum(dim=1),
    }

    stats = {}
    for pair_name, dots in pairs.items():
        stats[f"{pair_name}_cosine_mean"] = dots.mean()
        stats[f"{pair_name}_cosine_abs_mean"] = dots.abs().mean()
        stats[f"{pair_name}_cosine_squared_mean"] = dots.pow(2).mean()

    stats["overall_cosine_abs_mean"] = torch.stack([dots.abs().mean() for dots in pairs.values()]).mean()
    stats["overall_cosine_squared_mean"] = torch.stack([dots.pow(2).mean() for dots in pairs.values()]).mean()
    return stats


def _attention_entropy(weights):
    weights = weights.clamp_min(1e-12)
    return -(weights * weights.log()).sum(dim=-1)


def _record_sequence_attention_stats(totals, counts, weights, batch_size, category, modality, layer_idx):
    # weights: (B, heads, T, T). Diagonal means a token attends to itself.
    entropy = _attention_entropy(weights)
    diagonal = weights.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
    token_count = weights.size(-1)
    if token_count > 1:
        diagonal_sum = weights.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        off_diagonal = (weights.sum(dim=(-1, -2)) - diagonal_sum) / (token_count * (token_count - 1))
    else:
        off_diagonal = diagonal

    metrics = {
        "mean_entropy": entropy,
        "mean_self_position_attention": diagonal,
        "mean_other_position_attention": off_diagonal,
    }

    for metric, values in metrics.items():
        _add_weighted(
            totals,
            counts,
            (category, metric, modality, str(layer_idx), "all", ""),
            values.mean().item(),
            batch_size,
        )
        for head_idx in range(weights.size(1)):
            _add_weighted(
                totals,
                counts,
                (category, metric, modality, str(layer_idx), str(head_idx), ""),
                values[:, head_idx].mean().item(),
                batch_size,
            )


def _record_token_attention_stats(totals, counts, weights, tokens, batch_size, category, layer_idx):
    # weights: (B, heads, Q, K). Mean over queries gives attention paid to each token.
    entropy = _attention_entropy(weights)
    mean_by_key = weights.mean(dim=2)

    _add_weighted(
        totals,
        counts,
        (category, "mean_entropy", "all_queries", str(layer_idx), "all", ""),
        entropy.mean().item(),
        batch_size,
    )

    for token_idx, token in enumerate(tokens):
        _add_weighted(
            totals,
            counts,
            (category, "mean_attention", "all_queries", str(layer_idx), "all", token),
            mean_by_key[:, :, token_idx].mean().item(),
            batch_size,
        )
        for head_idx in range(weights.size(1)):
            _add_weighted(
                totals,
                counts,
                (category, "mean_attention", "all_queries", str(layer_idx), str(head_idx), token),
                mean_by_key[:, head_idx, token_idx].mean().item(),
                batch_size,
            )


def analyze_model(model, loader, device, split="analysis", stage="final", global_step=None, max_batches=5):
    was_training = model.training
    model.eval()

    totals = defaultdict(float)
    counts = defaultdict(float)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            batch_size = text.size(0)

            output = model(text, audio, vision, return_attention=True)
            analysis = _extract_analysis(output)

            features = analysis.get("features")
            if features:
                for metric, value in _orthogonality_batch_stats(features).items():
                    _add_weighted(totals, counts, ("orthogonality", metric, "", "", "", ""), value.item(), batch_size)

            for modality, layer_attentions in analysis.get("self_attention", {}).items():
                for layer_idx, weights in enumerate(layer_attentions):
                    _record_sequence_attention_stats(
                        totals,
                        counts,
                        weights.detach(),
                        batch_size,
                        "self_attention",
                        modality,
                        layer_idx,
                    )

            fusion_info = analysis.get("fusion_attention")
            if fusion_info:
                for layer_idx, weights in enumerate(fusion_info["weights"]):
                    _record_token_attention_stats(
                        totals,
                        counts,
                        weights.detach(),
                        fusion_info["tokens"],
                        batch_size,
                        "fusion_attention",
                        layer_idx,
                    )

            for query_name, attention_info in analysis.get("cross_attention", {}).items():
                weights = attention_info["weights"]
                keys = attention_info["keys"]
                if weights is None:
                    continue

                weights = weights.detach()
                if weights.dim() == 3:
                    weights = weights.unsqueeze(1)
                # (B, heads, 1, modalities) -> (B, heads, modalities)
                weights = weights.squeeze(2)

                entropy = _attention_entropy(weights)
                _add_weighted(
                    totals,
                    counts,
                    ("attention_entropy", "mean_entropy", query_name, "", "all", ""),
                    entropy.mean().item(),
                    batch_size,
                )

                for key_idx, modality in enumerate(keys):
                    overall_value = weights[:, :, key_idx].mean().item()
                    _add_weighted(
                        totals,
                        counts,
                        ("cross_attention_overall", "mean_attention", query_name, "", "all", modality),
                        overall_value,
                        batch_size,
                    )

                    for head_idx in range(weights.size(1)):
                        head_value = weights[:, head_idx, key_idx].mean().item()
                        _add_weighted(
                            totals,
                            counts,
                            ("cross_attention_head", "mean_attention", query_name, "", str(head_idx), modality),
                            head_value,
                            batch_size,
                        )

    if was_training:
        model.train()

    rows = []
    for (category, metric, query, layer, head, attended_modality), total in sorted(totals.items()):
        count = counts[(category, metric, query, layer, head, attended_modality)]
        rows.append({
            "stage": stage,
            "split": split,
            "global_step": global_step,
            "category": category,
            "metric": metric,
            "query": query,
            "layer": layer,
            "head": head,
            "attended_modality": attended_modality,
            "value": total / count if count else 0.0,
        })

    return rows


def save_diagnostics(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def run_diagnostics(model, loader, device, model_name, stage, split="test", global_step=None, max_batches=5, output_dir="outputs/metrics"):
    rows = analyze_model(
        model,
        loader,
        device,
        split=split,
        stage=stage,
        global_step=global_step,
        max_batches=max_batches,
    )
    out_path = os.path.join(output_dir, "diagnostics", f"{model_name}_{stage}_{split}.csv")
    save_diagnostics(rows, out_path)
    return out_path


class DiagnosticRecorder:
    def __init__(self, model_name, loader, device, snapshot_batches=None, max_batches=5, split="test", output_dir="outputs/metrics"):
        self.model_name = model_name
        self.loader = loader
        self.device = device
        self.snapshot_batches = set(snapshot_batches or [])
        self.max_batches = max_batches
        self.split = split
        self.output_dir = output_dir
        self.global_step = 0
        self.paths = []

    def step(self, model):
        self.global_step += 1
        if self.global_step in self.snapshot_batches:
            self.record(model, stage=f"batch_{self.global_step}")

    def record(self, model, stage):
        path = run_diagnostics(
            model,
            self.loader,
            self.device,
            self.model_name,
            stage=stage,
            split=self.split,
            global_step=self.global_step,
            max_batches=self.max_batches,
            output_dir=self.output_dir,
        )
        self.paths.append(path)
        return path

    def finalize(self, model):
        path = run_diagnostics(
            model,
            self.loader,
            self.device,
            self.model_name,
            stage="final",
            split=self.split,
            global_step=self.global_step,
            max_batches=None,
            output_dir=self.output_dir,
        )
        self.paths.append(path)
        return path
