import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F


def _append_batch_metrics(batch_metrics, row):
    if batch_metrics is not None:
        batch_metrics.append(row)


def _extract_primary_logits(model_output):
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output


def _compute_modality_ablation_losses(model, text_batch, audio_batch, vision_batch, labels, criterion):
    """Diagnostic-only losses: do not affect gradients or model updates."""
    was_training = model.training
    model.eval()
    zero_text = torch.zeros_like(text_batch)
    zero_audio = torch.zeros_like(audio_batch)
    zero_vision = torch.zeros_like(vision_batch)

    with torch.no_grad():
        text_logits = _extract_primary_logits(model(text_batch, zero_audio, zero_vision))
        audio_logits = _extract_primary_logits(model(zero_text, audio_batch, zero_vision))
        vision_logits = _extract_primary_logits(model(zero_text, zero_audio, vision_batch))
        losses = {
            "text_only_loss": criterion(text_logits, labels).item(),
            "audio_only_loss": criterion(audio_logits, labels).item(),
            "vision_only_loss": criterion(vision_logits, labels).item(),
        }

    if was_training:
        model.train()
    return losses


def train_standard_epoch(model, loader, optimizer, criterion, device, max_grad_norm, diagnostics=None, epoch=None, batch_metrics=None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    sample_count = 0

    for batch_idx, batch in enumerate(loader, start=1):
        text_batch = batch["text"].to(device)
        audio_batch = batch["audio"].to(device)
        vision_batch = batch["vision"].to(device)
        labels = batch["label3"].to(device)

        logits = model(text_batch, audio_batch, vision_batch)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if diagnostics is not None:
            diagnostics.step(model)

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        batch_accuracy = (predictions == labels).float().mean().item()
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        sample_count += batch_size

        row = {
            "epoch": epoch,
            "batch": batch_idx,
            "global_step": diagnostics.global_step if diagnostics is not None else None,
            "total_loss": loss.item(),
            "batch_acc": batch_accuracy,
        }
        if batch_metrics is not None:
            row.update(_compute_modality_ablation_losses(model, text_batch, audio_batch, vision_batch, labels, criterion))
        _append_batch_metrics(batch_metrics, row)

    return total_loss / sample_count, total_correct / sample_count


def train_ortho_fusion_epoch(model, loader, optimizer, criterion, device, ortho_weight: float = 0.1, max_grad_norm: float = 1.0, diagnostics=None, epoch=None, batch_metrics=None):
    model.train()
    total_loss = 0.0
    classification_loss_sum = 0.0
    orthogonality_loss_sum = 0.0
    total_correct = 0
    sample_count = 0

    for batch_idx, batch in enumerate(loader, start=1):
        text_batch = batch["text"].to(device)
        audio_batch = batch["audio"].to(device)
        vision_batch = batch["vision"].to(device)
        labels = batch["label3"].to(device)

        logits, text_features, audio_features, vision_features = model(text_batch, audio_batch, vision_batch)
        classification_loss = criterion(logits, labels)

        text_features = F.normalize(text_features, p=2, dim=1)
        audio_features = F.normalize(audio_features, p=2, dim=1)
        vision_features = F.normalize(vision_features, p=2, dim=1)
        text_audio_similarity = (text_features * audio_features).sum(dim=1)
        text_vision_similarity = (text_features * vision_features).sum(dim=1)
        audio_vision_similarity = (audio_features * vision_features).sum(dim=1)
        orthogonality_loss = (
            text_audio_similarity.pow(2).mean()
            + text_vision_similarity.pow(2).mean()
            + audio_vision_similarity.pow(2).mean()
        )

        loss = classification_loss + ortho_weight * orthogonality_loss

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if diagnostics is not None:
            diagnostics.step(model)

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        batch_accuracy = (predictions == labels).float().mean().item()
        total_loss += loss.item() * batch_size
        classification_loss_sum += classification_loss.item() * batch_size
        orthogonality_loss_sum += orthogonality_loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        sample_count += batch_size

        row = {
            "epoch": epoch,
            "batch": batch_idx,
            "global_step": diagnostics.global_step if diagnostics is not None else None,
            "total_loss": loss.item(),
            "classification_loss": classification_loss.item(),
            "ortho_loss_raw": orthogonality_loss.item(),
            "ortho_loss_weighted": orthogonality_loss.item() * ortho_weight,
            "batch_acc": batch_accuracy,
        }
        if batch_metrics is not None:
            row.update(_compute_modality_ablation_losses(model, text_batch, audio_batch, vision_batch, labels, criterion))
        _append_batch_metrics(batch_metrics, row)

    avg_classification_loss = classification_loss_sum / sample_count
    avg_orthogonality_loss = orthogonality_loss_sum / sample_count
    print(f"ortho_loss = {avg_orthogonality_loss * ortho_weight}")

    return (
        total_loss / sample_count,
        total_correct / sample_count,
        avg_classification_loss,
        avg_orthogonality_loss,
        avg_orthogonality_loss * ortho_weight,
    )


def train_aux_fusion_epoch(model, loader, optimizer, criterion, device, aux_weight: float = 0.05, max_grad_norm: float = 1.0, diagnostics=None, epoch=None, batch_metrics=None):
    model.train()
    total_loss = 0.0
    main_loss_sum = 0.0
    text_auxiliary_loss_sum = 0.0
    audio_auxiliary_loss_sum = 0.0
    vision_auxiliary_loss_sum = 0.0
    total_correct = 0
    sample_count = 0

    for batch_idx, batch in enumerate(loader, start=1):
        text_batch = batch["text"].to(device)
        audio_batch = batch["audio"].to(device)
        vision_batch = batch["vision"].to(device)
        labels = batch["label3"].to(device)

        main_logits, text_logits, audio_logits, vision_logits = model(text_batch, audio_batch, vision_batch)
        main_loss = criterion(main_logits, labels)
        text_auxiliary_loss = criterion(text_logits, labels)
        audio_auxiliary_loss = criterion(audio_logits, labels)
        vision_auxiliary_loss = criterion(vision_logits, labels)

        auxiliary_loss = text_auxiliary_loss + audio_auxiliary_loss + vision_auxiliary_loss
        loss = main_loss + aux_weight * auxiliary_loss

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if diagnostics is not None:
            diagnostics.step(model)

        predictions = main_logits.argmax(dim=1)
        batch_size = labels.size(0)
        batch_accuracy = (predictions == labels).float().mean().item()
        total_loss += loss.item() * batch_size
        main_loss_sum += main_loss.item() * batch_size
        text_auxiliary_loss_sum += text_auxiliary_loss.item() * batch_size
        audio_auxiliary_loss_sum += audio_auxiliary_loss.item() * batch_size
        vision_auxiliary_loss_sum += vision_auxiliary_loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        sample_count += batch_size

        row = {
            "epoch": epoch,
            "batch": batch_idx,
            "global_step": diagnostics.global_step if diagnostics is not None else None,
            "total_loss": loss.item(),
            "main_loss": main_loss.item(),
            "aux_text_loss": text_auxiliary_loss.item(),
            "aux_audio_loss": audio_auxiliary_loss.item(),
            "aux_video_loss": vision_auxiliary_loss.item(),
            "aux_loss_weighted": auxiliary_loss.item() * aux_weight,
            "batch_acc": batch_accuracy,
        }
        if batch_metrics is not None:
            row.update(_compute_modality_ablation_losses(model, text_batch, audio_batch, vision_batch, labels, criterion))
        _append_batch_metrics(batch_metrics, row)

    avg_main_loss = main_loss_sum / sample_count
    avg_text_auxiliary_loss = text_auxiliary_loss_sum / sample_count
    avg_audio_auxiliary_loss = audio_auxiliary_loss_sum / sample_count
    avg_vision_auxiliary_loss = vision_auxiliary_loss_sum / sample_count
    avg_weighted_auxiliary_loss = aux_weight * (
        avg_text_auxiliary_loss + avg_audio_auxiliary_loss + avg_vision_auxiliary_loss
    )

    return (
        total_loss / sample_count,
        total_correct / sample_count,
        avg_main_loss,
        avg_text_auxiliary_loss,
        avg_audio_auxiliary_loss,
        avg_vision_auxiliary_loss,
        avg_weighted_auxiliary_loss,
    )


# Backward-compatible aliases for older notebooks or scripts.
train_epoch = train_standard_epoch
train_orthogonality_epoch = train_ortho_fusion_epoch
train_epoch_ortho = train_ortho_fusion_epoch
train_auxiliary_epoch = train_aux_fusion_epoch
train_epoch_aux = train_aux_fusion_epoch
