import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

def _log_batch(batch_metrics, row):
    if batch_metrics is not None:
        batch_metrics.append(row)


def _first_logits(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def _modality_only_losses(model, text, audio, vision, labels, criterion):
    """Diagnostic-only losses: do not affect gradients or model updates."""
    was_training = model.training
    model.eval()
    zeros_text = torch.zeros_like(text)
    zeros_audio = torch.zeros_like(audio)
    zeros_vision = torch.zeros_like(vision)

    with torch.no_grad():
        text_logits = _first_logits(model(text, zeros_audio, zeros_vision))
        audio_logits = _first_logits(model(zeros_text, audio, zeros_vision))
        vision_logits = _first_logits(model(zeros_text, zeros_audio, vision))
        losses = {
            "text_only_loss": criterion(text_logits, labels).item(),
            "audio_only_loss": criterion(audio_logits, labels).item(),
            "vision_only_loss": criterion(vision_logits, labels).item(),
        }

    if was_training:
        model.train()
    return losses


def train_epoch(model, loader, optimizer, criterion, device, max_grad_norm, diagnostics=None, epoch=None, batch_metrics=None):
    model.train()
    total_loss = 0.0
    total_acc = 0
    n_samples = 0

    for batch_idx, batch in enumerate(loader, start=1):
        t = batch["text"].to(device)
        a = batch["audio"].to(device)
        v = batch["vision"].to(device)
        y = batch["label3"].to(device)

        logits = model(t, a, v)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if diagnostics is not None:
            diagnostics.step(model)

        preds = logits.argmax(dim=1)
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_acc += (preds == y).sum().item()
        n_samples += bs
        row = {
            "epoch": epoch,
            "batch": batch_idx,
            "global_step": diagnostics.global_step if diagnostics is not None else None,
            "total_loss": loss.item(),
            "batch_acc": (preds == y).float().mean().item(),
        }
        if batch_metrics is not None:
            row.update(_modality_only_losses(model, t, a, v, y, criterion))
        _log_batch(batch_metrics, row)

    return total_loss / n_samples, total_acc / n_samples


def train_epoch_ortho(model, loader, optimizer, criterion, device, ortho_weight: float = 0.1, max_grad_norm: float = 1.0, diagnostics=None, epoch=None, batch_metrics=None):
    model.train()
    total_loss = 0.0
    cls_loss_sum = 0.0
    total_acc = 0
    n_samples = 0
    ortho_loss_sum = 0.0

    for batch_idx, batch in enumerate(loader, start=1):
        t = batch["text"].to(device)
        a = batch["audio"].to(device)
        v = batch["vision"].to(device)
        y = batch["label3"].to(device)

        # Forward pass
        logits, t_feat, a_feat, v_feat = model(t, a, v)

        # Classification loss
        cls_loss = criterion(logits, y)

        # Orthogonality loss
        t_feat = F.normalize(t_feat, p=2, dim=1)
        a_feat = F.normalize(a_feat, p=2, dim=1)
        v_feat = F.normalize(v_feat, p=2, dim=1)
        dot1 = (t_feat * a_feat).sum(dim=1)
        dot2 = (t_feat * v_feat).sum(dim=1)
        dot3 = (a_feat * v_feat).sum(dim=1)
        ortho_loss = ((dot1) ** 2).mean() + ((dot2) ** 2).mean() + ((dot3) ** 2).mean()

        # Total loss
        loss = cls_loss + ortho_weight * ortho_loss

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if diagnostics is not None:
            diagnostics.step(model)

        preds = logits.argmax(dim=1)
        bs = y.size(0)
        total_loss += loss.item() * bs
        cls_loss_sum += cls_loss.item() * bs
        ortho_loss_sum += ortho_loss.item() * bs
        total_acc += (preds == y).sum().item()
        n_samples += bs
        row = {
            "epoch": epoch,
            "batch": batch_idx,
            "global_step": diagnostics.global_step if diagnostics is not None else None,
            "total_loss": loss.item(),
            "classification_loss": cls_loss.item(),
            "ortho_loss_raw": ortho_loss.item(),
            "ortho_loss_weighted": ortho_loss.item() * ortho_weight,
            "batch_acc": (preds == y).float().mean().item(),
        }
        if batch_metrics is not None:
            row.update(_modality_only_losses(model, t, a, v, y, criterion))
        _log_batch(batch_metrics, row)

    avg_cls_loss = cls_loss_sum / n_samples
    avg_ortho_loss = ortho_loss_sum / n_samples
    print(f'ortho_loss = {avg_ortho_loss * ortho_weight}')

    return (
        total_loss / n_samples,
        total_acc / n_samples,
        avg_cls_loss,
        avg_ortho_loss,
        avg_ortho_loss * ortho_weight,
    )


def train_epoch_aux(model, loader, optimizer, criterion, device, aux_weight: float = 0.05, max_grad_norm: float = 1.0, diagnostics=None, epoch=None, batch_metrics=None):
    model.train()
    total_loss = 0.0
    main_loss_sum = 0.0
    loss_text_sum = 0.0
    loss_audio_sum = 0.0
    loss_video_sum = 0.0
    total_acc = 0
    n_samples = 0

    for batch_idx, batch in enumerate(loader, start=1):
        t = batch["text"].to(device)
        a = batch["audio"].to(device)
        v = batch["vision"].to(device)
        y = batch["label3"].to(device)

        # Forward pass
        logits, logits_text, logits_audio, logits_video = model(t, a, v)

        # Losses
        main_loss = criterion(logits, y)
        loss_text = criterion(logits_text, y)
        loss_audio = criterion(logits_audio, y)
        loss_video = criterion(logits_video, y)

        # Combine losses
        aux_loss = loss_text + loss_audio + loss_video
        loss = main_loss + aux_weight * aux_loss

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if diagnostics is not None:
            diagnostics.step(model)

        preds = logits.argmax(dim=1)
        bs = y.size(0)
        total_loss += loss.item() * bs
        main_loss_sum += main_loss.item() * bs
        loss_text_sum += loss_text.item() * bs
        loss_audio_sum += loss_audio.item() * bs
        loss_video_sum += loss_video.item() * bs
        total_acc += (preds == y).sum().item()
        n_samples += bs
        row = {
            "epoch": epoch,
            "batch": batch_idx,
            "global_step": diagnostics.global_step if diagnostics is not None else None,
            "total_loss": loss.item(),
            "main_loss": main_loss.item(),
            "aux_text_loss": loss_text.item(),
            "aux_audio_loss": loss_audio.item(),
            "aux_video_loss": loss_video.item(),
            "aux_loss_weighted": aux_loss.item() * aux_weight,
            "batch_acc": (preds == y).float().mean().item(),
        }
        if batch_metrics is not None:
            row.update(_modality_only_losses(model, t, a, v, y, criterion))
        _log_batch(batch_metrics, row)

    avg_main = main_loss_sum / n_samples
    avg_text = loss_text_sum / n_samples
    avg_audio = loss_audio_sum / n_samples
    avg_video = loss_video_sum / n_samples
    avg_aux_weighted = aux_weight * (avg_text + avg_audio + avg_video)

    return total_loss / n_samples, total_acc / n_samples, avg_main, avg_text, avg_audio, avg_video, avg_aux_weighted
