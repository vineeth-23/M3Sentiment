import torch


def evaluate_standard_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    sample_count = 0

    with torch.no_grad():
        for batch in loader:
            text_batch = batch["text"].to(device)
            audio_batch = batch["audio"].to(device)
            vision_batch = batch["vision"].to(device)
            labels = batch["label3"].to(device)

            logits = model(text_batch, audio_batch, vision_batch)
            loss = criterion(logits, labels)

            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            sample_count += batch_size

    return total_loss / sample_count, total_correct / sample_count


def evaluate_orthogonality_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    sample_count = 0

    with torch.no_grad():
        for batch in loader:
            text_batch = batch["text"].to(device)
            audio_batch = batch["audio"].to(device)
            vision_batch = batch["vision"].to(device)
            labels = batch["label3"].to(device)

            logits, _, _, _ = model(text_batch, audio_batch, vision_batch)
            loss = criterion(logits, labels)

            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            sample_count += batch_size

    return total_loss / sample_count, total_correct / sample_count


def evaluate_auxiliary_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    sample_count = 0

    with torch.no_grad():
        for batch in loader:
            text_batch = batch["text"].to(device)
            audio_batch = batch["audio"].to(device)
            vision_batch = batch["vision"].to(device)
            labels = batch["label3"].to(device)

            logits, _, _, _ = model(text_batch, audio_batch, vision_batch)
            loss = criterion(logits, labels)

            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            sample_count += batch_size

    return total_loss / sample_count, total_correct / sample_count


# Backward-compatible aliases for older notebooks or scripts.
eval_epoch = evaluate_standard_epoch
eval_epoch_ortho = evaluate_orthogonality_epoch
eval_epoch_aux = evaluate_auxiliary_epoch
