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


def collect_confusion_matrix(model, loader, device, num_classes=3):
    model.eval()
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)

    with torch.no_grad():
        for batch in loader:
            text_batch = batch["text"].to(device)
            audio_batch = batch["audio"].to(device)
            vision_batch = batch["vision"].to(device)
            labels = batch["label3"].to(device)

            model_output = model(text_batch, audio_batch, vision_batch)
            logits = model_output[0] if isinstance(model_output, tuple) else model_output
            predictions = logits.argmax(dim=1)

            for actual_label, predicted_label in zip(labels.view(-1), predictions.view(-1)):
                confusion_matrix[int(actual_label.item()), int(predicted_label.item())] += 1

    return confusion_matrix.cpu().tolist()


def confusion_matrix_rows(confusion_matrix, class_names=None):
    class_names = class_names or ["negative", "neutral", "positive"]
    rows = []
    for actual_idx, actual_name in enumerate(class_names):
        row_total = sum(confusion_matrix[actual_idx])
        for predicted_idx, predicted_name in enumerate(class_names):
            count = confusion_matrix[actual_idx][predicted_idx]
            rows.append({
                "actual_label": actual_name,
                "predicted_label": predicted_name,
                "count": count,
                "row_percent": count / row_total if row_total else 0.0,
            })
    return rows


# Backward-compatible aliases for older notebooks or scripts.
eval_epoch = evaluate_standard_epoch
eval_epoch_ortho = evaluate_orthogonality_epoch
eval_epoch_aux = evaluate_auxiliary_epoch
