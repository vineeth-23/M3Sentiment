import torch

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0
    n_samples = 0

    with torch.no_grad():
        for batch in loader:
            # Move data to the specified device
            t = batch["text"].to(device)
            a = batch["audio"].to(device)
            v = batch["vision"].to(device)
            y = batch["label3"].to(device)

            # Forward pass
            logits = model(t, a, v)
            loss = criterion(logits, y)

            # Predictions and metrics
            preds = logits.argmax(dim=1)
            bs = y.size(0)
            total_loss += loss.item() * bs
            total_acc += (preds == y).sum().item()
            n_samples += bs

    return total_loss / n_samples, total_acc / n_samples


def eval_epoch_ortho(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0
    n_samples = 0

    with torch.no_grad():
        for batch in loader:
            # Move data to the specified device
            t = batch["text"].to(device)
            a = batch["audio"].to(device)
            v = batch["vision"].to(device)
            y = batch["label3"].to(device)

            # Forward pass with orthogonality outputs
            logits, _, _, _ = model(t, a, v)
            loss = criterion(logits, y)

            # Predictions and metrics
            preds = logits.argmax(dim=1)
            bs = y.size(0)
            total_loss += loss.item() * bs
            total_acc += (preds == y).sum().item()
            n_samples += bs

    return total_loss / n_samples, total_acc / n_samples


def eval_epoch_aux(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0
    n_samples = 0

    with torch.no_grad():
        for batch in loader:
            # Move data to the specified device
            t = batch["text"].to(device)
            a = batch["audio"].to(device)
            v = batch["vision"].to(device)
            y = batch["label3"].to(device)

            # Forward pass with auxiliary outputs
            logits, _, _, _ = model(t, a, v)
            loss = criterion(logits, y)

            # Predictions and metrics
            preds = logits.argmax(dim=1)
            bs = y.size(0)
            total_loss += loss.item() * bs
            total_acc += (preds == y).sum().item()
            n_samples += bs

    return total_loss / n_samples, total_acc / n_samples