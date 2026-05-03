import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from m3sentiment.config import Config
from m3sentiment.data_loaders import build_mosei_dataloaders
from m3sentiment.models.late_fusion import LateFusionTransformer
from m3sentiment.models.cross_modal_attention import LateFusionWithCrossModal
from m3sentiment.models.orthogonality import LateFusionWithCrossModalOrtho
from m3sentiment.models.auxiliary_heads import LateFusionWithCrossModalAuxHeads
from m3sentiment.training import (
    train_auxiliary_epoch,
    train_orthogonality_epoch,
    train_standard_epoch,
)
from m3sentiment.evaluation import (
    evaluate_auxiliary_epoch,
    evaluate_orthogonality_epoch,
    evaluate_standard_epoch,
)
from m3sentiment.diagnostics import DiagnosticRecorder
import pandas as pd

# Function to save metrics to CSV
def write_metrics_csv(metrics, fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    pd.DataFrame(metrics).to_csv(fname, index=False)


def save_final_model_weights(model, fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    torch.save(model.state_dict(), fname)


def _load_torch_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_checkpoint_path(args, model_name):
    return os.path.join(args.checkpoint_dir, f"{model_name}_latest.pt")


def save_epoch_checkpoint(args, model_name, epoch, model, optimizer, scheduler, metrics, batch_metrics, diagnostics):
    path = build_checkpoint_path(args, model_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "metrics": metrics,
            "batch_metrics": batch_metrics,
            "diagnostics_global_step": diagnostics.global_step if diagnostics is not None else 0,
        },
        path,
    )
    return path


def restore_training_state_if_requested(args, model_name, model, optimizer, scheduler, diagnostics, device):
    if not args.resume:
        return 1, [], []

    path = args.resume_checkpoint or build_checkpoint_path(args, model_name)
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}; starting {model_name} from epoch 1.")
        return 1, [], []

    checkpoint = _load_torch_checkpoint(path, device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    if diagnostics is not None:
        diagnostics.global_step = checkpoint.get("diagnostics_global_step", 0)

    last_epoch = checkpoint.get("epoch", 0)
    print(f"Resumed {model_name} from {path} after epoch {last_epoch}.")
    return last_epoch + 1, checkpoint.get("metrics", []), checkpoint.get("batch_metrics", [])


def parse_diagnostic_snapshot_steps(raw):
    if raw is None:
        return [1, 5, 10, 25, 50, 100, 250, 500, 1000]
    if not raw.strip():
        return []
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def average_batch_columns_for_epoch(batch_rows, epoch, columns):
    rows = [row for row in batch_rows if row.get("epoch") == epoch]
    averages = {}
    for column in columns:
        values = [row[column] for row in rows if column in row and row[column] is not None]
        if values:
            averages[column] = sum(values) / len(values)
    return averages


def choose_diagnostic_loader(args, train_loader, val_loader, test_loader):
    if args.diagnostics_split == "train":
        return train_loader
    if args.diagnostics_split == "val":
        return val_loader
    return test_loader


def create_diagnostic_recorder(args, model_name, loader, device):
    if args.no_diagnostics:
        return None
    return DiagnosticRecorder(
        model_name=model_name,
        loader=loader,
        device=device,
        snapshot_batches=parse_diagnostic_snapshot_steps(args.analysis_snapshot_batches),
        max_batches=args.analysis_max_batches,
        split=args.diagnostics_split,
        output_dir=args.metrics_dir,
    )

# Run the Late Fusion Transformer model (Baseline 1).
def run_late_fusion_experiment(config, dataset_path, args):
    train_loader, val_loader, test_loader = build_mosei_dataloaders(dataset_path, config.train["batch_size"])

    # Infer input dimensions from one batch
    sample = next(iter(train_loader))
    text_dim, audio_dim, vision_dim = (
        sample["text"].shape[-1],
        sample["audio"].shape[-1],
        sample["vision"].shape[-1]
    )

    # Initialize the model
    model = LateFusionTransformer(
        text_dim, audio_dim, vision_dim,
        hidden_dim=config.model["hidden_dim"],
        n_heads=config.model["n_heads"],
        n_layers=config.model["n_layers"],
        dropout=config.model["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.train["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    diagnostics_loader = choose_diagnostic_loader(args, train_loader, val_loader, test_loader)
    diagnostics = create_diagnostic_recorder(args, "baseline1", diagnostics_loader, device)

    epoch_metrics = []
    batch_metrics = []
    start_epoch, epoch_metrics, batch_metrics = restore_training_state_if_requested(
        args, "baseline1", model, optimizer, scheduler, diagnostics, device
    )
    for epoch in range(start_epoch, config.train["epochs"] + 1):
        # Train and evaluate
        train_loss, train_acc = train_standard_epoch(
            model, train_loader, optimizer, criterion, device, config.train["max_grad_norm"],
            diagnostics=diagnostics, epoch=epoch, batch_metrics=batch_metrics
        )
        val_loss, val_acc = evaluate_standard_epoch(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate_standard_epoch(model, test_loader, criterion, device)
        modality_losses = average_batch_columns_for_epoch(
            batch_metrics,
            epoch,
            ["text_only_loss", "audio_only_loss", "vision_only_loss"],
        )

        # Log metrics
        epoch_metrics.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            **modality_losses,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc
        })

        scheduler.step(val_loss)
        if diagnostics is not None and not args.no_epoch_diagnostics:
            diagnostics.record(model, stage=f"epoch_{epoch}")
        write_metrics_csv(epoch_metrics, os.path.join(args.metrics_dir, "baseline1_metrics.csv"))
        write_metrics_csv(batch_metrics, os.path.join(args.metrics_dir, "baseline1_batch_metrics.csv"))
        checkpoint = save_epoch_checkpoint(
            args, "baseline1", epoch, model, optimizer, scheduler, epoch_metrics, batch_metrics, diagnostics
        )
        print(f"Epoch {epoch:02d}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  checkpoint={checkpoint}")

    # Final evaluation and save results
    test_loss, test_acc = evaluate_standard_epoch(model, test_loader, criterion, device)
    print(f"\nTest ▶ loss={test_loss:.4f} acc={test_acc:.4f}")
    if diagnostics is not None:
        diagnostics.finalize(model)
    save_final_model_weights(model, os.path.join(args.model_dir, "baseline1.pth"))
    write_metrics_csv(epoch_metrics, os.path.join(args.metrics_dir, "baseline1_metrics.csv"))
    write_metrics_csv(batch_metrics, os.path.join(args.metrics_dir, "baseline1_batch_metrics.csv"))

# Run the Late Fusion with Cross-Modal Attention model (Baseline 2).
def run_cross_modal_experiment(config, dataset_path, args):
    train_loader, val_loader, test_loader = build_mosei_dataloaders(dataset_path, config.train["batch_size"])

    # Infer input dimensions from one batch
    sample = next(iter(train_loader))
    text_dim, audio_dim, vision_dim = (
        sample["text"].shape[-1],
        sample["audio"].shape[-1],
        sample["vision"].shape[-1]
    )

    # Initialize the model
    model = LateFusionWithCrossModal(
        text_dim, audio_dim, vision_dim,
        hidden_dim=config.model["hidden_dim"],
        n_heads=config.model["n_heads"],
        n_layers=config.model["n_layers"],
        dropout=config.model["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.train["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    diagnostics_loader = choose_diagnostic_loader(args, train_loader, val_loader, test_loader)
    diagnostics = create_diagnostic_recorder(args, "baseline2", diagnostics_loader, device)

    epoch_metrics = []
    batch_metrics = []
    start_epoch, epoch_metrics, batch_metrics = restore_training_state_if_requested(
        args, "baseline2", model, optimizer, scheduler, diagnostics, device
    )
    for epoch in range(start_epoch, config.train["epochs"] + 1):
        # Train and evaluate
        train_loss, train_acc = train_standard_epoch(
            model, train_loader, optimizer, criterion, device, config.train["max_grad_norm"],
            diagnostics=diagnostics, epoch=epoch, batch_metrics=batch_metrics
        )
        val_loss, val_acc = evaluate_standard_epoch(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate_standard_epoch(model, test_loader, criterion, device)
        modality_losses = average_batch_columns_for_epoch(
            batch_metrics,
            epoch,
            ["text_only_loss", "audio_only_loss", "vision_only_loss"],
        )

        # Log metrics
        epoch_metrics.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            **modality_losses,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc
        })

        scheduler.step(val_loss)
        if diagnostics is not None and not args.no_epoch_diagnostics:
            diagnostics.record(model, stage=f"epoch_{epoch}")
        write_metrics_csv(epoch_metrics, os.path.join(args.metrics_dir, "baseline2_metrics.csv"))
        write_metrics_csv(batch_metrics, os.path.join(args.metrics_dir, "baseline2_batch_metrics.csv"))
        checkpoint = save_epoch_checkpoint(
            args, "baseline2", epoch, model, optimizer, scheduler, epoch_metrics, batch_metrics, diagnostics
        )
        print(f"Epoch {epoch:02d}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  checkpoint={checkpoint}")

    # Final evaluation and save results
    test_loss, test_acc = evaluate_standard_epoch(model, test_loader, criterion, device)
    print(f"\nTest ▶ loss={test_loss:.4f} acc={test_acc:.4f}")
    if diagnostics is not None:
        diagnostics.finalize(model)
    save_final_model_weights(model, os.path.join(args.model_dir, "baseline2.pth"))
    write_metrics_csv(epoch_metrics, os.path.join(args.metrics_dir, "baseline2_metrics.csv"))
    write_metrics_csv(batch_metrics, os.path.join(args.metrics_dir, "baseline2_batch_metrics.csv"))

# Run the Late Fusion with Orthogonality model (Improved 1).
def run_orthogonality_experiment(config, dataset_path, args):
    train_loader, val_loader, test_loader = build_mosei_dataloaders(dataset_path, config.train["batch_size"])

    # Infer input dimensions
    sample = next(iter(train_loader))
    text_dim, audio_dim, vision_dim = (
        sample["text"].shape[-1],
        sample["audio"].shape[-1],
        sample["vision"].shape[-1]
    )

    model = LateFusionWithCrossModalOrtho(
        text_dim, audio_dim, vision_dim,
        hidden_dim=config.model["hidden_dim"],
        n_heads=config.model["n_heads"],
        n_layers=config.model["n_layers"],
        dropout=config.model["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.train["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    ortho_weight = config.train.get("ortho_weight", 0.01)
    max_grad_norm = config.train.get("max_grad_norm", 1.0)

    print(f"Ortho Weight = {ortho_weight}")
    diagnostics_loader = choose_diagnostic_loader(args, train_loader, val_loader, test_loader)
    diagnostics = create_diagnostic_recorder(args, "improved_ortho", diagnostics_loader, device)

    epoch_metrics = []
    batch_metrics = []
    start_epoch, epoch_metrics, batch_metrics = restore_training_state_if_requested(
        args, "improved_ortho", model, optimizer, scheduler, diagnostics, device
    )
    for epoch in range(start_epoch, config.train["epochs"] + 1):
        # Train and evaluate
        train_loss, train_acc, classification_loss, raw_ortho_loss, weighted_ortho_loss = train_orthogonality_epoch(
            model, train_loader, optimizer, criterion, device, ortho_weight, max_grad_norm,
            diagnostics=diagnostics, epoch=epoch, batch_metrics=batch_metrics
        )
        val_loss, val_acc = evaluate_orthogonality_epoch(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate_orthogonality_epoch(model, test_loader, criterion, device)
        modality_losses = average_batch_columns_for_epoch(
            batch_metrics,
            epoch,
            ["text_only_loss", "audio_only_loss", "vision_only_loss"],
        )

        # Log metrics
        epoch_metrics.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "classification_loss": classification_loss,
            "ortho_loss_raw": raw_ortho_loss,
            "ortho_loss_weighted": weighted_ortho_loss,
            **modality_losses,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc
        })

        scheduler.step(val_loss)
        if diagnostics is not None and not args.no_epoch_diagnostics:
            diagnostics.record(model, stage=f"epoch_{epoch}")
        write_metrics_csv(epoch_metrics, os.path.join(args.metrics_dir, "ortho_metrics.csv"))
        write_metrics_csv(batch_metrics, os.path.join(args.metrics_dir, "ortho_batch_metrics.csv"))
        checkpoint = save_epoch_checkpoint(
            args, "improved_ortho", epoch, model, optimizer, scheduler, epoch_metrics, batch_metrics, diagnostics
        )
        print(f"Epoch {epoch:02d}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  checkpoint={checkpoint}")

    # Final evaluation and save results
    test_loss, test_acc = evaluate_orthogonality_epoch(model, test_loader, criterion, device)
    print(f"\nTest ▶ loss={test_loss:.4f} acc={test_acc:.4f}")
    if diagnostics is not None:
        diagnostics.finalize(model)
    save_final_model_weights(model, os.path.join(args.model_dir, "improved_ortho.pth"))
    write_metrics_csv(epoch_metrics, os.path.join(args.metrics_dir, "ortho_metrics.csv"))
    write_metrics_csv(batch_metrics, os.path.join(args.metrics_dir, "ortho_batch_metrics.csv"))

# Run the Late Fusion with Auxiliary Heads model (Improved 2).
def run_auxiliary_experiment(config, dataset_path, args):
    train_loader, val_loader, test_loader = build_mosei_dataloaders(dataset_path, config.train["batch_size"])

    # Infer input dimensions
    sample = next(iter(train_loader))
    text_dim, audio_dim, vision_dim = (
        sample["text"].shape[-1],
        sample["audio"].shape[-1],
        sample["vision"].shape[-1]
    )

    model = LateFusionWithCrossModalAuxHeads(
        text_dim, audio_dim, vision_dim,
        hidden_dim=config.model["hidden_dim"],
        n_heads=config.model["n_heads"],
        n_layers=config.model["n_layers"],
        dropout=config.model["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.train["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    diagnostics_loader = choose_diagnostic_loader(args, train_loader, val_loader, test_loader)
    diagnostics = create_diagnostic_recorder(args, "improved_aux", diagnostics_loader, device)

    epoch_metrics = []
    batch_metrics = []
    start_epoch, epoch_metrics, batch_metrics = restore_training_state_if_requested(
        args, "improved_aux", model, optimizer, scheduler, diagnostics, device
    )
    for epoch in range(start_epoch, config.train["epochs"] + 1):
        # Train and evaluate
        train_loss, train_acc, main_loss, text_aux_loss, audio_aux_loss, vision_aux_loss, weighted_aux_loss = train_auxiliary_epoch(
            model, train_loader, optimizer, criterion, device, config.train["aux_weight"], config.train["max_grad_norm"],
            diagnostics=diagnostics, epoch=epoch, batch_metrics=batch_metrics
        )
        val_loss, val_acc = evaluate_auxiliary_epoch(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate_auxiliary_epoch(model, test_loader, criterion, device)
        modality_losses = average_batch_columns_for_epoch(
            batch_metrics,
            epoch,
            ["text_only_loss", "audio_only_loss", "vision_only_loss"],
        )

        # Log metrics
        epoch_metrics.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "main_loss": main_loss,
            "aux_text_loss": text_aux_loss,
            "aux_audio_loss": audio_aux_loss,
            "aux_video_loss": vision_aux_loss,
            "aux_loss_weighted": weighted_aux_loss,
            **modality_losses,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc
        })

        scheduler.step(val_loss)
        if diagnostics is not None and not args.no_epoch_diagnostics:
            diagnostics.record(model, stage=f"epoch_{epoch}")
        write_metrics_csv(epoch_metrics, os.path.join(args.metrics_dir, "aux_metrics.csv"))
        write_metrics_csv(batch_metrics, os.path.join(args.metrics_dir, "aux_batch_metrics.csv"))
        checkpoint = save_epoch_checkpoint(
            args, "improved_aux", epoch, model, optimizer, scheduler, epoch_metrics, batch_metrics, diagnostics
        )
        print(f"Epoch {epoch:02d}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  checkpoint={checkpoint}")

    # Final evaluation and save results
    test_loss, test_acc = evaluate_auxiliary_epoch(model, test_loader, criterion, device)
    print(f"\nTest ▶ loss={test_loss:.4f} acc={test_acc:.4f}")
    if diagnostics is not None:
        diagnostics.finalize(model)
    save_final_model_weights(model, os.path.join(args.model_dir, "improved_aux.pth"))
    write_metrics_csv(epoch_metrics, os.path.join(args.metrics_dir, "aux_metrics.csv"))
    write_metrics_csv(batch_metrics, os.path.join(args.metrics_dir, "aux_batch_metrics.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--data", default="data/aligned_mosei_dataset.pkl")
    parser.add_argument('--run1', action='store_true', help='Run the Late Fusion model (baseline1)')
    parser.add_argument('--run2', action='store_true', help='Run the Cross Attention model (baseline2)')
    parser.add_argument('--run3', action='store_true', help='Run the Orthogonality model (improved3)')
    parser.add_argument('--run4', action='store_true', help='Run the Aux Heads model (improved4)')
    parser.add_argument('--no-diagnostics', action='store_true', help='Disable representation and cross-attention diagnostics')
    parser.add_argument('--analysis-max-batches', type=int, default=5, help='Number of batches used for training-stage diagnostic snapshots')
    parser.add_argument('--analysis-snapshot-batches', default="1,5,10,25,50,100,250,500,1000", help='Comma-separated global batch steps for diagnostic snapshots')
    parser.add_argument('--diagnostics-split', choices=["train", "val", "test"], default="train", help='Dataset split used for diagnostic analysis')
    parser.add_argument('--no-epoch-diagnostics', action='store_true', help='Disable per-epoch diagnostic snapshots')
    parser.add_argument('--metrics-dir', default="outputs/metrics", help='Directory used for metric and diagnostic CSV files')
    parser.add_argument('--model-dir', default="outputs/model_weights", help='Directory used for final trained model weights')
    parser.add_argument('--checkpoint-dir', default="outputs/checkpoints", help='Directory used for resumable training checkpoints')
    parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint for the selected model')
    parser.add_argument('--resume-checkpoint', default=None, help='Optional explicit checkpoint path to resume from')
    args = parser.parse_args()

    config = Config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.run1:
        print("Running Late Fusion Transformer (Baseline 1)")
        run_late_fusion_experiment(config, args.data, args)
    if args.run2:
        print("Running Cross Attention Model (Baseline 2)")
        run_cross_modal_experiment(config, args.data, args)
    if args.run3:
        print("Running Orthogonality Model (Improved 3)")
        run_orthogonality_experiment(config, args.data, args)
    if args.run4:
        print("Running Auxiliary Heads Model (Improved 4)")
        run_auxiliary_experiment(config, args.data, args)
