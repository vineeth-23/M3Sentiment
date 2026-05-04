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
from m3sentiment.models.cross_modal import CrossModalFusionTransformer
from m3sentiment.models.ortho_fusion import OrthoFusionTransformer
from m3sentiment.models.aux_fusion import AuxFusionTransformer
from m3sentiment.training import (
    train_aux_fusion_epoch,
    train_ortho_fusion_epoch,
    train_standard_epoch,
)
from m3sentiment.evaluation import (
    collect_confusion_matrix,
    confusion_matrix_rows,
    evaluate_aux_fusion_epoch,
    evaluate_ortho_fusion_epoch,
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



def render_confusion_matrix_svg(confusion_matrix, out_path, title, class_names=None):
    class_names = class_names or ["negative", "neutral", "positive"]
    width, height = 720, 620
    left, top = 190, 120
    cell_size = 115
    max_count = max((count for row in confusion_matrix for count in row), default=1) or 1

    def svg_text(x, y, text, size=13, weight="400", anchor="middle", fill="#111827"):
        text = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return (
            f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}" fill="{fill}">{text}</text>'
        )

    body = [
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 40, title, size=22, weight="700"),
        svg_text(left + cell_size * 1.5, 82, "Predicted label", size=15, weight="700"),
        svg_text(35, top + cell_size * 1.5, "Actual label", size=15, weight="700", anchor="start"),
    ]

    for idx, class_name in enumerate(class_names):
        body.append(svg_text(left + idx * cell_size + cell_size / 2, top - 22, class_name, size=13, weight="700"))
        body.append(svg_text(left - 18, top + idx * cell_size + cell_size / 2 + 5, class_name, size=13, weight="700", anchor="end"))

    for actual_idx, row in enumerate(confusion_matrix):
        row_total = sum(row)
        for predicted_idx, count in enumerate(row):
            intensity = count / max_count
            red = int(239 - intensity * 170)
            green = int(246 - intensity * 170)
            blue = 255
            fill = f"rgb({red},{green},{blue})"
            x = left + predicted_idx * cell_size
            y = top + actual_idx * cell_size
            percent = count / row_total if row_total else 0.0
            body.append(f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" fill="{fill}" stroke="#ffffff" stroke-width="3"/>')
            body.append(svg_text(x + cell_size / 2, y + cell_size / 2 - 8, count, size=22, weight="700"))
            body.append(svg_text(x + cell_size / 2, y + cell_size / 2 + 20, f"{percent:.1%}", size=12, fill="#374151"))

    body.append(svg_text(left, height - 60, "Each row sums across predictions for one true class; percentages are row-normalized.", size=12, anchor="start", fill="#4b5563"))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as svg_file:
        svg_file.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" font-family="Arial, sans-serif">\n'
            + "\n".join(body)
            + "\n</svg>\n"
        )


def save_test_confusion_matrix(model, test_loader, device, args, model_name, display_name):
    class_names = ["negative", "neutral", "positive"]
    matrix = collect_confusion_matrix(model, test_loader, device, num_classes=len(class_names))
    rows = confusion_matrix_rows(matrix, class_names=class_names)

    csv_path = os.path.join(args.metrics_dir, "confusion_matrices", f"{model_name}_test_confusion_matrix.csv")
    svg_path = os.path.join(args.plot_dir, f"{model_name}_test_confusion_matrix.svg")
    write_metrics_csv(rows, csv_path)
    render_confusion_matrix_svg(matrix, svg_path, f"{display_name} Test Confusion Matrix", class_names)
    return csv_path, svg_path

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

# Run the Late Fusion Transformer model (Late Fusion).
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
    diagnostics = create_diagnostic_recorder(args, "late_fusion", diagnostics_loader, device)

    epoch_metrics = []
    batch_metrics = []
    start_epoch, epoch_metrics, batch_metrics = restore_training_state_if_requested(
        args, "late_fusion", model, optimizer, scheduler, diagnostics, device
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
        write_metrics_csv(epoch_metrics, os.path.join(args.metrics_dir, "late_fusion_metrics.csv"))
        write_metrics_csv(batch_metrics, os.path.join(args.metrics_dir, "late_fusion_batch_metrics.csv"))
        checkpoint = save_epoch_checkpoint(
            args, "late_fusion", epoch, model, optimizer, scheduler, epoch_metrics, batch_metrics, diagnostics
        )
        print(f"Epoch {epoch:02d}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  checkpoint={checkpoint}")

    # Final evaluation and save results
    test_loss, test_acc = evaluate_standard_epoch(model, test_loader, criterion, device)
    print(f"\nTest ▶ loss={test_loss:.4f} acc={test_acc:.4f}")
    save_test_confusion_matrix(model, test_loader, device, args, "late_fusion", "Late Fusion")
    if diagnostics is not None:
        diagnostics.finalize(model)
    save_final_model_weights(model, os.path.join(args.model_dir, "late_fusion.pth"))
    write_metrics_csv(epoch_metrics, os.path.join(args.metrics_dir, "late_fusion_metrics.csv"))
    write_metrics_csv(batch_metrics, os.path.join(args.metrics_dir, "late_fusion_batch_metrics.csv"))

# Run the Late Fusion with Cross-Modal Attention model (Cross-Modal Fusion).
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
    model = CrossModalFusionTransformer(
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
    diagnostics = create_diagnostic_recorder(args, "cross_modal", diagnostics_loader, device)

    epoch_metrics = []
    batch_metrics = []
    start_epoch, epoch_metrics, batch_metrics = restore_training_state_if_requested(
        args, "cross_modal", model, optimizer, scheduler, diagnostics, device
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
        write_metrics_csv(epoch_metrics, os.path.join(args.metrics_dir, "cross_modal_metrics.csv"))
        write_metrics_csv(batch_metrics, os.path.join(args.metrics_dir, "cross_modal_batch_metrics.csv"))
        checkpoint = save_epoch_checkpoint(
            args, "cross_modal", epoch, model, optimizer, scheduler, epoch_metrics, batch_metrics, diagnostics
        )
        print(f"Epoch {epoch:02d}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  checkpoint={checkpoint}")

    # Final evaluation and save results
    test_loss, test_acc = evaluate_standard_epoch(model, test_loader, criterion, device)
    print(f"\nTest ▶ loss={test_loss:.4f} acc={test_acc:.4f}")
    save_test_confusion_matrix(model, test_loader, device, args, "cross_modal", "Cross-Modal Fusion")
    if diagnostics is not None:
        diagnostics.finalize(model)
    save_final_model_weights(model, os.path.join(args.model_dir, "cross_modal.pth"))
    write_metrics_csv(epoch_metrics, os.path.join(args.metrics_dir, "cross_modal_metrics.csv"))
    write_metrics_csv(batch_metrics, os.path.join(args.metrics_dir, "cross_modal_batch_metrics.csv"))

# Run the Late Fusion with Ortho Fusion model (Improved 1).
def run_ortho_fusion_experiment(config, dataset_path, args):
    train_loader, val_loader, test_loader = build_mosei_dataloaders(dataset_path, config.train["batch_size"])

    # Infer input dimensions
    sample = next(iter(train_loader))
    text_dim, audio_dim, vision_dim = (
        sample["text"].shape[-1],
        sample["audio"].shape[-1],
        sample["vision"].shape[-1]
    )

    model = OrthoFusionTransformer(
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
    diagnostics = create_diagnostic_recorder(args, "ortho_fusion", diagnostics_loader, device)

    epoch_metrics = []
    batch_metrics = []
    start_epoch, epoch_metrics, batch_metrics = restore_training_state_if_requested(
        args, "ortho_fusion", model, optimizer, scheduler, diagnostics, device
    )
    for epoch in range(start_epoch, config.train["epochs"] + 1):
        # Train and evaluate
        train_loss, train_acc, classification_loss, raw_ortho_loss, weighted_ortho_loss = train_ortho_fusion_epoch(
            model, train_loader, optimizer, criterion, device, ortho_weight, max_grad_norm,
            diagnostics=diagnostics, epoch=epoch, batch_metrics=batch_metrics
        )
        val_loss, val_acc = evaluate_ortho_fusion_epoch(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate_ortho_fusion_epoch(model, test_loader, criterion, device)
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
        write_metrics_csv(epoch_metrics, os.path.join(args.metrics_dir, "ortho_fusion_metrics.csv"))
        write_metrics_csv(batch_metrics, os.path.join(args.metrics_dir, "ortho_fusion_batch_metrics.csv"))
        checkpoint = save_epoch_checkpoint(
            args, "ortho_fusion", epoch, model, optimizer, scheduler, epoch_metrics, batch_metrics, diagnostics
        )
        print(f"Epoch {epoch:02d}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  checkpoint={checkpoint}")

    # Final evaluation and save results
    test_loss, test_acc = evaluate_ortho_fusion_epoch(model, test_loader, criterion, device)
    print(f"\nTest ▶ loss={test_loss:.4f} acc={test_acc:.4f}")
    save_test_confusion_matrix(model, test_loader, device, args, "ortho_fusion", "Ortho Fusion")
    if diagnostics is not None:
        diagnostics.finalize(model)
    save_final_model_weights(model, os.path.join(args.model_dir, "ortho_fusion.pth"))
    write_metrics_csv(epoch_metrics, os.path.join(args.metrics_dir, "ortho_fusion_metrics.csv"))
    write_metrics_csv(batch_metrics, os.path.join(args.metrics_dir, "ortho_fusion_batch_metrics.csv"))

# Run the Late Fusion with Aux Fusion model (Improved 2).
def run_aux_fusion_experiment(config, dataset_path, args):
    train_loader, val_loader, test_loader = build_mosei_dataloaders(dataset_path, config.train["batch_size"])

    # Infer input dimensions
    sample = next(iter(train_loader))
    text_dim, audio_dim, vision_dim = (
        sample["text"].shape[-1],
        sample["audio"].shape[-1],
        sample["vision"].shape[-1]
    )

    model = AuxFusionTransformer(
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
    diagnostics = create_diagnostic_recorder(args, "aux_fusion", diagnostics_loader, device)

    epoch_metrics = []
    batch_metrics = []
    start_epoch, epoch_metrics, batch_metrics = restore_training_state_if_requested(
        args, "aux_fusion", model, optimizer, scheduler, diagnostics, device
    )
    for epoch in range(start_epoch, config.train["epochs"] + 1):
        # Train and evaluate
        train_loss, train_acc, main_loss, text_aux_loss, audio_aux_loss, vision_aux_loss, weighted_aux_loss = train_aux_fusion_epoch(
            model, train_loader, optimizer, criterion, device, config.train["aux_weight"], config.train["max_grad_norm"],
            diagnostics=diagnostics, epoch=epoch, batch_metrics=batch_metrics
        )
        val_loss, val_acc = evaluate_aux_fusion_epoch(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate_aux_fusion_epoch(model, test_loader, criterion, device)
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
        write_metrics_csv(epoch_metrics, os.path.join(args.metrics_dir, "aux_fusion_metrics.csv"))
        write_metrics_csv(batch_metrics, os.path.join(args.metrics_dir, "aux_fusion_batch_metrics.csv"))
        checkpoint = save_epoch_checkpoint(
            args, "aux_fusion", epoch, model, optimizer, scheduler, epoch_metrics, batch_metrics, diagnostics
        )
        print(f"Epoch {epoch:02d}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  checkpoint={checkpoint}")

    # Final evaluation and save results
    test_loss, test_acc = evaluate_aux_fusion_epoch(model, test_loader, criterion, device)
    print(f"\nTest ▶ loss={test_loss:.4f} acc={test_acc:.4f}")
    save_test_confusion_matrix(model, test_loader, device, args, "aux_fusion", "Aux Fusion")
    if diagnostics is not None:
        diagnostics.finalize(model)
    save_final_model_weights(model, os.path.join(args.model_dir, "aux_fusion.pth"))
    write_metrics_csv(epoch_metrics, os.path.join(args.metrics_dir, "aux_fusion_metrics.csv"))
    write_metrics_csv(batch_metrics, os.path.join(args.metrics_dir, "aux_fusion_batch_metrics.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--data", default="data/aligned_mosei_dataset.pkl")
    parser.add_argument('--run1', action='store_true', help='Run the Late Fusion model (late_fusion)')
    parser.add_argument('--run2', action='store_true', help='Run the Cross Attention model (cross_modal)')
    parser.add_argument('--run3', action='store_true', help='Run the Ortho Fusion model (ortho_fusion)')
    parser.add_argument('--run4', action='store_true', help='Run the Aux Fusion model (aux_fusion)')
    parser.add_argument('--no-diagnostics', action='store_true', help='Disable representation and cross-attention diagnostics')
    parser.add_argument('--analysis-max-batches', type=int, default=5, help='Number of batches used for training-stage diagnostic snapshots')
    parser.add_argument('--analysis-snapshot-batches', default="1,5,10,25,50,100,250,500,1000", help='Comma-separated global batch steps for diagnostic snapshots')
    parser.add_argument('--diagnostics-split', choices=["train", "val", "test"], default="train", help='Dataset split used for diagnostic analysis')
    parser.add_argument('--no-epoch-diagnostics', action='store_true', help='Disable per-epoch diagnostic snapshots')
    parser.add_argument('--metrics-dir', default="outputs/metrics", help='Directory used for metric and diagnostic CSV files')
    parser.add_argument('--model-dir', default="outputs/model_weights", help='Directory used for final trained model weights')
    parser.add_argument('--checkpoint-dir', default="outputs/checkpoints", help='Directory used for resumable training checkpoints')
    parser.add_argument('--plot-dir', default="outputs/plots/diagnostics", help='Directory used for generated diagnostic SVG plots')
    parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint for the selected model')
    parser.add_argument('--resume-checkpoint', default=None, help='Optional explicit checkpoint path to resume from')
    args = parser.parse_args()

    config = Config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.run1:
        print("Running Late Fusion Transformer (Late Fusion)")
        run_late_fusion_experiment(config, args.data, args)
    if args.run2:
        print("Running Cross Attention Model (Cross-Modal Fusion)")
        run_cross_modal_experiment(config, args.data, args)
    if args.run3:
        print("Running Ortho Fusion Model")
        run_ortho_fusion_experiment(config, args.data, args)
    if args.run4:
        print("Running Aux Fusion Model")
        run_aux_fusion_experiment(config, args.data, args)
