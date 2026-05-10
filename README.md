# M3Sentiment

## Overview

This project trains and analyzes transformer-based multimodal sentiment models on CMU-MOSEI. It compares late fusion, cross-modal attention, orthogonality regularization, and auxiliary-supervision variants while also tracking what the models learn internally.

The project focuses on two questions:

- Which model performs best for 3-way sentiment classification?
- How do text, audio, and vision representations, losses, and attention patterns change during training?

The sentiment classes are:

- `0`: negative
- `1`: neutral
- `2`: positive

## Quickstart

From the project root:

```bash
pip install -r requirements.txt
python data/scripts/get_data.py
python scripts/train_models.py --run1 --run2 --run3 --run4
python scripts/plot_diagnostics.py
python scripts/demo_predict.py --split test --index 0
```

## Model Variants

- **Late Fusion** combines separately encoded text, audio, and vision representations with a final fusion transformer.
- **Cross-Modal Fusion** lets each modality attend to the other modalities before classification.
- **Ortho Fusion** adds an orthogonality penalty that pushes text/audio/vision representations to become less redundant.
- **Aux Fusion** adds text-only, audio-only, and vision-only auxiliary classifiers during training.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Major Software Requirements

The main software requirements are listed in `requirements.txt`:

- Python 3
- PyTorch (`torch`)
- NumPy (`numpy`)
- pandas (`pandas`)
- Matplotlib (`matplotlib`)
- Hugging Face Transformers (`transformers`)
- gdown (`gdown`)

Training can run on CPU, but a CUDA-enabled GPU is recommended for Colab or longer local runs. In Colab, use `!nvidia-smi` to confirm that a GPU is available.

For Colab, install requirements from the project root:

```bash
!pip install -r requirements.txt
```

## Google Colab Workflow

Mount Google Drive and move into the project folder:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/M3Sentiment
!ls
```

Install dependencies:

```bash
!pip install -r requirements.txt
!pip install gdown
```

Download and verify the prepared dataset:

```bash
!python data/scripts/get_data.py
!ls -lh data/aligned_mosei_dataset.pkl
```

Create output directories and confirm GPU availability:

```bash
!mkdir -p outputs/metrics outputs/model_weights outputs/checkpoints outputs/plots/diagnostics
!nvidia-smi
```

Train the four model variants:

```bash
!python scripts/train_models.py --run1
!python scripts/train_models.py --run2
!python scripts/train_models.py --run3
!python scripts/train_models.py --run4
```

Generate diagnostic plots:

```bash
!python scripts/plot_diagnostics.py
```

Check generated artifacts:

```bash
!ls -lh outputs/metrics
!ls -lh outputs/model_weights
!ls -lh outputs/checkpoints
!ls -lh outputs/plots/diagnostics | head
!ls -lh outputs/metrics/confusion_matrices
!ls -lh outputs/plots/diagnostics/*confusion_matrix.svg
```

## Dataset

The expected processed dataset path is:

```text
data/aligned_mosei_dataset.pkl
```

The project trains on a prepared aligned CMU-MOSEI pickle file. The dataset download link is stored in `data/scripts/get_data.py`, and the helper downloads the file into the expected path.

To download the prepared dataset with the included helper:

```bash
python data/scripts/get_data.py
```

If you download the file manually, place it at:

```text
data/aligned_mosei_dataset.pkl
```

The processed pickle should contain `train`, `valid`, and `test` splits. Each split should contain aligned `text`, `audio`, `vision`, and `classification_labels` arrays. The training loader normalizes all modalities using training-split statistics and reuses those statistics for validation and test splits.

The CMU-MultimodalSDK processing utilities are under:

```text
data/scripts/
```

## Trained Models

The trained model weights from completed runs are saved under:

```text
outputs/model_weights/
```

Expected trained model files:

- `outputs/model_weights/late_fusion.pth`
- `outputs/model_weights/cross_modal.pth`
- `outputs/model_weights/ortho_fusion.pth`
- `outputs/model_weights/aux_fusion.pth`

Resumable training checkpoints are saved under:

```text
outputs/checkpoints/
```

The trained models were trained on the prepared dataset at:

```text
data/aligned_mosei_dataset.pkl
```

Use `python data/scripts/get_data.py` to download the same prepared dataset file.

## Training

Run commands from the project root.

Late Fusion:

```bash
python scripts/train_models.py --run1
```

Cross-Modal Fusion:

```bash
python scripts/train_models.py --run2
```

Ortho Fusion model:

```bash
python scripts/train_models.py --run3
```

Aux Fusion model:

```bash
python scripts/train_models.py --run4
```

All models:

```bash
python scripts/train_models.py --run1 --run2 --run3 --run4
```

Default training settings are in `configs/default.json`. The current default run uses batch size `64`, learning rate `2e-5`, `40` epochs, hidden size `128`, `4` attention heads, `2` transformer layers, and dropout `0.1`.

## Diagnostics

Diagnostics run by default on the training split. They include:

- modality-only diagnostic losses: `text_only_loss`, `audio_only_loss`, `vision_only_loss`
- within-modality self-attention summaries
- Late Fusion fusion attention
- cross-modal attention for Cross-Modal Fusion, Ortho Fusion, and Aux Fusion models
- orthogonality/cosine-similarity measurements between text/audio/vision representations
- batch snapshots at `1, 5, 10, 25, 50, 100, 250, 500, 1000`
- epoch-level snapshots
- final trained-model snapshots

Disable diagnostics:

```bash
python scripts/train_models.py --run3 --no-diagnostics
```

Use a different diagnostics split:

```bash
python scripts/train_models.py --run3 --diagnostics-split test
```

Disable per-epoch diagnostic snapshots for a faster run:

```bash
python scripts/train_models.py --run3 --no-epoch-diagnostics
```

## Resume Training

Training saves resumable checkpoints after every epoch in:

```text
outputs/checkpoints/
```

If Colab disconnects, resume the same model:

```bash
python scripts/train_models.py --run3 --resume
```

## Plotting

Generate SVG visualizations from saved CSVs:

```bash
python scripts/plot_diagnostics.py
```

Plots are written to:

```text
outputs/plots/diagnostics/
```

## Inference Demo

Run prediction on one processed dataset item using the trained Ortho Fusion model:

```bash
python scripts/demo_predict.py --split test --index 0
```

Example with explicit paths:

```bash
python scripts/demo_predict.py \
  --data data/aligned_mosei_dataset.pkl \
  --model-path outputs/model_weights/ortho_fusion.pth \
  --split test \
  --index 25
```

This demo expects processed CMU-MOSEI feature tensors. It does not run directly on a raw video file.

## Configuration

The default config is:

```text
configs/default.json
```

Use a custom config:

```bash
python scripts/train_models.py --config configs/my_experiment.json --run1
```

## Outputs

- Final trained model weights: `outputs/model_weights/`
- Resume checkpoints: `outputs/checkpoints/`
- Epoch and batch metrics: `outputs/metrics/`
- Diagnostic CSVs: `outputs/metrics/diagnostics/`
- SVG plots: `outputs/plots/diagnostics/`

Important presentation plots:

- Test accuracy comparison: `outputs/plots/diagnostics/test_accuracy_by_model.svg`
- Validation accuracy comparison: `outputs/plots/diagnostics/validation_accuracy_by_model.svg`
- Test loss comparison: `outputs/plots/diagnostics/test_loss_by_model.svg`
- Ortho Fusion confusion matrix: `outputs/plots/diagnostics/ortho_fusion_test_confusion_matrix.svg`
- Ortho Fusion orthogonality trend: `outputs/plots/diagnostics/ortho_fusion_orthogonality_over_training.svg`
- Ortho Fusion attention heatmap: `outputs/plots/diagnostics/ortho_fusion_final_head_attention_heatmap.svg`

## Results Summary

In the saved experiment outputs, Ortho Fusion achieved the strongest test performance:

| Model | Final Test Accuracy | Best Test Accuracy |
|---|---:|---:|
| Late Fusion | 67.03% | 67.44% |
| Cross-Modal Fusion | 67.40% | 67.93% |
| Ortho Fusion | 68.02% | 68.17% |
| Aux Fusion | 67.10% | 67.50% |

The main findings are that Ortho Fusion performs best overall, text is the strongest individual modality, and neutral sentiment is the hardest class to classify.

## Project Structure

```text
M3Sentiment/
├── configs/
│   └── default.json
├── data/
│   ├── aligned_mosei_dataset.pkl
│   └── scripts/
│       ├── get_data.py
│       ├── process_mosei.py
│       └── export_from_csd.py
├── outputs/
│   ├── checkpoints/
│   ├── metrics/
│   │   └── diagnostics/
│   ├── model_weights/
│   └── plots/
│       └── diagnostics/
├── scripts/
│   ├── demo_predict.py
│   ├── train_models.py
│   └── plot_diagnostics.py
├── src/
│   └── m3sentiment/
│       ├── attention_layers.py
│       ├── config.py
│       ├── data_loaders.py
│       ├── dataset.py
│       ├── diagnostics.py
│       ├── evaluation.py
│       ├── training.py
│       └── models/
│           ├── late_fusion.py
│           ├── cross_modal.py
│           ├── ortho_fusion.py
│           └── aux_fusion.py
├── requirements.txt
└── README.md
```
