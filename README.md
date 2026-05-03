# M3Sentiment

## Overview

This project trains and analyzes transformer-based multimodal sentiment models on CMU-MOSEI. It compares late fusion, cross-modal attention, orthogonality regularization, and auxiliary-supervision variants while also tracking what the models learn internally.

The project focuses on two questions:

- Which model performs best for 3-way sentiment classification?
- How do text, audio, and vision representations, losses, and attention patterns change during training?

## Model Variants

- **Baseline 1: Late Fusion Transformer** combines separately encoded text, audio, and vision representations with a final fusion transformer.
- **Baseline 2: Cross-Modal Attention** lets each modality attend to the other modalities before classification.
- **Improved 1: Orthogonality Loss** adds a penalty that pushes text/audio/vision representations to become less redundant.
- **Improved 2: Auxiliary Heads** adds text-only, audio-only, and vision-only auxiliary classifiers during training.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For Colab, install requirements from the project root:

```bash
!pip install -r requirements.txt
```

## Dataset

The expected processed dataset path is:

```text
data/aligned_mosei_dataset.pkl
```

To download the prepared dataset with the included helper:

```bash
python data/scripts/get_data.py
```

If you download the file manually, place it at:

```text
data/aligned_mosei_dataset.pkl
```

The CMU-MultimodalSDK processing utilities are under:

```text
data/scripts/
```

## Training

Run commands from the project root.

Baseline 1:

```bash
python scripts/train_models.py --run1
```

Baseline 2:

```bash
python scripts/train_models.py --run2
```

Orthogonality model:

```bash
python scripts/train_models.py --run3
```

Auxiliary-heads model:

```bash
python scripts/train_models.py --run4
```

All models:

```bash
python scripts/train_models.py --run1 --run2 --run3 --run4
```

## Diagnostics

Diagnostics run by default on the training split. They include:

- modality-only diagnostic losses: `text_only_loss`, `audio_only_loss`, `vision_only_loss`
- within-modality self-attention summaries
- Baseline 1 fusion attention
- cross-modal attention for Baseline 2, Orthogonality, and Auxiliary models
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
│           ├── cross_modal_attention.py
│           ├── orthogonality.py
│           └── auxiliary_heads.py
├── requirements.txt
└── README.md
```
