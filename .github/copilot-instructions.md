# Copilot Instructions for DLF-Lite

- This repo is a PyTorch-based multimodal classification project built around the CMU-MOSEI dataset.
- The main execution entrypoint is `run.py`; it selects one or more models to train using `--run1`, `--run2`, `--run3`, `--run4`.
- `config.json` is the only configuration source used by `Config` in `config.py`. Any new hyperparameter should be added there and accessed via `cfg.train[...]` or `cfg.model[...]`.
- Data loading is centralized in `utils.py` and `dataset.py`: `MoseiDataset` normalizes text/audio/vision using train-split statistics, and valid/test splits must be built with the same stats.

## Architecture and model structure

- There are four model variants:
  - `model_baseline1.py`: `LateFusionTransformer` encodes each modality separately, pools each sequence, then fuses 3 modality embeddings through a small transformer.
  - `model_baseline2.py`: `LateFusionWithCrossModal` uses per-modality transformer encoders plus cross-modal attention between modality embeddings.
  - `improved_ortho.py`: same as baseline2 but returns `(logits, t_feat, a_feat, v_feat)` for orthogonality regularization in `train.py`.
  - `improved_aux.py`: same as baseline2 but adds auxiliary classifiers for each unimodal representation and combines auxiliary losses in `train_epoch_aux`.
- Models all expect input shapes `(B, 50, D_text/audio/vision)` and output 3-class logits.

## Project conventions

- Training and evaluation are separated into `train.py` and `eval.py`. `run.py` orchestrates model instantiation, optimizer/scheduler creation, and CSV logging.
- The saved checkpoints are hard-coded to `models/<modelname>.pth`; metrics CSV files are saved to `csv/<modelname>_metrics.csv`.
- There is no automated test framework in the repo, so preserve the existing command patterns and do lightweight manual validation via the `run.py` command line.
- The dataset file path default is `data/aligned_mosei_dataset.pkl`; if missing, code will raise `FileNotFoundError` from `dataset.py`.

## Useful patterns for changes

- Use `nn.TransformerEncoderLayer` and `nn.TransformerEncoder` for sequence encoding in all model files.
- Cross-modal attention is implemented via `nn.MultiheadAttention` with `batch_first=True` and the query shape `(B, 1, H)`.
- For improved model variants, keep the main logits separate from auxiliary outputs and apply the auxiliary loss only in `train_epoch_aux`.
- `train.py` uses `clip_grad_norm_` consistently; preserve this if adding new training variants.

## Commands to run

- Install dependencies: `pip install -r requirements.txt`
- Train Baseline 1: `python run.py --run1`
- Train Baseline 2: `python run.py --run2`
- Train Improved Orthogonality: `python run.py --run3`
- Train Improved Auxiliary Heads: `python run.py --run4`
- Train all models: `python run.py --run1 --run2 --run3 --run4`

## Important notes

- `run.py` chooses device via `torch.cuda.is_available()`.
- `config.json` is required and must contain both `train` and `model` keys.
- The code assumes `models/` and `csv/` directories exist before saving; these are not created automatically.
