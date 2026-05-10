import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from m3sentiment.config import Config
from m3sentiment.dataset import MoseiDataset
from m3sentiment.models.ortho_fusion import OrthoFusionTransformer


LABEL_NAMES = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


def load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_datasets(data_path, split):
    train_dataset = MoseiDataset(data_path, split="train")
    if split == "train":
        return train_dataset
    return MoseiDataset(data_path, split=split, stats=train_dataset.stats)


def main():
    parser = argparse.ArgumentParser(
        description="Run sentiment inference on one processed CMU-MOSEI item."
    )
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--data", default="data/aligned_mosei_dataset.pkl")
    parser.add_argument("--model-path", default="outputs/model_weights/ortho_fusion.pth")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    config = Config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = build_datasets(args.data, args.split)
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"Index {args.index} is outside the {args.split} split with {len(dataset)} items.")

    sample = dataset[args.index]
    text = sample["text"].unsqueeze(0).to(device)
    audio = sample["audio"].unsqueeze(0).to(device)
    vision = sample["vision"].unsqueeze(0).to(device)
    actual_label = int(sample["label3"].item())

    model = OrthoFusionTransformer(
        text.shape[-1],
        audio.shape[-1],
        vision.shape[-1],
        hidden_dim=config.model["hidden_dim"],
        n_heads=config.model["n_heads"],
        n_layers=config.model["n_layers"],
        dropout=config.model["dropout"],
    ).to(device)

    model.load_state_dict(load_state_dict(args.model_path, device))
    model.eval()

    with torch.no_grad():
        model_output = model(text, audio, vision)
        logits = model_output[0] if isinstance(model_output, tuple) else model_output
        probabilities = F.softmax(logits, dim=1)[0]
        predicted_label = int(torch.argmax(probabilities).item())

    print(f"Split: {args.split}")
    print(f"Item index: {args.index}")
    print(f"Predicted sentiment: {LABEL_NAMES[predicted_label]}")
    print(f"Actual sentiment: {LABEL_NAMES[actual_label]}")
    print("Confidence scores:")
    for label_id, score in enumerate(probabilities.tolist()):
        print(f"  {LABEL_NAMES[label_id]}: {score:.4f}")


if __name__ == "__main__":
    main()
