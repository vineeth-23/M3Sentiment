import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class MoseiDataset(Dataset):
    """Loads one normalized CMU-MOSEI split from the aligned pickle file."""

    def __init__(self, pkl_path, split="train", stats=None):
        if not os.path.isfile(pkl_path):
            raise FileNotFoundError(f"File not found: {pkl_path}")

        with open(pkl_path, "rb") as dataset_file:
            raw_dataset = pickle.load(dataset_file)
        split_data = raw_dataset[split]

        text_features = split_data["text"]
        audio_features = np.nan_to_num(split_data["audio"], nan=0.0, posinf=0.0, neginf=0.0)
        vision_features = split_data["vision"]
        sentiment_labels = split_data["classification_labels"].astype(np.int64)

        if split == "train":
            stats = {
                "t_mean": text_features.mean((0, 1)),
                "t_std": text_features.std((0, 1)) + 1e-6,
                "a_mean": audio_features.mean((0, 1)),
                "a_std": audio_features.std((0, 1)) + 1e-6,
                "v_mean": vision_features.mean((0, 1)),
                "v_std": vision_features.std((0, 1)) + 1e-6,
            }
        elif stats is None:
            raise RuntimeError("Need train-split stats for valid/test splits")

        self.text_features = (text_features - stats["t_mean"]) / stats["t_std"]
        self.audio_features = (audio_features - stats["a_mean"]) / stats["a_std"]
        self.vision_features = (vision_features - stats["v_mean"]) / stats["v_std"]
        self.sentiment_labels = sentiment_labels
        self.stats = stats

    def __len__(self):
        return len(self.sentiment_labels)

    def __getitem__(self, index):
        return {
            "text": torch.from_numpy(self.text_features[index]).float(),
            "audio": torch.from_numpy(self.audio_features[index]).float(),
            "vision": torch.from_numpy(self.vision_features[index]).float(),
            "label3": torch.tensor(self.sentiment_labels[index], dtype=torch.long),
        }
