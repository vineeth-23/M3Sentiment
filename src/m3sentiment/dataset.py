import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import os


class MoseiDataset(Dataset):
    """
    A PyTorch Dataset class for loading and preprocessing the CMU-MOSEI dataset.

    Args:
        pkl_path (str): Path to the pickle file containing the dataset.
        split (str): Dataset split to load ("train", "valid", or "test").
        stats (dict, optional): Precomputed statistics (mean and std) for normalization.
                                Required for "valid" and "test" splits.
    """
    def __init__(self, pkl_path, split="train", stats=None):
        # Check if the pickle file exists
        if not os.path.isfile(pkl_path):
            raise FileNotFoundError(f"File not found: {pkl_path}")

        # Load the dataset from the pickle file
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        sd = data[split]  # Get the split-specific data

        # Extract modalities and labels
        text_np = sd["text"]    # Text data: shape (N, 50, 786)
        audio_np = sd["audio"]  # Audio data: shape (N, 50, 74), may contain ±inf
        vision_np = sd["vision"]  # Vision data: shape (N, 50, 35)
        labels_np = sd['classification_labels'].astype(np.int64)  # Labels: shape (N,)

        # Clean ±inf values in audio data
        audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute statistics (mean and std) for normalization if in the "train" split
        if split == "train":
            stats = {
                "t_mean": text_np.mean((0, 1)),  # Mean of text data
                "t_std": text_np.std((0, 1)) + 1e-6,  # Std of text data
                "a_mean": audio_np.mean((0, 1)),  # Mean of audio data
                "a_std": audio_np.std((0, 1)) + 1e-6,  # Std of audio data
                "v_mean": vision_np.mean((0, 1)),  # Mean of vision data
                "v_std": vision_np.std((0, 1)) + 1e-6,  # Std of vision data
            }
        elif stats is None:
            # Raise an error if stats are not provided for "valid" or "test" splits
            raise RuntimeError("Need train-split stats for valid/test splits")

        # Normalize each modality using z-score normalization
        self.X_text = (text_np - stats["t_mean"]) / stats["t_std"]
        self.X_audio = (audio_np - stats["a_mean"]) / stats["a_std"]
        self.X_vision = (vision_np - stats["v_mean"]) / stats["v_std"]
        self.y3 = labels_np  # Classification labels
        self.stats = stats  # Store stats for reference

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.y3)

    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - "text": Normalized text data (torch.Tensor)
                - "audio": Normalized audio data (torch.Tensor)
                - "vision": Normalized vision data (torch.Tensor)
                - "label3": Classification label (torch.Tensor)
        """
        return {
            "text": torch.from_numpy(self.X_text[idx]).float(),
            "audio": torch.from_numpy(self.X_audio[idx]).float(),
            "vision": torch.from_numpy(self.X_vision[idx]).float(),
            "label3": torch.tensor(self.y3[idx], dtype=torch.long),
        }