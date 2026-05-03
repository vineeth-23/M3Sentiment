from torch.utils.data import DataLoader

from m3sentiment.dataset import MoseiDataset


def build_mosei_dataloaders(dataset_path, batch_size):
    train_dataset = MoseiDataset(dataset_path, split="train", stats=None)
    normalization_stats = train_dataset.stats

    validation_dataset = MoseiDataset(dataset_path, split="valid", stats=normalization_stats)
    test_dataset = MoseiDataset(dataset_path, split="test", stats=normalization_stats)

    return (
        DataLoader(train_dataset, batch_size, shuffle=True),
        DataLoader(validation_dataset, batch_size),
        DataLoader(test_dataset, batch_size),
    )


# Backward-compatible alias for older notebooks or scripts.
get_data_loaders = build_mosei_dataloaders
