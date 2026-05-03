from torch.utils.data import DataLoader
from m3sentiment.dataset import MoseiDataset

def get_data_loaders(dataset_path, batch_size):
    # Build train dataset to compute stats
    train_ds = MoseiDataset(dataset_path, split="train", stats=None)
    stats = train_ds.stats

    # Build validation and test datasets using train stats
    val_ds = MoseiDataset(dataset_path, split="valid", stats=stats)
    test_ds = MoseiDataset(dataset_path, split="test", stats=stats)

    # Return DataLoaders
    return (
        DataLoader(train_ds, batch_size, shuffle=True),
        DataLoader(val_ds, batch_size),
        DataLoader(test_ds, batch_size)
    )
