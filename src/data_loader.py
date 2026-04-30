import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from .utils import get_logger

logger = get_logger(__name__)

class LunarPitDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = np.array(labels)
        self.transform = transform

        self.pit_indices = np.where(self.labels == 1)[0]
        self.neg_indices = np.where(self.labels == 0)[0]

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        npy_path = self.file_paths[idx]
        image_array = np.load(npy_path)

        tensor = torch.from_numpy(image_array).float().unsqueeze(0)
        tensor = tensor.repeat(3, 1, 1)

        if tensor.max() > 1.0:
            tensor /= 255.0

        if self.transform:
            tensor = self.transform(tensor)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor, label
    

class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_pits_per_batch = batch_size // 2
        self.n_negs_per_batch = batch_size - self.n_pits_per_batch
        self.n_batches = len(dataset.pit_indices) // self.n_pits_per_batch

    def __iter__(self):
        for _ in range(self.n_batches):
            p_indices = np.random.choice(self.dataset.pit_indices, self.n_pits_per_batch, replace=False)
            n_indices = np.random.choice(self.dataset.neg_indices, self.n_negs_per_batch, replace=False)

            batch = np.concatenate([p_indices, n_indices])
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches

def get_dataloaders(data_dir="data/processed/dataset", batch_size=32, val_split=0.2, transform=None):
    base_path = Path(data_dir)

    pit_files = list((base_path / "pits").glob("*.npy"))
    neg_files = list((base_path / "negatives").glob("*.npy"))

    all_files = pit_files + neg_files
    all_labels = [1] * len(pit_files) + [0] * len(neg_files)
    logger.info(f"Found: {len(pit_files)} Pits and {len(neg_files)} Negatives")

    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files, all_labels,
        test_size=val_split,
        stratify=all_labels,
        random_state=67
    )

    logger.info(f"Split: {len(train_files)} Train-Images, {len(val_files)} Val-Images")

    train_dataset = LunarPitDataset(train_files, train_labels, transform=transform)
    val_dataset = LunarPitDataset(val_files, val_labels, transform=transform)

    train_sampler = BalancedBatchSampler(train_dataset, batch_size)
    train_loader = DataLoader(
        train_dataset, 
        batch_sampler=train_sampler, 
        num_workers=4
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader
