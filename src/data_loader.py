import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
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
        image_array = np.load(npy_path).astype(np.float32)

        valid_mask = image_array > -32752
        valid_data = image_array[valid_mask]

        if valid_data.size > 0:
            f_min, f_max = valid_data.min(), valid_data.max()
            
            if f_max > f_min:
                image_array = np.clip((image_array - f_min) / (f_max - f_min + 1e-6), 0, 1)
            else:
                image_array = np.zeros_like(image_array)
        else:
            image_array = np.zeros_like(image_array)
            
        tensor = torch.from_numpy(image_array).float().unsqueeze(0).repeat(3, 1, 1)
        
        if self.transform:
            tensor = self.transform(tensor)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor, label
    

class LunarPitHFDataset(Dataset):
    """Wrapper for Hugging Face Dataset format loaded from disk."""
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

        # Retrieve labels for the batch sampler
        labels = np.array(self.hf_dataset["label"])
        self.pit_indices = np.where(labels == 1)[0]
        self.neg_indices = np.where(labels == 0)[0]

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        npy = np.array(item["npy_data"], dtype=np.float32)
        tensor = torch.from_numpy(npy).float().unsqueeze(0).repeat(3, 1, 1)

        if self.transform:
            tensor = self.transform(tensor)

        label = torch.tensor(item["label"], dtype=torch.long)
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
            yield batch.tolist()

    def __len__(self):
        return self.n_batches


def get_dataloaders(data_dir="data/processed/dataset", batch_size=32, transform=None):
    base_path = Path(data_dir)

    # 1. Check if Hugging Face dataset format exists at base_path
    if (base_path / "dataset_dict.json").exists():
        logger.info(f"Loading Hugging Face dataset from disk: {base_path}")
        from datasets import load_from_disk
        ds = load_from_disk(str(base_path))

        train_dataset = LunarPitHFDataset(ds["train"], transform=transform)
        # Use validation split if available, otherwise fallback to test
        val_split = "validation" if "validation" in ds else "test"
        val_dataset = LunarPitHFDataset(ds[val_split], transform=transform)

        logger.info("Loaded from Hugging Face dataset:")
        logger.info(
            f"  TRAIN split ({len(train_dataset)} samples): "
            f"{len(train_dataset.pit_indices)} pits, {len(train_dataset.neg_indices)} negatives"
        )
        logger.info(
            f"  VAL split ({val_split}) ({len(val_dataset)} samples): "
            f"{len(val_dataset.pit_indices)} pits, {len(val_dataset.neg_indices)} negatives"
        )
    else:
        # 2. Fall back to folder-based legacy loader
        def load_from_split(split_name):
            split_path = base_path / split_name
            p_files = list((split_path / "pits").glob("*.npy"))
            n_files = list((split_path / "negatives").glob("*.npy"))

            files = p_files + n_files
            labels = [1] * len(p_files) + [0] * len(n_files)
            return files, labels

        train_files, train_labels = load_from_split("train")
        val_files, val_labels = load_from_split("test")

        logger.info("Loaded from Split-Folders (legacy):")
        logger.info(f"  TRAIN: {len(train_files)} Images")
        logger.info(f"  VAL:   {len(val_files)} Images")

        train_dataset = LunarPitDataset(train_files, train_labels, transform=transform)
        val_dataset = LunarPitDataset(val_files, val_labels, transform=transform)

    train_sampler = BalancedBatchSampler(train_dataset, batch_size)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader
