from typing import Tuple
from data.dataset import FTTDataset
from torch.utils.data import DataLoader
import torch
def create_dataloaders(
        train_dataset : FTTDataset,
        validation_dataset : FTTDataset,
        test_dataset : FTTDataset,
        num_workers : int = 0,
        batch_size : int = 512, 
        seed : int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:

    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator)

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True) 

    return train_loader, validation_loader, test_loader