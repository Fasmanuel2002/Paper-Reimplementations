from dataset.dataset import C4SpanCorruptionDataset, collate_fn
from typing import Tuple
from torch.utils.data import DataLoader
from functools import partial

def create_dataloaders(train_dataset : C4SpanCorruptionDataset,
                       validation_dataset : C4SpanCorruptionDataset,
                       test_dataset : C4SpanCorruptionDataset,
                       pad_id : int,
                       num_workers : int = 4,
                       batch_size : int = 128) -> Tuple[DataLoader, DataLoader ,DataLoader]:

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_id=pad_id),
        num_workers=num_workers
        pin_memory=True, #Faster Transfer for GPU
        persistent_workers=True, #Workers stay alive between epochs,
        drop_last=True #If the batch is little
    )

    
    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, pad_id=pad_id),
        num_workers=num_workers
    )
    test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=partial(collate_fn, pad_id=pad_id),
            num_workers=num_workers
    )

    return train_loader, validation_loader, test_loader