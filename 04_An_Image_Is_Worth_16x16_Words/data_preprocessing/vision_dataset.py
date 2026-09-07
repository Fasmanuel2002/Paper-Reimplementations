from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import os
from typing import Optional, Tuple
from torchvision.datasets import ImageFolder

class imagenette2Dataset:
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    def __init__(self, data_path : str = "imagenette2", image_size : int = 224, batch_size : int = 32):
        self.data_path = data_path
        self.image_size = image_size
        self.batch_size = batch_size
    
    def transforms(self, image_size: int=224, is_train : Optional[bool] = True) -> T.Compose:
        if is_train:
            return T.Compose([
                T.RandomResizedCrop(image_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
            ])
        else: 
            resize_dim = int(self.image_size * (256 / image_size))
            return T.Compose([
                T.Resize(resize_dim),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
            ])
    
    def create_dataloaders(self, num_workers : int = 4) -> Tuple[DataLoader, DataLoader]:
        train_transform = self.transforms(image_size=self.image_size, is_train=True)
        validation_transform = self.transforms(image_size=self.image_size, is_train=False)
        
        train_dataset = ImageFolder(root=os.path.join(self.data_path, "train"), transform=train_transform)
        validation_dataset = ImageFolder(root=os.path.join(self.data_path, "val"), transform=validation_transform)
        
        train_dataloader = DataLoader(train_dataset, 
                                      batch_size=self.batch_size, 
                                      shuffle=True,
                                      num_workers=num_workers,
                                      pin_memory=True
                                      )
        validation_dataloader = DataLoader(validation_dataset, 
                                           batch_size=self.batch_size, 
                                           shuffle=False,
                                           num_workers=num_workers,
                                           pin_memory=True
                                           )
        return train_dataloader, validation_dataloader