import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from model.ViT import VisionTransformer
from typing import Tuple

def train_function(
    vision_model : VisionTransformer,
    data_loader : DataLoader,
    optimizer : optim.AdamW,
    device : str) -> float:
    vision_model.train() #Putting the model for training
    epoch_loss = 0.0
    for index, (images, labels) in enumerate(data_loader):
        src_images = images.to(device) # shape (Batch_size, in_channels, Height = image_size, width = image_size)
        
        trg_labels = labels.to(device)
        
        optimizer.zero_grad() #Making all the gradients 0 so it can update
        
        logits, loss, _ = vision_model(src_images, trg_labels) # Shape -> (batch_size, num_classes)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()

    return epoch_loss / len(data_loader)

def validation_function(
    vision_model : VisionTransformer,
    data_loader : DataLoader,
    device : str) -> Tuple[float, float]:
    vision_model.eval() #Putting the model for evaluation
    epoch_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for index, (images, labels) in enumerate(data_loader):
            src_images = images.to(device) # shape (Batch_size, in_channels, Height = image_size, width = image_size)
            
            trg_labels = labels.to(device)
            
            logits, loss, _ = vision_model(src_images, trg_labels) # Shape -> (batch_size, num_classes)
            
            epoch_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == trg_labels).sum().item()
            total_samples += trg_labels.size(0)
        
        avg_loss = epoch_loss / len(data_loader)
        accuracy = float(correct_predictions / total_samples)
        

    return avg_loss, accuracy
