import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from model.ViT import VisionTransformer
from typing import Tuple
from sklearn.metrics import accuracy_score, classification_report, f1_score


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

def test_function(
    vision_model : VisionTransformer,
    test_loader : DataLoader,
    device : str
    ):
    
    vision_model.eval()
    test_loss = 0.0
    y_pred_all_predictions = []
    y_true_all_labels = []
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for index, (images, labels) in enumerate(test_loader):
            src_images = images.to(device)
            trg_labels = labels.to(device)
            
            logits, loss, attention_weights = vision_model(src_images, trg_labels)
            
            predictions = torch.argmax(logits, dim=1)
            
            y_pred_all_predictions.extend(predictions.cpu().numpy())
            
            y_true_all_labels.extend(trg_labels.cpu().numpy())
            
            test_correct += (predictions == trg_labels).sum().item()
    
            test_total += trg_labels.numel()
            
            test_loss += loss.item()
    
    test_loss /= len(test_loader)
    test_accuracy = test_correct / max(test_total, 1)
    
    macro_f1 = f1_score(y_true_all_labels, y_pred_all_predictions, average="macro")
    weighted_f1 = f1_score(y_true_all_labels, y_pred_all_predictions, average="weighted")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}\n")
    
    
    print("Classification Report")
    print(classification_report(y_true_all_labels, y_pred_all_predictions, digits=4))

    return test_loss, test_accuracy, macro_f1, weighted_f1
    
    
