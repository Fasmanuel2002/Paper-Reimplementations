import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from model.FT_transformer import FTTransformer
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, f1_score, precision_score,roc_auc_score, recall_score
from typing import Tuple
from utils.metrics import search_best_f1_threshold

def train_function(
        ft_transformer : FTTransformer,
        train_loader : DataLoader,
        optimizer : optim.AdamW,
        criterion : nn.BCEWithLogitsLoss,
        device) -> float:

    ft_transformer.train()
    epoch_loss = 0.0
    total_samples = 0
    
    for (x_categorical, x_continous, y_labels) in train_loader:
        x_categorical, x_continous,  y_labels = x_categorical.to(device), x_continous.to(device),  y_labels.to(device).float().reshape(-1, 1)
        optimizer.zero_grad(set_to_none=True) #Put all gradients into zero

        logits, _ = ft_transformer(x_categorical, x_continous)

        loss = criterion(logits, y_labels)
        
        batch_samples = y_labels.size(0)

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(ft_transformer.parameters(), max_norm=1.0) #To delete vanish gradient

        optimizer.step()

        epoch_loss += loss.item() * batch_samples
        
        total_samples += batch_samples

    return epoch_loss / total_samples


def validation_function(
        ft_transformer : FTTransformer,
        validation_loader : DataLoader,
        criterion : nn.BCEWithLogitsLoss,
        thresholds : np.ndarray,
        device : str) -> Tuple:

    ft_transformer.eval()
    all_val_y_probs = []
    all_val_y_true = []
    epoch_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for (x_categorical, x_continous, y_labels) in validation_loader:
            x_categorical, x_continous,  y_labels = x_categorical.to(device), x_continous.to(device),  y_labels.to(device).float().reshape(-1, 1)

            logits, _ = ft_transformer(x_categorical, x_continous)
            
            loss = criterion(logits, y_labels)

            probabilities = torch.sigmoid(logits)

            all_val_y_probs.extend(probabilities.squeeze(1).cpu().numpy())
            
            all_val_y_true.extend(y_labels.squeeze(1).cpu().numpy())
            
            batch_samples = y_labels.size(0)
            
            epoch_loss += loss.item() * batch_samples    
            total_samples += batch_samples
        
        
        
        all_val_y_probs = np.asarray(all_val_y_probs)
        
        all_val_y_true = np.asarray(all_val_y_true).astype(int)

        val_epoch_f1, val_best_threshold, val_auprc, val_auroc = search_best_f1_threshold(all_val_y_true, all_val_y_probs, thresholds)

        val_total_loss = epoch_loss / total_samples

        best_predictions = (all_val_y_probs >= val_best_threshold).astype(int)
        
        val_epoch_accuracy = accuracy_score(all_val_y_true, best_predictions)
        
        val_epoch_precision = precision_score(all_val_y_true, best_predictions, zero_division=0)
        
        val_epoch_recall = recall_score(all_val_y_true, best_predictions, zero_division=0)
    
    return val_total_loss, val_epoch_accuracy, val_epoch_f1,val_best_threshold, val_auprc, val_auroc, val_epoch_precision, val_epoch_recall


def test_function(
        ft_transformer : FTTransformer,
        test_loader : DataLoader,
        criterion : nn.BCEWithLogitsLoss,
        threshold : float,
        device : str) -> Tuple:

    ft_transformer.eval()
    all_test_y_probs = []
    all_test_y_true = []
    epoch_loss = 0.0
    total_samples = 0
    with torch.inference_mode():
        for (x_categorical, x_continous, y_labels) in test_loader:
            x_categorical, x_continous,  y_labels = x_categorical.to(device), x_continous.to(device),  y_labels.to(device).float().reshape(-1, 1)
            logits, _ = ft_transformer(x_categorical, x_continous)
            loss = criterion(logits, y_labels)

            probabilities = torch.sigmoid(logits)

            all_test_y_probs.extend(probabilities.squeeze(1).cpu().numpy())
            
            all_test_y_true.extend(y_labels.squeeze(1).cpu().numpy())
            
            batch_samples = y_labels.size(0)

            epoch_loss += loss.item() * batch_samples 
            total_samples += batch_samples

        all_test_y_probs = np.asarray(all_test_y_probs)
        
        all_test_y_true = np.asarray(all_test_y_true).astype(int)
        
        predictions = (all_test_y_probs >= threshold).astype(int)
        
        test_total_loss = epoch_loss / total_samples
        
        test_f1 = f1_score(all_test_y_true, predictions, average="binary", zero_division=0)
        
        test_epoch_accuracy = accuracy_score(all_test_y_true, predictions)
        
        test_epoch_precision = precision_score(all_test_y_true, predictions, zero_division=0)
        
        test_epoch_recall = recall_score(all_test_y_true, predictions, zero_division=0)

        test_auprc = average_precision_score(all_test_y_true, all_test_y_probs)
        
        test_auroc = roc_auc_score(all_test_y_true, all_test_y_probs)

    return test_total_loss, test_epoch_accuracy, test_f1, test_auprc, test_auroc, test_epoch_precision, test_epoch_recall
