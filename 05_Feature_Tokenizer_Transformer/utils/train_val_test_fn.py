import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from model.FT_transformer import FTTransformer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from typing import Tuple
def train_function(
        ft_transformer : FTTransformer,
        train_loader : DataLoader,
        optimizer : optim.AdamW,
        criterion : nn.BCEWithLogitsLoss,
        device) -> float:

    ft_transformer.train()
    epoch_loss = 0.0

    for (x_categorical, x_continous, y_labels) in train_loader:
        x_categorical, x_continous,  y_labels = x_categorical.to(device), x_continous.to(device), y_labels.to(device)
        y_labels = y_labels.unsqueeze(1)
        optimizer.zero_grad() #Put all gradients into zero

        logits, _ = ft_transformer(x_categorical, x_continous)

        loss = criterion(logits, y_labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(ft_transformer.parameters(), max_norm=1.0) #To delete vanish gradient

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)


def validation_function(
        ft_transformer : FTTransformer,
        validation_loader : DataLoader,
        criterion : nn.BCEWithLogitsLoss,
        device : str) -> Tuple:

    ft_transformer.eval()
    all_val_y_pred = []
    all_val_y_true = []
    epoch_loss = 0.0
    with torch.no_grad():
        for (x_categorical, x_continous, y_labels) in validation_loader:
            x_categorical, x_continous,  y_labels = x_categorical.to(device), x_continous.to(device), y_labels.to(device)
            y_labels = y_labels.unsqueeze(1)

            logits, _ = ft_transformer(x_categorical, x_continous)
            loss = criterion(logits, y_labels)

            probabilities = torch.sigmoid(logits)

            predictions = (probabilities >= 0.5).int() #Threshold to obtain 0 or 1

            all_val_y_pred.extend(predictions.detach().cpu().numpy())
            all_val_y_true.extend(y_labels.detach().cpu().numpy())
            epoch_loss += loss.item()    
        
        val_total_loss = epoch_loss / len(validation_loader)
        val_epoch_accuracy = accuracy_score(all_val_y_true, all_val_y_pred)
        val_epoch_f1 = f1_score(all_val_y_true, all_val_y_pred, average="binary")


    return val_total_loss, val_epoch_accuracy, val_epoch_f1


def test_function(
        ft_transformer : FTTransformer,
        test_loader : DataLoader,
        criterion : nn.BCEWithLogitsLoss,
        device : str) -> Tuple:

    ft_transformer.eval()
    all_test_y_pred = []
    all_test_y_true = []
    epoch_loss = 0.0
    with torch.no_grad():
        for (x_categorical, x_continous, y_labels) in test_loader:
            x_categorical, x_continous,  y_labels = x_categorical.to(device), x_continous.to(device), y_labels.to(device)
            y_labels = y_labels.unsqueeze(1)
            logits, _ = ft_transformer(x_categorical, x_continous)
            loss = criterion(logits, y_labels)

            probabilities = torch.sigmoid(logits)

            predictions = (probabilities >= 0.5).int() #Threshold to obtain 0 or 1

            all_test_y_pred.extend(predictions.detach().cpu().numpy())
            all_test_y_true.extend(y_labels.detach().cpu().numpy())
            epoch_loss += loss.item()    
        
        val_total_loss = epoch_loss / len(test_loader)
        val_epoch_accuracy = accuracy_score(all_test_y_true, all_test_y_pred)
        val_epoch_f1 = f1_score(all_test_y_true, all_test_y_pred, average="binary")


    return val_total_loss, val_epoch_accuracy, val_epoch_f1