import torch 
import tqdm
import torch.nn as nn


from utils.early_stopping import EarlyStopping
from data.dataloader import create_dataloaders
from data.preprocess import Preprocess
from model.FT_transformer import FTTransformer
from utils.train_val_test_fn import train_function, validation_function
from data.dataset import FTTDataset
from torch.utils.tensorboard import SummaryWriter

import pandas as pd

def main():
    # Hyperparameters
    n_layers = 1             
    d_dimensionality = 177
    n_heads = 3
    dropout = 0.1
    patience = 5
    batch_size = 512
    seed = 42
    num_workers = 4
    learning_rate  = 3e-4
    num_epochs = 100
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("GPU is avaible:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Model of GPU:", torch.cuda.get_device_name(0))

    torch.manual_seed(seed)

    df = pd.read_csv("train.csv")

    df_train, df_val, df_test, encoder_columns, class_embeddings_sizes = Preprocess().run_pipeline(df)

    train_dataset = FTTDataset(df_train, encoder_columns)
    validation_dataset = FTTDataset(df_val, encoder_columns)
    test_dataset = FTTDataset(df_test, encoder_columns)

    train_loader, val_loader, _ = create_dataloaders(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        seed=seed)

    print(len(train_dataset.X_continual_values))
    #All the embeddings classes sizes
    sizes_list = list(class_embeddings_sizes.values())
    continues_values_columns_len = len(df_train.columns) - len(encoder_columns) - 1#CLS token
    FT_transformer = FTTransformer(number_class_embedding_sizes=sizes_list,
                                        num_continual_values_columns=continues_values_columns_len,
                                        dimension_model=d_dimensionality,
                                        number_heads=n_heads,
                                        number_layers=n_layers,
                                        num_classes=1,
                                        dropout=dropout,
                                        device=device).to(device)

    #Class Imbalance in target
    num_negatives = (df_train["target"] == 0).sum()
    num_positives = (df_train["target"] == 1).sum()
    
    #Class imbalance in target
    pos_weight = torch.tensor([num_negatives / num_positives], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    
    optimizer = torch.optim.AdamW(FT_transformer.parameters(), lr=learning_rate, weight_decay=0.05)
    early_stopping = EarlyStopping(patience=patience, mode="max", verbose=True, path='best_model.pt')
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs, eta_min=1e-6)

    print(f"Model parameters: {sum(p.numel() for p in FT_transformer.parameters())/1e6:.2f} M parameters")

    tensor_board_writer = SummaryWriter(log_dir=f"runs")

    for epoch in tqdm.tqdm(range(num_epochs), desc="Training Progress"):
        train_loss = train_function(FT_transformer, train_loader, optimizer, criterion, device)
        
        val_loss, val_epoch_accuracy, val_epoch_f1 = validation_function(FT_transformer, val_loader, criterion, device)
    
        print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f} | Val Acc: {val_epoch_accuracy:.4f} | Binary F1: {val_epoch_f1:.4f}")
        
        # Step the learning rate scheduler every epoch
        lr_scheduler.step()
        
        early_stopping(val_epoch_f1, FT_transformer)
        tensor_board_writer.add_scalars("Loss", {
                    "Train": train_loss,
                    "Val": val_loss
                    }, epoch)
        
        tensor_board_writer.add_scalar("Metric/Val-Acc", val_epoch_accuracy, epoch)
        tensor_board_writer.add_scalar("Metric/Binary-F1", val_epoch_f1, epoch)

        
        if early_stopping.early_stop:
            print("Finished because of the early Stopping")
            break 
    tensor_board_writer.close()

if __name__ == "__main__":
    main()