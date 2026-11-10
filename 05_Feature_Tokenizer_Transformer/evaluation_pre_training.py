import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from data.dataloader import create_dataloaders
from data.preprocess import Preprocess
from model.FT_transformer import FTTransformer
from utils.train_val_test_fn import validation_function, test_function
from data.dataset import FTTDataset
import matplotlib.pyplot as plt
import seaborn as sns
def main():
    checkpoint_path = "best_model.pt"   
    n_layers = 8                       
    d_dimensionality = 512              
    n_heads = 8
    dropout = 0.1
    batch_size = 2048
    seed = 42
    num_workers = 4
    thresholds = np.linspace(0.01, 0.99, 99)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)

    df = pd.read_csv("train.csv")
    df_train, df_val, df_test, encoder_columns, class_embeddings_sizes = Preprocess().run_pipeline(df)

    train_dataset = FTTDataset(df_train, encoder_columns)
    validation_dataset = FTTDataset(df_val, encoder_columns)
    test_dataset = FTTDataset(df_test, encoder_columns)

    _, val_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        seed=seed)

    sizes_list = list(class_embeddings_sizes.values())
    continues_values_columns_len = len(df_train.columns) - len(encoder_columns) - 1
    FT_transformer = FTTransformer(number_class_embedding_sizes=sizes_list,
                                   num_continual_values_columns=continues_values_columns_len,
                                   dimension_model=d_dimensionality,
                                   number_heads=n_heads,
                                   number_layers=n_layers,
                                   num_classes=1,
                                   dropout=dropout,
                                   device=device).to(device)

    FT_transformer.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    criterion = nn.BCEWithLogitsLoss() 

    
    _, _, val_f1, best_threshold, _, val_auroc, _, _ = validation_function(
        FT_transformer, val_loader, criterion, thresholds, device)
    print(f"Val  | Best Threshold: {best_threshold:.2f} | F1: {val_f1:.4f} | Norm Gini: {2*val_auroc-1:.4f}")

    test_loss, test_acc, test_f1, test_auprc, test_auroc, test_precision, test_recall = test_function(FT_transformer, test_loader, criterion, best_threshold, device)

    print(f"Test | Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f}")
    print(f"Test | AUPRC: {test_auprc:.4f} | AUROC: {test_auroc:.4f} | Norm Gini: {2*test_auroc-1:.4f} | Precision: {test_precision:.4f} | Recall: {test_recall:.4f}")

    #Creating the attention maps for the test set
    x_categorical, x_continous, _ = next(iter(test_loader))
    x_categorical = x_categorical.to(device)
    x_continuous = x_continous.to(device)

    with torch.no_grad():
        _, attention_maps = FT_transformer(x_categorical, x_continuous)

    #Extract the last layer for attention maps in the first index of the batch
    sample_attention = attention_maps[0, -1]
    avg_head_attention = torch.mean(sample_attention, dim=-1).cpu().numpy()

    # Attention [CLS] token for each feature
    cls_attention_to_features = avg_head_attention[0, 1:]
    features_names = list(encoder_columns) + [col for col in df_train.columns if col not in encoder_columns and col != 'target']

    # Plotting the attention scores for each feature
    plt.figure(figsize=(12, 6))
    sns.barplot(x=features_names, y=cls_attention_to_features, palette="magma")
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Features")
    plt.ylabel("Attention Score")
    plt.title("Feature Importance based on Attention Scores")
    plt.savefig("attention_scores.png", bbox_inches='tight', dpi=300)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
