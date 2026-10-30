import torch
import pandas as pd
from typing import Any, Dict

class FTTDataset(torch.utils.data.Dataset):
    def __init__(self, df : pd.DataFrame, encoder_columns):
        Y_target = df["target"]
        X_input = df.drop(columns=["target"])
        X_categorical_values = list(encoder_columns.keys())
        X_continual_values = [values for values in X_input.columns if values not in X_categorical_values]

        self.X_categorical_values = torch.tensor(df[X_categorical_values].values, dtype=torch.long)

        self.X_continual_values = torch.tensor(df[X_continual_values].values, dtype=torch.float32)
        
        self.Y_target = torch.tensor(Y_target.to_numpy(), dtype=torch.float32)

    def __getitem__(self, index) -> Any:
        return self.X_categorical_values[index], self.X_continual_values[index], self.Y_target[index] 

    def __len__(self):
        return len(self.Y_target)


