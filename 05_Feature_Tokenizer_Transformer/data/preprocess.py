import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Preprocess:
    def __init__(self, df_porto_dataset : pd.DataFrame) -> None:
        self.df_porto_dataset = df_porto_dataset.copy()
    
    
    def drop_columns(self, df_porto_dataset : pd.DataFrame) -> pd.DataFrame:
        return df_porto_dataset.drop(columns="id")
        
    def replace_missing(self, df_porto_dataset : pd.DataFrame) -> pd.DataFrame:
        return df_porto_dataset.replace(-1, np.nan)
    
    def encode_categorical_features(self, df_porto_dataset : pd.DataFrame):
        encoder_columns = {}
        class_embeddings_size = {}
        categorical_binary_cols = [col for col in df_porto_dataset.columns if col.endswith(("_bin", "_cat"))]
        df_encoded = df_porto_dataset.copy()
        
        for column in categorical_binary_cols:
            label_encoder = LabelEncoder()
            
            df_encoded[column] = label_encoder.fit_transform(df_encoded[column].fillna("missing").astype(str))
            
            encoder_columns[column] = label_encoder
            
            class_embeddings_size[column] = len(label_encoder.classes_)
        
        return df_encoded, encoder_columns, class_embeddings_size
    
    def missing_continues_values(self, df_porto_dataset : pd.DataFrame):
        continuos_values_cols = [col for col in df_porto_dataset.columns if not col.endswith(("_bin", "_cat")) and not col == "target"]
        for column in continuos_values_cols:
            df_porto_dataset[column] = df_porto_dataset[column].fillna(df_porto_dataset[column].median()) 
        return df_porto_dataset
    
    