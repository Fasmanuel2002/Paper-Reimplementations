from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

class Preprocess:
    
    def run_pipeline(self, df_porto_dataset : pd.DataFrame):
            df_porto_dataset = self.drop_columns(df_porto_dataset)
            df_porto_dataset = self.replace_missing(df_porto_dataset)
            train_data , dev_data = self.split_train_dev(df_porto_dataset)
            train_data, encoder_columns, class_embeddings_sizes, medians = self.encode_features_train(train_data)
            dev_dataset = self.transform_features_val_test(dev_data, medians=medians, encoder_columns=encoder_columns)
            val_dataset, test_dataset = self.split_val_test(dev_dataset)
            return train_data, val_dataset, test_dataset, encoder_columns, class_embeddings_sizes
    
    def drop_columns(self, df_porto_dataset : pd.DataFrame) -> pd.DataFrame:
        return df_porto_dataset.drop(columns="id")
        
    def replace_missing(self, df_porto_dataset : pd.DataFrame) -> pd.DataFrame:
        return df_porto_dataset.replace(-1, np.nan)
    
    def split_train_dev(self, df_porto_dataset : pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_data, dev = train_test_split(df_porto_dataset, test_size=0.20, stratify=df_porto_dataset["target"], random_state=42)    
        return train_data, dev
    
    def split_val_test(self, dev : pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
            val_data, test_data = train_test_split(dev, test_size=0.50, stratify=dev["target"], random_state=42)    
            return val_data, test_data
            
    def encode_features_train(self, df_porto_dataset_train : pd.DataFrame):
        medians = {}
        encoder_columns = {}
        class_embeddings_size = {}
        categorical_binary_cols = [col for col in df_porto_dataset_train.columns if col.endswith(("_bin", "_cat"))]
        continuos_values_cols = [col for col in df_porto_dataset_train.columns if not col.endswith(("_bin", "_cat")) and not col == "target"]
        df_encoded = df_porto_dataset_train.copy()
        
        for column in continuos_values_cols:
            medians[column] = df_porto_dataset_train[column].median()
            df_encoded[column] = df_porto_dataset_train[column].fillna(medians[column]) 
       
        for column in categorical_binary_cols:
            values = df_encoded[column].fillna("missing").astype(str)
            
            label_encoder = LabelEncoder()
            
            label_encoder.fit(pd.concat([values, pd.Series(["missing", "unknown"])]))
            
            df_encoded[column] = label_encoder.transform(values)
            
            encoder_columns[column] = label_encoder
            
            class_embeddings_size[column] = len(label_encoder.classes_)
        
        return df_encoded, encoder_columns, class_embeddings_size, medians
    
    def transform_features_val_test(self, df_porto_dataset : pd.DataFrame, medians : dict, encoder_columns : dict):
        df = df_porto_dataset.copy()
        categorical_binary_cols = [col for col in df.columns if col.endswith(("_bin", "_cat"))]
        continuos_values_cols = [col for col in df.columns if not col.endswith(("_bin", "_cat")) and not col == "target"]
        
        for column in continuos_values_cols:
            df[column] = df[column].fillna(medians[column])
        
        for column in categorical_binary_cols:
            values = df[column].fillna("missing").astype(str)
            classes = encoder_columns[column].classes_
            values = values.where(values.isin(classes), "unknown")
            df[column] = encoder_columns[column].transform(values)
        return df
    
    
        