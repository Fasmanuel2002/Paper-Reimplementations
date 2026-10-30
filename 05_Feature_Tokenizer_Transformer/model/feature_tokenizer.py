import torch
import torch.nn as nn

class FeatureTokenizer(nn.Module):
    """
    The Feature Tokenizer from the paper Revisiting Deep Learning Models for Tabular Data part 3.3 FT-Transformer
    """
    def __init__(self, class_embedding_sizes : list, num_continual_values_columns : 
                 int, dimension_model : int = 64):
        super().__init__()

        self.categorical_embeddigns = nn.ModuleList([
            nn.Embedding(num_embeddings=num_classes, embedding_dim=dimension_model) for num_classes in class_embedding_sizes]) 

        self.continual_projections = nn.ModuleList([
            nn.Linear(in_features=1, out_features=dimension_model) for _ in range(num_continual_values_columns)])

        #CLS for clasiffication
        self.class_token = nn.Parameter(torch.zeros(1, 1, dimension_model))

    def forward(self, x_categorical : torch.Tensor, x_continous : torch.Tensor):
        B, C = x_categorical.shape # Batch size , Channels
        B_cont, C_cont = x_continous.shape
        tokens = []

        for i, embedding in enumerate(self.categorical_embeddigns):
            col_tensor = x_categorical[:, i] #Every Batch [512, 1]
            token = embedding(col_tensor).unsqueeze(1) #[512, 1, 64], B,T,C
            tokens.append(token)

        for i, projection in enumerate(self.continual_projections):
            col_tensor = x_continous[:, i].unsqueeze(1) #[512, 1]
            token = projection(col_tensor).unsqueeze(1) #[512, 1, 64]
            tokens.append(token)

        # Concat everything so the tensor shape its [512, (31 + 26 columns -> 57), 64]
        tokens = torch.concat(tokens, dim=1)
        cls_tokens = self.class_token.expand(B,-1,-1) #B, T, C [512, (11 + 13 + 1 cls token -> 58), 65]
        x = torch.cat((cls_tokens, tokens), dim=1) 
        return x #B, T, C