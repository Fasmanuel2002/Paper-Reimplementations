import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .feature_tokenizer import FeatureTokenizer

class FTTransformer(nn.Module):
    """Pytorch implementation of FT-transformer"""
    def __init__(self, 
        number_class_embedding_sizes: list, 
        num_continual_values_columns: int, 
        dimension_model: int, 
        number_heads: int, 
        number_layers: int, 
        num_classes: int = 1,
        dropout: float = 0.2,
        device : str = ""):
        super().__init__()

        self.device = device

        self.tokenizer = FeatureTokenizer(class_embedding_sizes=number_class_embedding_sizes,
                                          num_continual_values_columns=num_continual_values_columns,
                                          dimension_model=dimension_model)

        self.blocks = nn.ModuleList([
            Block(
                number_heads=number_heads,
                n_embeddings=dimension_model,
                dropout=dropout
            ) for _ in range(number_layers)
        ])
        self.last_layer_normalization = nn.LayerNorm(dimension_model)

        self.classification_head = nn.Linear(dimension_model, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear): 
            nn.init.normal_(module.weight, mean=0.0, std=0.02) #Gaussian initialization of the weights
            if module.bias is not None:
                nn.init.zeros_(module.bias) #Make the biases zero
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02) #Gaussian initialization of the weights
    
        
    def forward(self, x_categorical_values : torch.Tensor, x_continuas_values : torch.Tensor):
        x = self.tokenizer(x_categorical_values, x_continuas_values)
        all_atention_weights = []

        for block in self.blocks:
            x, attention_weight = block(x)
            all_atention_weights.append(attention_weight)

        cls_token_output = x[:, 0, :] # [B, 64] its the first token form the 25

        logits = self.classification_head(self.last_layer_normalization(cls_token_output))

        attention_stack = torch.stack(all_atention_weights, dim=1)

        return logits, attention_stack


class Block(nn.Module):
    def __init__(self, number_heads : int, n_embeddings : int, dropout : float):
        super().__init__()
        head_size = n_embeddings // number_heads #Always must be the 
        assert n_embeddings % number_heads == 0, f"The number of embeddings: {n_embeddings}, must be divisionable for the number of heads {number_heads}"
        self.head_attention = MultiHeadAttention(
            number_heads=number_heads,
            head_size=head_size,
            n_embeddings=n_embeddings,
            dropout=dropout)
        
        self.last_fnn = MLP(n_embeddings=n_embeddings, dropout=dropout)

        self.first_layer_normalization = nn.LayerNorm(n_embeddings)

        self.first_second_normalization = nn.LayerNorm(n_embeddings)
    def forward(self, x : torch.Tensor):
        multi_head_attention_layer, attention_weights = self.head_attention(self.first_layer_normalization(x))
        x = x + multi_head_attention_layer #First residual connection + MHA
        x = x + self.last_fnn(self.first_second_normalization(x)) #Second residual connection + MLP == last fnn

        return x, attention_weights
class MLP(nn.Module):
    """
    A feed-forward network for per token level idenpently 
    self-attention is the communication when they gather all the data, it needs to think about that data individually
    """
    def __init__(self, n_embeddings : int, dropout : float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embeddings, n_embeddings * 4),
            nn.GELU(),
            nn.Linear(4 * n_embeddings, n_embeddings),
            nn.Dropout(dropout)
        )
    def forward(self, x : torch.Tensor):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, number_heads : int, head_size : int, n_embeddings : int , dropout : int ):
        super().__init__()

        self.heads = nn.ModuleList([
            HeadAttention(head_size=head_size, n_embeddings=n_embeddings, dropout=dropout)
        for _ in range(number_heads)])

        self.proj = nn.Linear(n_embeddings, n_embeddings)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x : torch.Tensor):
        outs, attention_weights = zip(*[head(x) for head in self.heads])

        out = torch.cat(outs, dim=-1)#Concat every output

        out = self.dropout(self.proj(out))

        attention_weights = torch.stack(attention_weights, dim=-1)

        return out, attention_weights



class HeadAttention(nn.Module):
    """Some Information about AttentionHead"""
    def __init__(self, head_size : int ,  n_embeddings : int, dropout : float):
        super(HeadAttention, self).__init__()
        self.head_size = head_size
        self.queries = nn.Linear(in_features=n_embeddings, out_features=head_size, bias=False)
        self.keys = nn.Linear(in_features=n_embeddings, out_features=head_size, bias=False)
        self.values = nn.Linear(in_features=n_embeddings, out_features=head_size, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x : torch.Tensor):
        B,T,C = x.shape #Batch Size, Sequence Lenght, Channels
        q = self.queries(x) #Batch Size, Sequence Lenght, Num_heads
        k = self.keys(x) #Batch Size, Sequence Lenght, Num_heads
        v = self.values(x) #Batch Size, Sequence Lenght, Num_heads

        wei_matrix = (q @ k.transpose(-2,-1)) / (self.head_size ** 0.5) #B, T, C @ B, C, T -> B, T, T

        wei_matrix = F.softmax(wei_matrix, dim=-1) 

        attention_weigts = wei_matrix

        wei_matrix = self.dropout(wei_matrix)

        output = wei_matrix @ v #B, T, T @ B,T,C -> B, T, C 
        
        return output, attention_weigts