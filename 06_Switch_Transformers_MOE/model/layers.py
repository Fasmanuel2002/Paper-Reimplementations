
from torch import nn
import torch
import torch.nn.functional as F


class MLP(nn.Module):
    """
    A feed-forward network for per token level idenpently 
    self-attention is the communication when they gather all the data, it needs to think about that data individually
    """
    def __init__(self, n_embeddings: int, dimensionality_ff : int, dropout : float) -> None:
        super().__init__()

        self.fnn = nn.Sequential(
              nn.Linear(n_embeddings, dimensionality_ff, bias=False),
              nn.ReLU(),
              nn.Linear(dimensionality_ff, n_embeddings, bias=False),
              nn.Dropout(p=dropout)
        )

    def forward(self, x : torch.Tensor):
        return self.fnn(x)



class MultiHeadAttention(nn.Module):
    """
    Multi-head attention is to run in parallel multiple heat attention
    Multi-head attention allows the model to jointly attend information from different 
    representation subspaces at different position
    """
    def __init__(self, num_heads : int, head_size : int, n_embeddings : int,  dropout : float) -> None:
        super().__init__()
        """
        Args:
            num_heads (int): Number of heads of self attention
            head_size (int): Size of the projection space for this head (d_k)
        """

        self.heads = nn.ModuleList([AttentionHead(
            head_size=head_size,
            n_embeddings=n_embeddings,
            dropout=dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embeddings, n_embeddings, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x_q : torch.Tensor, x_kv : torch.Tensor, attention_mask: torch.Tensor):
        outs, attention_weights = zip(*[head(x_q, x_kv, attention_mask) for head in self.heads])
    
        
        out = torch.cat(outs, dim=-1)
        out = self.dropout(self.proj(out))
        
        attention_weights = torch.stack(attention_weights, dim=1)
        return out, attention_weights


class AttentionHead(nn.Module):
    """
    Self-Attention head  and Cross-Attention from Attention is all you need Transformer
    """
    def __init__(self, head_size : int ,  n_embeddings : int,  dropout : float):
            super(AttentionHead, self).__init__()
            self.head_size = head_size
            self.queries = nn.Linear(in_features=n_embeddings, out_features=head_size, bias=False)
            self.keys = nn.Linear(in_features=n_embeddings, out_features=head_size, bias=False)
            self.values = nn.Linear(in_features=n_embeddings, out_features=head_size, bias=False)
            self.dropout = nn.Dropout(p=dropout)

    
    def forward(self, x_q : torch.Tensor, x_kv : torch.Tensor, attention_mask: torch.Tensor):
        q = self.queries(x_q) #Batch Size, T_q, Head_size
        k = self.keys(x_kv) #Batch Size, T_KV, Head_size
        v = self.values(x_kv) #Batch Size, T_KV , Head_size
    
        wei_matrix = (q @ k.transpose(-2,-1)) / (self.head_size ** 0.5) #B, T_q, C @ B, C, T_KV -> B, T_q, T_KV
        wei_matrix = wei_matrix.masked_fill(attention_mask == 0, float("-inf")) # mask: (B, 1, T) or (T, T), 1 = attend, 0 = block
        wei_matrix = F.softmax(wei_matrix, dim=-1) 
    
        attention_weigts = wei_matrix
    
        wei_matrix = self.dropout(wei_matrix)
    
        output = wei_matrix @ v #B, T, T @ B,T,C -> B, T, C 
            
        return output, attention_weigts