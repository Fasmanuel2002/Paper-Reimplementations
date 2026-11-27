


from torch import nn
import torch
import torch.nn.functional as F





class MLP(nn.Module):
    """
    A feed-forward network for per token level idenpently 
    self-attention is the communication when they gather all the data, it needs to think about that data individually
    """
    def __init__(self, n_embeddings : int, dropout : float) -> None:
        super().__init__()

        self.fnn = nn.Sequential(
              nn.Linear(n_embeddings, n_embeddings * 4),
              nn.GELU(),
              nn.Linear(4 * n_embeddings, n_embeddings),
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
    def __init__(self, num_heads : int, d_model : int, n_embeddings : int,  dropout : float) -> None:
        super().__init__()
        """
        Args:
            num_heads (int): Number of heads of self attention
            head_size (int): Size of the projection space for this head (d_k)
        """

        self.heads = nn.ModuleList([SelfAttention(
            d_model=d_model,
            n_embeddings=n_embeddings,
            dropout=dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embeddings, n_embeddings)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x : torch.Tensor, attention_mask : torch.Tensor):
        outs, attention_weights = zip(*[head(x, attention_mask) for head in self.heads])
    
        
        out = torch.cat(outs, dim=-1)
        out = self.dropout(self.proj(out))
        
        attention_weights = torch.stack(attention_weights, dim=1)
        return out, attention_weights


class SelfAttention(nn.Module):
    """
    Self-Attention head from Attention is all you need Transformer
    """
    def __init__(self, d_model : int ,  n_embeddings : int, dropout : float):
            super(SelfAttention, self).__init__()
            self.d_model = d_model
            self.queries = nn.Linear(in_features=n_embeddings, out_features=d_model)
            self.keys = nn.Linear(in_features=n_embeddings, out_features=d_model)
            self.values = nn.Linear(in_features=n_embeddings, out_features=d_model)
            self.dropout = nn.Dropout(p=dropout)

    
    def forward(self, x : torch.Tensor, attention_mask: torch.Tensor):

        B,T,C = x.shape #Batch Size, Sequence Lenght, Channels
        q = self.queries(x) #Batch Size, Sequence Lenght, Num_heads
        k = self.keys(x) #Batch Size, Sequence Lenght, Num_heads
        v = self.values(x) #Batch Size, Sequence Lenght, Num_heads
    
        wei_matrix = (q @ k.transpose(-2,-1)) / (self.d_model ** 0.5) #B, T, C @ B, C, T -> B, T, T
        wei_matrix = wei_matrix.masked_fill(attention_mask == 0, float("-inf")) # mask: (B, 1, T) or (T, T), 1 = attend, 0 = block
        wei_matrix = F.softmax(wei_matrix, dim=-1) 
    
        attention_weigts = wei_matrix
    
        wei_matrix = self.dropout(wei_matrix)
    
        output = wei_matrix @ v #B, T, T @ B,T,C -> B, T, C 
            
        return output, attention_weigts