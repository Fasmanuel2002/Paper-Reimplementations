


from torch import nn
import torch
import torch.nn.functional as F
from model.layers import MultiHeadAttention, MLP


class Encoder(nn.Module):
    """
    Encoder transformer from attention is all you need
    """
    def __init__(self, num_embeddings : int, sequence_lenght : int, num_heads : int, n_layers : int, dimensionality_ff : int, token_embedding_table : nn.Embedding, dropout : float) -> None:
        super().__init__()
        self.token_embedding_table = token_embedding_table
        self.positional_encoding = nn.Embedding(num_embeddings=sequence_lenght, embedding_dim=num_embeddings) #The positional Encoding
        self.encoder_blocks = nn.ModuleList([EncoderBlock(n_embeddings=num_embeddings, num_heads=num_heads, dimensionality_ff=dimensionality_ff, dropout=dropout) for _ in range(n_layers)])
        self.final_norm = nn.RMSNorm(num_embeddings)

        
    def forward(self, idx : torch.Tensor, attention_mask : torch.Tensor, return_attention : bool = False):
        B, T = idx.shape #idx and targes are both (B,T) tensor of integers
        tokens_embeddings = self.token_embedding_table(idx)
        positional_embeddings = self.positional_encoding(torch.arange(T, device=idx.device))
        x = tokens_embeddings + positional_embeddings
        all_attention_weights = []
        
        if return_attention:
        
            for encoder_block in self.encoder_blocks:
                x, attention_weights = encoder_block(x, attention_mask)
                all_attention_weights.append(attention_weights)

            attention_stack = torch.stack(all_attention_weights, dim=1)
            x = self.final_norm(x)
            return x, attention_stack

        else:
            for encoder_block in self.encoder_blocks:
                x, attention_weights = encoder_block(x, attention_mask)
            
            x = self.final_norm(x)
            return x
        
       

class EncoderBlock(nn.Module):
    def __init__(self, n_embeddings : int, num_heads : int, dimensionality_ff : int,dropout : float) -> None:
        super().__init__()
        head_size = n_embeddings // num_heads
        assert n_embeddings % num_heads == 0, f"The dimensionality of the model: {n_embeddings}, must be divisionable for the number of heads {num_heads}"

        self.multi_head_attention = MultiHeadAttention(
            num_heads=num_heads,
            head_size=head_size,
            n_embeddings=n_embeddings,
            dropout=dropout)
        self.fnn = MLP(n_embeddings=n_embeddings, dimensionality_ff=dimensionality_ff, dropout=dropout)
        self.ln1 = nn.RMSNorm(n_embeddings)
        self.ln2 = nn.RMSNorm(n_embeddings)

    def forward(self, x : torch.Tensor, attention_mask : torch.Tensor):
        head = self.ln1(x)
        mha, attention_weights = self.multi_head_attention(head,head, attention_mask)
        x = x + mha #Residual COnnection + MHA
        x = x + self.fnn(self.ln2(x))
        return x, attention_weights


