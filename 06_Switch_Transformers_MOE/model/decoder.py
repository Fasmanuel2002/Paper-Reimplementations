
from torch import nn
import torch
import torch.nn.functional as F
from model.layers import MultiHeadAttention, MLP


class Decoder(nn.Module):
    """
    Decoder transformer from attention is all you need
    """
    def __init__(self, num_embeddings : int, sequence_lenght : int, num_heads : int, n_layers : int, dimensionality_ff : int, token_embedding_table : nn.Embedding ,dropout : float) -> None:
        super().__init__()
        self.token_embedding_table = token_embedding_table
        self.positional_encoding = nn.Embedding(num_embeddings=sequence_lenght, embedding_dim=num_embeddings)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(n_embeddings=num_embeddings, number_heads=num_heads, dimensionality_ff=dimensionality_ff, dropout=dropout) for _ in range(n_layers)])
        self.final_norm = nn.RMSNorm(num_embeddings)

    def forward(self, idx : torch.Tensor, encoder_mask : torch.Tensor, casual_mask : torch.Tensor, enc_hidden_states : torch.Tensor):
        B, T = idx.shape
        tokens_embeddings = self.token_embedding_table(idx)

        positional_embeddings = self.positional_encoding(torch.arange(T, device=idx.device))
        x = tokens_embeddings + positional_embeddings

        attention_weights = None
        for decoder_block in self.decoder_blocks:
            x, attention_weights = decoder_block(x, encoder_mask, casual_mask, enc_hidden_states)

        x = self.final_norm(x)
        return x, attention_weights


class DecoderBlock(nn.Module):
    """
        DecoderBlock transformer from attention is all you need
    """
    def __init__(self, n_embeddings : int, number_heads : int, dimensionality_ff : int, dropout : float) -> None:
        super().__init__()
        head_size = n_embeddings // number_heads
        assert n_embeddings % number_heads == 0, f"The number of embeddings: {n_embeddings}, must be divisionable for the number of heads {number_heads}"
        
        #First Block Masked Multi-Head Attention of the Decoder
        self.self_attention = MultiHeadAttention(num_heads=number_heads, head_size=head_size, n_embeddings=n_embeddings, dropout=dropout)
        self.ln1 = nn.RMSNorm(n_embeddings)
        
        #Second Block Cross Multi-Head Attention of the Decoder
        self.cross_attention = MultiHeadAttention(num_heads=number_heads, head_size=head_size, n_embeddings=n_embeddings, dropout=dropout)
        self.ln2 = nn.RMSNorm(n_embeddings)

        #Third Block MLP
        self.fnn = MLP(n_embeddings=n_embeddings, dimensionality_ff=dimensionality_ff, dropout=dropout)
        self.ln3 = nn.RMSNorm(n_embeddings)


        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x : torch.Tensor, encoder_mask : torch.Tensor, casual_mask : torch.Tensor, enc_hidden_states : torch.Tensor):
        #First Block Masked Multi-Head Attention of the Decoder
        head = self.ln1(x)
        self_attention, _ = self.self_attention(head, head, casual_mask)
        x = x + self_attention
        
        #Second Block Cross Multi-Head Attention of the Decoder
        head2 = self.ln2(x)
        casual_self_attention, attention_weights = self.cross_attention(head2, enc_hidden_states, encoder_mask)
        x = x + casual_self_attention

        #Third Block MLP
        head3 = self.ln3(x)
        x = x + self.fnn(head3)
        
        return x, attention_weights