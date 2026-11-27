from torch import nn
import torch
import torch.nn.functional as F
from model.encoder import Encoder
from model.decoder import Decoder
import numpy as np
import math


class T5_Transformer(nn.Module):
    """
    Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)
    """
    def __init__(self,vocab_size : int, num_embeddings : int, sequence_lenght : int, num_heads : int, n_layers : int, dimensionality_ff : int, dropout : float) -> None:
        super().__init__()
        self.sequence_lenght = sequence_lenght
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=num_embeddings) #The embedding table
        self.encoder = Encoder(num_embeddings=num_embeddings, sequence_lenght=sequence_lenght, num_heads=num_heads, n_layers=n_layers, dimensionality_ff=dimensionality_ff, token_embedding_table=self.token_embedding_table, dropout=dropout)
        self.decoder = Decoder(num_embeddings=num_embeddings, sequence_lenght=sequence_lenght, num_heads=num_heads, n_layers=n_layers, dimensionality_ff=dimensionality_ff, token_embedding_table=self.token_embedding_table, dropout=dropout)
        self.lm_head = nn.Linear(num_embeddings, vocab_size, bias=False)
        self.tril: torch.Tensor
        self.register_buffer('tril', torch.tril(torch.ones(sequence_lenght, sequence_lenght)))

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            self.switch_init_(module)
        
        elif isinstance(module, nn.Embedding):
             nn.init.normal_(module.weight, mean=0.0, std=0.02) #Gaussian initialization of the weights
        
    def switch_init_(self, linear : nn.Linear, s : float = 0.1):
        fan_in = linear.weight.shape[1] #The fan in (out_features, in_features)
        std = math.sqrt(s / fan_in)
        nn.init.trunc_normal_(linear.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)
        
        
    def forward(self, inputs_ids : torch.Tensor, attention_mask : torch.Tensor , decoder_inputs_ids : torch.Tensor, labels=None):
        B, T_dec  = decoder_inputs_ids.shape
        enc_mask = attention_mask.unsqueeze(1)
        
        x  = self.encoder(inputs_ids, enc_mask)
        causal_mask = self.tril[:T_dec, :T_dec]
        x, attention_weights = self.decoder(decoder_inputs_ids, enc_mask, causal_mask, x)
        
        logits = self.lm_head(x)

        if labels is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C )# 2D (B*T, C)
            labels = labels.view(B*T)
            loss = F.cross_entropy(logits, target=labels)



        return logits, loss, attention_weights
