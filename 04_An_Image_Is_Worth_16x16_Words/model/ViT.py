from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class VisionTransformer(nn.Module):
    def __init__(self, n_heads: int, d_dimensionality: int, mlp_size: int,
                 n_layers: int, image_size: int, patch_size: int, in_channels: int, 
                 num_classes: int, dropout: float, device: str) -> None:
        super().__init__()
        
        self.device = device
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.patch_embedding = PatchEmbedding(d_dimensionality=d_dimensionality, image_size=image_size, patch_size=patch_size, in_channels=in_channels, dropout=dropout)
        self.encoder_blocks = nn.ModuleList([Block(n_heads=n_heads,
                                      d_dimensionality=d_dimensionality,
                                      mlp_size=mlp_size,
                                      dropout=dropout) for _ in range(n_layers)])
        
        self.layer_normalization_final = nn.LayerNorm(d_dimensionality)
        self.final_head_mlp = nn.Linear(d_dimensionality, num_classes)
        
        self.apply(self._init_weights)
        
        
    def _init_weights(self, module : nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.2)
            if module.bias is not None:
                nn.init.zeros_(module.bias) #Make the biases zero
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02) #Gaussian initialization of the weights
            
            
    def forward(self, inputs, targets : Optional[torch.Tensor] = None):
        x = self.patch_embedding(inputs)
        
        all_attention_weights = []
        for block in self.encoder_blocks:
            x, attention_weights = block(x)
            all_attention_weights.append(attention_weights)
        
        x = self.layer_normalization_final(x)
        cls_token_final = x[:, 0] # Shape: (B, d_dimensionality)
        logits = self.final_head_mlp(cls_token_final) # Shape: (B, num_classes)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        #Stack all the attention for make the matrix
        attention_stack = torch.stack(all_attention_weights, dim=1)
        
        return logits, loss, attention_stack
            
        
        
class Block(nn.Module):
    def __init__(self, n_heads : int, d_dimensionality : int, mlp_size : int, dropout : float) -> None:
        super().__init__()
        head_size = d_dimensionality // n_heads
        assert d_dimensionality % n_heads == 0, f"The dimensionality of the images: {d_dimensionality}, must be divisionable for the number of heads {n_heads}"
        
        self.multi_head_attention = MultiHeadAttention(
            n_heads=n_heads,
            d_dimensionality=d_dimensionality,
            head_size=head_size, 
            dropout=dropout)
        
        self.fnn = MLP(d_dimensionality=d_dimensionality, mlp_size=mlp_size, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_dimensionality)
        self.ln2 = nn.LayerNorm(d_dimensionality)
        

    def forward(self, x):
        residual_connection_1 = x
        x, attention_weigths = self.multi_head_attention(self.ln1(x))
        x = x + residual_connection_1 # MHA + first residual connection
        
        residual_connection_2 = x
        x = self.fnn(self.ln2(x))
        x = x + residual_connection_2 #MLP + second residual connection
        
        return x, attention_weigths
    
class MLP(nn.Module):
    def __init__(self, d_dimensionality : int, mlp_size : int, dropout : float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_dimensionality, mlp_size)
        self.fc2 = nn.Linear(mlp_size, d_dimensionality)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        #ReLU completely removes negative values vs GELU smoothly reduces negative values and keeps some of them.
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads : int, d_dimensionality : int, head_size : int, dropout : float) -> None:
        super().__init__()
        
        self.heads = nn.ModuleList([HeadAttention(
            d_dimensionality=d_dimensionality,
            head_size=head_size,
            dropout=dropout
        ) for _ in range(n_heads)])
        assert head_size == d_dimensionality // n_heads, "The headsize must be diviseble by the number of heads and the headsize must be the same"
        self.proj = nn.Linear(d_dimensionality, d_dimensionality)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out_puts, attenton = zip(*[head(x) for head in self.heads])
        out = torch.cat(out_puts, dim=-1)
        out = self.dropout(self.proj(out))
        attention_wieghts = torch.stack(attenton, dim=1)
        return out, attention_wieghts
        

class HeadAttention(nn.Module):
    def __init__(self, d_dimensionality : int, head_size : int, dropout : float) -> None:
        super().__init__()
        """
        Args:
            d_dimensionality (int): Dimensionality of the input of the images (C / d_model)
            head_size (int): Size of the projection space for this head (d_k)
        """
        self.head_size = head_size
        self.queries = nn.Linear(in_features=d_dimensionality, out_features=head_size, bias=False) # What Im looking for?
        self.keys = nn.Linear(in_features=d_dimensionality, out_features=head_size, bias=False) #What do I contain?
        self.values = nn.Linear(in_features=d_dimensionality, out_features=head_size, bias=False) #If you make me attention, what I will give you?
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x : torch.Tensor):
        B, N, C = x.shape
        q = self.queries(x)
        k = self.keys(x)
        v = self.values(x)
        
        wei = (q @ k.transpose(-2, -1)) / (self.head_size ** 0.5) # The formula for attention is all you need B, N, C @ B, C, N -> B, N, N
        
        wei = F.softmax(wei, dim=-1)
        
        attention_weigthts = wei
        
        wei = self.dropout(wei)
        
        output = wei @ v
        
        return output, attention_weigthts

class PatchEmbedding(nn.Module):
    def __init__(self, d_dimensionality : int, image_size : int, patch_size : int, in_channels : int, dropout : float) -> None:
        super().__init__()
        self.number_patches = (image_size // patch_size) ** 2 #The formula to get the number of patches from the paper N = (Height * witdh) / patch_size
        self.projection = nn.Conv2d(in_channels=in_channels, out_channels=d_dimensionality, kernel_size=patch_size, stride=patch_size) # For workin with images, its to make the patches and the dimensionality into d_dimensionality out, H_out and wieghit_out
        self.classification_token = nn.Parameter(torch.randn(1,1, d_dimensionality))
        self.position_embedding = nn.Parameter(torch.randn(1, self.number_patches + 1, d_dimensionality)) #This is because the number of patches will be the size
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x : torch.Tensor):
        B, C, H, W = x.shape
        x = self.projection(x) #B, d_dimensionality, (H, W)
        x = x.flatten(2) #B, d_dimensionality, N_sequence_lenght
        x = x.transpose(-2, -1) #B,N,d_dimensionality
        class_token = self.classification_token.expand(B, -1, -1) #[CLS] = (B, 1, d_dimensionality)
        x = torch.cat([class_token, x], dim=1) # (B, N+1, d_dimensionality)
        x = x + self.position_embedding # (B, N+1, d_dimensionality)
        x = self.dropout(x)
        return x #(B, N+1, d_dimensionality)

        
        
        