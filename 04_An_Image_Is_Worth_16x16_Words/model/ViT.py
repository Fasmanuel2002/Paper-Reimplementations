import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim






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
        return x

        
        
        