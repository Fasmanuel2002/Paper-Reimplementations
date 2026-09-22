# PyTorch Reimplementation of "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ViT Encoder-Only)

This directory contains a PyTorch reimplementation based on the paper [An Image Is Worth 16×16 Words](https://arxiv.org/abs/2010.1192) (Dosovitskiy et al. 2020).

This project reimplements a pre-trained and fine-tuned Vision Transformer (Encoder-only architecture). It focuses on the methodology described in Section 3 of the paper. This pre-trained model is designed for classifying 10 classes from the [imagenette2](https://www.kaggle.com/datasets/adityakane/imagenette2) dataset using the `[CLS]` token. 

The resulting model features around *21.5 million* parameters and was pre-trained and fine-tuned leveraging an *NVIDIA L4 GPU*.

## Image Processing / Patch Embedding
The standard Transformer receives as input a 1D sequence of token embeddings. To handle 2D images, the image $x \in \mathbb{R}^{H \times W \times C}$ is reshaped into a sequence of flattened 2D patches $x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$, where (H, W) are the resolution of the original image, C is the number of channels (P, P) is the resolution of each image patch, and $N = HW / P^2$ are the resulting number of the patches. As the original paper tell


In this reimplementation for the **ImageNette2** dataset, input images of resolution $224 \times 224$ are split into a $14 \times 14$ grid using a patch size of $16 \times 16$, yielding a total of **196 patches** per image. As demonstrated in `vision_dataset.py` and `ViT.py` in the `PatchEmbedding` class .



![vision_transformer_overview](../assets/Vision_transformer_patch_embedding.png)
**Figure 1: Model Overview (Taken from original paper)** As explained in the image preprocessing section, the input image is split into fixed-size patches. The model then linearly embeds each patch, adds position embeddings, and feeds the resulting sequence of vectors to a standard Transformer encoder.

## Transformer Encoder Architecture
![vision_transformer_encoder_only](../assets/TransformerEncoderOnly_VIT.png)

**Figure 2: Transfomer Encoder (Taken from original paper)**: The Transformer Encoder Only architecture


The model architecture utilizes the standard Transformer encoder from [Attention Is All You Need](https://arxiv.org/abs/1706.03762). It consists of alternating layers of Multi-Head Self-Attention (MSA) and Multi-Layer Perceptron (MLP) blocks. Following modern architecture designs, Layer Normalization (LN) is applied *before* every block (Pre-Norm), and residual connections are applied *after* every block.

The internal forward pass is structured as follows:
* **Patch + Position Embedding** concatenated with a learnable `[CLS]` token
* **Layer Normalization** applied before both the attention and MLP blocks
* **Multi-Head Self-Attention (MSA)** to model global visual context between image patches
* **Multi-Layer Perceptron (MLP)**  with GELU activation function
* **Residual Connections** adding the input of each sub-layer to its output
* **Final Classification Head**: A linear projection layer that takes the final state of the `[CLS]` token and outputs the raw class logits 


## Pre-Training Details & Results

The ViT model was trained from scratch on the **ImageNette2** dataset a 10-class dataset to learn base visual representations.

### Model Hyperparameters
* **Layers:** 12
* **Hidden Dimensionality:** 384
* **MLP Size:** 1536
* **Attention Heads:** 6
* **Patch Size:** 16x16
* **Dropout:** 0.1
* **Number of Classes:** 10

### Training Setup
* **Optimizer:** AdamW 
* **Batch Size:** 64
* **Image Resolution:** 224x224 (3 channels)
* **Regularization:** Early Stopping (Patience: 5 epochs) and Weight Decay ($L_2$ regularization: 0.05)
* **Learning Rate Scheduler:** CosineAnnealingLR
* **Loss Function:** Cross-Entropy Loss

