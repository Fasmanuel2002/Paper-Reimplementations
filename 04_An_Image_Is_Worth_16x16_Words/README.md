# PyTorch Reimplementation of "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ViT Encoder-Only)

This directory contains a PyTorch reimplementation based on the paper [An Image Is Worth 16×16 Words](https://arxiv.org/abs/2010.1192) (Dosovitskiy et al. 2020).

This project reimplements a pre-trained and fine-tuned Vision Transformer (Encoder-only architecture). It focuses on the methodology described in Section 3 of the paper. This pre-trained model is designed for classifying 10 classes from the [imagenette2](https://www.kaggle.com/datasets/adityakane/imagenette2) dataset using the `[CLS]` token. 

The resulting model features around *21.5 million* parameters and was pre-trained and fine-tuned leveraging an *NVIDIA L4 GPU*.

## Image Processing / Patch Embedding
The standard Transformer receives as input a 1D sequence of token embeddings. To handle 2D images, the image $x \in \mathbb{R}^{H \times W \times C}$ is reshaped into a sequence of flattened 2D patches $x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$, where (H, W) are the resolution of the original image, C is the number of channels (P, P) is the resolution of each image patch, and $N = HW / P^2$ are the resulting number of the patches. As the original paper tell


In this reimplementation for the **ImageNette2** dataset, input images of resolution $224 \times 224$ are split into a $14 \times 14$ grid using a patch size of $16 \times 16$, yielding a total of **196 patches** per image. As demonstrated in `vision_dataset.py` and `ViT.py` in the `PatchEmbedding` class .