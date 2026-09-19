import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plots a confusion matrix using seaborn heatmap.

    Args:
        y_true (list or array): True labels.
        y_pred (list or array): Predicted labels.
        class_names (list): List of class names corresponding to the labels.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png", bbox_inches='tight', dpi=300)
    plt.show()
    print("Confusion matrix saved to confusion_matrix.png")



def visualize_attention_weights(image_tensor : torch.Tensor, attention_weights, patch_size : int = 16):
    
    
    #Make the avarage of the attention weights across all heads, so we have and consesus and layers and eliminate the batch dimension
    avg_attention_weights = torch.mean(attention_weights, dim=1).squeeze(0).cpu().detach()
    
    # Extarct the attention weights corresponding to the class token[CLS] and we take the column to see all the atention to take a classification decision
    cls_attention = avg_attention_weights[0, 1:]  # Exclude the class token itself
    
    # Reshape the vector 1D (196) to a 2D grid (14x14) for visualization
    grid_size = int(np.sqrt(cls_attention.size(0))) # 196 patches -> 14x14 grid
    cls_attention_2d = cls_attention.view(grid_size, grid_size).numpy() # 196 patches -> 14x14 grid
    
    # Make the weights to be as the image of the same size as the input image
    original_size = grid_size * patch_size
    cls_attention_resized = cv2.resize(src=cls_attention_2d, dsize=(original_size, original_size)) # Resize the attention map to match the original image size 14x14 -> 224x224
    
    # Normalize the attention weights for better visualization
    cls_attention_normalized = (cls_attention_resized - np.min(cls_attention_resized)) / (np.max(cls_attention_resized) - np.min(cls_attention_resized))
    
    # Desnormalize the input image tensor to convert it back to a displayable format
    img = image_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean  # Denormalize
    img = np.clip(img, 0, 1)  # Clip values to [0, 1] range for display
    
    # Display the original image and the attention map side by side
    fig, axest = plt.subplots(1, 2, figsize=(12, 6))
    
    axest[0].imshow(img)
    axest[0].set_title('Original Image')
    axest[0].axis('off')
    
    # Dislpay attention map with a color map
    axest[1].imshow(img, alpha=0.5)  # Show the original image with some transparency
    axest[1].imshow(cls_attention_normalized, cmap='jet', alpha=0.5)  # Overlay the attention map
    axest[1].set_title('Attention Map Overlay, [CLS]')
    axest[1].axis('off')
    
    plt.savefig("attention_map_overlay.png", bbox_inches='tight', dpi=300)
    print("Attention map saved to attention_map_overlay.png")
    plt.show()
    