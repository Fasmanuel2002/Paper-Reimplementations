import torch
import torchvision
from utils.EarlyStopping import EarlyStopping
from data_preprocessing.vision_dataset import imagenette2Dataset
from model.ViT import VisionTransformer
import tqdm
from utils.train_val_test_fn import test_function
from torch.utils.tensorboard import SummaryWriter # type: ignore


def main():
    # Hyperparameters
    n_layers = 12             
    d_dimensionality = 384    
    mlp_size = 1536           
    n_heads = 6
    dropout = 0.1
    patch_size = 16
    input_channels = 3
    image_size = 224
    n_classes = 10
    batch_size = 64
    patience = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("GPU is avaible:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Model of GPU:", torch.cuda.get_device_name(0))
        
    torch.manual_seed(1337)
    
    test_dataset = imagenette2Dataset(data_path="imagenette_test", image_size=224, batch_size=batch_size)
    
    test_loader = test_dataset.create_test_dataloader()
    
    #Create the Vision Transformer model
    vision_transformer = VisionTransformer(n_layers=n_layers,
                                           d_dimensionality=d_dimensionality,
                                           mlp_size=mlp_size,
                                           n_heads=n_heads,
                                           dropout=dropout,
                                           in_channels=input_channels,
                                           patch_size=patch_size,
                                           image_size=image_size,
                                           num_classes=n_classes,
                                           device=device).to(device) # pyright: ignore[reportArgumentType]
    
    vision_transformer.load_state_dict(torch.load('best_model_vit_small.pt', map_location=device, weights_only=True))
    vision_transformer.eval()
    
    test_loss, test_accuracy, macro_f1, weighted_f1 = test_function(vision_transformer, test_loader, device) # type: ignore


if __name__ == "__main__":
    main()