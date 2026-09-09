
import torch
from utils.EarlyStopping import EarlyStopping
from data_preprocessing.vision_dataset import imagenette2Dataset
from model.ViT import VisionTransformer
import tqdm
from utils.train_val_fn import train_function, validation_function
from torch.utils.tensorboard import SummaryWriter # type: ignore

def main():
    # Hyperparameters
    n_layers = 6
    d_dimensionality = 256
    mlp_size = 768
    n_heads = 4
    dropout = 0.1
    patch_size = 16
    input_channels = 3
    image_size = 224
    n_classes = 10
    batch_size = 32 
    patience = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("GPU is avaible:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Model of GPU:", torch.cuda.get_device_name(0))
        
    torch.manual_seed(1337)
    
    #number of epochs for training
    num_epochs = 50
    early_stopping = EarlyStopping(patience=patience, mode="max" ,verbose=True, path='best_model.pt')

    
    #Load the dataset
    dataset = imagenette2Dataset(data_path="imagenette2", image_size=224, batch_size=batch_size) 

    #Create the dataloaders
    train_loader, validation_loader = dataset.create_dataloaders()
    
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
                                       device=device).to(device)
    
    optimizer = torch.optim.AdamW(vision_transformer.parameters(), lr=3e-3, weight_decay=0.3)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=50, 
        eta_min=1e-6
    )
    
    print(f"Model parameters: {sum(p.numel() for p in vision_transformer.parameters())/1e6:.2f} M parameters")
    
    print("Training is starting")
    
    tensor_board_writer = SummaryWriter(log_dir=f"runs")
    
    for epoch in tqdm.tqdm(range(num_epochs), desc="Training Progress"):
        train_loss = train_function(vision_transformer, train_loader, optimizer, device)
        
        val_loss, val_acc = validation_function(vision_transformer, validation_loader, device )
        
        # Step the learning rate scheduler every epoch
        lr_scheduler.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc * 100:.2f}%")
        
        early_stopping(val_acc, vision_transformer)
        tensor_board_writer.add_scalars("Loss", {
                    "Train": train_loss,
                    "Val": val_loss
                    }, epoch)
        
        if early_stopping.early_stop:
            print("Finished because of the early Stopping")
            break 
    tensor_board_writer.close()
    
if __name__ == "__main__":
    main()