import torch
import torchvision
from utils.EarlyStopping import EarlyStopping
from data_preprocessing.vision_dataset import imagenette2Dataset
from model.ViT import VisionTransformer
import tqdm
from utils.train_val_test_fn import train_function, validation_function
from torch.utils.tensorboard import SummaryWriter # type: ignore
import os
import torch.nn as nn

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
    num_epochs = 50
    early_stopping = EarlyStopping(patience=patience, mode="max" ,verbose=True, path='fine_tuned_model.pt')
    
    
    #Load the dataset
    dataset_birds_images = imagenette2Dataset(data_path="bird_species_dataset", image_size=224, batch_size=batch_size) 
    
    #Create the dataloaders
    train_loader, validation_loader = dataset_birds_images.create_dataloaders()
    
    #Counting the total of classes of the birds for the finetuning -> 525
    number_classes_birds = len(next(os.walk("bird_species_dataset/train"))[1])
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
    
   
    vision_transformer.load_state_dict(torch.load('best_model_vit_small.pt', map_location=device, weights_only=False))
    
    
    #Froze all the previous layers because we are making the finetuning
    for param in vision_transformer.parameters():
        param.requires_grad = False
        
    vision_transformer.final_head_mlp = nn.Linear(in_features=d_dimensionality, out_features=number_classes_birds).to(device)
    
    optimizer_fine_tuning = torch.optim.AdamW(vision_transformer.final_head_mlp.parameters(), lr=3e-4, weight_decay=0.05)
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_fine_tuning, 
            T_max=100, 
            eta_min=1e-6)
    
    print(f"Model parameters that are being Finetuned: {sum(p.numel() for p in vision_transformer.final_head_mlp.parameters())/1e6:.2f} M parameters")
    
    print("Finetuning of the Vision Transformer is starting")
    
    
    tensor_board_writer = SummaryWriter(log_dir=f"runs_finetuning")
    
    for epoch in tqdm.tqdm(range(num_epochs), desc="Training Progress"):
            train_loss = train_function(vision_transformer, train_loader, optimizer_fine_tuning, device) # type: ignore
            
            val_loss, val_acc = validation_function(vision_transformer, validation_loader, device ) # type: ignore
            
            # Step the learning rate scheduler every epoch
            lr_scheduler.step()
            
            print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc * 100:.2f}%")
            
            early_stopping(val_acc, vision_transformer)
            tensor_board_writer.add_scalars("Loss", {
                        "Train": train_loss,
                        "Val": val_loss
                        }, epoch)
            
            tensor_board_writer.add_scalar("Accuracy/Val", val_acc, epoch)
            # Grab a single batch from the validation loader
            images, labels = next(iter(validation_loader))
            
            img_grid = torchvision.utils.make_grid(images[:16], normalize=True)
            tensor_board_writer.add_image("Sample_Validation_Images", img_grid, epoch)
                    
            if early_stopping.early_stop:
                print("Finished because of the early Stopping")
                break 
    tensor_board_writer.close()


if __name__ == "__main__":
    main()