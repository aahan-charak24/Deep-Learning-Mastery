import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Country211
from torchvision.transforms import v2
import os


#transforms
def get_transforms():
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype = torch.float32, scale = True),
        v2.RandomResizedCrop((224, 224), (0.5, 1)),
        v2.RandomHorizontalFlip(0.5),
        v2.RandAugment(num_ops = 2, magnitude = 9),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    ])
    
    test_transform = v2.Compose([
        v2.ToImage(),
         v2.ToDtype(dtype = torch.float32, scale = True),
         v2.Resize((224, 224)), 
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform


#get dataloaders
def get_dataloaders(data_dir : str, batch_size : int, num_workers:int = os.cpu_count(), split = "train"):
    train_tf, test_tf = get_transforms()
    
    if split == "train":
        #dataset
        train_set = Country211(root = data_dir, split = "train", transform = train_tf, download = True)
        val_set = Country211(root = data_dir, split = "valid", transform = test_tf, download = True)
        train_loader= DataLoader(train_set, batch_size, shuffle = True, num_workers = num_workers, pin_memory = True,
                                     persistent_workers = True if num_workers > 0 else False)
        val_loader = DataLoader(val_set, batch_size, shuffle = False, num_workers = num_workers,  pin_memory = True,
                                     persistent_workers = True if num_workers > 0 else False)
        
        return train_loader, val_loader

    test_set = Country211(root = data_dir, split = "test", transform = test_tf, download = True)
    
    #loader
    #pin memory: # Enables fast CPU-to-GPU direct memory transfer (DMA)
    #persistent_workers: # Keeps background workers alive between epochs to avoid respawn overhead
    
    test_loader = DataLoader(test_set, batch_size, shuffle = False, num_workers = num_workers , pin_memory = True,
                             persistent_workers = True if num_workers > 0 else False)
    

    
    return test_loader
    
    