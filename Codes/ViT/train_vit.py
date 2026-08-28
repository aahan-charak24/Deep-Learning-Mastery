import torch
import torch.nn as nn
from vit import ViT
from data import get_dataloaders
import torch.optim as optim
from utils import save_training_curves
from tqdm import tqdm
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

#global variables
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1337)

#hyperparameters
batch_size = 32
lr = 3e-4
n_epochs = 2
history = {
    "train_loss":[],
    "valid_loss":[],
    "train_acc" : [],
    "valid_acc": []
}
config_path  = "vit_config.json"
checkpoint_path = "model/best_model_vit.pth"

if __name__ == "__main__":
    #dataloader
    train_loader, val_loader = get_dataloaders("dataset/", batch_size, num_workers = 4, split = "train")
    
    n_classes = len(train_loader.dataset.classes)
    
    #model
    model = ViT(config_path, (3, 224, 224), n_classes)
    model = model.to(device)
    
    
    #optim
    optim = optim.AdamW(model.parameters(), lr, weight_decay=0.05)
    
    #warmups 
    warmup_epochs = 5
    sch1 = LinearLR(optim, start_factor = 1e-3, total_iters = warmup_epochs)
    sch2 = CosineAnnealingLR(optim, T_max = n_epochs - warmup_epochs, eta_min = 1e-6)
    scheduler = SequentialLR(optim, schedulers = [sch1, sch2], milestones = [warmup_epochs])
    
    loss_fn = nn.CrossEntropyLoss()
    
    best_acc = -1000
  
    #model train
    for i in range(n_epochs):
        loss_ep = 0
        correct = 0
        total_samples = 0
        #train loop
        model.train()
        for xb, yb in tqdm(train_loader):
            
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss_ep += loss.item() * xb.size(0)
            loss.backward()
            optim.step()
            pred = torch.argmax(pred, dim = 1)
            correct += (pred == yb).sum().item()
            total_samples += yb.size(0)
        
        #once through full batch done
        train_acc = correct / total_samples
        history["train_acc"].append(train_acc)
        train_loss = loss_ep / total_samples
        history['train_loss'].append(train_loss)
        

        
        #val loop
        with torch.no_grad():
            loss_ep = 0
            correct = 0
            total_samples = 0
            model.eval()
            for xb, yb in tqdm(val_loader):
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss_ep += loss.item() * xb.size(0)
                pred = torch.argmax(pred, dim = 1)
                correct += (pred == yb).sum().item()
                total_samples += yb.size(0)
            #full batch done
            val_acc = correct / total_samples
            val_loss = loss_ep / total_samples
            history['valid_acc'].append(val_acc)
            history['valid_loss'].append(val_loss)
            
            #checkpoints
            if val_acc > best_acc:
                print(f"Best val acc improved from {best_acc} to {val_acc}")
                best_acc = val_acc
                torch.save(model.state_dict(), checkpoint_path)
        
        print(f"Epoch {i+1}, train loss: {train_loss:.6f} train acc: {train_acc:.6f}, val loss: {val_loss:.6f} val acc:{val_acc:.6f}")
        
        #scheduler
        scheduler.step()
    #save plot
    save_training_curves(history)
    
            
            

            
            

    #model save checkpoints etc


