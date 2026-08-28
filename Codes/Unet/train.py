import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from models.unet import Unet
from albumentations.pytorch import ToTensorV2
from utils.dataset import KVASIR
import torch.optim as optim
from tqdm import tqdm
from models.unet import DiceLoss
import config

#faster convolutions
torch.backends.cudnn.benchmark = True

#training transform
train_transform = A.Compose([
    A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
    A.HorizontalFlip(p = 0.5),
    A.VerticalFlip(p = 0.2),
    A.ShiftScaleRotate(
        shift_range =  (-0.08, 0.08),
        scale_range = (-0.2, 0.2),
        rotate_range = (-25, 25),
        p = 0.4
        ),
    
    A.Normalize(
        mean = (0.485, 0.456, 0.406),
        std = (0.229, 0.224, 0.225)
    ),
    ToTensorV2()
    
])



#train dataset
train_dataset = KVASIR(
    image_dir = config.TRAIN_IMG_DIR, 
    mask_dir = config.TRAIN_MASK_DIR, 
    transform = train_transform,
    )


#train loader
train_loader = DataLoader(train_dataset, batch_size = config.BATCH_SIZE, shuffle = True)


#unet model
model = Unet(num_classes = 1).to(device=config.DEVICE)

#loss function
criterion = nn.BCEWithLogitsLoss()
diceLoss = DiceLoss()

#optimizer
optimizer = optim.Adam(lr = config.LR, params = model.parameters())

#training loop
for epoch in range(config.EPOCHS):
    #set model to train
    model.train()
    
    #epoch loss calc
    epoch_loss = 0
    
    loop = tqdm(
        train_loader, desc = f"Epoch [{epoch+1} / {config.EPOCHS}]"
    )
    
    #loop through batches
    for images, masks in loop:
        
        #images to gpu
        images = images.to(config.DEVICE)
        
        #mask to gpu
        masks = masks.float().to(config.DEVICE)
        
        #forward pass
        outputs = model(images)
        
        #loss
        loss = criterion(outputs, masks) + diceLoss(outputs, masks)
        
        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #add to epoch loss
        epoch_loss += loss.item()
        
        #tqdm loop to display output
        loop.set_postfix(loss = loss.item())
        
    #average loss for batch
    batch_loss = epoch_loss / len(train_loader)
    
    #print epoch info
    print(f"Epoch [{epoch + 1} / {config.EPOCHS}]")
    print(f"Average train loss: {batch_loss:.4f}")


#save the trained model
torch.save(model.state_dict(), config.MODEL_SAVE_PATH)