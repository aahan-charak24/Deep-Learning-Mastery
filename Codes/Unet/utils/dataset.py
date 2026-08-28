import os
import cv2
import torch
import numpy as np


from torch.utils.data import Dataset

class KVASIR(Dataset):
    #constructor
    def __init__(self, image_dir, mask_dir, transform = None):
        
        super(KVASIR, self).__init__()
        
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        #get the images
        self.images = sorted(os.listdir(self.image_dir))
        
    
    #length function
    def __len__(self):
        return len(self.images)
    
    #get a image example
    def __getitem__(self, index):
        #get image name
        image_name = self.images[index]
        
        #path of image
        image_path = os.path.join(self.image_dir, image_name)
        #mask path
        mask_path = os.path.join(self.mask_dir, image_name)
        
        #get the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        #get the mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        #convert mask to appropriate format and normalize
        mask = mask.astype(np.float32)
        mask = mask / 255.0
        
        #check if transform or not
        if self.transform is not None:
            
            augmented = self.transform(
                image = image, mask = mask
            )

            image = augmented['image']
            mask = augmented['mask'].float()
            
        mask = mask.float().unsqueeze(0)
        
        return image, mask
        
        
        
        
        
        
        
        
    