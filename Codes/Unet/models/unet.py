import torch
import torch.nn as nn

#double convolution operation
#we pad 1 to ensure that the final segmentation map is the same size as the input image
def double_convolution(in_channels, out_channels):
    conv_op = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace = True),
        nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace = True)
    )
    
    return conv_op


#main unet module
class Unet(nn.Module):
    def __init__(self, num_classes):
        super(Unet, self).__init__()
        
        #maxpool layer
        self.maxpool2d = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        #contraction path
        #downsampling
        self.down_convolution_1 = double_convolution(3, 64)
        self.down_convolution_2 = double_convolution(64, 128)
        self.down_convolution_3 = double_convolution(128, 256)
        self.down_convolution_4 = double_convolution(256, 512)
        self.down_convolution_5 = double_convolution(512, 1024)
        
        #expanding path
        #upsampling
        self.up_transpose_1 = nn.ConvTranspose2d(
            in_channels = 1024, out_channels = 512, kernel_size = 2, stride = 2
        )
        self.up_convolution_1 = double_convolution(1024, 512)
        
        self.up_transpose_2 = nn.ConvTranspose2d(
            in_channels = 512, out_channels = 256, kernel_size = 2, stride = 2
        )
        self.up_convolution_2 = double_convolution(512, 256)
        
        self.up_transpose_3 = nn.ConvTranspose2d(
            in_channels = 256, out_channels = 128, kernel_size = 2, stride = 2
        )
        
        self.up_convolution_3 = double_convolution(256, 128)
        
        self.up_transpose_4 = nn.ConvTranspose2d(
            in_channels = 128, out_channels = 64, kernel_size = 2, stride = 2
        )
        
        self.up_convolution_4 = double_convolution(128, 64)
        
        #output layer
        self.out = nn.Conv2d(in_channels = 64, out_channels = num_classes, kernel_size = 1)
        
    
    #forward pass
    def forward(self, X):
        #downsampling
        down_1 = self.down_convolution_1(X)
        down_2 =  self.maxpool2d(down_1)
        down_3 = self.down_convolution_2(down_2)
        down_4 = self.maxpool2d(down_3)
        down_5 = self.down_convolution_3(down_4)
        down_6 = self.maxpool2d(down_5)
        down_7 = self.down_convolution_4(down_6)
        down_8 = self.maxpool2d(down_7)
        down_9 = self.down_convolution_5(down_8)
        
        #upsampling
        up_1 = self.up_transpose_1(down_9)

        x = self.up_convolution_1(torch.cat([down_7, up_1], 1))
        
        up_2 = self.up_transpose_2(x)
        x = self.up_convolution_2(torch.cat([down_5, up_2], 1))
        
        up_3 = self.up_transpose_3(x)
        x = self.up_convolution_3(torch.cat([down_3, up_3], 1))
        
        up_4 = self.up_transpose_4(x)
        x = self.up_convolution_4(torch.cat([down_1, up_4], 1))
        
        out = self.out(x)
        
        return out
        
#dice loss for optimization 
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-8):
        super().__init__()
        self.smooth = smooth
    #forward
    def forward(self, preds, targets):
        #get the sigmoid outputs [0-1]
        preds = torch.sigmoid(preds)
        
        #flatten per image
        preds = preds.view(preds.size(0), -1)
        targets  = targets.view(targets.size(0), -1)

        #intersection (overlapping region, where 1 both that overlaps)
        intersection = (preds * targets).sum()
        #union
        union = preds.sum() + targets.sum()
        #dice score (2* iou)
        dice = ( 2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice.mean()      
        
        
        
# if __name__ == "__main__":
#     input_image = torch.rand((1, 3, 512, 512))
#     model = Unet(num_classes=10)
#     #total parameters in the model
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f'Total parameters: {total_params}')
    
#     #total trainable parameters
#     total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
#     print(f'Total trainable parameters: {total_trainable_params}')
    
#     outputs = model(input_image)
    
#     print(outputs.shape)
    

