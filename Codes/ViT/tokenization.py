import torch.nn as nn
import torch

class PatchTokenization(nn.Module):
  '''
  Module to perform patch-tokenization given a batch of images

  Input:
  img_sz: tuple(int channels, int height, int weight) (dimensions of the input image)
  patch_size: int (patch_size of the grids)
  embed_dim: int (dimensions of the tokens)

  '''
  def __init__(self, img_sz : tuple[int, int, int], patch_size : int, embed_dim : int):
    super().__init__()
    self.img_sz = img_sz
    self.patch_size = patch_size
    self.embed_dim = embed_dim
    C, H, W = self.img_sz
    assert H % self.patch_size == 0, "Image height must be evenly divisible by patch size"
    assert W % self.patch_size == 0, "Image width my be evenly divisible by patch size"
    self.num_patches = int((H*W)/ self.patch_size ** 2)

    #projection to embedding dimensions
    self.proj = nn.Conv2d(in_channels = C, out_channels = self.embed_dim, kernel_size=self.patch_size, stride = self.patch_size)

  def forward(self, x : torch.Tensor) -> torch.Tensor:
    x = self.proj(x) # (B, embed_dim, H/P_SIZE, W/P_SIZE)
    x = x.flatten(2) # (B, embed_dim, num_patches)
    x = x.transpose(1, 2) # (B, num_patches, embed_dim)

    return x




