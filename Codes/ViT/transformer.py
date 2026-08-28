import torch
import torch.nn as nn
from torch.nn import functional as f

#self attention head
class Head(nn.Module):
  '''
  Calculates self-attention scores given a batch of input sequences.

  Constructor inputs:
  head_size: (int) size of output attention head.
  embed_dim: (int) size of input sequence embeddings
  '''
  def __init__(self, head_size : int, embed_dim : int):
    super().__init__()
    self.head_size = head_size
    self.query = nn.Linear(embed_dim, head_size, bias = False)
    self.key = nn.Linear(embed_dim, head_size, bias = False)
    self.value = nn.Linear(embed_dim, head_size, bias = False)

  def forward(self, x:torch.Tensor) -> torch.Tensor:
    q = self.query(x)
    k = self.key(x)
    v = self.value(x)
    #att weight
    wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5 #(B, T, head_size) * (B, head_size, T) => (B, T, T)
    wei = f.softmax(wei, dim = -1)
    out = wei @ v # (B, T, T) * (B,T ,head_size) => (B, T, head_size)
    return out



#mhsa module
class MHSA(nn.Module):
  '''
  Calculates mhsa output given an input batch sequence.
  
  Constructor inputs:
  n_heads: (int) number of input heads
  embed_dim: (int) input embedding dimensions
  head_size: (int) size of each sa head

  '''
  def __init__(self, n_heads : int, embed_dim : int, head_size : int):
    super().__init__()
    self.n_heads = n_heads
    self.heads = nn.ModuleList([Head(head_size, embed_dim) for _ in range(n_heads)])
    self.proj = nn.Linear(embed_dim, embed_dim)
    self.dropout = nn.Dropout(0.2)

  def forward(self, x:torch.Tensor)-> torch.Tensor:
    x = torch.cat([head(x) for head in self.heads], dim = -1)
    out = self.dropout(self.proj(x))
    return out


#transformer encoder
class Encoder(nn.Module):
  '''
  Encoder block of a transformer:
  Constructor params:
  embed_dim: (int) embedding dimension of the batch of input sequence
  n_heads: (int) number of heads in the mhsa layer

  '''
  def __init__(self, embed_dim : int, n_heads : int):
    super().__init__()
    self.embed_dim = embed_dim
    assert embed_dim % n_heads == 0, "embedding dimensions must be perfectly divisble by number of heads"
    head_size = embed_dim // n_heads
    self.ln1 = nn.LayerNorm(embed_dim)
    self.mhsa = MHSA(n_heads, embed_dim, head_size)
    self.ln2 = nn.LayerNorm(embed_dim)
    self.mlp = nn.Sequential(
        nn.Linear(embed_dim, embed_dim * 4),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(embed_dim *4, embed_dim),
        nn.Dropout(0.2)
    )
    
  def forward(self, x : torch.Tensor) -> torch.Tensor:
    x = x + self.mhsa(self.ln1(x))
    x = x + self.mlp(self.ln2(x))
    return x
