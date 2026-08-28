import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
  '''
  Calculate the positinal embedding matrix.

  Inputs:
  seq_len: (int) max sequence length of tokens to expect
  embedding_dim: (int) dimension of the embedding vector

  '''
  def __init__(self, embedding_dim : int, seq_len : int = 5000):

    super().__init__()
    self.seq_len = seq_len
    self.embedding_dim = embedding_dim
    # k: (seq_len, 1)
    k = torch.arange(seq_len, dtype = torch.float32).unsqueeze(1)
    #i: (1, embedding_dim // 2)
    i = torch.arange(0, embedding_dim, 2, dtype = torch.float32)

    #div: (1, embedding_dim // 2)
    div = 1.0 / ((10000) ** (i / embedding_dim))

    # (seq_len, embedding_dim)
    P = torch.zeros(seq_len, embedding_dim, dtype = torch.float32)

    #even dims
    P[:, ::2] = torch.sin(k*div) #broadcasted mult to (seq_len, embedding_dim//2)
    #odd dims
    P[:, 1::2] = torch.cos(k*div)

    #(1, seq_len, embedding_dim)
    P = P.unsqueeze(0)

    #register buffer: tensor not a parameter, modules state part
    #persistent false: dont include this buffer when saving model state
    self.register_buffer('pe', P, persistent = True)


  def forward(self, x : torch.Tensor) -> torch.Tensor:
    '''
    Return the positional embedding vector.
    If seq len is smaller than the positional vector size. trim the resulting positional vector.
    Inputs:
    x -> (torch.Tensor) input batch of embedded tokens

    Returns:
    pe -> (torch.tensor) batch broadcastable positional embedding matrix
    '''
    return self.pe[:, :x.size(1)] # (1, x_seq(size), embed_dim)

