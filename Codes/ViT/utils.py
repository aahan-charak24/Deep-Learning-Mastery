import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt


def save_training_curves(history, filename="images/training_curves.png"):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Create a figure with 2 subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 2. Plot Loss Curves (Left Subplot)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['valid_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 3. Plot Accuracy Curves (Right Subplot)
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['valid_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=10)
    ax2.set_ylabel('Accuracy', fontsize=10)
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # 4. Clean up layout and save to disk
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close() # Closes the figure window to free up system memory
    print(f"Training curves successfully saved to {filename}")

#blank token
def append_blank_token(xb : torch.Tensor, device)->torch.Tensor:
  '''
  Given an input image token tensor, append a blank token across its token dimension.
  Input: xb => A batch of input image tokens

  Return b_t: tensor with blank token appended to the input
  '''
  B, _, C = xb.shape
  b_t = torch.zeros(B, 1, C)
  b_t = b_t.to(device)
  #concatenate
  b_t = torch.cat((b_t, xb), dim = 1)

  return b_t

#function to calculate positional encoding
def get_positional_matrix(seq_len : int, embedding_dim : int) -> torch.Tensor:
    '''
    Get positional matrix for positional embeddings
    
    Inputs:
    seq_len : (int) sequence length
    embedding_dim: (int) dimension of the embeddings
    
    Return:
    (torch.Tensor) positional embedding matrix
    
    '''
  #positional matrix
    P = torch.zeros(seq_len, embedding_dim)
    i = 0

    for k in range(seq_len):
        for j in range(embedding_dim):
            i = int(math.floor(j / 2))
            if j % 2 == 0:
                P[k][j] = math.sin(k * (1/(10000 ** ((2 * i) / embedding_dim))))
            else:
                P[k][j] = math.cos(k * (1/(10000 ** ((2 *i) / embedding_dim))))

    return P


#vectorized
def get_positional_matrix_vectorized(seq_len : int, embedding_dim : int):
    '''
        Get positional matrix for positional embeddings
        
        Inputs:
        seq_len : (int) sequence length
        embedding_dim: (int) dimension of the embeddings
        
        Return:
        (torch.Tensor) positional embedding matrix
        
    '''
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

    return P




