import torch
import torch.nn as nn
from torch.nn import functional as f
from transformer import Encoder
from tokenization import PatchTokenization
from embeddings import PositionalEmbedding
from utils import append_blank_token
import json






class ViT(nn.Module):
    '''
    ViT architecture model from original paper.
    Modified a bit
    
    Constructor params:
    config_path : str containg config files for ViT transfomrer
    img_sz : tuple containing image size (C, H, W)
    n_classes: number of classification heads
    
    '''

    def __init__(self, config_path : str, img_sz : tuple[int, int, int], n_classes:int = 10):
        super().__init__()
        #load params
        with open(config_path, "r", encoding = "utf-8") as fh:
            config = json.load(fh)
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #hyperparameters
        n_heads = config["n_heads"]
        embed_dim = config["embed_dim"]
        n_blocks = config["n_blocks"]
        patch_size = config["patch_size"]
        seq_len = (img_sz[1] * img_sz[2]) // patch_size **2
        self.n_classes = n_classes
        #patch tokenization
        self.pt = PatchTokenization(img_sz, patch_size, embed_dim)
        #Positional encoding
        self.pe = PositionalEmbedding(embed_dim, seq_len + 1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        #blocks
        self.blocks = nn.Sequential(*[Encoder(embed_dim, n_heads) for _ in range(n_blocks)])
        #classification head
        self.cls_head = nn.Sequential( 
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, n_classes)
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.pt(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1 )
        x = torch.cat((cls_tokens, x), dim = 1)
        x = x + self.pe(x)
        x = self.blocks(x)
        #get the classification token
        x = x[:,0]
        out = self.cls_head(x)

        return out






