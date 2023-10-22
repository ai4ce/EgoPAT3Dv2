import torch
from torch import nn, Tensor
import math

class PositionalEncoding(nn.Module):
    # sine wave positional ecoding
    def __init__(self, d_model: int,  max_len: int = 100):
        super().__init__() 

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model) # pe is of size [seq_length, batch_size, embedding_dim]
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # this makes parameters not affected by optimizer

    def forward(self, x: Tensor, idx: int) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embedding_dim]
        """
        if torch.count_nonzero(x) == 0: # don't add PE to zero padding
            return x
        else:
            x = x + self.pe[idx]
            return x