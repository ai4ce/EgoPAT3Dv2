import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    # sine wave positional ecoding
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__() 
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # this makes parameters not affected by optimizer

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerPureEncoder(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, mask_flag: bool = True):
        '''
        d_model (int): the number of expected features in the input (required).
        nhead (int): the number of heads in the multiheadattention models (required).
        dim_hid (int): the dimension of the feedforward network model (default=2048).
        nlayers (int): the number of layers in the encoder
        mask_flag (bool): whether to use a flag or not.
        '''
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=dropout, batch_first=False)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        self.d_model = d_model
        self.mask_flag = mask_flag # whether we apply masking on the encoder side
        # self.decoder = nn.Linear(d_model, ntoken)

        # self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor, padding_mask: Tensor = None) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size, d_model]
            nopeak_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, d_model]
        """
        src = self.pos_encoder(src)

        if self.mask_flag:
            output = self.transformer_encoder(src, mask = src_mask, src_key_padding_mask = padding_mask)
        else:
            output = self.transformer_encoder(src)
        # output = self.decoder(output)
        return output
