import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from .multiHeadAttention import MultiHeadAttention
from .attention import attention
from .utils import positional_encoding

class Decoder_Layer(nn.Module):
    """
    Decoder Block
    
    Parameters
    ----------
    d_model:
        Dimension of the input vector
    nhead:
        Number of heads
    dropout:
        The dropout value
    """
    
    def __init__(self,
                 d_model,
                 nhead,
                 dropout):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        self.Attention = MultiHeadAttention(d_model, nhead)
                
        self.feedForward = nn.Sequential(
            nn.Linear(d_model,64),
            nn.ReLU(),
            nn.Linear(64,d_model),
            nn.Dropout(dropout)
            )
        
        self.layerNorm1 = nn.LayerNorm(d_model)
        self.layerNorm2 = nn.LayerNorm(d_model)
        
    def forward(self, q, kv, mask):
        
        # Attention
        residual = q
        x = self.Attention(query=q, key=kv, value=kv, mask=mask)
        x = self.dropout(x)
        x = self.layerNorm1(x + residual)
        
        # Feed Forward
        residual = x
        x = self.feedForward(x)
        x = self.layerNorm2(x + residual)
        
        return x

class Decoder_p(nn.Module):
    """
    Decoder Block

    Parameters
    ----------
    d_model:
        Dimension of the input vector
    nhead:
        Number of heads
    num_decoder_layers:
        Number of decoder layers to stack
    dropout:
        The dropout value
    """
    def __init__(self,
                 d_model,
                 nhead,
                 num_decoder_layers,
                 dropout):
        super().__init__()

        self.decoder_layers = nn.ModuleList([Decoder_Layer(d_model,nhead,dropout)
                                             for _ in range(num_decoder_layers)])

    def forward(self, q, kv, mask, pred_time):
        # Positional Embedding
        q = q + positional_encoding(
            q.shape[0], q.shape[1], q.shape[2], pred_time)

        # Decoder Layers
        output = q
        for layer in self.decoder_layers:
            output = layer(output, kv, mask)


        return output