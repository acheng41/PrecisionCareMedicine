import torch
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def positional_encoding(batch_size, length, d_model):
    """
    Modified Positional Encoding for each visit

    Parameters
    ----------
    batch_size:
        Number of subjects in batch
    length:
        Number of sequences
    d_model:
        Dimension of the model vector
    """

    pe = torch.zeros(batch_size, length, d_model).to(device)
    position = torch.arange(0, length).unsqueeze(-1).to(device)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).reshape(1, 1, -1).to(device)
    pe[:, :, 0::2] = torch.sin(position * div_term)
    pe[:, :, 1::2] = torch.cos(position * div_term)
    return pe