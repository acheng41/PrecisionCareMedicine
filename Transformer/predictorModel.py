class Transformer(nn.Module):
    """
    An adaptation of the transformer model (Attention is All you Need)
    for survival analysis.

    Parameters
    ----------
    d_long:
        Number of longitudinal outcomes
    d_base:
        Number of baseline / time-independent covariates
    d_model:
        Dimension of the input vector (post embedding)
    nhead:
        Number of heads
    num_decoder_layers:
        Number of decoder layers to stack
    dropout:
        The dropout value
    """
    def __init__(self,
                 d_long,
                 d_base,
                 d_output,
                 d_model = 32,
                 nhead = 4,
                 num_decoder_layers = 3,
                 dropout = 0.2):
        super().__init__()

        self.decoder = Decoder(d_long, d_base, d_model, nhead, num_decoder_layers, dropout)

        self.decoder_pred = Decoder_p(d_model, nhead, 4, dropout)

        self.RELU = nn.ReLU()

        self.pred = nn.Sequential(
            nn.Linear(d_model, d_output)
            )


    def forward(self, long, base, mask, obs_time, pred_time):
        # Decoder Layers
        x = self.decoder(long, base, mask, obs_time)

        # Decoder Layer with prediction time embedding
        x = self.decoder_pred(x, x, mask, pred_time) #hidden state for next visit

        # Output layer
        long = self.pred(x)

        return long