from __future__ import annotations

import logging
import torch
from torch import nn

logger = logging.getLogger(__name__)


class LSTMAutoencoder(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, latent_dim: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.to_output = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.encoder(x)
        latent = self.to_latent(self.dropout(h_n[-1]))  # (batch, latent_dim)
        hidden_seed = self.to_hidden(latent)  # (batch, hidden_dim)
        repeated = hidden_seed.unsqueeze(1).repeat(
            1, x.size(1), 1
        )  # (batch, seq_len, hidden_dim)
        dec_out, _ = self.decoder(repeated)
        recon = self.to_output(dec_out)  # no activation on output
        return recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.encoder(x)
        return self.to_latent(h_n[-1])

    def decode(self, latent: torch.Tensor, seq_len: int) -> torch.Tensor:
        hidden_seed = self.to_hidden(latent)
        repeated = hidden_seed.unsqueeze(1).repeat(1, seq_len, 1)
        dec_out, _ = self.decoder(repeated)
        return self.to_output(dec_out)
