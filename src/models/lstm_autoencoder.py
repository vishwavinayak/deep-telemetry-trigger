from __future__ import annotations

import logging
from typing import Tuple

import torch
from torch import nn

logger = logging.getLogger(__name__)


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.to_output = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, seq, feat)
        enc_out, (h_n, _) = self.encoder(x)
        # Use last hidden state as compressed representation
        latent = self.to_latent(self.dropout(h_n[-1])).unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(latent)
        recon = self.to_output(self.dropout(dec_out))
        return recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.encoder(x)
        return self.to_latent(h_n[-1])

    def decode(self, latent: torch.Tensor, seq_len: int) -> torch.Tensor:
        latent_seq = latent.unsqueeze(1).repeat(1, seq_len, 1)
        dec_out, _ = self.decoder(latent_seq)
        return self.to_output(dec_out)
