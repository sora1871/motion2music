# scripts/model_def.py
from __future__ import annotations
import torch
import torch.nn as nn

class LSTMAutoEncoder(nn.Module):
    """
    Minimal LSTM AutoEncoder.
    Args:
        input_dim: 次元数（例：99）
        hidden_dim: LSTMの隠れ次元
        latent_dim: 圧縮表現の次元
        num_layers: LSTM層数
        dropout: num_layers>1 の場合のみ有効
    """
    def __init__(
        self,
        input_dim: int = 99,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        do = dropout if num_layers > 1 else 0.0

        # Encoder: (B, T, input_dim) -> (B, T, hidden_dim) -> (B, latent_dim)
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=do,
            bidirectional=False,
        )
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)

        # Decoder: latent -> hidden -> (B, T, input_dim)
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=input_dim,      # 直接 input_dim を出す
            num_layers=num_layers,
            batch_first=True,
            dropout=do,
            bidirectional=False,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim)
        _, (h_n, _) = self.encoder_lstm(x)      # h_n: (num_layers, B, hidden_dim)
        h_last = h_n[-1]                         # (B, hidden_dim)
        z = self.encoder_fc(h_last)              # (B, latent_dim)
        return z

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        # z: (B, latent_dim)
        B = z.size(0)
        h0 = torch.tanh(self.decoder_fc(z)).unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, B, hidden_dim)
        c0 = torch.zeros_like(h0)
        # デコーダ入力はゼロ埋め（学習時はteacher forcingなどに差し替え可）
        dec_in = torch.zeros(B, seq_len, self.hidden_dim, device=z.device)              # (B, T, hidden_dim)
        out, _ = self.decoder_lstm(dec_in, (h0, c0))                                    # (B, T, input_dim)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim)
        z = self.encode(x)
        recon = self.decode(z, seq_len=x.size(1))
        return recon
