"""Recurrent Autoencoder with a ConvGRU memory.

Input/output: 64x64 grayscale (1 channel). The ``use_memory`` flag toggles the
ConvGRU + Combiner path so that a "no-memory" baseline can be evaluated with the
same weights (by resetting the hidden state each frame).
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """conv3x3(stride) -> BN -> ReLU -> conv3x3 -> BN, with a projected shortcut."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class Encoder(nn.Module):
    """1x64x64 -> 128x8x8."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.block1 = ResidualBlock(32, 64, stride=2)    # 32x32
        self.block2 = ResidualBlock(64, 128, stride=2)   # 16x16
        self.block3 = ResidualBlock(128, 128, stride=2)  # 8x8

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class ConvGRUCell(nn.Module):
    """Convolutional GRU cell (hidden=128, kernel 3x3)."""

    def __init__(self, input_dim=128, hidden_dim=128, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv_u = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=pad)
        self.conv_r = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=pad)
        self.conv_h = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=pad)

    def forward(self, z, h_prev):
        combined = torch.cat([z, h_prev], dim=1)
        u = torch.sigmoid(self.conv_u(combined))
        r = torch.sigmoid(self.conv_r(combined))
        combined_r = torch.cat([z, r * h_prev], dim=1)
        h_tilde = torch.tanh(self.conv_h(combined_r))
        h = (1 - u) * h_prev + u * h_tilde
        return h


class Decoder(nn.Module):
    """128x8x8 -> 1x64x64 (Sigmoid)."""

    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block1 = ResidualBlock(128, 128)
        self.block2 = ResidualBlock(128, 64)
        self.block3 = ResidualBlock(64, 32)
        self.out_conv = nn.Conv2d(32, 1, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.up(self.block1(x))   # 16x16
        x = self.up(self.block2(x))   # 32x32
        x = self.up(self.block3(x))   # 64x64
        x = torch.sigmoid(self.out_conv(x))
        return x


class RecurrentAutoencoder(nn.Module):
    def __init__(self, hidden_dim=128, use_memory=True):
        super().__init__()
        self.use_memory = use_memory
        self.hidden_dim = hidden_dim
        self.encoder = Encoder()
        self.gru = ConvGRUCell(128, hidden_dim, 3)
        self.combiner = nn.Conv2d(128 + hidden_dim, 128, 1)
        self.decoder = Decoder()

    def forward(self, x_seq, h=None):
        """x_seq: (B, T, 1, 64, 64) -> (x_hat_seq (B, T, 1, 64, 64), h_new)."""
        B, T, C, H, W = x_seq.shape
        z = self.encoder(x_seq.reshape(B * T, C, H, W))         # (B*T, 128, 8, 8)
        z = z.reshape(B, T, 128, z.shape[-2], z.shape[-1])

        outputs = []
        if self.use_memory:
            for t in range(T):
                z_t = z[:, t]
                if h is None:
                    h = torch.zeros_like(z_t)
                h = self.gru(z_t, h)
                z_comb = self.combiner(torch.cat([z_t, h], dim=1))
                outputs.append(self.decoder(z_comb))
        else:
            for t in range(T):
                outputs.append(self.decoder(z[:, t]))

        x_hat_seq = torch.stack(outputs, dim=1)
        return x_hat_seq, h
