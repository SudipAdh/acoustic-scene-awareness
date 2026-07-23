"""Neural network architectures.

Three networks, each serving a distinct task in the project:

  MLP           baseline classifier — flattens the spectrogram, throwing away
                all time-frequency structure. Exists to quantify what the CNN's
                inductive bias is actually worth.
  AudioCNN      main classifier — 2-D convolutions over the mel-spectrogram.
  ConvAutoencoder  novelty detector — trained only to reconstruct known sounds,
                so unfamiliar sounds reconstruct badly and can be flagged.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


# ---------------------------------------------------------------------------
# Baseline: multi-layer perceptron
# ---------------------------------------------------------------------------
class MLPBaseline(nn.Module):
    """A fully-connected net over the flattened spectrogram.

    Every input pixel gets its own weight, so the model has no notion that two
    neighbouring time frames are related. It also carries far more parameters
    than the CNN despite being the weaker model — the comparison the report
    uses to motivate convolution.
    """

    def __init__(
        self,
        n_mels: int = C.N_MELS,
        n_frames: int = C.N_FRAMES,
        n_classes: int = 10,
        hidden: tuple[int, ...] = (512, 256),
        dropout: float = 0.5,
    ):
        super().__init__()
        in_dim = n_mels * n_frames

        layers: list[nn.Module] = [nn.Flatten()]
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Main classifier: convolutional neural network
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU -> MaxPool, the standard CNN building block."""

    def __init__(self, in_ch: int, out_ch: int, pool: int = 2, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(pool)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.pool(x)
        return self.drop(x)


class AudioCNN(nn.Module):
    """Four-block CNN over log-mel spectrograms.

    Early filters learn local time-frequency texture (onsets, harmonic stacks,
    broadband noise); deeper blocks compose those into class-level patterns.

    Global average pooling replaces a large flatten+dense head. That keeps the
    parameter count ~10x below the MLP and makes the model robust to a sound
    occurring at a different point in the clip.

    `return_embedding=True` exposes the 256-d penultimate features, which the
    clustering / t-SNE analysis uses to show what the network learned.
    """

    def __init__(self, n_classes: int = 10, dropout: float = 0.3, width: int = 32):
        super().__init__()
        w = width
        self.block1 = ConvBlock(1, w, pool=2)
        self.block2 = ConvBlock(w, w * 2, pool=2, dropout=dropout / 2)
        self.block3 = ConvBlock(w * 2, w * 4, pool=2, dropout=dropout / 2)
        self.block4 = ConvBlock(w * 4, w * 8, pool=2, dropout=dropout)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(w * 8, n_classes)
        self.embed_dim = w * 8

    def forward(self, x, return_embedding: bool = False):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        emb = self.gap(x).flatten(1)
        logits = self.fc(self.dropout(emb))

        if return_embedding:
            return logits, emb
        return logits


# ---------------------------------------------------------------------------
# Novelty detection: convolutional autoencoder
# ---------------------------------------------------------------------------
class ConvAutoencoder(nn.Module):
    """Compresses a spectrogram to a small latent code and rebuilds it.

    Trained only on the ten known UrbanSound8K classes, it learns to reconstruct
    those well. A sound it has never encountered (a smashing window, a crying
    baby) falls outside that learned manifold and reconstructs poorly, so the
    per-clip reconstruction error works directly as a novelty score.
    """

    def __init__(self, latent_ch: int = 32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),   # -> 32 x 87
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # -> 16 x 44
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, latent_ch, 3, stride=2, padding=1),  # -> 8 x 22
            nn.BatchNorm2d(latent_ch),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_ch, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        # Transposed convolutions can overshoot by a pixel; crop to match input.
        return out[..., : x.shape[-2], : x.shape[-1]]

    def encode(self, x):
        return self.encoder(x)

    @torch.no_grad()
    def reconstruction_error(self, x) -> torch.Tensor:
        """Per-sample mean squared error — the novelty score."""
        recon = self(x)
        return ((recon - x) ** 2).flatten(1).mean(dim=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(name: str, n_classes: int = 10, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "mlp":
        return MLPBaseline(n_classes=n_classes, **kwargs)
    if name == "cnn":
        return AudioCNN(n_classes=n_classes, **kwargs)
    if name in ("autoencoder", "ae"):
        return ConvAutoencoder(**kwargs)
    raise ValueError(f"unknown model: {name}")
