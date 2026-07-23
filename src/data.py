"""Audio loading, mel-spectrogram feature extraction and PyTorch datasets.

The expensive part of this project is turning ~10k wav files into
mel-spectrograms. That is done once by `precompute_features()` and cached to
.npy, after which training reads features straight from memory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config as C


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def load_us8k_metadata() -> pd.DataFrame:
    """UrbanSound8K metadata, one row per clip, including its official fold."""
    df = pd.read_csv(C.US8K_META)
    df["path"] = df.apply(
        lambda r: C.US8K_AUDIO / f"fold{r['fold']}" / r["slice_file_name"], axis=1
    )
    return df


def load_esc50_metadata() -> pd.DataFrame:
    """ESC-50 metadata. Used only as a source of *unseen* sounds for the
    novelty-detection experiment."""
    df = pd.read_csv(C.ESC50_META)
    df["path"] = df["filename"].apply(lambda f: C.ESC50_AUDIO / f)
    return df


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def load_audio(path, sr: int = C.SAMPLE_RATE) -> np.ndarray:
    """Load a clip as mono at `sr`, then pad/trim to exactly CLIP_SECONDS.

    Fixing the length is what makes every spectrogram the same shape, which the
    CNN needs. Clips shorter than 4 s are zero-padded; longer ones are cut.
    """
    import librosa

    y, _ = librosa.load(path, sr=sr, mono=True)
    if len(y) < C.N_SAMPLES:
        y = np.pad(y, (0, C.N_SAMPLES - len(y)), mode="constant")
    else:
        y = y[: C.N_SAMPLES]
    return y


def audio_to_melspec(y: np.ndarray, sr: int = C.SAMPLE_RATE) -> np.ndarray:
    """Convert a waveform to a log-scaled mel-spectrogram.

    This is the core "audio becomes an image" step: the result is a
    (N_MELS, N_FRAMES) matrix of energy per frequency band over time, which a
    2-D CNN can process exactly like a grayscale image.

    Decibel scaling matters — human loudness perception is logarithmic, and
    without it a handful of loud frames dominate the input range.
    """
    import librosa

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=C.N_FFT,
        hop_length=C.HOP_LENGTH,
        n_mels=C.N_MELS,
        fmin=C.F_MIN,
        fmax=C.F_MAX,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def extract_features(path) -> np.ndarray:
    return audio_to_melspec(load_audio(path))


def precompute_features(df: pd.DataFrame, cache_name: str, force: bool = False):
    """Extract mel-spectrograms for every row of `df` and cache them.

    Returns (X, meta) where X has shape (N, N_MELS, N_FRAMES).
    """
    from tqdm import tqdm

    cache_x = C.PROCESSED_DIR / f"{cache_name}_X.npy"
    cache_meta = C.PROCESSED_DIR / f"{cache_name}_meta.csv"

    if cache_x.exists() and cache_meta.exists() and not force:
        X = np.load(cache_x)
        meta = pd.read_csv(cache_meta)
        print(f"[cache] loaded {cache_name}: X={X.shape}")
        return X, meta

    feats, keep = [], []
    for i, row in tqdm(
        df.iterrows(), total=len(df), desc=f"extracting {cache_name}", unit="clip"
    ):
        try:
            feats.append(extract_features(row["path"]))
            keep.append(i)
        except Exception as exc:  # a handful of US8K files are known to be odd
            print(f"  ! skipped {row['path']}: {exc}")

    X = np.stack(feats)
    meta = df.loc[keep].drop(columns=["path"]).reset_index(drop=True)

    np.save(cache_x, X)
    meta.to_csv(cache_meta, index=False)
    print(f"[cache] saved {cache_name}: X={X.shape} -> {cache_x}")
    return X, meta


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def compute_norm_stats(X: np.ndarray) -> tuple[float, float]:
    """Mean/std computed on the TRAIN split only, then applied to val/test.

    Computing these over the whole dataset would leak test statistics into
    training, so they are always derived from the training fold.
    """
    return float(X.mean()), float(X.std() + 1e-8)


# ---------------------------------------------------------------------------
# Augmentation (spectrogram domain)
# ---------------------------------------------------------------------------
def spec_augment(
    spec: np.ndarray,
    rng: np.random.Generator,
    n_freq_masks: int = 1,
    n_time_masks: int = 1,
    freq_mask_max: int = 8,
    time_mask_max: int = 16,
) -> np.ndarray:
    """SpecAugment: blank out random frequency bands and time spans.

    Forces the network to use the whole time-frequency pattern instead of
    latching onto one narrow band, which measurably reduces overfitting on a
    dataset this small.
    """
    spec = spec.copy()
    n_mels, n_frames = spec.shape
    fill = spec.min()

    for _ in range(n_freq_masks):
        f = int(rng.integers(0, freq_mask_max + 1))
        if f > 0 and n_mels > f:
            f0 = int(rng.integers(0, n_mels - f))
            spec[f0 : f0 + f, :] = fill

    for _ in range(n_time_masks):
        t = int(rng.integers(0, time_mask_max + 1))
        if t > 0 and n_frames > t:
            t0 = int(rng.integers(0, n_frames - t))
            spec[:, t0 : t0 + t] = fill

    return spec


def time_shift(spec: np.ndarray, rng: np.random.Generator, max_frac: float = 0.2):
    """Roll the clip in time — a siren is a siren regardless of when it starts."""
    n_frames = spec.shape[1]
    shift = int(rng.integers(-int(n_frames * max_frac), int(n_frames * max_frac) + 1))
    return np.roll(spec, shift, axis=1)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class SpectrogramDataset(Dataset):
    """Serves normalised (1, N_MELS, N_FRAMES) tensors and integer labels."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        mean: float,
        std: float,
        augment: bool = False,
        seed: int = C.SEED,
    ):
        self.X = X
        self.y = y.astype(np.int64)
        self.mean = mean
        self.std = std
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        spec = self.X[idx]
        if self.augment:
            spec = time_shift(spec, self.rng)
            spec = spec_augment(spec, self.rng)
        spec = (spec - self.mean) / self.std
        return torch.from_numpy(spec).unsqueeze(0).float(), int(self.y[idx])


def make_fold_split(X, meta, test_fold: int, val_fold: int | None = None):
    """Split by UrbanSound8K's official folds.

    Clips sliced from the same source recording share a fold, so splitting by
    fold (rather than randomly) is what keeps the evaluation honest.
    """
    if val_fold is None:
        val_fold = test_fold % C.N_FOLDS + 1

    folds = meta["fold"].values
    labels = meta["classID"].values

    te = folds == test_fold
    va = folds == val_fold
    tr = ~(te | va)

    return (X[tr], labels[tr]), (X[va], labels[va]), (X[te], labels[te])
