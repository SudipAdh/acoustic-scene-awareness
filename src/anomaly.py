"""Novel-sound detection with a convolutional autoencoder.

The classifier is closed-set: shown a smashing window it must still answer with
one of its ten labels, usually confidently and wrongly. For an assistive system
that is the dangerous failure mode.

The fix here is an autoencoder trained *only* on the ten known classes. It learns
to rebuild those, so an unfamiliar sound reconstructs badly and its reconstruction
error becomes a usable "I have not heard this before" score.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config as C
from .data import SpectrogramDataset
from .models import ConvAutoencoder
from .train import set_seed


def train_autoencoder(X_train, mean, std, epochs=30, batch_size=C.BATCH_SIZE,
                      lr=1e-3, device=C.DEVICE, verbose=True, X_val=None):
    """Fit the autoencoder on known-class spectrograms only."""
    set_seed(C.SEED)

    dummy = np.zeros(len(X_train), dtype=np.int64)
    dl = DataLoader(SpectrogramDataset(X_train, dummy, mean, std, augment=False),
                    batch_size=batch_size, shuffle=True, drop_last=True)

    dl_val = None
    if X_val is not None:
        dl_val = DataLoader(
            SpectrogramDataset(X_val, np.zeros(len(X_val), dtype=np.int64), mean, std),
            batch_size=batch_size)

    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": []}
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, _ in dl:
            xb = xb.to(device)
            loss = criterion(model(xb), xb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0); n += xb.size(0)
        scheduler.step()
        history["train_loss"].append(total / n)

        if dl_val is not None:
            model.eval()
            vt, vn = 0.0, 0
            with torch.no_grad():
                for xb, _ in dl_val:
                    xb = xb.to(device)
                    vt += criterion(model(xb), xb).item() * xb.size(0); vn += xb.size(0)
            history["val_loss"].append(vt / vn)

        if verbose and (ep % 5 == 0 or ep == 1):
            msg = f"  AE ep{ep:3d}/{epochs} train {history['train_loss'][-1]:.5f}"
            if history["val_loss"]:
                msg += f"  val {history['val_loss'][-1]:.5f}"
            print(msg)

    if verbose:
        print(f"  AE trained in {time.time()-t0:.0f}s")
    return model, history


@torch.no_grad()
def reconstruction_scores(model, X, mean, std, device=C.DEVICE,
                          batch_size=C.BATCH_SIZE) -> np.ndarray:
    """Per-clip reconstruction error — higher means less familiar."""
    model.eval()
    dl = DataLoader(SpectrogramDataset(X, np.zeros(len(X), dtype=np.int64), mean, std),
                    batch_size=batch_size)
    out = []
    for xb, _ in dl:
        out.append(model.reconstruction_error(xb.to(device)).cpu().numpy())
    return np.concatenate(out)


def evaluate_novelty(known_scores: np.ndarray, novel_scores: np.ndarray,
                     percentile: float = 95.0) -> dict:
    """Score the detector as a binary problem: novel (1) vs known (0).

    The threshold is set at a percentile of the KNOWN scores, so it can be
    calibrated without ever seeing a novel example — which is the situation a
    deployed system is actually in.
    """
    from sklearn.metrics import (average_precision_score, roc_auc_score, roc_curve)

    y = np.concatenate([np.zeros(len(known_scores)), np.ones(len(novel_scores))])
    s = np.concatenate([known_scores, novel_scores])

    auc = float(roc_auc_score(y, s))
    ap = float(average_precision_score(y, s))
    fpr, tpr, _ = roc_curve(y, s)

    thr = float(np.percentile(known_scores, percentile))
    tp = int((novel_scores > thr).sum())
    fn = int((novel_scores <= thr).sum())
    fp = int((known_scores > thr).sum())
    tn = int((known_scores <= thr).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "roc_auc": auc,
        "average_precision": ap,
        "threshold": thr,
        "threshold_percentile": percentile,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "known_mean": float(known_scores.mean()),
        "novel_mean": float(novel_scores.mean()),
    }


def softmax_confidence_baseline(probs_known: np.ndarray,
                                probs_novel: np.ndarray) -> dict:
    """Baseline: use 1 - max(softmax) as the novelty score.

    Included so the report can show whether the autoencoder actually beats the
    obvious "the classifier is unsure" heuristic, rather than assuming it does.
    """
    from sklearn.metrics import roc_auc_score

    known = 1.0 - probs_known.max(axis=1)
    novel = 1.0 - probs_novel.max(axis=1)
    y = np.concatenate([np.zeros(len(known)), np.ones(len(novel))])
    s = np.concatenate([known, novel])
    return {
        "roc_auc": float(roc_auc_score(y, s)),
        "known_mean_conf": float(probs_known.max(axis=1).mean()),
        "novel_mean_conf": float(probs_novel.max(axis=1).mean()),
    }
