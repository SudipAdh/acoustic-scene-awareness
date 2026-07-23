"""Evaluation: aggregated cross-validation metrics and embedding extraction."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config as C
from .data import SpectrogramDataset, compute_norm_stats, make_fold_split
from .train import predict, train_one_fold


def evaluate_cv_predictions(X, meta, model_name="cnn", folds=None,
                            epochs=C.EPOCHS, augment=True, **kwargs):
    """Train per fold and pool the held-out predictions.

    Because every clip is in exactly one test fold, pooling gives one honest
    prediction for the entire dataset — the right basis for an aggregate
    confusion matrix and per-class scores.
    """
    folds = folds or list(range(1, C.N_FOLDS + 1))
    all_true, all_pred, all_prob, results = [], [], [], []

    for f in folds:
        model, res = train_one_fold(
            X, meta, test_fold=f, model_name=model_name,
            epochs=epochs, augment=augment, **kwargs
        )
        results.append(res)

        (Xtr, ytr), _, (Xte, yte) = make_fold_split(X, meta, f)
        mean, std = compute_norm_stats(Xtr)
        dl = DataLoader(SpectrogramDataset(Xte, yte, mean, std), batch_size=C.BATCH_SIZE)
        yt, yp, pr = predict(model, dl, C.DEVICE)

        all_true.append(yt); all_pred.append(yp); all_prob.append(pr)

    return (np.concatenate(all_true), np.concatenate(all_pred),
            np.concatenate(all_prob), results)


def full_metrics(y_true, y_pred, class_names) -> dict:
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score,
                                 precision_score, recall_score)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_f1": {
            c: float(v) for c, v in zip(
                class_names, f1_score(y_true, y_pred, average=None, zero_division=0)
            )
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0, digits=3
        ),
    }


@torch.no_grad()
def extract_embeddings(model, X, y, mean, std, device=C.DEVICE, batch_size=C.BATCH_SIZE):
    """Pull the CNN's 256-d penultimate features for clustering / t-SNE."""
    model.eval()
    ds = SpectrogramDataset(X, y, mean, std, augment=False)
    dl = DataLoader(ds, batch_size=batch_size)

    embs, labels = [], []
    for xb, yb in dl:
        _, emb = model(xb.to(device), return_embedding=True)
        embs.append(emb.cpu().numpy())
        labels.append(yb.numpy())

    return np.concatenate(embs), np.concatenate(labels)
