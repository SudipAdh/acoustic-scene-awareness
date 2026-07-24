"""Training engine: single-fold training plus the official 10-fold protocol."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config as C
from .data import SpectrogramDataset, compute_norm_stats, make_fold_split
from .models import build_model, count_parameters


def set_seed(seed: int = C.SEED) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed) if torch.backends.mps.is_available() else None


@dataclass
class FoldResult:
    fold: int
    model: str
    best_val_acc: float
    test_acc: float
    test_f1_macro: float
    epochs_run: int
    train_seconds: float
    n_params: int
    history: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Epoch loops
# ---------------------------------------------------------------------------
def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def predict(model, loader, device):
    """Returns (y_true, y_pred, probabilities) for a whole loader."""
    model.eval()
    ys, ps, probs = [], [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        p = torch.softmax(logits, dim=1)
        ys.append(yb.numpy())
        ps.append(logits.argmax(1).cpu().numpy())
        probs.append(p.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(probs)


# ---------------------------------------------------------------------------
# Single fold
# ---------------------------------------------------------------------------
def train_one_fold(
    X,
    meta,
    test_fold: int,
    model_name: str = "cnn",
    epochs: int = C.EPOCHS,
    batch_size: int = C.BATCH_SIZE,
    lr: float = C.LEARNING_RATE,
    augment: bool = True,
    device: torch.device = C.DEVICE,
    n_classes: int = 10,
    verbose: bool = True,
    save_path=None,
) -> tuple[nn.Module, FoldResult]:
    """Train one model on the folds outside `test_fold`, evaluate on it."""
    from sklearn.metrics import f1_score

    set_seed(C.SEED + test_fold)

    (Xtr, ytr), (Xva, yva), (Xte, yte) = make_fold_split(X, meta, test_fold)

    # Normalisation statistics come from the training split only.
    mean, std = compute_norm_stats(Xtr)

    ds_tr = SpectrogramDataset(Xtr, ytr, mean, std, augment=augment)
    ds_va = SpectrogramDataset(Xva, yva, mean, std, augment=False)
    ds_te = SpectrogramDataset(Xte, yte, mean, std, augment=False)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size)
    dl_te = DataLoader(ds_te, batch_size=batch_size)

    model = build_model(model_name, n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=C.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val, best_state, patience, t0 = 0.0, None, 0, time.time()
    epochs_run = 0

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, dl_tr, criterion, optimizer, device, True)
        va_loss, va_acc = _run_epoch(model, dl_va, criterion, optimizer, device, False)
        scheduler.step()
        epochs_run = ep

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        if va_acc > best_val:
            best_val = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if verbose and (ep % 5 == 0 or ep == 1):
            print(
                f"  fold{test_fold} {model_name} ep{ep:3d}/{epochs} "
                f"train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}"
            )

        if patience >= C.EARLY_STOP_PATIENCE:
            if verbose:
                print(f"  fold{test_fold} early stop at epoch {ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_pred, _ = predict(model, dl_te, device)
    test_acc = float((y_true == y_pred).mean())
    test_f1 = float(f1_score(y_true, y_pred, average="macro"))

    result = FoldResult(
        fold=test_fold,
        model=model_name,
        best_val_acc=best_val,
        test_acc=test_acc,
        test_f1_macro=test_f1,
        epochs_run=epochs_run,
        train_seconds=time.time() - t0,
        n_params=count_parameters(model),
        history=history,
    )

    if save_path:
        torch.save(
            {"state_dict": model.state_dict(), "mean": mean, "std": std,
             "model_name": model_name, "n_classes": n_classes},
            save_path,
        )

    if verbose:
        print(
            f"  fold{test_fold} {model_name} -> test acc {test_acc:.4f} "
            f"f1 {test_f1:.4f} ({result.train_seconds:.0f}s)"
        )

    return model, result


# ---------------------------------------------------------------------------
# Full cross-validation
# ---------------------------------------------------------------------------
def cross_validate(
    X,
    meta,
    model_name: str = "cnn",
    folds=None,
    epochs: int = C.EPOCHS,
    augment: bool = True,
    tag: str | None = None,
    **kwargs,
):
    """Run the official 10-fold protocol and report mean +/- std.

    Two conveniences for long unattended runs, both controlled by environment
    variables so they never affect the results:

    * ``COOLDOWN_SEC`` pauses between folds to let the GPU shed heat. Folds are
      independent, so this changes timing only, never the numbers.
    * ``RESUME=1`` skips a configuration whose cached JSON already covers the
      requested folds, so a restarted run does not redo finished work. Results
      are deterministic, so a skipped config is identical to a rerun.
    """
    import os

    folds = folds or list(range(1, C.N_FOLDS + 1))
    cooldown = float(os.environ.get("COOLDOWN_SEC", "0"))

    name = tag or f"{model_name}_aug{int(augment)}"
    out = C.LOGS_DIR / f"cv_{name}.json"

    if os.environ.get("RESUME") == "1" and out.exists():
        cached = json.loads(out.read_text())
        if set(cached.get("folds", [])) >= set(folds):
            print(f"[resume] {name} already complete "
                  f"({cached['acc_mean']*100:.2f}%) -- skipping")
            return cached

    results: list[FoldResult] = []

    print(f"\n{'='*70}\n{model_name.upper()}  |  folds={folds}  augment={augment}"
          f"{f'  cooldown={cooldown:.0f}s' if cooldown else ''}\n{'='*70}")

    for i, f in enumerate(folds):
        _, res = train_one_fold(
            X, meta, test_fold=f, model_name=model_name,
            epochs=epochs, augment=augment, **kwargs
        )
        results.append(res)

        if cooldown and i < len(folds) - 1:
            print(f"  [cooldown] pausing {cooldown:.0f}s to let the GPU cool "
                  f"(fold {i+1}/{len(folds)} done)")
            time.sleep(cooldown)

    accs = np.array([r.test_acc for r in results])
    f1s = np.array([r.test_f1_macro for r in results])

    summary = {
        "model": model_name,
        "augment": augment,
        "folds": folds,
        "acc_mean": float(accs.mean()), "acc_std": float(accs.std()),
        "f1_mean": float(f1s.mean()), "f1_std": float(f1s.std()),
        "n_params": results[0].n_params,
        "total_seconds": float(sum(r.train_seconds for r in results)),
        "per_fold": [asdict(r) for r in results],
    }

    print(
        f"\n{model_name.upper()} SUMMARY  acc {accs.mean():.4f} +/- {accs.std():.4f} | "
        f"macro-F1 {f1s.mean():.4f} +/- {f1s.std():.4f} | params {results[0].n_params:,}"
    )

    name = tag or f"{model_name}_aug{int(augment)}"
    out = C.LOGS_DIR / f"cv_{name}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"saved -> {out}")

    return summary
