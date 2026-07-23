"""Figure generation for the report.

All figures share one visual system: a validated categorical palette, a single
blue sequential ramp for magnitude, recessive grid/axis ink, and direct labels
so identity never rests on colour alone.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from . import config as C

# --------------------------------------------------------------------------
# Design tokens (validated: all-pairs CVD dE 9.2, normal-vision 24.0, light)
# --------------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#4a3aa7", "#e34948", "#6da7ec", "#c98500"]

SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
            "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
            "#184f95", "#104281", "#0d366b"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

STATUS = {"good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b"}


def use_report_style() -> None:
    """Apply the shared style. Call once before generating figures."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.dpi": 130,
        "savefig.dpi": 300,          # print-resolution figures for the report
        "savefig.bbox": "tight",
    })


def _despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def save(fig, name: str) -> str:
    """Save a figure to results/figures as PNG (report) and PDF (vector)."""
    path = C.FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    fig.savefig(C.FIGURES_DIR / f"{name}.pdf")
    plt.close(fig)
    print(f"  figure -> {path}")
    return str(path)


# --------------------------------------------------------------------------
# Dataset overview
# --------------------------------------------------------------------------
def plot_class_distribution(counts: dict[str, int], name="fig_class_distribution",
                            title="UrbanSound8K clips per class"):
    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    order = np.argsort(values)[::-1]
    labels = [labels[i].replace("_", " ") for i in order]
    values = [values[i] for i in order]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.barh(labels[::-1], values[::-1], color=SERIES[0], height=0.62)
    for b, v in zip(bars, values[::-1]):
        ax.text(v + max(values) * 0.012, b.get_y() + b.get_height() / 2,
                f"{v:,}", va="center", ha="left", fontsize=9, color=INK_2)

    ax.set_xlabel("Number of clips")
    ax.set_title(title, loc="left", pad=12)
    ax.set_xlim(0, max(values) * 1.12)
    ax.grid(axis="y", visible=False)
    _despine(ax)
    return save(fig, name)


def plot_spectrogram_grid(specs, titles, name="fig_spectrogram_examples",
                          suptitle="Log-mel spectrograms: each sound has a distinct signature",
                          ncols=5):
    n = len(specs)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, spec, t in zip(axes, specs, titles):
        ax.imshow(spec, origin="lower", aspect="auto", cmap="magma",
                  extent=[0, C.CLIP_SECONDS, 0, C.N_MELS])
        ax.set_title(t.replace("_", " "), fontsize=9.5, pad=5)
        ax.set_xticks([0, 2, 4]); ax.set_yticks([])
        ax.grid(False)
        ax.tick_params(labelsize=8)
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(suptitle, fontsize=12, fontweight="bold", x=0.02, ha="left", y=1.0)
    fig.supxlabel("Time (s)", fontsize=9, color=INK_2)
    fig.supylabel("Mel frequency band", fontsize=9, color=INK_2)
    fig.tight_layout()
    return save(fig, name)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def plot_training_curves(histories: dict[str, dict], name="fig_training_curves"):
    """histories: {model_label: history_dict}"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    for i, (label, h) in enumerate(histories.items()):
        c = SERIES[i]
        ep = range(1, len(h["train_loss"]) + 1)
        ax1.plot(ep, h["train_loss"], color=c, lw=2, ls="--", alpha=0.55)
        ax1.plot(ep, h["val_loss"], color=c, lw=2, label=label)
        ax2.plot(ep, [a * 100 for a in h["train_acc"]], color=c, lw=2, ls="--", alpha=0.55)
        ax2.plot(ep, [a * 100 for a in h["val_acc"]], color=c, lw=2, label=label)
        # direct label at the end of the validation curve
        ax2.annotate(label, (len(h["val_acc"]), h["val_acc"][-1] * 100),
                     xytext=(4, 0), textcoords="offset points",
                     color=c, fontsize=9, fontweight="bold", va="center")

    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("Loss  (dashed = train, solid = validation)", loc="left", pad=10)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy  (dashed = train, solid = validation)", loc="left", pad=10)
    ax1.legend(loc="upper right")
    for ax in (ax1, ax2):
        _despine(ax)
    fig.tight_layout()
    return save(fig, name)


def plot_confusion_matrix(cm, classes, name="fig_confusion_matrix",
                          title="CNN confusion matrix (10-fold aggregate)", normalize=True):
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
    m = cm.astype(float)
    if normalize:
        m = m / np.clip(m.sum(axis=1, keepdims=True), 1, None) * 100

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    im = ax.imshow(m, cmap=cmap, vmin=0, vmax=100 if normalize else m.max())

    labels = [c.replace("_", " ") for c in classes]
    ax.set_xticks(range(len(classes)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title, loc="left", pad=12)
    ax.grid(False)

    # value labels — identity never rests on colour alone
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            if v >= 0.5:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if v > 55 else INK_2,
                        fontweight="bold" if i == j else "normal")

    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("% of actual class" if normalize else "clips", color=INK_2, fontsize=9)
    cb.outline.set_visible(False)
    fig.tight_layout()
    return save(fig, name)


def plot_model_comparison(results: dict[str, dict], name="fig_model_comparison",
                          n_folds: int | None = None):
    """results: {label: {'acc_mean','acc_std','f1_mean','f1_std','n_params'}}"""
    labels = list(results.keys())
    x = np.arange(len(labels))
    w = 0.36

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.3),
                                   gridspec_kw={"width_ratios": [1.55, 1]})

    acc = [results[l]["acc_mean"] * 100 for l in labels]
    accs = [results[l]["acc_std"] * 100 for l in labels]
    f1 = [results[l]["f1_mean"] * 100 for l in labels]
    f1s = [results[l]["f1_std"] * 100 for l in labels]

    b1 = ax1.bar(x - w / 2, acc, w, yerr=accs, capsize=3, color=SERIES[0], label="Accuracy")
    b2 = ax1.bar(x + w / 2, f1, w, yerr=f1s, capsize=3, color=SERIES[1], label="Macro F1")

    headroom = max(v + e for v, e in zip(acc + f1, accs + f1s))
    # Labels clear the error-bar cap, not just the bar top.
    for bars, vals, errs in ((b1, acc, accs), (b2, f1, f1s)):
        for b, v, e in zip(bars, vals, errs):
            ax1.text(b.get_x() + b.get_width() / 2, v + e + headroom * 0.035, f"{v:.1f}",
                     ha="center", fontsize=9, color=INK_2, fontweight="bold")

    fold_note = f"{n_folds}-fold " if n_folds else ""
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("Score (%)")
    ax1.set_ylim(0, headroom * 1.30)
    ax1.set_title(f"Performance ({fold_note}mean ± sd)", loc="left", pad=10)
    # Legend on the title line but right-aligned: clear of both the
    # left-aligned title and every bar.
    ax1.legend(loc="lower right", bbox_to_anchor=(1.0, 1.005), ncols=2,
               frameon=False, borderaxespad=0)
    ax1.grid(axis="x", visible=False)
    _despine(ax1)

    params = [results[l]["n_params"] / 1e6 for l in labels]
    b3 = ax2.bar(x, params, 0.5, color=SERIES[2])
    for b, v in zip(b3, params):
        ax2.text(b.get_x() + b.get_width() / 2, v + max(params) * 0.03,
                 f"{v:.2f}M", ha="center", fontsize=9, color=INK_2, fontweight="bold")
    ax2.set_xticks(x, labels)
    ax2.set_ylabel("Trainable parameters (millions)")
    ax2.set_ylim(0, max(params) * 1.22)
    ax2.set_title("Model size", loc="left", pad=10)
    ax2.grid(axis="x", visible=False)
    _despine(ax2)

    fig.tight_layout()
    return save(fig, name)


def plot_per_class_f1(per_class: dict[str, float], name="fig_per_class_f1"):
    items = sorted(per_class.items(), key=lambda kv: kv[1])
    labels = [k.replace("_", " ") for k, _ in items]
    vals = [v * 100 for _, v in items]
    colors = [STATUS["critical"] if v < 70 else (STATUS["warning"] if v < 85 else SERIES[0])
              for v in vals]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.barh(labels, vals, color=colors, height=0.62)
    for b, v in zip(bars, vals):
        ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.1f}",
                va="center", fontsize=9, color=INK_2)
    ax.set_xlabel("F1 score (%)")
    ax.set_xlim(0, 105)
    ax.set_title("Per-class F1 — which sounds the CNN finds hard", loc="left", pad=12)
    ax.grid(axis="y", visible=False)
    _despine(ax)
    return save(fig, name)


# --------------------------------------------------------------------------
# Embeddings / clustering
# --------------------------------------------------------------------------
def plot_tsne(emb2d, labels, class_names, name="fig_tsne_embeddings",
              title="What the CNN learned: t-SNE of 256-d embeddings"):
    """Colour + a direct centroid label per class, so identity is never
    carried by colour alone (10 classes exceeds a safe colour-only cap)."""
    fig, ax = plt.subplots(figsize=(8.2, 6.8))

    for i, cname in enumerate(class_names):
        m = labels == i
        if not m.any():
            continue
        ax.scatter(emb2d[m, 0], emb2d[m, 1], s=9, alpha=0.55,
                   color=SERIES[i % len(SERIES)], linewidths=0)
        cx, cy = emb2d[m, 0].mean(), emb2d[m, 1].mean()
        ax.annotate(cname.replace("_", " "), (cx, cy), fontsize=9.5, fontweight="bold",
                    ha="center", va="center", color=INK,
                    bbox=dict(boxstyle="round,pad=0.28", fc=SURFACE,
                              ec=SERIES[i % len(SERIES)], lw=1.4, alpha=0.92))

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, loc="left", pad=12)
    ax.grid(False)
    _despine(ax, keep=())
    fig.tight_layout()
    return save(fig, name)


# --------------------------------------------------------------------------
# Anomaly / novelty detection
# --------------------------------------------------------------------------
def plot_anomaly_scores(known, novel, threshold=None, name="fig_anomaly_scores"):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(0, np.percentile(np.concatenate([known, novel]), 99), 60)

    ax.hist(known, bins=bins, color=SERIES[0], alpha=0.75, label="Known sounds (UrbanSound8K)")
    ax.hist(novel, bins=bins, color=SERIES[1], alpha=0.75, label="Novel sounds (ESC-50, unseen)")

    if threshold is not None:
        ax.axvline(threshold, color=STATUS["critical"], lw=2, ls="--")
        ax.annotate(f"threshold {threshold:.3f}", (threshold, ax.get_ylim()[1] * 0.92),
                    xytext=(7, 0), textcoords="offset points",
                    color=STATUS["critical"], fontsize=9, fontweight="bold")

    ax.set_xlabel("Autoencoder reconstruction error")
    ax.set_ylabel("Number of clips")
    ax.set_title("Novelty detection: unseen sounds reconstruct worse", loc="left", pad=12)
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    _despine(ax)
    return save(fig, name)


def plot_roc(fpr, tpr, auc, name="fig_anomaly_roc"):
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1.4, ls="--")
    ax.plot(fpr, tpr, color=SERIES[0], lw=2.4)
    ax.annotate(f"AUC = {auc:.3f}", (0.55, 0.22), fontsize=12, fontweight="bold", color=SERIES[0])
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Novel-sound detection ROC", loc="left", pad=12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    _despine(ax)
    fig.tight_layout()
    return save(fig, name)


def plot_reconstructions(originals, recons, titles, name="fig_ae_reconstructions"):
    n = len(originals)
    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 4.6))
    axes = np.atleast_2d(axes)
    for j in range(n):
        for row, (data, tag) in enumerate(((originals[j], "input"), (recons[j], "rebuilt"))):
            ax = axes[row, j]
            ax.imshow(data, origin="lower", aspect="auto", cmap="magma")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if row == 0:
                ax.set_title(titles[j], fontsize=9.5, pad=5)
            if j == 0:
                ax.set_ylabel(tag, fontsize=9.5, color=INK_2)
    fig.suptitle("Autoencoder reconstructions — known sounds rebuild cleanly, novel ones do not",
                 fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    return save(fig, name)
