#!/usr/bin/env python
"""Render captured experiment output as macOS-terminal-style PNGs.

The report's Experiment Evidence appendix shows the pipeline executing on the
development machine. This takes the real stdout of ``run_experiments.py`` and
lays it out as terminal windows, so the evidence is legible in print rather
than a low-resolution photo of a screen.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures"

BG = "#1e1e2e"
BAR = "#2b2b3d"
FG = "#e6e6e6"
GREEN = "#7ee787"
BLUE = "#79c0ff"
YELLOW = "#f2cc60"
DIM = "#9aa2b1"
DOTS = ["#ff5f56", "#ffbd2e", "#27c93f"]

MONO = {"family": "monospace", "fontsize": 8.2}


def colour_for(line: str) -> str:
    s = line.strip()
    if s.startswith("=") or "STAGE:" in line or "SUMMARY" in line or "EXPERIMENT RUN" in s:
        return YELLOW
    if "->" in line or "saved" in line or "figure" in line or "DONE" in line:
        return GREEN
    if any(k in line for k in ("device", "torch", "host", "platform", "machine", "python", "accuracy", "ROC-AUC", "ARI")):
        return BLUE
    if s.startswith("#") or "precision" in line or "macro avg" in line or "weighted avg" in line:
        return DIM
    return FG


def render(lines, title, dest):
    n = len(lines)
    fig_h = 0.185 * n + 0.7
    fig = plt.figure(figsize=(9.2, fig_h), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_facecolor(BG)

    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.012",
                                mutation_aspect=fig_h / 9.2, fc=BG, ec="none"))
    bar_h = 0.34 / fig_h
    ax.add_patch(plt.Rectangle((0, 1 - bar_h), 1, bar_h, fc=BAR, ec="none"))
    for i, c in enumerate(DOTS):
        ax.add_patch(plt.Circle((0.018 + i * 0.022, 1 - bar_h / 2), 0.006 * 9.2 / fig_h * 1.0,
                                fc=c, ec="none", transform=ax.transAxes))
    ax.text(0.5, 1 - bar_h / 2, title, color=DIM, ha="center", va="center",
            family="monospace", fontsize=8.5)

    y = 1 - bar_h - 0.03
    step = (y) / (n + 0.5)
    for line in lines:
        ax.text(0.02, y, line.rstrip("\n") or " ", color=colour_for(line),
                ha="left", va="top", **MONO)
        y -= step

    fig.savefig(dest, facecolor=BG, bbox_inches=None)
    plt.close(fig)
    print(f"  wrote {dest.relative_to(ROOT)} ({n} lines)")


def main():
    log = (ROOT / "scripts" / "_run_quick.log")
    if not log.exists():
        # fall back to scratchpad copy path passed by the caller
        import sys
        log = Path(sys.argv[1])
    text = log.read_text()
    lines = [l for l in text.splitlines()]

    # Split the run into three legible windows by stage boundaries.
    def block(start_key, end_key=None):
        out, on = [], False
        for l in lines:
            if start_key in l:
                on = True
            if on:
                out.append(l)
            if on and end_key and end_key in l and start_key not in l:
                break
        return out

    env = []
    for l in lines:
        env.append(l)
        if "epochs    :" in l:
            break

    compare = block("STAGE: compare", "comparison_summary.json")
    final = block("STAGE: final", "final_metrics.json")
    tail = block("STAGE: cluster")

    render(env + [""] + compare,
           "python run_experiments.py  —  compare stage (MLP vs CNN)",
           OUT / "fig_evidence_run.png")
    render(final,
           "python run_experiments.py  —  final CNN evaluation",
           OUT / "fig_evidence_metrics.png")
    render(tail,
           "python run_experiments.py  —  clustering + novelty detection",
           OUT / "fig_evidence_novelty.png")


if __name__ == "__main__":
    main()
