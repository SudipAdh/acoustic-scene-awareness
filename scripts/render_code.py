#!/usr/bin/env python
"""Render key source sections as editor-style PNGs for the code appendix.

The submission notice asks for screenshots of the important code sections, not
the whole codebase. This produces clean, syntax-highlighted captures of the
five sections the report leans on, framed like an editor window so they read as
screenshots rather than pasted text.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Token

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures"

BG = "#1e1e2e"
BAR = "#2b2b3d"
GUTTER = "#6b7280"
TITLE = "#9aa2b1"
DOTS = ["#ff5f56", "#ffbd2e", "#27c93f"]

# A compact one-dark-ish palette keyed by Pygments token type.
PALETTE = {
    Token.Keyword: "#c678dd",
    Token.Keyword.Namespace: "#c678dd",
    Token.Name.Builtin: "#56b6c2",
    Token.Name.Function: "#61afef",
    Token.Name.Class: "#e5c07b",
    Token.Name.Decorator: "#e5c07b",
    Token.Name.Builtin.Pseudo: "#e06c75",
    Token.Literal.String: "#98c379",
    Token.Literal.String.Doc: "#7f848e",
    Token.Literal.String.Affix: "#98c379",
    Token.Literal.Number: "#d19a66",
    Token.Comment: "#7f848e",
    Token.Comment.Single: "#7f848e",
    Token.Operator: "#56b6c2",
    Token.Punctuation: "#abb2bf",
    Token.Name: "#abb2bf",
    Token.Text: "#abb2bf",
}
DEFAULT = "#abb2bf"
CW = 0.00706  # width of one monospace char in axes units at fontsize 8


def tok_colour(tt):
    t = tt
    while t is not None:
        if t in PALETTE:
            return PALETTE[t]
        t = t.parent
    return DEFAULT


def render(code, title, dest, first_line=1):
    lines = code.rstrip("\n").split("\n")
    n = len(lines)
    fig_h = 0.168 * n + 0.62
    fig = plt.figure(figsize=(9.4, fig_h), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_facecolor(BG)

    bar_h = 0.32 / fig_h
    ax.add_patch(plt.Rectangle((0, 1 - bar_h), 1, bar_h, fc=BAR, ec="none"))
    for i, c in enumerate(DOTS):
        ax.add_patch(plt.Circle((0.016 + i * 0.02, 1 - bar_h / 2), 0.0052,
                                fc=c, ec="none"))
    ax.text(0.5, 1 - bar_h / 2, title, color=TITLE, ha="center", va="center",
            family="monospace", fontsize=8.5)

    # Tokenise per line so we can lay out spans left-to-right.
    all_toks = list(lex(code, PythonLexer()))
    per_line, cur = [[]], 0
    for tt, val in all_toks:
        parts = val.split("\n")
        for j, p in enumerate(parts):
            if j > 0:
                per_line.append([]); cur += 1
            if p:
                per_line[cur].append((tt, p))

    y = 1 - bar_h - 0.028
    step = y / (n + 0.6)
    gx = 0.012
    x0 = 0.055
    for idx, spans in enumerate(per_line[:n]):
        ax.text(gx, y, f"{first_line + idx:>3}", color=GUTTER, ha="left", va="top",
                family="monospace", fontsize=7.6)
        x = x0
        for tt, p in spans:
            ax.text(x, y, p, color=tok_colour(tt), ha="left", va="top",
                    family="monospace", fontsize=8.0)
            x += len(p) * CW
        y -= step

    fig.savefig(dest, facecolor=BG)
    plt.close(fig)
    print(f"  wrote {dest.relative_to(ROOT)} ({n} lines)")


def slice_file(rel, start, end):
    text = (ROOT / rel).read_text().split("\n")
    return "\n".join(text[start - 1:end]), start


def main():
    jobs = [
        ("src/data.py", 56, 79, "src/data.py — log-mel spectrogram front-end",
         "fig_code_melspec.png"),
        ("src/data.py", 209, 235, "src/data.py — official UrbanSound8K fold split",
         "fig_code_foldsplit.png"),
        ("src/models.py", 76, 118, "src/models.py — AudioCNN (convolutional classifier)",
         "fig_code_cnn.png"),
        ("src/models.py", 120, 171, "src/models.py — ConvAutoencoder (novelty detection)",
         "fig_code_autoencoder.png"),
        ("src/train.py", 191, 243, "src/train.py — 10-fold cross-validation loop",
         "fig_code_crossval.png"),
    ]
    for rel, a, b, title, name in jobs:
        code, first = slice_file(rel, a, b)
        render(code, title, OUT / name, first_line=first)


if __name__ == "__main__":
    main()
