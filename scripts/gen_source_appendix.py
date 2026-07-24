#!/usr/bin/env python
"""Generate the full-source LaTeX appendix from the project's modules.

The assignment brief requires the code in its totality to appear in the
appendix. This dumps every source file into assignment_report/source_appendix.tex
as syntax-highlighted listings, which the report \\inputs. Regenerate whenever
the source changes so the appendix never drifts from the code.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FILES = [
    ("src/config.py", "Configuration and hyperparameters"),
    ("src/data.py", "Data pipeline: loading, mel-spectrograms, augmentation"),
    ("src/models.py", "Network architectures: MLP, CNN, autoencoder"),
    ("src/train.py", "Training loop and 10-fold cross-validation"),
    ("src/evaluate.py", "Evaluation and embedding extraction"),
    ("src/anomaly.py", "Autoencoder novelty detection"),
    ("src/cluster.py", "k-means and t-SNE clustering"),
    ("src/viz.py", "Figure generation"),
    ("run_experiments.py", "Staged experiment runner"),
    ("demo/live_demo.py", "Real-time microphone demo"),
]


def main() -> None:
    out = [r"% Auto-generated. Regenerate: python scripts/gen_source_appendix.py", ""]
    for path, desc in FILES:
        code = (ROOT / path).read_text().rstrip()
        safe = path.replace("_", r"\_")
        out += [
            r"\subsection{\texttt{%s}}" % safe,
            desc + ".",
            r"\begin{lstlisting}[caption={%s (\texttt{%s})}]" % (desc, safe),
            code,
            r"\end{lstlisting}",
            "",
        ]
    dest = ROOT / "assignment_report" / "source_appendix.tex"
    dest.write_text("\n".join(out))
    lines = sum(len((ROOT / p).read_text().splitlines()) for p, _ in FILES)
    print(f"wrote {dest.relative_to(ROOT)}: {len(FILES)} files, {lines} lines")


if __name__ == "__main__":
    main()
