#!/usr/bin/env python
"""End-to-end pipeline check that does not need UrbanSound8K.

Builds a small 10-class dataset from ESC-50 shaped exactly like UrbanSound8K
(fold / classID / class columns) and runs training, evaluation, clustering and
novelty detection over it. The point is to prove the code path works; the
accuracy numbers here are not results.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C  # noqa: E402
from src.data import load_esc50_metadata, precompute_features  # noqa: E402

SMOKE_CLASSES = ["dog", "rain", "crying_baby", "door_wood_knock", "helicopter",
                 "chainsaw", "rooster", "sea_waves", "clock_tick", "sneezing"]


def build():
    df = load_esc50_metadata()
    sub = df[df["category"].isin(SMOKE_CLASSES)].copy().reset_index(drop=True)

    cmap = {c: i for i, c in enumerate(sorted(SMOKE_CLASSES))}
    sub["classID"] = sub["category"].map(cmap)
    sub["class"] = sub["category"]
    # ESC-50 ships 5 folds; remap to 10 so the US8K fold logic is exercised.
    sub["fold"] = ((sub["fold"] - 1) * 2 + (np.arange(len(sub)) % 2)) + 1

    print(f"smoke set: {len(sub)} clips, {sub['classID'].nunique()} classes, "
          f"folds {sorted(sub['fold'].unique())}")
    return sub


def main():
    print(f"device: {C.DEVICE}")
    sub = build()
    X, meta = precompute_features(sub, "smoke", force=False)
    print(f"features: {X.shape}")

    from src.train import cross_validate

    folds = [1, 2]
    epochs = 8

    print("\n--- MLP ---")
    r_mlp = cross_validate(X, meta, "mlp", folds=folds, epochs=epochs,
                           augment=False, tag="smoke_mlp")
    print("\n--- CNN ---")
    r_cnn = cross_validate(X, meta, "cnn", folds=folds, epochs=epochs,
                           augment=True, tag="smoke_cnn")

    print("\n--- evaluation + figures ---")
    from src import viz
    from src.evaluate import evaluate_cv_predictions, extract_embeddings, full_metrics

    yt, yp, _, _ = evaluate_cv_predictions(X, meta, "cnn", folds=folds,
                                           epochs=epochs, augment=True)
    names = sorted(SMOKE_CLASSES)
    m = full_metrics(yt, yp, names)
    print(f"pooled accuracy {m['accuracy']:.3f} macro-F1 {m['f1_macro']:.3f}")

    viz.use_report_style()
    viz.plot_confusion_matrix(np.array(m["confusion_matrix"]), names,
                              name="smoke_confusion", title="SMOKE confusion matrix")
    viz.plot_per_class_f1(m["per_class_f1"], name="smoke_per_class_f1")
    viz.plot_model_comparison({"MLP": r_mlp, "CNN": r_cnn}, name="smoke_comparison")

    print("\n--- clustering ---")
    from src.cluster import clustering_metrics, run_kmeans, run_tsne
    from src.data import compute_norm_stats, make_fold_split
    from src.train import train_one_fold

    model, _ = train_one_fold(X, meta, test_fold=1, model_name="cnn",
                              epochs=epochs, augment=True, verbose=False)
    (Xtr, ytr), _, (Xte, yte) = make_fold_split(X, meta, 1)
    mean, std = compute_norm_stats(Xtr)
    emb, labels = extract_embeddings(model, Xte, yte, mean, std)
    cl, _ = run_kmeans(emb, n_clusters=10)
    cm_ = clustering_metrics(emb, cl, labels)
    print(f"ARI {cm_['adjusted_rand_index']:.3f} NMI {cm_['normalized_mutual_info']:.3f} "
          f"silhouette {cm_['silhouette']:.3f}")
    pts, idx = run_tsne(emb, perplexity=12, max_samples=400)
    viz.plot_tsne(pts, labels[idx], names, name="smoke_tsne", title="SMOKE t-SNE")

    print("\n--- anomaly ---")
    from src.anomaly import evaluate_novelty, reconstruction_scores, train_autoencoder

    # Hold out two classes entirely as "novel"
    novel_ids = {0, 1}
    known_m = ~np.isin(meta["classID"].values, list(novel_ids))
    novel_m = np.isin(meta["classID"].values, list(novel_ids))
    Xk, Xn = X[known_m], X[novel_m]
    split = int(len(Xk) * 0.8)
    mean2, std2 = compute_norm_stats(Xk[:split])

    ae, _ = train_autoencoder(Xk[:split], mean2, std2, epochs=8, verbose=True)
    sk = reconstruction_scores(ae, Xk[split:], mean2, std2)
    sn = reconstruction_scores(ae, Xn, mean2, std2)
    res = evaluate_novelty(sk, sn)
    print(f"novelty ROC-AUC {res['roc_auc']:.3f} F1 {res['f1']:.3f}")
    viz.plot_anomaly_scores(sk, sn, threshold=res["threshold"], name="smoke_anomaly")
    viz.plot_roc(np.array(res["fpr"]), np.array(res["tpr"]), res["roc_auc"],
                 name="smoke_roc")

    print("\nSMOKE TEST PASSED — every stage ran end to end.")


if __name__ == "__main__":
    main()
