#!/usr/bin/env python
"""End-to-end experiment runner.

Stages (select with --stage, default: all)

  prep     extract and cache mel-spectrograms for both datasets
  explore  dataset figures (class balance, example spectrograms)
  compare  MLP vs CNN, and CNN with/without augmentation
  final    full 10-fold CNN run -> confusion matrix, per-class F1, saved model
  cluster  k-means + t-SNE over the CNN's learned embeddings
  anomaly  autoencoder novelty detection vs unseen ESC-50 sounds

Examples
  python run_experiments.py --stage prep
  python run_experiments.py --stage compare --folds 1 2 3 --epochs 25
  python run_experiments.py                      # everything, full protocol
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from src import config as C
from src import viz
from src.data import (load_esc50_metadata, load_us8k_metadata,
                      compute_norm_stats, make_fold_split, precompute_features)


def _save(name: str, obj) -> None:
    p = C.LOGS_DIR / f"{name}.json"
    p.write_text(json.dumps(obj, indent=2, default=str))
    print(f"  saved -> {p}")


def banner(msg: str) -> None:
    print(f"\n{'='*74}\n  {msg}\n{'='*74}")


# ---------------------------------------------------------------------------
def stage_prep(force=False):
    banner("STAGE: prep — extracting mel-spectrograms")

    us8k = load_us8k_metadata()
    print(f"UrbanSound8K: {len(us8k)} clips, {us8k['class'].nunique()} classes")
    X, meta = precompute_features(us8k, "us8k", force=force)

    esc = load_esc50_metadata()
    novel = esc[esc["category"].isin(C.ESC50_NOVEL_CLASSES)].reset_index(drop=True)
    print(f"ESC-50 novel subset: {len(novel)} clips, {novel['category'].nunique()} classes")
    Xn, mn = precompute_features(novel, "esc50_novel", force=force)

    _save("prep_summary", {
        "us8k_shape": list(X.shape),
        "us8k_class_counts": us8k["class"].value_counts().to_dict(),
        "esc50_novel_shape": list(Xn.shape),
        "esc50_novel_classes": sorted(novel["category"].unique().tolist()),
        "feature": {"sample_rate": C.SAMPLE_RATE, "n_mels": C.N_MELS,
                    "n_fft": C.N_FFT, "hop_length": C.HOP_LENGTH,
                    "clip_seconds": C.CLIP_SECONDS, "n_frames": C.N_FRAMES},
    })
    return X, meta


# ---------------------------------------------------------------------------
def stage_explore(X, meta):
    banner("STAGE: explore — dataset figures")
    viz.use_report_style()

    us8k = load_us8k_metadata()
    viz.plot_class_distribution(us8k["class"].value_counts().to_dict())

    # one representative spectrogram per class
    specs, titles = [], []
    for cid in range(10):
        rows = np.where(meta["classID"].values == cid)[0]
        if len(rows):
            specs.append(X[rows[0]])
            titles.append(C.US8K_CLASSES[cid])
    viz.plot_spectrogram_grid(specs, titles, ncols=5)

    # the same class under augmentation, to illustrate what the model sees
    from src.data import spec_augment, time_shift
    rng = np.random.default_rng(C.SEED)
    base = X[np.where(meta["classID"].values == 8)[0][0]]  # siren
    variants = [base, time_shift(base, rng), spec_augment(base, rng),
                spec_augment(time_shift(base, rng), rng)]
    viz.plot_spectrogram_grid(
        variants, ["original", "time shift", "SpecAugment", "both"],
        name="fig_augmentation", ncols=4,
        suptitle="Augmentation: the same siren, varied so the CNN cannot memorise it")


# ---------------------------------------------------------------------------
def stage_compare(X, meta, folds, epochs):
    banner("STAGE: compare — MLP vs CNN, and augmentation ablation")
    from src.train import cross_validate, train_one_fold

    results = {}
    results["MLP"] = cross_validate(X, meta, "mlp", folds=folds, epochs=epochs,
                                    augment=False, tag="mlp")
    results["CNN"] = cross_validate(X, meta, "cnn", folds=folds, epochs=epochs,
                                    augment=False, tag="cnn_noaug")
    results["CNN + augment"] = cross_validate(X, meta, "cnn", folds=folds, epochs=epochs,
                                              augment=True, tag="cnn_aug")

    viz.use_report_style()
    viz.plot_model_comparison(results, n_folds=len(folds))

    # training curves on a single representative fold
    f = folds[0]
    hists = {}
    for label, (mname, aug) in {
        "MLP": ("mlp", False),
        "CNN": ("cnn", False),
        "CNN + augment": ("cnn", True),
    }.items():
        _, res = train_one_fold(X, meta, test_fold=f, model_name=mname,
                                epochs=epochs, augment=aug, verbose=False)
        hists[label] = res.history
    viz.plot_training_curves(hists)

    _save("comparison_summary", results)
    return results


# ---------------------------------------------------------------------------
def stage_final(X, meta, folds, epochs):
    banner("STAGE: final — full CNN evaluation")
    from src.evaluate import evaluate_cv_predictions, full_metrics

    y_true, y_pred, y_prob, per_fold = evaluate_cv_predictions(
        X, meta, "cnn", folds=folds, epochs=epochs, augment=True)

    m = full_metrics(y_true, y_pred, C.US8K_CLASSES)
    print("\n" + m["report"])
    print(f"Accuracy {m['accuracy']:.4f} | macro-F1 {m['f1_macro']:.4f}")

    viz.use_report_style()
    viz.plot_confusion_matrix(np.array(m["confusion_matrix"]), C.US8K_CLASSES)
    viz.plot_per_class_f1(m["per_class_f1"])

    _save("final_metrics", {k: v for k, v in m.items() if k != "report"})
    (C.LOGS_DIR / "final_classification_report.txt").write_text(m["report"])

    # Train and persist one deployable model (held-out fold 10) for the demo.
    from src.train import train_one_fold
    banner("training deployable model for the live demo (test fold 10)")
    model, res = train_one_fold(X, meta, test_fold=10, model_name="cnn",
                                epochs=epochs, augment=True,
                                save_path=C.MODELS_DIR / "cnn_demo.pt")
    print(f"demo model saved: test acc {res.test_acc:.4f}")
    return m, model


# ---------------------------------------------------------------------------
def stage_cluster(X, meta, epochs):
    banner("STAGE: cluster — k-means + t-SNE on CNN embeddings")
    from src.cluster import (cluster_purity, clustering_metrics, run_kmeans, run_tsne)
    from src.evaluate import extract_embeddings
    from src.train import train_one_fold

    test_fold = 10
    model, _ = train_one_fold(X, meta, test_fold=test_fold, model_name="cnn",
                              epochs=epochs, augment=True, verbose=False)

    (Xtr, ytr), _, (Xte, yte) = make_fold_split(X, meta, test_fold)
    mean, std = compute_norm_stats(Xtr)

    # Embeddings come from the held-out fold — never data the CNN trained on.
    emb, labels = extract_embeddings(model, Xte, yte, mean, std)
    print(f"embeddings: {emb.shape}")

    clusters, _ = run_kmeans(emb, n_clusters=10)
    metrics = clustering_metrics(emb, clusters, labels)
    purity = cluster_purity(clusters, labels, C.US8K_CLASSES)
    print(f"ARI {metrics['adjusted_rand_index']:.3f} | "
          f"NMI {metrics['normalized_mutual_info']:.3f} | "
          f"silhouette {metrics['silhouette']:.3f}")

    pts, idx = run_tsne(emb)
    viz.use_report_style()
    viz.plot_tsne(pts, labels[idx], C.US8K_CLASSES)

    _save("cluster_metrics", {"metrics": metrics, "cluster_purity": purity})
    return metrics


# ---------------------------------------------------------------------------
def stage_anomaly(X, meta, epochs):
    banner("STAGE: anomaly — autoencoder novelty detection")
    from src.anomaly import (evaluate_novelty, reconstruction_scores,
                             softmax_confidence_baseline, train_autoencoder)
    from src.data import SpectrogramDataset
    from src.train import predict, train_one_fold
    from torch.utils.data import DataLoader

    Xn = np.load(C.PROCESSED_DIR / "esc50_novel_X.npy")
    print(f"novel (unseen) clips: {Xn.shape}")

    test_fold = 10
    (Xtr, ytr), (Xva, yva), (Xte, yte) = make_fold_split(X, meta, test_fold)
    mean, std = compute_norm_stats(Xtr)

    ae, hist = train_autoencoder(Xtr, mean, std, epochs=epochs, X_val=Xva)

    known = reconstruction_scores(ae, Xte, mean, std)
    novel = reconstruction_scores(ae, Xn, mean, std)
    res = evaluate_novelty(known, novel)
    print(f"AE novelty  ROC-AUC {res['roc_auc']:.3f} | AP {res['average_precision']:.3f} | "
          f"P {res['precision']:.3f} R {res['recall']:.3f} F1 {res['f1']:.3f}")

    # Baseline: does classifier under-confidence already do this job?
    clf, _ = train_one_fold(X, meta, test_fold=test_fold, model_name="cnn",
                            epochs=epochs, augment=True, verbose=False)
    dl_k = DataLoader(SpectrogramDataset(Xte, yte, mean, std), batch_size=C.BATCH_SIZE)
    dl_n = DataLoader(SpectrogramDataset(Xn, np.zeros(len(Xn), dtype=np.int64), mean, std),
                      batch_size=C.BATCH_SIZE)
    _, _, pk = predict(clf, dl_k, C.DEVICE)
    _, _, pn = predict(clf, dl_n, C.DEVICE)
    base = softmax_confidence_baseline(pk, pn)
    print(f"softmax baseline ROC-AUC {base['roc_auc']:.3f} "
          f"(mean confidence: known {base['known_mean_conf']:.3f}, "
          f"novel {base['novel_mean_conf']:.3f})")

    viz.use_report_style()
    viz.plot_anomaly_scores(known, novel, threshold=res["threshold"])
    viz.plot_roc(np.array(res["fpr"]), np.array(res["tpr"]), res["roc_auc"])

    # visual side-by-side of reconstructions
    with torch.no_grad():
        def recon(arr, i):
            x = torch.from_numpy((arr[i] - mean) / std).float()[None, None].to(C.DEVICE)
            return x[0, 0].cpu().numpy(), ae(x)[0, 0].cpu().numpy()

        pairs = [recon(Xte, 0), recon(Xte, 1), recon(Xn, 0), recon(Xn, 1)]
    viz.plot_reconstructions([p[0] for p in pairs], [p[1] for p in pairs],
                             ["known", "known", "novel", "novel"])

    torch.save({"state_dict": ae.state_dict(), "mean": mean, "std": std,
                "threshold": res["threshold"]}, C.MODELS_DIR / "autoencoder.pt")

    _save("anomaly_results", {
        "autoencoder": {k: v for k, v in res.items() if k not in ("fpr", "tpr")},
        "softmax_baseline": base,
        "ae_history": hist,
    })
    return res


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    choices=["all", "prep", "explore", "compare", "final",
                             "cluster", "anomaly"])
    ap.add_argument("--folds", type=int, nargs="+", default=None,
                    help="folds to evaluate (default: all 10)")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke run: 1 fold, few epochs")
    ap.add_argument("--force-prep", action="store_true")
    args = ap.parse_args()

    folds = args.folds or list(range(1, C.N_FOLDS + 1))
    epochs = args.epochs
    if args.quick:
        folds, epochs = [1], 3
        print(">>> QUICK MODE: 1 fold, 3 epochs")

    print(f"device: {C.DEVICE} | torch {torch.__version__}")
    print(f"folds: {folds} | epochs: {epochs}")
    t0 = time.time()

    X, meta = stage_prep(force=args.force_prep)
    s = args.stage

    if s in ("all", "explore"):
        stage_explore(X, meta)
    if s in ("all", "compare"):
        stage_compare(X, meta, folds, epochs)
    if s in ("all", "final"):
        stage_final(X, meta, folds, epochs)
    if s in ("all", "cluster"):
        stage_cluster(X, meta, epochs)
    if s in ("all", "anomaly"):
        stage_anomaly(X, meta, epochs)

    banner(f"DONE in {(time.time()-t0)/60:.1f} min — figures in {C.FIGURES_DIR}")


if __name__ == "__main__":
    main()
