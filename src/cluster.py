"""Unsupervised analysis of the CNN's learned representation.

The classifier is trained with labels, but the *embedding space* it builds can be
examined without them. If k-means on those embeddings recovers the sound classes,
the network has learned a genuinely structured representation rather than a bare
decision rule — and the classes it confuses show up as overlapping clusters.
"""
from __future__ import annotations

import numpy as np

from . import config as C


def run_tsne(embeddings: np.ndarray, n_components: int = 2,
             perplexity: float = 30.0, max_samples: int = 3000,
             seed: int = C.SEED):
    """Project embeddings to 2-D for visualisation.

    t-SNE is O(n^2)-ish, so large inputs are subsampled. Returns (points, index)
    where `index` selects the rows that were kept.
    """
    from sklearn.manifold import TSNE

    rng = np.random.default_rng(seed)
    idx = (rng.choice(len(embeddings), max_samples, replace=False)
           if len(embeddings) > max_samples else np.arange(len(embeddings)))

    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                init="pca", learning_rate="auto", random_state=seed)
    return tsne.fit_transform(embeddings[idx]), idx


def run_kmeans(embeddings: np.ndarray, n_clusters: int = 10, seed: int = C.SEED):
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    return km.fit_predict(embeddings), km


def clustering_metrics(embeddings: np.ndarray, cluster_labels: np.ndarray,
                       true_labels: np.ndarray) -> dict:
    """Compare discovered clusters against the true classes.

    ARI/NMI are label-permutation invariant, which matters because cluster 3 has
    no reason to correspond to class 3. Silhouette judges cluster separation
    without using labels at all.
    """
    from sklearn.metrics import (adjusted_rand_score,
                                 normalized_mutual_info_score, silhouette_score)

    return {
        "adjusted_rand_index": float(adjusted_rand_score(true_labels, cluster_labels)),
        "normalized_mutual_info": float(normalized_mutual_info_score(true_labels, cluster_labels)),
        "silhouette": float(silhouette_score(embeddings, cluster_labels)),
        "n_clusters": int(len(np.unique(cluster_labels))),
    }


def cluster_purity(cluster_labels: np.ndarray, true_labels: np.ndarray,
                   class_names: list[str]) -> dict:
    """For each cluster, the dominant true class and how pure the cluster is."""
    out = {}
    for c in np.unique(cluster_labels):
        m = cluster_labels == c
        vals, counts = np.unique(true_labels[m], return_counts=True)
        top = vals[counts.argmax()]
        out[int(c)] = {
            "dominant_class": class_names[int(top)],
            "purity": float(counts.max() / counts.sum()),
            "size": int(m.sum()),
        }
    return out
