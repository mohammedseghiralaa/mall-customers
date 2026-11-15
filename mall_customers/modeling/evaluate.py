import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, silhouette_score,
                             davies_bouldin_score, calinski_harabasz_score)
from mall_customers.config import PROFILE_COLUMNS, RANDOM_STATE, N_INIT


def compute_metrics(X, labels):
    return {
        "silhouette": silhouette_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
    }


def profile_clusters(df):
    profile = df.groupby("cluster")[PROFILE_COLUMNS].mean().round(1)
    profile["size"] = df["cluster"].value_counts().sort_index()
    return profile


def stability_score(X, k, n_runs=20, frac=0.8):
    """Average ARI between K-Means fitted on two random subsamples."""
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(X)
    size = int(frac * n)
    scores = []
    for _ in range(n_runs):
        idx_a = rng.choice(n, size=size, replace=False)
        idx_b = rng.choice(n, size=size, replace=False)
        km_a = KMeans(n_clusters=k, n_init=N_INIT,
                      random_state=int(rng.integers(1_000_000))).fit(X[idx_a])
        km_b = KMeans(n_clusters=k, n_init=N_INIT,
                      random_state=int(rng.integers(1_000_000))).fit(X[idx_b])
        scores.append(adjusted_rand_score(km_a.predict(X), km_b.predict(X)))
    return float(np.mean(scores)), float(np.std(scores))