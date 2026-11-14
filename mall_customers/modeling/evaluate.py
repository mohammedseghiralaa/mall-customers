from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score)
from mall_customers.config import PROFILE_COLUMNS


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
