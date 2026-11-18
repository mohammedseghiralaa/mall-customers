import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from mall_customers.config import FIGURES_DIR


def _save(name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")


def plot_elbow_silhouette(ks, inertias, silhouettes):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ks, inertias, marker="o")
    ax[0].set_title("Elbow method (inertia)")
    ax[1].plot(ks, silhouettes, marker="o")
    ax[1].set_title("Silhouette score")
    for a in ax:
        a.set_xlabel("k")
    _save("elbow_silhouette")
    plt.show()


def plot_clusters(X, labels, centers):
    plt.figure(figsize=(7, 5))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")
    plt.scatter(centers[:, 0], centers[:, 1], c="red", marker="X",
                s=200, label="centers")
    plt.xlabel("Income (scaled)")
    plt.ylabel("Spending score (scaled)")
    plt.title("Customer clusters")
    plt.legend()
    _save("clusters")
    plt.show()


def plot_dendrogram(Z, k):
    threshold = (Z[-k, 2] + Z[-(k - 1), 2]) / 2        # 1. computes the height (6.9 for k=5)
    plt.figure(figsize=(11, 5))
    dendrogram(Z, truncate_mode="lastp", p=30,
               color_threshold=threshold)               # 2. colors the branches below the line
    plt.axhline(threshold, color="red", linestyle="--",
                label=f"cut for k={k}")                 # 3. draws the red line
    plt.title("Hierarchical clustering dendrogram (Ward)")
    plt.xlabel("Cluster size (number of customers)")
    plt.ylabel("Merge distance")
    plt.legend()
    _save("dendrogram")
    plt.show()