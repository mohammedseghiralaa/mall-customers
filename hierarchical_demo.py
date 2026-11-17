import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

from mall_customers.config import K_RANGE, PROFILE_COLUMNS, REPORTS_DIR, RANDOM_STATE
from mall_customers.dataset import load_data
from mall_customers.features import prepare_features
from mall_customers.modeling.train import train_kmeans
from mall_customers.modeling.hierarchical import build_linkage, cut_tree, merge_heights
from mall_customers.plots import plot_dendrogram

K = 5  # starting value, check it against the tables below

df = load_data()
X, _ = prepare_features(df)

# 1. Build the whole tree (Ward linkage)
Z = build_linkage(X, method="ward")

# 2. Merge heights: a big jump means two far-apart groups were merged
print("Merge heights (the merge that reduces k clusters to k-1):")
for i, h in enumerate(merge_heights(Z, n=10)):
    k = i + 2
    print(f"  {k} -> {k - 1} clusters: height {h:.2f}")

# 3. Silhouette for each possible cut
print("\nSilhouette for each cut:")
for k in K_RANGE:
    labels = cut_tree(Z, k)
    print(f"  k={k}: silhouette {silhouette_score(X, labels):.3f}")

# 4. Final cut and evaluation
labels_h = cut_tree(Z, K)
print(f"\nHierarchical, k={K}")
print("  silhouette:    ", round(silhouette_score(X, labels_h), 3))
print("  davies-bouldin:", round(davies_bouldin_score(X, labels_h), 3))

# 5. Compare with K-Means
labels_km = train_kmeans(X, K).labels_
print("\nARI vs K-Means:", round(adjusted_rand_score(labels_km, labels_h), 3))
print("\nAgreement table (rows: hierarchical, columns: K-Means):")
print(pd.crosstab(labels_h, labels_km, rownames=["hier"], colnames=["kmeans"]))

# 6. Profile the hierarchical clusters
df["cluster_hier"] = labels_h
profile = df.groupby("cluster_hier")[PROFILE_COLUMNS].mean().round(1)
profile["size"] = df["cluster_hier"].value_counts().sort_index()
print("\nHierarchical cluster profile:\n", profile)

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
profile.to_csv(REPORTS_DIR / "hierarchical_profile.csv")

# 7. Dendrogram
plot_dendrogram(Z, K)