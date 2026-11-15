import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from mall_customers.config import FIGURES_DIR, RANDOM_STATE, N_INIT
from mall_customers.dataset import load_data

FEATURES_3D = ["Annual Income (k$)", "Spending Score (1-100)", "Age"]
K = 6  # best k for these 3 features in compare_features.py

df = load_data()
X = StandardScaler().fit_transform(df[FEATURES_3D])

# 1. K-Means on all 3 features
labels = KMeans(n_clusters=K, n_init=N_INIT, random_state=RANDOM_STATE).fit_predict(X)

# 2. PCA: 3 features -> 2 axes
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

print("Explained variance per axis:", pca.explained_variance_ratio_.round(3))
print("Total kept:", round(pca.explained_variance_ratio_.sum(), 3))
print("\nLoadings (how each axis is built):")
print(pd.DataFrame(pca.components_, columns=FEATURES_3D, index=["PC1", "PC2"]).round(2))

# 3. Plot
plt.figure(figsize=(7, 5))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="viridis")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} of variance)")
plt.title(f"K-Means (k={K}) on 3 features, shown with PCA")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURES_DIR / "pca_clusters.png", dpi=150, bbox_inches="tight")
plt.show()