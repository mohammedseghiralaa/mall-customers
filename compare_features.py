from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

from mall_customers.config import K_RANGE, RANDOM_STATE, N_INIT
from mall_customers.dataset import load_data

feature_sets = {
    "income + spending": ["Annual Income (k$)", "Spending Score (1-100)"],
    "income + spending + age": ["Annual Income (k$)", "Spending Score (1-100)", "Age"],
    "age + spending": ["Age", "Spending Score (1-100)"],
    "age + income": ["Age", "Annual Income (k$)"],
}

df = load_data()

print(f"{'features':28} {'best k':>6} {'silhouette':>11} {'davies-bouldin':>15}")
for name, cols in feature_sets.items():
    X = StandardScaler().fit_transform(df[cols])

    best_k, best_sil, best_labels = None, -1, None
    for k in K_RANGE:
        labels = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE).fit_predict(X)
        sil = silhouette_score(X, labels)
        if sil > best_sil:
            best_k, best_sil, best_labels = k, sil, labels

    db = davies_bouldin_score(X, best_labels)
    print(f"{name:28} {best_k:>6} {best_sil:>11.3f} {db:>15.3f}")
    