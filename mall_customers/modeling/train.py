from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mall_customers.config import K_RANGE, RANDOM_STATE, N_INIT


def evaluate_k_values(X):
    inertias, silhouettes = [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE).fit(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, km.labels_))
    return list(K_RANGE), inertias, silhouettes


def train_kmeans(X, k):
    return KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE).fit(X)