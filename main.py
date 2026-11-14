from mall_customers.config import PROCESSED_DATA_DIR
from mall_customers.dataset import load_data, check_data
from mall_customers.features import prepare_features
from mall_customers.modeling.train import evaluate_k_values, train_kmeans
from mall_customers.modeling.evaluate import compute_metrics, profile_clusters
from mall_customers.plots import plot_elbow_silhouette, plot_clusters

def main():
    # 1. Load and check the data
    df = load_data()
    check_data(df)

    # 2. Select features and scale them
    X, scaler = prepare_features(df)

    # 3. Find k (elbow + silhouette)
    ks, inertias, silhouettes = evaluate_k_values(X)
    plot_elbow_silhouette(ks, inertias, silhouettes)
    best_k = ks[silhouettes.index(max(silhouettes))]
    print(f"\nBest k by silhouette: {best_k}")

    # 4. Train the final model
    km = train_kmeans(X, best_k)
    df["cluster"] = km.labels_

    # 5. Evaluate and profile the clusters
    print("\nMetrics:", compute_metrics(X, km.labels_))
    print("\nCluster profile:\n", profile_clusters(df))

    # 6. Plot and save the results
    plot_clusters(X, km.labels_, km.cluster_centers_)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_DIR / "clustered_customers.csv", index=False)


if __name__ == "__main__":
    main()

