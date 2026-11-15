import json

import pandas as pd

from mall_customers.config import PROCESSED_DATA_DIR, REPORTS_DIR
from mall_customers.dataset import load_data, check_data
from mall_customers.features import prepare_features
from mall_customers.modeling.train import evaluate_k_values, train_kmeans
from mall_customers.modeling.evaluate import (compute_metrics, profile_clusters,
                                              stability_score)  # NEW
from mall_customers.plots import plot_elbow_silhouette, plot_clusters


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and check the data
    df = load_data()
    check_data(df)

    # 2. Select features and scale them
    X, scaler = prepare_features(df)

    # 3. Find k (elbow + silhouette + stability)
    ks, inertias, silhouettes = evaluate_k_values(X)
    plot_elbow_silhouette(ks, inertias, silhouettes)
    best_k = ks[silhouettes.index(max(silhouettes))]
    print(f"\nBest k by silhouette: {best_k}")

    stabilities = [stability_score(X, k)[0] for k in ks]  # NEW

    k_table = pd.DataFrame({
        "k": ks,
        "inertia": inertias,
        "silhouette": silhouettes,
        "stability": stabilities,                          # NEW
    })
    k_table.to_csv(REPORTS_DIR / "k_selection.csv", index=False)
    print("\nK selection:\n", k_table.round(3))             # NEW

    # 4. Train the final model
    km = train_kmeans(X, best_k)
    df["cluster"] = km.labels_

    # 5. Evaluate and profile the clusters
    metrics = compute_metrics(X, km.labels_)
    profile = profile_clusters(df)
    stab_mean, stab_std = stability_score(X, best_k)       # NEW
    print("\nMetrics:", metrics)
    print(f"\nStability (ARI): {stab_mean:.3f} +/- {stab_std:.3f}")  # NEW
    print("\nCluster profile:\n", profile)

    results = {"best_k": int(best_k)}
    results.update({name: float(v) for name, v in metrics.items()})
    results["stability_mean"] = stab_mean                  # NEW
    results["stability_std"] = stab_std                    # NEW
    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    profile.to_csv(REPORTS_DIR / "cluster_profile.csv")

    # 6. Plot and save the results
    plot_clusters(X, km.labels_, km.cluster_centers_)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_DIR / "clustered_customers.csv", index=False)


if __name__ == "__main__":
    main()