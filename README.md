# Customer Segmentation with K-Means

Unsupervised machine learning project that groups 200 mall customers into meaningful segments using **K-Means clustering**, so a shop can adapt its marketing to each type of customer.

The project covers the full workflow: data checks, feature scaling, choosing the number of clusters (elbow, silhouette and stability), evaluation, cluster profiling, feature comparison and PCA visualization.

## Results at a glance

- **5 customer segments** found from annual income and spending score
- **Silhouette score:** 0.555 (reasonable separation)
- **Davies-Bouldin index:** 0.572 (low is good)
- **Stability (ARI):** 0.976 +/- 0.018 (the same groups are found on random subsamples)

![Customer clusters](reports/figures/clusters.png)

## Dataset

[Mall Customers dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) (Kaggle): 200 customers, no missing values, no duplicates.

| Column | Description |
|---|---|
| `CustomerID` | Unique identifier (not used) |
| `Genre` | Gender (not used) |
| `Age` | Age in years (used to describe the clusters) |
| `Annual Income (k$)` | Yearly income in thousands of dollars (**feature**) |
| `Spending Score (1-100)` | Score assigned by the mall (**feature**) |

Put `Mall_Customers.csv` in `data/raw/` before running the project.

## Method

1. **Data check:** shape, missing values, duplicates, summary statistics.
2. **Feature selection and scaling:** `Annual Income` and `Spending Score`, standardized with `StandardScaler` so both features count equally in the distance.
3. **Choosing k:** K-Means (with k-means++ initialization, `n_init=10`) was run for k = 2 to 10 and compared using three tools:
   - **Elbow method** (inertia)
   - **Silhouette score**
   - **Stability:** K-Means is fitted on two random 80% subsamples, both models label all customers, and the agreement is measured with the Adjusted Rand Index. This is repeated 20 times.
4. **Evaluation:** silhouette, Davies-Bouldin and Calinski-Harabasz scores.
5. **Profiling:** the average income, spending score and age of each cluster, so each group can be named and interpreted.

## Choosing k

| k | Inertia | Silhouette | Stability |
|---|---|---|---|
| 2 | 269.7 | 0.321 | 0.224 |
| 3 | 157.7 | 0.467 | 1.000 |
| 4 | 108.9 | 0.494 | 0.775 |
| **5** | **65.6** | **0.555** | **0.976** |
| 6 | 55.1 | 0.540 | 0.924 |
| 7 | 44.9 | 0.528 | 0.881 |
| 8 | 37.2 | 0.455 | 0.821 |
| 9 | 32.4 | 0.457 | 0.803 |
| 10 | 30.0 | 0.443 | 0.750 |

**k = 5** was chosen because the three tools agree: the elbow bends at 5, the silhouette is highest at 5, and the clusters are very stable (0.976).

![Elbow and silhouette](reports/figures/elbow_silhouette.png)

## Customer segments

| Cluster | Avg. income (k$) | Avg. spending | Avg. age | Size | Segment | Suggested action |
|---|---|---|---|---|---|---|
| 0 | 55.3 | 49.5 | 42.7 | 81 | Mainstream customers | Regular promotions and loyalty points |
| 1 | 86.5 | 82.1 | 32.7 | 39 | Best customers | VIP program, early access to new products |
| 2 | 25.7 | 79.4 | 25.3 | 22 | Young enthusiasts | Trendy products, social media campaigns |
| 3 | 88.2 | 17.1 | 41.1 | 35 | Wealthy but careful | Premium products and personalized offers (largest untapped potential) |
| 4 | 26.3 | 20.9 | 45.2 | 23 | Budget-conscious | Discounts and value bundles |

Cluster numbers are arbitrary and can change between runs. The segment names are an interpretation of each group's averages.

## Experiments

### Which features give the best clusters?

| Features | Best k | Silhouette | Davies-Bouldin |
|---|---|---|---|
| **Income + spending** | **5** | **0.555** | **0.572** |
| Income + spending + age | 6 | 0.428 | 0.825 |
| Age + spending | 2 | 0.472 | 0.845 |

Adding age made the clusters less separated, so the final model uses income and spending only. Age is used afterwards to describe the segments.

### PCA visualization (3 features)

To see the 3-feature clustering in 2D, PCA keeps 77.6% of the variance (PC1: 44.3%, PC2: 33.3%). PC2 is almost exactly income, and PC1 contrasts age with spending score. The plot shows that age only splits the middle group in a fuzzy way, which confirms the choice above.

![PCA clusters](reports/figures/pca_clusters.png)

## Project structure

```
Mall_Customers/
├── data/
│   ├── raw/                    # Mall_Customers.csv (input)
│   └── processed/              # clustered_customers.csv (output)
├── reports/
│   ├── figures/                # saved plots
│   ├── cluster_profile.csv     # average of each feature per cluster
│   ├── k_selection.csv         # inertia, silhouette, stability for each k
│   └── metrics.json            # final metrics
├── mall_customers/
│   ├── config.py               # paths and settings
│   ├── dataset.py              # load and check data
│   ├── features.py             # select and scale features
│   ├── plots.py                # plots
│   └── modeling/
│       ├── train.py            # try k values, train K-Means
│       └── evaluate.py         # metrics, profiling, stability
├── main.py                     # runs the full pipeline
├── compare_features.py         # feature set comparison
├── pca_demo.py                 # PCA visualization
└── requirements.txt
```

## How to run

```bash
git clone https://github.com/mohammedseghiralaa/mall-customers.git
cd mall-customers

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# put Mall_Customers.csv in data/raw/ then:
python main.py                   # full pipeline
python compare_features.py       # feature comparison
python pca_demo.py               # PCA plot
```

Close each plot window to let the script continue. Results are saved in `reports/` and `data/processed/`.

## Limitations and next steps

- Only 200 customers: results are less certain than with a large dataset.
- K-Means assumes round clusters of similar size and is sensitive to outliers.
- Next steps: compare with DBSCAN, hierarchical clustering and Gaussian Mixture Models, and try the same pipeline on a larger retail dataset (for example using RFM features).

## Tools

Python, pandas, NumPy, scikit-learn, matplotlib.
