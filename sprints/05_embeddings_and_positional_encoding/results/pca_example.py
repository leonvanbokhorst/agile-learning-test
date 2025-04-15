import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

# --- Configuration ---
N_SAMPLES = 500
N_FEATURES = 10  # Original number of dimensions
N_COMPONENTS = 2  # Target number of dimensions
CENTERS = 4  # Number of clusters to generate
RANDOM_STATE = 42

# --- Generate Sample Data ---
# Create high-dimensional data with some cluster structure
X, y = make_blobs(
    n_samples=N_SAMPLES,
    centers=CENTERS,
    n_features=N_FEATURES,
    random_state=RANDOM_STATE,
)
print(f"Original data shape: {X.shape}")  # (n_samples, n_features)

# --- Apply PCA ---
print(f"\nApplying PCA to reduce dimensions to {N_COMPONENTS}...")

# Instantiate PCA
# n_components specifies the target number of dimensions
pca = PCA(n_components=N_COMPONENTS)

# Fit PCA to the data and transform the data
# fit_transform() combines fitting and transformation in one step
X_pca = pca.fit_transform(X)

print(f"Transformed data shape: {X_pca.shape}")  # (n_samples, n_components)
print(f"Explained variance ratio by component: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")

# --- Plot Results ---
print("\nPlotting the PCA-transformed data...")

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", alpha=0.7)
plt.title(f"PCA Result ({N_COMPONENTS} Components)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(
    handles=scatter.legend_elements()[0],
    labels=[f"Cluster {i}" for i in range(CENTERS)],
)
plt.grid(True, linestyle="--", alpha=0.5)

# Add text showing total explained variance
plt.text(
    0.95,
    0.01,
    f"Total Variance Explained: {np.sum(pca.explained_variance_ratio_):.2%}",
    verticalalignment="bottom",
    horizontalalignment="right",
    transform=plt.gca().transAxes,
    fontsize=9,
)

plt.show()

print("\nPCA Example Complete.")
print("Note: PCA finds linear projections that maximize variance.")
print("The plot shows the data projected onto the first two principal components.")
