import numpy as np
import matplotlib.pyplot as plt
import umap  # Make sure umap-learn is installed (`pip install umap-learn`)
from sklearn.datasets import make_blobs
import time

# --- Configuration ---
N_SAMPLES = 500
N_FEATURES = 10  # Original number of dimensions
N_COMPONENTS = 2  # Target number of dimensions
CENTERS = 4  # Number of clusters to generate
RANDOM_STATE = 42
N_NEIGHBORS = 15  # UMAP specific: Number of neighbors to consider
MIN_DIST = 0.1  # UMAP specific: Minimum distance between embedded points

# --- Generate Sample Data ---
# Create high-dimensional data with some cluster structure
X, y = make_blobs(
    n_samples=N_SAMPLES,
    centers=CENTERS,
    n_features=N_FEATURES,
    random_state=RANDOM_STATE,
)
print(f"Original data shape: {X.shape}")  # (n_samples, n_features)

# --- Apply UMAP ---
print(f"\nApplying UMAP to reduce dimensions to {N_COMPONENTS}...")
print(f"(Using n_neighbors={N_NEIGHBORS}, min_dist={MIN_DIST})\n")

# Instantiate UMAP
# Key parameters:
# - n_neighbors: Controls how UMAP balances local versus global structure.
#                Smaller values focus more on local structure, larger values on global.
# - n_components: Target dimensionality (usually 2 or 3).
# - min_dist: Controls how tightly UMAP is allowed to pack points together.
#             Smaller values mean tighter clusters.
# - metric: Distance metric used in the high-dimensional space (e.g., 'euclidean').
# - random_state: For reproducibility.

reducer = umap.UMAP(
    n_neighbors=N_NEIGHBORS,
    n_components=N_COMPONENTS,
    min_dist=MIN_DIST,
    random_state=RANDOM_STATE,
)

# Fit UMAP to the data and transform it
start_time = time.time()
X_umap = reducer.fit_transform(X)
end_time = time.time()

print(f"Transformed data shape: {X_umap.shape}")  # (n_samples, n_components)
print(f"UMAP computation time: {end_time - start_time:.2f} seconds")

# --- Plot Results ---
print("\nPlotting the UMAP-transformed data...")

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap="viridis", alpha=0.7)
plt.title(
    f"UMAP Result ({N_COMPONENTS} Components, Neighbors={N_NEIGHBORS}, MinDist={MIN_DIST})"
)
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.legend(
    handles=scatter.legend_elements()[0],
    labels=[f"Cluster {i}" for i in range(CENTERS)],
)
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

print("\nUMAP Example Complete.")
print("Note: UMAP is non-linear and aims to balance local and global structure.")
print(
    "It's often faster than t-SNE and can provide good cluster separation and global layout."
)
