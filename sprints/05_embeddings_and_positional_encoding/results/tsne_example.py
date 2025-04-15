import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs
import time

# --- Configuration ---
N_SAMPLES = 500
N_FEATURES = 10  # Original number of dimensions
N_COMPONENTS = 2  # Target number of dimensions
CENTERS = 4  # Number of clusters to generate
RANDOM_STATE = 42
PERPLEXITY = 30  # t-SNE specific parameter, relates to number of nearest neighbors
LEARNING_RATE = "auto"  # Common default for t-SNE
N_ITER = 1000  # Number of optimization iterations

# --- Generate Sample Data ---
# Create high-dimensional data with some cluster structure
X, y = make_blobs(
    n_samples=N_SAMPLES,
    centers=CENTERS,
    n_features=N_FEATURES,
    random_state=RANDOM_STATE,
)
print(f"Original data shape: {X.shape}")  # (n_samples, n_features)

# --- Apply t-SNE ---
print(f"\nApplying t-SNE to reduce dimensions to {N_COMPONENTS}...")
print(f"(Using perplexity={PERPLEXITY}, iterations={N_ITER})\n")

# Instantiate t-SNE
# Key parameters:
# - n_components: Target dimensionality (usually 2 or 3)
# - perplexity: Related to the number of nearest neighbors considered for each point.
#                Typical values are between 5 and 50.
# - learning_rate: Controls how much points move in each iteration.
# - n_iter: Number of optimization iterations.
# - init: Method for initialization ('random' or 'pca'). PCA initialization can be faster/more stable.
# - random_state: For reproducibility.

tsne = TSNE(
    n_components=N_COMPONENTS,
    perplexity=PERPLEXITY,
    learning_rate=LEARNING_RATE,
    n_iter=N_ITER,
    init="pca",  # Using PCA initialization can often help
    random_state=RANDOM_STATE,
    n_jobs=-1,  # Use all available CPU cores
)

# Fit t-SNE to the data and transform it
start_time = time.time()
X_tsne = tsne.fit_transform(X)
end_time = time.time()

print(f"Transformed data shape: {X_tsne.shape}")  # (n_samples, n_components)
print(f"t-SNE KL divergence: {tsne.kl_divergence_:.4f}")
print(f"t-SNE computation time: {end_time - start_time:.2f} seconds")

# --- Plot Results ---
print("\nPlotting the t-SNE-transformed data...")

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="viridis", alpha=0.7)
plt.title(f"t-SNE Result ({N_COMPONENTS} Components, Perplexity={PERPLEXITY})")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(
    handles=scatter.legend_elements()[0],
    labels=[f"Cluster {i}" for i in range(CENTERS)],
)
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

print("\nt-SNE Example Complete.")
print("Note: t-SNE is non-linear and excels at revealing local cluster structure.")
print(
    "Distances between clusters and cluster sizes in the plot might not be globally meaningful."
)
