# Layer Normalization (LayerNorm)

## What is it?

Layer Normalization (`torch.nn.LayerNorm`) is a technique used to normalize the activations within a neural network layer.

Unlike Batch Normalization, which normalizes _across the batch dimension_ for each feature, Layer Normalization normalizes _across the feature dimension_ for each individual sample in the batch.

## How it Works

For a single input sample $x$ (which is typically a vector representing features, like an embedding), LayerNorm calculates the mean ($\mu$) and standard deviation ($\sigma$) across all the elements (features) within that sample:

$$ \mu = \frac{1}{H} \sum_{i=1}^{H} x_i $$
$$ \sigma^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2 $$

Where $H$ is the number of features (e.g., the embedding dimension `d_model`).

It then normalizes each feature $x_i$ using this mean and standard deviation:

$$ \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} $$

Where $\epsilon$ is a small value added for numerical stability (to avoid division by zero).

Finally, it applies learnable affine transformation parameters: a scale ($\gamma$) and a shift ($\beta$). These are vectors of the same size as the feature dimension, allowing the network to learn the optimal scale and mean for the normalized activations:

$$ y_i = \gamma_i \hat{x}_i + \beta_i $$

These $\gamma$ and $\beta$ parameters are learned during training along with the other model weights.

## Why Use It in Transformers?

LayerNorm is particularly favored in Transformers and other sequence models for several reasons:

1.  **Batch Size Independence:** Its calculations are independent of the batch size. It works identically whether the batch size is 1 or 1000. Batch Normalization struggles with small batch sizes as the batch statistics become noisy and unreliable.
2.  **Sequence Stability:** In models processing sequences (like RNNs or Transformers), the statistics of activations can vary significantly across different time steps or positions. LayerNorm normalizes each position's features independently, helping to stabilize dynamics and improve gradient flow, leading to smoother training.
3.  **Natural Fit:** It fits well within the standard Transformer block structure, often used in the "Add & Norm" steps following self-attention and feed-forward sub-layers.

## `torch.nn.LayerNorm` Usage

When initializing `nn.LayerNorm`, the crucial argument is `normalized_shape`. This specifies the dimensions over which the mean and standard deviation are calculated.

- For a typical Transformer input tensor of shape `(batch_size, seq_len, d_model)`, you would usually normalize over the last dimension (`d_model`). So, `normalized_shape` would be `d_model` or `(d_model,)`.

```python
import torch
import torch.nn as nn

batch_size = 4
seq_len = 10
d_model = 64 # Feature dimension

# Input tensor
x = torch.randn(batch_size, seq_len, d_model)

# Initialize LayerNorm to normalize over the last dimension (d_model)
layer_norm = nn.LayerNorm(normalized_shape=d_model)

# Apply LayerNorm
output = layer_norm(x)

print("Input shape:", x.shape) # (4, 10, 64)
print("Output shape:", output.shape) # (4, 10, 64)

# Check mean and std dev for one sample across the feature dimension
# Should be close to 0 and 1 respectively
print("Output mean (sample 0, pos 0):", output[0, 0, :].mean()) # ~0
print("Output std dev (sample 0, pos 0):", output[0, 0, :].std()) # ~1
```

This module automatically handles the creation and learning of the $\gamma$ and $\beta$ parameters.
