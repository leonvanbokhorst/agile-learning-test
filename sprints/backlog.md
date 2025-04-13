# Sprint Backlog

Ideas for future sprints after completing Sprint 1 (Setup & Basics).

## Potential Next Sprints

- **Sprint: Datasets & DataLoaders:**

  - Goal: Learn how to load and preprocess data efficiently using PyTorch's `Dataset` and `DataLoader` classes.
  - Tasks: Custom `Dataset` implementation, using built-in datasets, data augmentation/transformation pipelines.

- **Sprint: Building a Basic Neural Network:**

  - Goal: Combine `nn.Module`, layers (`Linear`, activation functions like `ReLU`), and loss functions (`MSELoss`, `CrossEntropyLoss`) to build and train a simple feed-forward network on a toy dataset (e.g., MNIST, simple regression).
  - Tasks: Define network architecture, implement training loop (forward pass, loss calc, backward pass, optimizer step), basic evaluation.

- **Sprint: Understanding Embeddings & Positional Encoding:**

  - Goal: Dive into the first key components of sequence models like Transformers.
  - Tasks: Implement `nn.Embedding`, understand its purpose, implement sinusoidal or learned positional encoding.

- **Sprint: Implementing Multi-Head Self-Attention:**

  - Goal: Tackle the core mechanism of the Transformer architecture.
  - Tasks: Implement scaled dot-product attention, implement multi-head attention wrapper, understand masking.

- **Sprint: Building the Transformer Block:**

  - Goal: Assemble the components (Multi-Head Attention, Add & Norm, Feed-Forward) into a full Transformer block.
  - Tasks: Combine sub-layers, ensure correct tensor shapes flow through.

- **Sprint: Assembling the GPT-2 Model:**

  - Goal: Stack Transformer blocks to create the full GPT-2 like architecture.
  - Tasks: Define the overall model structure, handle input/output processing.

- **Sprint: Training the GPT-2 Model (Small Scale):**

  - Goal: Implement the training loop specifically for the language model, train on a small text dataset.
  - Tasks: Prepare text data, implement causal masking, train and observe basic text generation.

- **Sprint: Evaluation & Generation:**
  - Goal: Learn how to evaluate the trained language model and generate text.
  - Tasks: Implement perplexity calculation, implement different text generation strategies (greedy, sampling).
