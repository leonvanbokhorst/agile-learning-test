# Sprint Backlog

## Completed Sprints

### Sprint 1: Setup & Basics

- [x] Environment setup with `pyproject.toml` and `uv`
- [x] Basic tensor operations and manipulations
- [x] Understanding of autograd and gradient computation
- [x] Implementation of a simple neural network with one hidden layer
- [x] Documentation of neural network concepts and architecture

### Sprint 2: Datasets & DataLoaders

- [x] Understanding PyTorch `Dataset` interface (`__len__`, `__getitem__`)
- [x] Implementing custom datasets
- [x] Working with `DataLoaders` (batching, shuffling, `num_workers`)
- [x] Applying `torchvision.transforms` (Compose, ToTensor, Normalize, basic augmentation)
- [x] Loading built-in datasets (`torchvision.datasets.MNIST`)
- [x] Handled `DataLoader` multiprocessing caveats (`if __name__ == '__main__':`)

### Sprint 3: Models & Training Loops

- [x] Defined basic `nn.Module` (Linear Regression example)
- [x] Understood `__init__` and `forward` methods.
- [x] Implemented common loss functions (`nn.MSELoss`, `nn.CrossEntropyLoss`).
- [x] Implemented optimizers (`torch.optim.Adam`).
- [x] Built a complete training loop (forward, loss, backward, step, zero_grad).
- [x] Integrated `tqdm` for progress visualization.
- [x] Implemented basic evaluation loop (`model.eval()`, `torch.no_grad()`).

### Sprint 4: Advanced Training Techniques & CNN Basics

- [x] Defined basic CNN architecture (`nn.Conv2d`, `nn.MaxPool2d`, etc.)
- [x] Implemented TensorBoard logging (`SummaryWriter`).
- [x] Implemented Learning Rate Scheduling (`CosineAnnealingLR`).
- [x] Implemented Early Stopping logic.
- [x] Combined techniques in a full MNIST training loop.

### Sprint 5: Embeddings & Positional Encoding

- [x] Implemented `nn.Embedding` and custom embeddings.
- [x] Implemented Sinusoidal and Learned Positional Encoding.
- [x] Understood and implemented embedding visualization concepts (PCA, t-SNE, UMAP).
- [x] Managed complex library dependencies for visualization tools.

## Next Sprint Focus (Sprint 6 Tentative)

### Sprint: Implementing Multi-Head Self-Attention

- Goal: Tackle the core mechanism of the Transformer architecture
- Tasks:
  - Implement scaled dot-product attention
  - Create multi-head attention wrapper
  - Understand and implement masking
  - Handle different attention patterns
  - Optimize attention computation

### Sprint: Building the Transformer Block

- Goal: Assemble components into a full Transformer block
- Tasks:
  - Combine attention with feed-forward networks
  - Implement layer normalization
  - Add residual connections
  - Create encoder and decoder blocks
  - Handle different block configurations

### Sprint: Assembling the GPT-2 Model

- Goal: Stack Transformer blocks to create the full architecture
- Tasks:
  - Define model structure
  - Implement input/output processing
  - Add tokenization
  - Handle model configuration
  - Implement model saving/loading

### Sprint: Training the GPT-2 Model

- Goal: Implement training pipeline for the language model
- Tasks:
  - Prepare text data pipeline
  - Implement training loop
  - Add evaluation metrics
  - Handle model checkpointing
  - Implement learning rate scheduling

### Sprint: Evaluation & Generation

- Goal: Learn model evaluation and text generation
- Tasks:
  - Implement perplexity calculation
  - Add different generation strategies
  - Create evaluation pipeline
  - Handle different decoding methods
  - Implement beam search

## Future Considerations

- Model optimization and quantization
- Deployment strategies
- Fine-tuning techniques
- Advanced training methods
- Model interpretability
