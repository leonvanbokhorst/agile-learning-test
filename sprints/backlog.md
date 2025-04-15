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

### Sprint 6: Multi-Head Attention

- [x] Implemented scaled dot-product attention function.
- [x] Understood and implemented attention masking (padding and look-ahead).
- [x] Implemented `MultiHeadAttention` module.
- [x] Tested module with masks.

### Sprint 7: Building the Transformer Block

- [x] Implemented Layer Normalization (`nn.LayerNorm`).
- [x] Implemented Residual Connections and the Add & Norm pattern.
- [x] Implemented the Position-wise Feed-Forward Network (`PositionWiseFeedForward`).
- [x] Implemented the `EncoderBlock` module.
- [x] Implemented the `DecoderBlock` module.
- [x] Tested blocks individually and stacked.

### Sprint 8: Assembling the GPT-2 Model

- [x] Defined overall `GPT` model architecture (`model.py`).
- [x] Implemented token embeddings and sinusoidal positional encoding (`positional_encoding.py`).
- [x] Created GPT-specific decoder block (`gpt_decoder_block.py`).
- [x] Stacked decoder blocks and added final output layer.
- [x] Enabled weight tying and added weight initialization.
- [x] Handled configuration using a `GPTConfig` dataclass (`config.py`).
- [x] Implemented checkpoint saving/loading (`utils.py`).
- [x] (Stretch) Integrated GPT-2 tokenizer using `tokenizers` library (`tokenizer.py`).

## Next Sprint Focus (Sprint 9 Tentative)

### Sprint: Training the GPT-2 Model

- Goal: Implement training pipeline for the language model
- Tasks:
  - Prepare text data pipeline (e.g., TinyShakespeare dataset)
    - Integrate `tokenizer.py` for encoding.
    - Create custom `Dataset` for sequential data.
    - Set up `DataLoader`.
  - Implement training loop for language modeling (predict next token).
    - Define appropriate loss function (e.g., `nn.CrossEntropyLoss`).
    - Select optimizer (e.g., AdamW).
  - Add evaluation metrics (e.g., perplexity calculation).
  - Refine model checkpointing (include optimizer state, epoch, etc.).
  - Implement learning rate scheduling (e.g., cosine decay with warmup).

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
