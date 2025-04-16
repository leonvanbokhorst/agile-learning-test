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

### Sprint 9: Training the GPT-2 Model

- [x] Prepared text data pipeline (TinyShakespeare dataset, `Dataset`, `DataLoader`).
- [x] Integrated GPT-2 tokenizer.
- [x] Implemented training loop for language modeling (`CrossEntropyLoss`, AdamW).
- [x] Added evaluation metrics (perplexity).
- [x] Refined model checkpointing (optimizer state, epoch, etc.).
- [x] Implemented learning rate scheduling (cosine decay with warmup).
- [x] Integrated TensorBoard logging.
- [x] Created main training script with `argparse`.

## Next Sprint Focus (Sprint 10 Tentative)

### Sprint 10: Pre-trained GPT-2, Text Generation & Basic Demo

- **Goal:** Load and interact with a pre-trained GPT-2 model, implement various text generation techniques, and create a simple local demo.
- **Tasks:**
  - Load pre-trained GPT-2 model weights and tokenizer (e.g., using Hugging Face `transformers` library).
  - Understand the configuration of the pre-trained model.
  - Implement text generation / sampling functions:
    - Greedy decoding.
    - Top-k sampling.
    - Nucleus (top-p) sampling.
  - Implement and understand temperature scaling for controlling randomness.
  - Write a script to generate text using the pre-trained model and different sampling strategies.
  - **Add:** Create a _basic local interactive demo_ (e.g., using `Gradio` or `Streamlit`) to input prompts/parameters and display generated text (Proof-of-Concept level).

## Future Sprints

### Sprint 11: Fine-tuning GPT-2 for Classification (Tentative)

- **Goal:** Adapt the pre-trained GPT-2 model for a sequence classification task.
- **Tasks:**
  - Choose a suitable classification dataset.
  - Add a classification head to the pre-trained GPT-2 model architecture.
  - Prepare data loaders for classification training.
  - Implement a fine-tuning loop, potentially freezing most of the base model initially.
  - Evaluate the fine-tuned classification model.

### Sprint 12: Fine-tuning GPT-2 for Generative Tasks (Tentative)

- **Goal:** Specialize the generative capabilities of the pre-trained GPT-2 model.
- **Tasks:**
  - Choose a specific domain or style dataset (e.g., poetry, code snippets, specific author).
  - Prepare data loaders suitable for generative fine-tuning.
  - Implement a fine-tuning loop focused on the language modeling objective.
  - Compare text generated by the fine-tuned model vs. the original pre-trained model.

### Sprint 13: Parameter-Efficient Fine-Tuning (PEFT - LoRA) (Tentative)

- **Goal:** Understand and implement LoRA for parameter-efficient fine-tuning.
- **Tasks:**
  - Learn the theory behind LoRA.
  - Implement/integrate LoRA layers (e.g., using `peft` library or custom layers).
  - Re-run a fine-tuning task (classification or generation) using LoRA.
  - Compare results (performance, parameters) against full fine-tuning.

### Sprint 14: Model Optimization (Quantization - PoC Level) (Tentative)

- **Goal:** Learn and apply post-training quantization to the GPT-2 model.
- **Tasks:**
  - Understand quantization concepts (INT8, dynamic vs. static).
  - Use PyTorch's quantization tools (`torch.quantization`).
  - Apply quantization to a GPT-2 model (pre-trained or fine-tuned).
  - Evaluate impact on model size, (estimated) inference speed, and performance (minimal deployment focus).

### Sprint 15: Exploring Other Architectures (Encoder-Decoder) (Tentative)

- **Goal:** Implement or dissect a basic Encoder-Decoder Transformer architecture.
- **Tasks:**
  - Revisit `EncoderBlock` and `DecoderBlock` from Sprint 7.
  - Assemble them into a full Encoder-Decoder model.
  - Understand the data flow for sequence-to-sequence tasks.
  - (Optional) Implement a simple training loop for a toy seq2seq task.

## Future Considerations

- Advanced Fine-tuning techniques (LoRA, prompt tuning, PEFT library)
- Model optimization and quantization (for the pre-trained/fine-tuned model)
- Deployment strategies
- Model interpretability
- More advanced generation techniques (e.g., beam search)
- Building larger or different Transformer architectures (e.g., Encoder-Decoder)
- Advanced Evaluation Metrics (BLEU, ROUGE, etc.)
- Knowledge Distillation
- Self-Supervised Learning Concepts (Masked LM, etc.)
- Scaling Training (DistributedDataParallel basics)
