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

### Sprint 10: Pre-trained GPT-2, Text Generation & Basic Demo

- [x] Loaded pre-trained GPT-2 model and tokenizer using `transformers`.
- [x] Implemented greedy, top-k, top-p sampling, and temperature scaling.
- [x] Built a basic interactive demo using `gradio`.
- [x] Implemented streaming output in the demo using `TextIteratorStreamer`.

### Sprint 11: Fine-tuning GPT-2 for Classification

- **Goal:** Adapt the pre-trained GPT-2 model for a sequence classification task (Fake News Detection).
- **Status:** Completed (See `sprints/11_finetune_gpt2_classification/README.md`)
- **Key Outcomes:**
  - [x] Selected and prepared `Pulk17/Fake-News-Detection-dataset`.
  - [x] Adapted `gpt2` tokenizer and created DataLoaders.
  - [x] Loaded `GPT2ForSequenceClassification` model.
  - [x] Implemented fine-tuning loop with periodic validation.
  - [x] Achieved **99.83%** accuracy on the test set after 1 epoch.

### Sprint 12: Fine-tuning GPT-2 for Generative Tasks

- **Goal:** Specialize the generative capabilities of the pre-trained GPT-2 model on custom `book.txt` data.
- **Status:** Completed (See `sprints/12_finetune_gpt2_generative/README.md`)
- **Key Outcomes:**
  - [x] Prepared custom text dataset (`book.txt`) and implemented `TextDataset`.
  - [x] Implemented generative fine-tuning loop with validation (perplexity) and checkpointing.
  - [x] Successfully ran fine-tuning (1 epoch) on the custom dataset.
  - [x] Implemented script to compare generation vs. original model.
  - [x] Observed qualitative differences, though specific style mimicry needs further work/epochs.

## Next Sprint Focus (Sprint 13 & beyond)

### Sprint 13: Parameter-Efficient Fine-Tuning (PEFT - LoRA) (Completed)

- **Goal:** Understand and implement LoRA for parameter-efficient fine-tuning.
- **Key Outcomes:**
  - Learned LoRA theory and setup (Hugging Face `peft`).
  - Ran `finetune_lora.py` achieving best val loss 0.2228 and perplexity 1.2496.
  - Trained only 294,912 parameters (~0.2364% of GPT-2).
  - Saved adapters in `sprints/13_peft_lora/results/checkpoints/lora_finetuned_model/`.

## Upcoming Sprint Focus

### Sprint 14: Model Optimization (Quantization - PoC Level) (Tentative)

- **Goal:** Learn and apply post-training quantization to the GPT-2 model.
- **Tasks:**
  - Understand quantization concepts (INT8, dynamic vs. static).
  - Use PyTorch's quantization tools (`torch.quantization`).
  - Apply quantization to a GPT-2 model (pre-trained or fine-tuned).
  - Evaluate impact on model size, (estimated) inference speed, and performance.

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
