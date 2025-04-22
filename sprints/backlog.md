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
  - [x] Implemented generative fine-tuning loop with validation (perplexity=1.1211) and checkpointing.
  - [x] Successfully ran full fine-tuning (1 epoch) on the custom dataset.
  - [x] Implemented script to compare generation vs. original model.

### Sprint 13: Parameter-Efficient Fine-Tuning (PEFT - LoRA)

- **Goal:** Understand and implement LoRA for parameter-efficient generative fine-tuning.
- **Status:** Completed (See `sprints/13_peft_lora/README.md`)
- **Key Outcomes:**
  - [x] Learned LoRA theory and setup using Hugging Face `peft`.
  - [x] Implemented and ran `finetune_lora.py` for generative task.
  - [x] Achieved best validation perplexity of **1.2537**.
  - [x] Trained only **294,912** parameters (~0.24% of GPT-2).
  - [x] Saved LoRA adapter checkpoint.
  - [x] Compared LoRA results (perplexity, param count) to Sprint 12 baseline.

### Sprint 14: Model Optimization (Quantization - PoC Level)

- **Goal:** Learn and apply post-training quantization (Dynamic, Static) to a GPT-2 model using PyTorch Eager Mode.
- **Status:** Completed (See `sprints/14_quantization/README.md`)
- **Key Outcomes:**
  - [x] Understood core quantization concepts (PTQ Dynamic/Static, INT8/FP16/BF16).
  - [x] Implemented PTQ Dynamic using `quantize_dynamic`.
  - [x] Implemented PTQ Static using wrapper, calibration, `prepare`/`convert`.
  - [x] Debugged backend/QConfig/layer targeting issues in Eager Mode.
  - [x] Achieved ~1.2x inference speedup on CPU vs FP32 for both methods.
  - [x] Documented process and learnings in notes and README.

### Sprint 15: Encoder-Decoder Architecture & Exposure Bias

- **Goal:** Assemble an Encoder-Decoder model and investigate training dynamics for seq2seq tasks.
- **Status:** Completed (See `sprints/15_encoder_decoder/README.md`)
- **Key Outcomes:**
  - [x] Built a functional Encoder-Decoder model from components.
  - [x] Identified and deeply analyzed exposure bias in a toy sequence reversal task.
  - [x] Experimented with multiple advanced training techniques (Scheduled Sampling, Professor Forcing, REINFORCE).
  - [x] Concluded standard MLE training is ill-suited for deterministic generation tasks.
  - [x] Implemented a direct algorithmic alternative (`torch.flip`).

### Sprint 16: Flirty Llama3

- **Goal:** Explores GRPO CoT Reasoning RL for cooking recipes.
- **Status:** In Progress (GRPO training with 1B model running; monitoring initial metrics)

### Sprint 17: Graph Neural Networks

- **Goal:** Implement a Graph Neural Network (GCN) in PyTorch.
- **Status:** In Progress (See `sprints/17_gnn_/README.md`)
- **Key Outcomes:**
  - [x] Implemented GCN in PyTorch.
  - [x] Trained GCN on Zachary's Karate Club social graph.
  - [ ] Extract node embeddings and cluster them using UMAP.
  - [ ] Feed embeddings to GPT-2 for generation.

- **Possible Extras:**
  - [ ] Implement GNN for arbitrary graph tasks.
  - [ ] Apply GNN to a more complex dataset (e.g. ZINC molecular dataset).
  - [ ] Apply GNN to a more complex task (e.g. node classification, graph classification).

## Backlog / Future Sprint Ideas

*(Items from previous sprints or new ideas)*

- Implement advanced memory management for DataLoaders.
- Explore alternative sequence modeling architectures (RNNs, LSTMs).
- Dive deeper into specific Transformer variants (e.g., BERT, RoBERTa).
- Implement Quantization Aware Training (QAT).
- Implement advanced PEFT methods (Adapters, Prompt Tuning, etc.).
- Build a more sophisticated RAG pipeline.
- Explore multi-modal models.

*(Ideas added from Sprint 16)*
- Experiment with GRPO hyperparameters (e.g., `beta`, learning rate, `k`) if reward progress stalls.

*(Ideas added from Sprint 17)*
- Explore more complex GNN architectures (e.g., Graph Attention Networks - GAT).
- Implement full RAG or graph querying capabilities using the dynamic KG embeddings stored in Faiss.
- Investigate and implement more robust entity linking/canonicalization techniques for the dynamic KG.
- Apply the dynamic KG pipeline to a larger, streaming text dataset.
- Use GNN embeddings for downstream tasks beyond visualization/indexing (e.g., link prediction, community detection).
- Integrate GNNs with other model types (e.g., GNN + Transformer for graph-based NLP).
