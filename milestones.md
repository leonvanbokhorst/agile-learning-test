# Project Milestones

## Sprint 1: Environment Setup & PyTorch Fundamentals

### Completed

- [x] Environment setup with `pyproject.toml` and `uv`
- [x] Basic tensor operations and manipulations
- [x] Understanding of autograd and gradient computation
- [x] Implementation of a simple neural network with one hidden layer
- [x] Documentation of neural network concepts and architecture
- [x] Comprehensive progress tracking and documentation

### Current Focus

- Deepening understanding of neural network components
- Exploring more complex network architectures
- Preparing for transformer architecture implementation

### Next Steps

- Moving towards transformer architecture components
  - Implementing attention mechanisms
  - Understanding positional encoding
  - Building multi-head attention
- Building towards the final LLM implementation
  - Implementing transformer blocks
  - Creating the full model architecture
  - Setting up training pipeline

### Documentation

- Created detailed notes in `sprints/01_setup_and_basics/notes/`
- Updated progress in README files
- Maintained skills and competencies log
- Documented neural network concepts in `02d_hidden_layer_network.md`

## Sprint 2: Datasets & DataLoaders

### Completed

- [x] Implementation of custom `Dataset` (`__len__`, `__getitem__`)
- [x] Understanding and configuration of `DataLoader` (batching, shuffling)
- [x] Understanding parallel loading (`num_workers`) and associated caveats (`if __name__ == '__main__':`)
- [x] Loading and using built-in datasets (`torchvision.datasets.MNIST`)
- [x] Applying basic data transformations (`Compose`, `ToTensor`, `Normalize`)
- [x] Documentation of Dataset and DataLoader concepts and examples
- [x] Implemented custom transform class ([results/04_custom_transform.py](results/04_custom_transform.py))
- [x] Implemented basic data augmentation (`torchvision.transforms` like `RandomRotation`)

### Key Insights

- Mastered the `Dataset` and `DataLoader` workflow, the core PyTorch mechanism for feeding data to models efficiently.
- Gained practical experience with essential `DataLoader` features like batching, shuffling, and parallel loading (`num_workers`), including platform-specific considerations (`if __name__ == '__main__':`).
- Successfully used `torchvision` to load standard datasets (MNIST) and apply crucial transformations (`ToTensor`, `Normalize`), including basic data augmentation (`RandomRotation`).
- Learned to define custom datasets and transforms, providing flexibility for non-standard data.
- Recognized the importance of applying appropriate transformations (especially normalization) and the distinction between training-time augmentation and validation/test-time preprocessing.

### Skipped/Deferred

- Deeper dive into DataLoader memory management.
- `torchtext` dataset handling (due to library deprecation/compatibility).

### Documentation

- Created notes in [sprints/02_datasets_and_dataloaders/notes/](sprints/02_datasets_and_dataloaders/notes/)
- Updated Sprint 2 `README.md`

## Sprint 3: Models & Training Loops

### Completed

- [x] Defined a basic `nn.Module` (Linear Regression example).
- [x] Understood `__init__` and `forward` methods.
- [x] Implemented common loss functions (`nn.MSELoss`).
- [x] Implemented optimizers (`torch.optim.Adam`).
- [x] Built a complete training loop (forward, loss, backward, step, zero_grad).
- [x] Integrated `tqdm` for progress visualization.
- [x] Implemented basic evaluation loop (`model.eval()`, `torch.no_grad()`).
- [x] Calculated validation loss to monitor generalization.

### Key Insights

- Mastered the fundamental PyTorch training pipeline.
- Understood the roles and interactions of models, loss functions, and optimizers.
- Recognized the importance of `model.train()` vs `model.eval()` modes.
- Appreciated the efficiency gains from `torch.no_grad()` during evaluation.
- Gained practical experience implementing and monitoring a training/validation cycle.

### Next Steps

- **Sprint 4: Advanced Training Techniques & MNIST Classification** (Tentative)
  - Building a more complex CNN for image classification.
  - Implementing techniques like learning rate scheduling, early stopping.
  - Using TensorBoard for visualization.
  - Training a model on the MNIST dataset.

### Documentation

- Created notes and results in [sprints/03_models_and_training_loops/](sprints/03_models_and_training_loops/)
- Filled out Sprint 3 `README.md` retrospective.

## Sprint 4: Advanced Training Techniques & MNIST Classification

### Completed

- [x] Defined a basic CNN architecture (`nn.Conv2d`, `nn.MaxPool2d`, etc.).
- [x] Implemented TensorBoard integration for logging metrics (`SummaryWriter`, `add_scalar`).
- [x] Implemented Learning Rate Scheduling (`CosineAnnealingLR`, `scheduler.step()`).
- [x] Implemented Early Stopping logic (monitoring validation loss, patience, saving best model).
- [x] Combined all components into a full training/validation loop for MNIST.
- [x] Successfully trained the CNN on MNIST, observing the effects of LR scheduling and early stopping.
- [x] Correctly handled multiprocessing issues with `DataLoader` (`if __name__ == '__main__':`).
- [x] Practiced running scripts as modules (`python -m ...`) for relative imports.

### Key Insights

- Understood the practical application and benefits of TensorBoard, LR scheduling, and early stopping for managing the training process.
- Gained experience debugging common PyTorch issues like `DataLoader` multiprocessing errors and import path problems.
- Reinforced understanding of the complete PyTorch workflow from data loading to model definition, training, and basic evaluation.
- Recognized that MNIST served as a practical exercise for learning these techniques, which are transferable to more complex tasks like sequence modeling.

### Next Steps

- **Sprint 5: Embeddings & Positional Encoding** (Tentative)
  - Understanding `nn.Embedding` for representing discrete tokens (like words).
  - Implementing different types of positional encoding to inject sequence order information.
  - Moving towards building the foundational components of sequence-to-sequence models and Transformers.

### Documentation

- Created notes and results in [sprints/04_advanced_training_mnist/](./sprints/04_advanced_training_mnist/)
- Filled out Sprint 4 `README.md`.

## Sprint 5: Embeddings & Positional Encoding

### Completed

- [x] Understood and implemented `nn.Embedding`.
- [x] Implemented a custom embedding layer for understanding.
- [x] Understood the rationale for positional encoding.
- [x] Implemented Sinusoidal Positional Encoding (`PositionalEncoding` module).
- [x] Added positional encoding to token embeddings.
- [x] Understood and implemented Learned Positional Embeddings.
- [x] Understood embedding visualization concepts (PCA, t-SNE, UMAP).
- [x] Implemented visualization examples.

### Key Insights

- Learned how discrete tokens are represented as trainable vectors using `nn.Embedding`.
- Understood the limitation of sequence models that don't inherently process order (like basic attention).
- Grasped the concept and implementation of adding fixed sinusoidal signals to embeddings to provide positional context.
- Understood the alternative approach of using a _learned_ `nn.Embedding` layer to represent positions, offering flexibility at the cost of fixed sequence length and more parameters.
- Learned how dimensionality reduction techniques (PCA, t-SNE, UMAP) can be used to visualize high-dimensional embeddings, each offering different trade-offs between preserving global structure and local clusters.
- Gained experience debugging Python environment dependency issues, particularly with complex libraries like Matplotlib.

### Next Steps

- [x] Conceptualizing Embedding Visualization.
- **Sprint 6: Multi-Head Self-Attention** (Completed)
  - Implementing scaled dot-product attention.
  - Building the multi-head attention mechanism.
  - Understanding masking.

### Documentation

- Created notes and results in [sprints/05_embeddings_and_positional_encoding/](./sprints/05_embeddings_and_positional_encoding/)
- Updated Sprint 5 `README.md`.

### Notes:

- [Embedding Basics](notes/01_nn_embedding_notes.md)
- [Custom Embeddings](notes/02_custom_embedding_notes.md)
- [Positional Encoding](notes/03_positional_encoding_notes.md)
- [Learned Positional Embeddings](notes/04_learned_positional_embeddings.md)
- [Embedding Visualization](notes/05_embedding_visualization_notes.md)
- [PCA Explained](notes/05a_pca_explained_novice.md)
- [t-SNE Explained](notes/05b_tsne_explained.md)
- [UMAP Explained](notes/05c_umap_explained_novice.md)

### Results:

- [nn.Embedding Basics](results/01_nn_embedding_basics.py)
- [Custom Embedding (Optional)](results/02_custom_embedding.py)
- [Sinusoidal Positional Encoding](results/03_positional_encoding.py)
- [Learned Positional Embedding Example](results/learned_pe_example.py)
- [PCA Visualization Example](results/pca_example.py)
- [t-SNE Visualization Example](results/tsne_example.py)
- [UMAP Visualization Example](results/umap_example.py)

## Sprint 6: Multi-Head Attention

### Completed

- [x] Implemented scaled dot-product attention function.
- [x] Understood and implemented attention masking (padding and look-ahead).
- [x] Implemented `MultiHeadAttention` module as `nn.Module`.
- [x] Used helper methods (`split_heads`, `combine_heads`) for clarity.
- [x] Tested module with and without masks.
- [x] Documented concepts in notes.

### Key Insights

- Mastered the core mechanism of Transformer attention.
- Understood how multiple heads allow attending to different representation subspaces.
- Gained practical experience with tensor manipulation for reshaping and transposing dimensions for parallel computation.
- Learned the importance and implementation of padding and causal masks.
- Successfully built a complex, reusable PyTorch module.

### Next Steps

- **Sprint 7: Building the Transformer Block** (Tentative)
  - Combining Multi-Head Attention with Feed-Forward Networks.
  - Implementing Layer Normalization.
  - Adding Residual Connections (Add & Norm).
  - Creating a complete Encoder or Decoder block.

### Documentation

- Created notes and results in [sprints/06_multi_head_attention/](./sprints/06_multi_head_attention/)
- Updated Sprint 6 `README.md`.

### Notes:

- [Scaled Dot-Product Attention](./sprints/06_multi_head_attention/notes/01_scaled_dot_product_attention.md)
- [Multi-Head Attention](./sprints/06_multi_head_attention/notes/02_multi_head_attention.md)
- [Attention Masking](./sprints/06_multi_head_attention/notes/03_attention_masking.md)

### Results:

- [Scaled Dot-Product Attention Implementation](./sprints/06_multi_head_attention/results/scaled_dot_product_attention.py)
- [Multi-Head Attention Implementation](./sprints/06_multi_head_attention/results/02_multi_head_attention.py)

## Sprint 7: Building the Transformer Block

### Completed

- [x] Understood and implemented Layer Normalization (`nn.LayerNorm`).
- [x] Understood and implemented Residual Connections (Add & Norm pattern).
- [x] Implemented the Position-wise Feed-Forward Network (FFN).
- [x] Assembled the complete Transformer Encoder Block module.
- [x] Assembled the complete Transformer Decoder Block module (including Masked Self-Attention and Cross-Attention).
- [x] Tested block functionality with dummy data, masks, and stacking.
- [x] Documented all components and block structures.

### Key Insights

- Gained hands-on experience combining multiple `nn.Module` components into larger, functional blocks.
- Mastered the standard Add & Norm pattern ubiquitous in Transformers.
- Deepened understanding of the data flow and specific roles of sub-layers within both Encoder and Decoder blocks.
- Successfully debugged integration issues between modules developed in different sprints.
- Recognized the importance of careful input/output management and mask handling, especially in the Decoder.

### Next Steps

- **Sprint 8: Assembling the GPT-2 Model** (Tentative)
  - Stacking Encoder/Decoder blocks to create the full architecture.
  - Implementing input embedding layers (token + positional).
  - Adding the final output layer (Linear + Softmax).
  - Handling model configuration (hyperparameters).

### Documentation

- Created notes and results in [sprints/07_transformer_block/](./sprints/07_transformer_block/)
- Updated Sprint 7 `README.md`.

### Notes:

- [Layer Normalization](./sprints/07_transformer_block/notes/01_layer_norm.md)
- [Layer Norm Output Explanation](./sprints/07_transformer_block/notes/01a_layer_norm_output_explanation.md)
- [Residual Connections](./sprints/07_transformer_block/notes/02_residual_connections.md)
- [Feed-Forward Network](./sprints/07_transformer_block/notes/03_feed_forward_network.md)
- [Encoder Block](./sprints/07_transformer_block/notes/04_encoder_block.md)
- [Decoder Block](./sprints/07_transformer_block/notes/05_decoder_block.md)

### Results:

- [Layer Norm Example](./sprints/07_transformer_block/results/01_layer_norm_example.py)
- [Add & Norm Example](./sprints/07_transformer_block/results/02_residual_connection_example.py)
- [Position-wise FFN](./sprints/07_transformer_block/results/positionwise_feed_forward.py)
- [Encoder Block Implementation](./sprints/07_transformer_block/results/encoder_block.py)
- [Decoder Block Implementation](./sprints/07_transformer_block/results/decoder_block.py)

## Sprint 8: Assembling the GPT-2 Model

### Completed

- [x] Defined overall `GPT` model architecture using stacked `GPTDecoderBlock`s.
- [x] Implemented token and positional embeddings (using `PositionalEncodingBatchFirst`).
- [x] Implemented final LayerNorm and output projection layer.
- [x] Enabled weight tying between embeddings and output layer.
- [x] Implemented weight initialization strategy.
- [x] Refactored model to use a `GPTConfig` dataclass for hyperparameter management.
- [x] Implemented checkpoint saving (`save_checkpoint`) and loading (`load_checkpoint`) utilities.
- [x] (Stretch) Added `tokenizers` dependency and implemented loading of standard GPT-2 tokenizer.

### Key Insights

- Successfully integrated components from previous sprints into a complete, cohesive model (`GPT`).
- Learned the structure of a decoder-only Transformer model (GPT-style).
- Gained experience with practical considerations like configuration management (dataclasses), weight initialization, weight tying, and checkpointing.
- Understood how to leverage external libraries (`tokenizers`) for standard components like tokenization.
- Practiced organizing code into logical modules (`model.py`, `config.py`, `utils.py`, etc.).

### Next Steps

- **Sprint 9: Training the GPT-2 Model** (Tentative)
  - Preparing a text dataset (e.g., TinyShakespeare).
  - Implementing a data pipeline using the GPT-2 tokenizer.
  - Building the training loop specifically for language modeling (predicting the next token).
  - Implementing evaluation metrics relevant to language models (e.g., perplexity).
  - Refining checkpointing to include optimizer state and training progress.

### Documentation

- Created notes and results in [sprints/08_gpt2_assembly/](./sprints/08_gpt2_assembly/)
- Updated Sprint 8 `README.md`.

### Notes:

- [Model Structure](./sprints/08_gpt2_assembly/notes/sprint_08/model_structure.md) (Note: You might need to recreate this file if deleted)
- (Create other notes as needed for config, utils, tokenizer)

### Results:

- [GPT Model](./sprints/08_gpt2_assembly/results/model.py)
- [GPT Decoder Block](./sprints/08_gpt2_assembly/results/gpt_decoder_block.py)
- [Positional Encoding](./sprints/08_gpt2_assembly/results/positional_encoding.py)
- [Config](./sprints/08_gpt2_assembly/results/config.py)
- [Utils (Checkpointing)](./sprints/08_gpt2_assembly/results/utils.py)
- [Tokenizer](./sprints/08_gpt2_assembly/results/tokenizer.py)

## Sprint 9: Training the GPT-2 Model

### Completed

- [x] Implemented a complete text data pipeline (download, split, tokenize, `Dataset`, `DataLoader`).
- [x] Built a full training loop for language modeling (forward, loss, backward, optim, gradient clipping).
- [x] Integrated evaluation (perplexity) within the training process.
- [x] Enhanced checkpointing to save/load full training state (model, optimizer, scheduler, steps).
- [x] Implemented learning rate scheduling (warmup + cosine decay).
- [x] Added TensorBoard logging for key metrics.
- [x] Created a configurable main training script (`train_gpt2.py`) using `argparse`.
- [x] Successfully ran training end-to-end, demonstrating the pipeline's functionality.
- [x] Gained practical insight into the computational cost of training transformers from scratch.

### Key Insights

- Successfully synthesized components from all previous sprints into a working end-to-end training system for a non-trivial model.
- Mastered the practical details of language model training, including data preparation, loss calculation, and evaluation metrics like perplexity.
- Learned the importance and implementation of advanced checkpointing and learning rate scheduling techniques.
- Developed a strong appreciation for the computational resources required for training even moderately sized transformer models from scratch, motivating the use of pre-trained models in future sprints.

### Next Steps

- **Sprint 10: Using Pre-trained Models & Text Generation** (Tentative)
  - Loading pre-trained GPT-2 weights (e.g., from Hugging Face).
  - Implementing text generation/sampling algorithms (greedy, top-k, nucleus).
  - Potentially exploring basic fine-tuning techniques on the pre-trained model.

### Documentation

- Created detailed notes and results in [sprints/09_train_gpt2/](./sprints/09_train_gpt2/)
- Updated Sprint 9 `README.md` including the retrospective.
