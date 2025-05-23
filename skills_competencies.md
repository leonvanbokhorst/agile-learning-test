# Skills & Competencies Log

## Baseline Assessment (Start of Project)

- **General Programming (Python):** Proficient (Assumed based on user input)
  - _Note: Demonstrated understanding of modern Python packaging (`pyproject.toml`, `src`-layout)._
- **Deep Learning Concepts:** Familiar with Practical Applications (User reports experience with training, fine-tuning (LoRA), and quantization (gguf) of DL models, especially LLMs)
  - _Goal: Deepen foundational understanding required for building models from scratch._
- **PyTorch Framework:** Familiar (User reports experience with training/fine-tuning existing models)
  - _Goal: Gain proficiency in *implementing* core components (Tensors, Autograd, nn.Module internals, DataLoader customization) from scratch, not just using high-level APIs._
- **Transformer Architecture:** Novice (Conceptual understanding likely present, but practical implementation from scratch is the goal)
  - _Goal: Understand and implement key components (Attention, Positional Encoding, etc.) in PyTorch._
- **LLM (GPT-like) Implementation:** Novice (in terms of building from scratch)
  - _User has conceptual knowledge and practical experience fine-tuning, quantizing, and deploying LLMs._
  - _Goal: Build a functional model from foundational PyTorch components._
- **Hugging Face Ecosystem:** Familiar / Proficient (User reports using it for datasets and models)
- **Virtual Environments (venv/conda):** Proficient (User reports daily usage)
  - _Note: Successfully used `uv` for environment creation and package installation._
- **Git / Version Control (Git/GitHub):** Proficient (User reports essential daily usage)

## Sprint 1 Progress (Setup & Basics)

- **Environment Setup:** Completed.

  - Configured `pyproject.toml` for project dependencies and metadata.
  - Utilized `uv` to install dependencies (including `torch` and dev tools) into the virtual environment.
  - Implemented standard `src`-layout to resolve packaging ambiguity.

- **Tensor Basics:** Completed exercises in `results/01_tensor_basics.py` covering:

  - Creating tensors (various methods)
  - Basic tensor operations (+, -, \*, /)
  - Indexing and slicing
  - Reshaping (`view`, `reshape`)
  - Permuting dimensions (`permute`)
  - Moving tensors between CPU/GPU (`.to()`, `.cuda()`, `.cpu()`)

- **Autograd & Gradients:** Completed exercises in `results/02b_autograd_scalar_example.py` and `results/02c_neural_network_gradients.py`

  - Understanding computation graphs and gradient tracking
  - Using `requires_grad=True` for automatic differentiation
  - Computing gradients with `.backward()`
  - Managing gradient accumulation with `.zero_grad()`
  - Understanding gradient flow in neural networks
  - Implementing gradient descent with learning rates

- **Neural Network Basics:** Completed exercises in `results/02d_hidden_layer_network.py`

  - Implementing a simple network with one hidden layer
  - Understanding parameter count and connections
  - Working with activation functions (ReLU)
  - Understanding signal flow through layers
  - Documenting network architecture and concepts
  - Creating comprehensive documentation in `notes/02d_hidden_layer_network.md`

- **Documentation & Progress Tracking:**
  - Maintaining detailed notes in the `notes/` directory
  - Updating sprint progress in README files
  - Tracking milestones and competencies
  - Creating clear, educational examples with comments

## Sprint 2 Progress (Datasets & DataLoaders)

- **Custom Datasets:** Completed exercises in [results/01_simple_dataset.py](results/01_simple_dataset.py) covering:

  - Implementing `torch.utils.data.Dataset` interface (`__len__`, `__getitem__`).
  - Handling data generation within the dataset.
  - Basic type hinting for datasets (`Dataset[tuple[torch.Tensor, torch.Tensor]]`).
  - Applying simple transforms within `__init__` (though `transform` argument is preferred).

- **DataLoaders:** Completed exercises in [results/01_simple_dataset.py](results/01_simple_dataset.py) and [results/02_dataloader_features.py](results/02_dataloader_features.py) covering:

  - Wrapping a `Dataset` with `torch.utils.data.DataLoader`.
  - Configuring `batch_size`.
  - Understanding and implementing `shuffle=True` for training data randomization.
  - Understanding `num_workers` for parallel data loading.
    - Recognizing performance implications (overhead vs. `__getitem__` complexity).
    - Implementing the `if __name__ == '__main__':` guard for multiprocessing compatibility (Windows/macOS).

- **Built-in Datasets:** Completed exercises in [results/03_builtin_datasets.py](results/03_builtin_datasets.py) using `torchvision`:

  - Loading standard datasets (e.g., `torchvision.datasets.MNIST`).
  - Understanding `root`, `train`, `download` parameters.
  - Automatic dataset downloading and caching.
  - Understanding standard dataset splits (train/test).

- **Data Transformations:** Used basic transforms in [results/01_simple_dataset.py](results/01_simple_dataset.py) and [results/03_builtin_datasets.py](results/03_builtin_datasets.py):

  - Using `torchvision.transforms.Compose` to chain transforms.
  - Using `torchvision.transforms.ToTensor()` to convert image data (PIL/NumPy) to tensors and scale.
  - Using `torchvision.transforms.Normalize()` for data normalization (with dataset-specific means/stds).
  - Applying transforms via the `transform` argument in `Dataset` constructors.
  - Implementing custom transform classes (`__call__` method) ([results/04_custom_transform.py](results/04_custom_transform.py)).
  - Implementing basic data augmentation using `torchvision.transforms` (e.g., `RandomRotation`) ([results/03_load_mnist_with_augmentation.py](results/03_load_mnist_with_augmentation.py), [notes/03_data_augmentation_guide.md](notes/03_data_augmentation_guide.md)).

- **Documentation:**

  - Created notes on Dataset basics ([notes/01_dataset_basics.md](notes/01_dataset_basics.md)).
  - Created notes on DataLoader features and built-in datasets ([notes/02_dataloader_and_builtin.md](notes/02_dataloader_and_builtin.md)).
  - Created notes on Data Augmentation ([notes/03_data_augmentation_guide.md](notes/03_data_augmentation_guide.md)).
  - Updated sprint `README.md` checklists.

- **Key Insights:**

  - Mastered the `Dataset` and `DataLoader` workflow, the core PyTorch mechanism for feeding data to models efficiently.
  - Gained practical experience with essential `DataLoader` features like batching, shuffling, and parallel loading (`num_workers`), including platform-specific considerations (`if __name__ == '__main__':`).
  - Successfully used `torchvision` to load standard datasets (MNIST) and apply crucial transformations (`ToTensor`, `Normalize`), including basic data augmentation (`RandomRotation`).
  - Learned to define custom datasets and transforms, providing flexibility for non-standard data.
  - Recognized the importance of applying appropriate transformations (especially normalization) and the distinction between training-time augmentation and validation/test-time preprocessing.

- **Skipped/Deferred Topics:**
  - Advanced memory management techniques for DataLoaders.
  - `torchtext` usage (due to deprecation and compatibility issues with current PyTorch version).

## Sprint 3 Progress (Models & Training Loops)

- **`nn.Module` Basics:** Completed work in [`sprints/03_models_and_training_loops/results/01_define_nn_module.py`](../sprints/03_models_and_training_loops/results/01_define_nn_module.py) and notes in [`sprints/03_models_and_training_loops/notes/01_define_nn_module_notes.md`](../sprints/03_models_and_training_loops/notes/01_define_nn_module_notes.md) covering:
  - Defining custom models by inheriting from `nn.Module`.
  - Understanding the role of `__init__` (layer definition, calling `super().__init__()`).
  - Understanding the role of `forward` (defining data flow).
  - Instantiating and using standard layers (`nn.Linear`, `nn.Flatten`, `nn.ReLU`).
  - Testing model structure and forward pass with dummy data.
- **Activation Functions & Non-Linearity:** Documented concepts in [`sprints/03_models_and_training_loops/notes/02_activation_functions_notes.md`](../sprints/03_models_and_training_loops/notes/02_activation_functions_notes.md) covering:

  - The necessity of non-linearity to learn complex patterns.
  - The limitation of stacking only linear layers.
  - The role of activation functions (ReLU, Sigmoid, Tanh) in introducing non-linearity.
  - The placement of activation functions (typically after linear layers).

- **Loss Functions:** Completed work in [`sprints/03_models_and_training_loops/results/02_loss_functions.py`](...) and notes in [`sprints/03_models_and_training_loops/notes/02_loss_functions_training_loop_notes.md`](...) covering:

  - Understanding the purpose of loss functions (measuring error).
  - Using common loss functions (`nn.CrossEntropyLoss`, `nn.MSELoss`).
  - Understanding the input/output shapes expected by loss functions.
  - Cross-Entropy specifics (combines LogSoftmax and NLLLoss).

- **Optimizers:** Completed work in [`sprints/03_models_and_training_loops/results/03_optimizers.py`](...) and notes in [`sprints/03_models_and_training_loops/notes/03_optimizers_notes.md`](...) covering:

  - Understanding the role of optimizers in updating weights based on gradients.
  - Instantiating common optimizers (`torch.optim.Adam`, `torch.optim.SGD`).
  - Understanding key hyperparameters (`lr`, `momentum`, `betas`).
  - Recognizing Adam/AdamW as common default choices.

- **Training Loop:** Completed work in [`sprints/03_models_and_training_loops/results/04_training_loop.py`](...) and notes in [`sprints/03_models_and_training_loops/notes/04_training_loop_notes.md`](...) covering:

  - Implementing the standard PyTorch training cycle: forward pass, loss calculation, `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`.
  - Iterating over a `DataLoader`.
  - Setting the model to training mode (`model.train()`).
  - Accumulating and logging epoch loss.
  - Integrating `tqdm` for batch-level progress visualization.

- **Basic Evaluation:** Completed work in [`sprints/03_models_and_training_loops/results/05_basic_evaluation.py`](...) and notes in [`sprints/03_models_and_training_loops/notes/05_basic_evaluation_notes.md`](...) covering:
  - Creating a separate validation dataset and `DataLoader`.
  - Implementing an evaluation loop.
  - Setting the model to evaluation mode (`model.eval()`) and understanding its importance (e.g., for Dropout, BatchNorm).
  - Disabling gradient calculation using `with torch.no_grad():` for efficiency.
  - Calculating metrics (e.g., validation loss) on the validation set.
  - Performing evaluation periodically during training (e.g., after each epoch).

_(Update this section as sprints are completed or significant learning occurs. Add specific skills or concepts learned under relevant headings.)_

## Sprint 4 Progress (Advanced Training & CNN Basics)

- **CNN Fundamentals:** Learned and implemented core CNN layers in [`sprints/04_advanced_training_mnist/results/define_cnn.py`](./sprints/04_advanced_training_mnist/results/define_cnn.py) and notes in [`sprints/04_advanced_training_mnist/notes/01_cnn_architecture.md`](./sprints/04_advanced_training_mnist/notes/01_cnn_architecture.md):

  - `nn.Conv2d`: Understanding kernels, channels, stride, padding.
  - `nn.ReLU`: Standard activation function.
  - `nn.MaxPool2d`: Down-sampling feature maps.
  - `nn.Flatten`: Preparing data for fully connected layers.
  - Understood basic hierarchical feature learning concept.
  - Briefly explored a more complex ResNet structure (`BasicBlock`, skip connections) in [`sprints/04_advanced_training_mnist/results/define_resnet_mnist.py`](./sprints/04_advanced_training_mnist/results/define_resnet_mnist.py).

- **TensorBoard Integration:** Learned to use `torch.utils.tensorboard.SummaryWriter` as documented in [`sprints/04_advanced_training_mnist/notes/02_tensorboard_basics.md`](./sprints/04_advanced_training_mnist/notes/02_tensorboard_basics.md):

  - Creating a `SummaryWriter` with timestamped log directories (`runs/...`).
  - Logging scalar values (`add_scalar`) like loss, accuracy, and learning rate during training.
  - Understanding how to launch and interpret the TensorBoard dashboard (`tensorboard --logdir=runs`).

- **Learning Rate Scheduling:** Learned concepts and implementation (notes in [`sprints/04_advanced_training_mnist/notes/03_learning_rate_scheduling.md`](./sprints/04_advanced_training_mnist/notes/03_learning_rate_scheduling.md)):

  - Understood the rationale for changing LR during training.
  - Implemented `torch.optim.lr_scheduler.CosineAnnealingLR`.
  - Correctly integrated `scheduler.step()` into the training loop.
  - Understood the concept of "annealing" in this context.
  - Connected PyTorch schedulers to Hugging Face `Trainer`'s `lr_scheduler_type`.

- **Early Stopping:** Learned concepts and implementation (notes in [`sprints/04_advanced_training_mnist/notes/04_early_stopping.md`](./sprints/04_advanced_training_mnist/notes/04_early_stopping.md)):

  - Understood the goal of preventing overfitting and saving time.
  - Implemented logic to monitor validation loss.
  - Used a `patience` counter to trigger stopping.
  - Saved the best model's `state_dict` based on validation performance.
  - Loaded the best weights after training stopped.

- **Integrated Training Loop:** Combined all above components in [`sprints/04_advanced_training_mnist/results/train_mnist_cnn.py`](./sprints/04_advanced_training_mnist/results/train_mnist_cnn.py):

  - Successfully trained a CNN on MNIST.
  - Observed the practical effects of LR scheduling and early stopping.

- **Python/PyTorch Practices:**
  - Resolved `DataLoader` multiprocessing errors using `if __name__ == '__main__':` guard.
  - Practiced running scripts as modules (`python -m ...`) to handle relative imports correctly.

## Sprint 5 Progress (Embeddings & Positional Encoding)

- **Embeddings:**
  - Understood the concept of word embeddings (`nn.Embedding`) for mapping discrete tokens to dense vectors.
  - Implemented `nn.Embedding` and understood its parameters.
  - Created a custom embedding layer from scratch for deeper understanding.
- **Sinusoidal Positional Encoding:**

  - Understood the need for positional information in sequence models (especially Transformers).
  - Implemented the sinusoidal positional encoding formula from "Attention Is All You Need".
  - Visualized positional encoding patterns.
  - Integrated positional encodings with token embeddings via addition.
  - Built a `PositionalEncoding` `nn.Module` incorporating dropout.
  - Used `register_buffer` for non-trainable parameters.

- **Learned Positional Encoding:**
  - Understood the concept of learning positional embeddings via `nn.Embedding`.
  - Implemented learned positional embeddings in PyTorch ([`sprints/05_embeddings_and_positional_encoding/results/learned_pe_example.py`](./sprints/05_embeddings_and_positional_encoding/results/learned_pe_example.py)).
  - Contrasted learned vs. sinusoidal PEs (flexibility vs. extrapolation, parameters). ([`sprints/05_embeddings_and_positional_encoding/notes/04_learned_positional_embeddings.md`](./sprints/05_embeddings_and_positional_encoding/notes/04_learned_positional_embeddings.md))
- **Embedding Visualization:**
  - Understood the purpose and concepts of dimensionality reduction for visualizing embeddings ([`sprints/05_embeddings_and_positional_encoding/notes/05_embedding_visualization_notes.md`](./sprints/05_embeddings_and_positional_encoding/notes/05_embedding_visualization_notes.md)).
  - Learned the core ideas behind PCA, t-SNE, and UMAP ([`notes/05a...`](./sprints/05_embeddings_and_positional_encoding/notes/05a_pca_explained_novice.md), [`notes/05b...`](./sprints/05_embeddings_and_positional_encoding/notes/05b_tsne_explained.md), [`notes/05c...`](./sprints/05_embeddings_and_positional_encoding/notes/05c_umap_explained_novice.md)).
  - Implemented basic visualization examples using `matplotlib`, `scikit-learn` (PCA, t-SNE), and `umap-learn` ([`results/pca_example.py`](./sprints/05_embeddings_and_positional_encoding/results/pca_example.py), [`results/tsne_example.py`](./sprints/05_embeddings_and_positional_encoding/results/tsne_example.py), [`results/umap_example.py`](./sprints/05_embeddings_and_positional_encoding/results/umap_example.py)).
  - Debugged complex dependency issues related to visualization libraries in `pyproject.toml`.

## Sprint 6 Progress (Multi-Head Attention)

- **Scaled Dot-Product Attention:**
  - Understood the concept, formula, and motivation (Q, K, V analogy).
  - Implemented the `scaled_dot_product_attention` function from scratch using PyTorch tensor operations ([`sprints/06_multi_head_attention/results/scaled_dot_product_attention.py`](./sprints/06_multi_head_attention/results/scaled_dot_product_attention.py)).
  - Included scaling by `sqrt(d_k)`.
  - Handled potential NaNs resulting from masking.
- **Attention Masking:**
  - Understood the purpose and implementation of Padding Masks (ignoring pad tokens).
  - Understood the purpose and implementation of Look-Ahead (Causal) Masks (preventing attention to future tokens).
  - Modified `scaled_dot_product_attention` to accept and apply boolean masks (`False` indicates masking).
  - Tested attention calculation with both mask types.
  - Documented masking concepts in [`sprints/06_multi_head_attention/notes/03_attention_masking.md`](./sprints/06_multi_head_attention/notes/03_attention_masking.md).
- **Multi-Head Attention:**
  - Understood the rationale for using multiple heads (attending to different subspaces).
  - Implemented the `MultiHeadAttention` `nn.Module` ([`sprints/06_multi_head_attention/results/02_multi_head_attention.py`](./sprints/06_multi_head_attention/results/02_multi_head_attention.py)).
  - Implemented linear projections for Q, K, V, and the final output.
  - Implemented `split_heads` and `combine_heads` helper methods for reshaping.
  - Correctly utilized the `scaled_dot_product_attention` function for parallel head computation.
  - Ensured the module handles input masks correctly.
- **PyTorch Best Practices:**
  - Continued use of type hints and comprehensive docstrings.
  - Implemented tests within `if __name__ == "__main__":` blocks.
  - Used `nn.Module` for building reusable components.
  - Handled tensor dimension manipulation (`transpose`, `view`).
- **Comparison:**
  - Compared LoRA results (perplexity, param count) to Sprint 12 baseline.

## Sprint 7 Progress (Building the Transformer Block)

- **Layer Normalization (`nn.LayerNorm`):**
  - Understood the concept, purpose (batch size independence, sequence stability), and mechanism.
  - Implemented `nn.LayerNorm` correctly, specifying `normalized_shape` for typical Transformer inputs.
  - Verified its effect (mean ~0, std dev ~1) on feature dimensions.
  - Documented in [`sprints/07_transformer_block/notes/01_layer_norm.md`](./sprints/07_transformer_block/notes/01_layer_norm.md).
- **Residual Connections & Add/Norm Pattern:**
  - Understood the motivation (gradient flow, identity mapping) and implementation of skip connections.
  - Implemented the common "Add & Norm" pattern: `LayerNorm(x + Dropout(Sublayer(x)))`.
  - Created a reusable wrapper concept (`AddNormWrapper` in example).
  - Documented in [`sprints/07_transformer_block/notes/02_residual_connections.md`](./sprints/07_transformer_block/notes/02_residual_connections.md).
- **Position-wise Feed-Forward Network (FFN):**
  - Understood the structure (Linear -> Activation -> Dropout -> Linear) and purpose.
  - Implemented the FFN as a reusable `nn.Module` (`PositionWiseFeedForward`).
  - Used GELU activation.
  - Documented in [`sprints/07_transformer_block/notes/03_feed_forward_network.md`](./sprints/07_transformer_block/notes/03_feed_forward_network.md).
- **Transformer Encoder Block:**
  - Assembled the `EncoderBlock` module using Multi-Head Self-Attention and FFN sub-layers, each wrapped in Add & Norm.
  - Correctly handled inputs, padding masks, and data flow.
  - Handled tuple output from `MultiHeadAttention`.
  - Tested shape maintenance and basic normalization with single and stacked blocks.
  - Documented in [`sprints/07_transformer_block/notes/04_encoder_block.md`](./sprints/07_transformer_block/notes/04_encoder_block.md).
- **Transformer Decoder Block:**
  - Assembled the `DecoderBlock` module with three sub-layers: Masked Self-Attention, Cross-Attention, and FFN, each wrapped in Add & Norm.
  - Correctly managed inputs (target sequence, encoder output) and masks (look-ahead target mask, encoder padding mask).
  - Differentiated between self-attention (Q=K=V=target) and cross-attention (Q=target_processed, K=V=encoder_output).
  - Tested shape maintenance and basic normalization with single and stacked blocks.
  - Documented in [`sprints/07_transformer_block/notes/05_decoder_block.md`](./sprints/07_transformer_block/notes/05_decoder_block.md).
- **Debugging & Integration:**
  - Practiced debugging import errors related to module structure and execution methods (`python -m ...`).
  - Debugged module interface mismatches (e.g., MHA arguments, tuple outputs).
  - Refactored code for clarity (e.g., renaming files).

## Sprint 8 Progress (Assembling the GPT-2 Model)

- **GPT Model Architecture:**
  - Defined the overall `GPT` model structure as a `nn.Module` ([`results/model.py`](./sprints/08_gpt2_assembly/results/model.py)).
  - Implemented token embeddings (`nn.Embedding`).
  - Implemented sinusoidal positional encoding (`PositionalEncodingBatchFirst`) and integrated it ([`results/positional_encoding.py`](./sprints/08_gpt2_assembly/results/positional_encoding.py)).
  - Created a GPT-specific decoder block (`GPTDecoderBlock`) by removing cross-attention from the previous `DecoderBlock` ([`results/gpt_decoder_block.py`](./sprints/08_gpt2_assembly/results/gpt_decoder_block.py)).
  - Stacked `GPTDecoderBlock` layers using `nn.ModuleList`.
  - Added final `LayerNorm` and output linear projection layer.
  - Understood and implemented weight tying between token embeddings and output projection.
- **Model Configuration:**
  - Created a `GPTConfig` dataclass to manage hyperparameters ([`results/config.py`](./sprints/08_gpt2_assembly/results/config.py)).
  - Refactored the `GPT` model to accept the `GPTConfig` object.
  - Implemented automatic calculation of `d_ff` in the config.
- **Weight Initialization:**
  - Implemented a `_init_weights` method for initializing Linear and Embedding layers according to GPT-2 practices.
  - Used `model.apply()` to apply the initialization.
- **Saving & Loading:**
  - Implemented `save_checkpoint` and `load_checkpoint` functions in [`results/utils.py`](./sprints/08_gpt2_assembly/results/utils.py).
  - Checkpoints save model `state_dict` and `config`.
  - Loading function reconstructs the model from the saved config.
- **Tokenization (Stretch Goal):**
  - Added `tokenizers` library to project dependencies (`pyproject.toml`).
  - Implemented `get_gpt2_tokenizer` function to load the standard GPT-2 tokenizer from Hugging Face Hub/cache ([`results/tokenizer.py`](./sprints/08_gpt2_assembly/results/tokenizer.py)).
  - Tested encoding and decoding with the loaded tokenizer.
- **PyTorch Practices:**
  - Used dataclasses for configuration.
  - Organized utility functions (`save/load`) into a separate file.
  - Added basic tests (`if __name__ == '__main__':`) for new modules.
  - Handled Python environment updates using `uv`.
  - Implemented helper methods within the model class (e.g., `get_num_params`).

## Sprint 9 Progress (Training the GPT-2 Model)

- **Text Data Pipeline:**
  - Implemented data preparation script (`prepare_data.py`) for downloading and splitting text datasets (TinyShakespeare).
  - Created a custom PyTorch `Dataset` (`TextDataset`) for sequential text data, handling tokenization and generating input/target pairs ([`results/dataset.py`](./sprints/09_train_gpt2/results/dataset.py)).
  - Configured `DataLoader` for efficient batching, shuffling, and optional parallel loading.
  - Integrated the GPT-2 tokenizer from Sprint 8.
- **Language Model Training Loop:**
  - Implemented a complete training loop for autoregressive language modeling ([`results/train_gpt2.py`](./sprints/09_train_gpt2/results/train_gpt2.py)).
  - Correctly used `nn.CrossEntropyLoss` for next-token prediction.
  - Integrated the `GPT` model and `GPTConfig` from Sprint 8.
  - Used `torch.optim.AdamW` optimizer with appropriate weight decay.
  - Implemented gradient clipping (`torch.nn.utils.clip_grad_norm_`).
- **Evaluation & Metrics:**
  - Implemented a function to calculate perplexity as the primary evaluation metric.
  - Set up an evaluation loop using `torch.no_grad()` and `model.eval()`.
- **Checkpointing Enhancements:**
  - Modified `save_checkpoint` and `load_checkpoint` utilities (`utils.py`) to handle optimizer state, learning rate scheduler state, training step/epoch, and validation loss.
  - Implemented logic to resume training from checkpoints.
- **Learning Rate Scheduling:**
  - Implemented a cosine decay learning rate scheduler with linear warmup.
  - Integrated the scheduler step correctly within the training loop.
- **Logging & Configuration:**
  - Added `SummaryWriter` logging (TensorBoard) for training loss, validation loss/perplexity, and learning rate.
  - Used `argparse` to configure training parameters (data paths, hyperparameters, device, logging, checkpointing).
  - Implemented device handling logic (`get_device`) to automatically select CUDA, MPS, or CPU.
- **End-to-End Execution:**
  - Created a main training script combining all components.
  - Successfully ran training experiments, gaining insights into compute requirements.
- **Documentation:**
  - Created detailed notes for each major component (data pipeline, training loop, perplexity, checkpointing, LR scheduling) in `sprints/09_train_gpt2/notes/`.

## Sprint 10 Progress (Pre-trained Models, Generation, Demo)

- **Hugging Face `transformers` Library:**
  - Successfully used `AutoTokenizer` and `AutoModelForCausalLM` to load pre-trained models (e.g., GPT-2).
  - Handled model configuration (`model.config`).
  - Managed tokenizer specifics (adding PAD token, resizing embeddings).
  - Gained familiarity with the high-level `model.generate()` API.
- **Text Generation Techniques:**
  - Understood and implemented Greedy Search.
  - Understood and implemented Temperature Scaling.
  - Understood and implemented Top-k Sampling.
  - Understood and implemented Nucleus (Top-p) Sampling.
  - Configured generation parameters (`do_sample`, `temperature`, `top_k`, `top_p`, `max_new_tokens`).
- **Streaming Output:**
  - Understood the concept of using Python generators (`yield`) for streaming.
  - Implemented streaming generation using `transformers.TextIteratorStreamer`.
  - Utilized `threading.Thread` to run blocking `model.generate` calls asynchronously.
- **Basic UI/Demo Building:**
  - Installed and used `gradio` library.
  - Built an interactive web UI using `gr.Blocks` with components like Textbox, Dropdown, Slider, Button.
  - Connected UI inputs/outputs to Python backend functions (including streaming generator function).
- **Python/Environment Practices:**
  - Reinforced understanding of Python imports (relative vs. absolute, running as module `-m`).
  - Gained further experience with `uv` dependency management (`uv add`, `uv sync`).

## Sprint 11 Progress (Fine-tuning for Classification)

- **Fine-tuning Concepts:**
  - Understood the process of adapting a pre-trained model for a downstream task ([`notes/06_finetuning_explained.md`](./sprints/11_finetune_gpt2_classification/notes/06_finetuning_explained.md)).
  - Differentiated between fine-tuning and training from scratch.
- **Sequence Classification with Transformers:**
  - Used `AutoModelForSequenceClassification` to load a pre-trained model with an added (randomly initialized) classification head.
  - Understood how the classification head processes sequence representations for prediction.
- **Hugging Face `datasets` Library:**
  - Loaded datasets by name from the Hub (`load_dataset`).
  - Handled cases where specified datasets were unavailable.
  - Used `.map()` for efficient tokenization.
  - Used `.cast_column()` to set feature types (`ClassLabel`).
  - Used `.train_test_split()` for creating stratified train/validation/test splits.
  - Used `.save_to_disk()` and `load_from_disk()` for processed datasets.
- **Tokenizer Fine-tuning Specifics:**
  - Handled `pad_token` assignment for models like GPT-2 that lack one by default.
  - Applied padding and truncation strategies suitable for classification.
- **Training Loop for Fine-tuning:**
  - Implemented a standard fine-tuning loop ([`results/06_finetune_loop.py`](./sprints/11_finetune_gpt2_classification/results/06_finetune_loop.py)).
  - Used `AdamW` optimizer.
  - Leveraged the model's built-in loss calculation when labels are provided.
  - Implemented periodic validation within epochs.
- **Evaluation Metrics:**
  - Calculated accuracy using `sklearn.metrics.accuracy_score`.
  - Generated and interpreted `sklearn.metrics.classification_report` (precision, recall, F1-score).
- **Device Handling:**
  - Implemented robust device checking (`cuda`, `mps`, `cpu`).
- **PyTorch Practices:**
  - Refactored evaluation logic into a reusable function.
  - Used `tqdm` for progress bars in training and evaluation.

## Sprint 12 Progress (Fine-tuning for Generation)

- **Generative Fine-tuning Concepts:**
  - Understood the difference between fine-tuning for classification vs. generation ([`notes/02_finetuning_setup.md`](./sprints/12_finetune_gpt2_generative/notes/02_finetuning_setup.md)).
  - Used `AutoModelForCausalLM` for loading the base model.
- **Data Preparation for Generation:**
  - Adapted the data preparation script (`prepare_data.py`) to tokenize raw text and save as `.bin` files for training/validation ([`results/prepare_data.py`](./sprints/12_finetune_gpt2_generative/results/prepare_data.py)).
  - Created a `TextDataset` using `numpy.memmap` for efficient loading of tokenized binary data ([`results/dataset.py`](./sprints/12_finetune_gpt2_generative/results/dataset.py)).
- **Generative Fine-tuning Loop:**
  - Implemented a fine-tuning loop using standard PyTorch practices ([`results/finetune_generative.py`](./sprints/12_finetune_gpt2_generative/results/finetune_generative.py)).
  - Leveraged the model's built-in loss calculation for causal language modeling when `labels` are provided.
  - Used `AdamW` optimizer and a learning rate scheduler (`get_scheduler`).
  - Implemented periodic evaluation based on validation loss and perplexity.
  - Included checkpointing to save the best model based on validation loss using `model.save_pretrained()`.
- **Model Comparison:**
  - Implemented a script (`generate_text.py`) to load the original pre-trained model and the fine-tuned checkpoint ([`results/generate_text.py`](./sprints/12_finetune_gpt2_generative/results/generate_text.py)).
  - Generated text from both models using identical prompts and parameters for qualitative comparison.
- **Practices:**
  - Practiced iterative development (refining evaluation frequency, handling dataset availability issues).

## Sprint 13 Progress (Parameter-Efficient Fine-Tuning - LoRA)

- **PEFT & LoRA Theory:**
  - Understood the motivation for Parameter-Efficient Fine-Tuning (PEFT).
  - Learned the core concepts of Low-Rank Adaptation (LoRA): freezing base weights, injecting low-rank adapter matrices (A, B), updating only adapters.
  - Understood the mechanism of low-rank decomposition (ΔW ≈ BA).
  - Documented concepts in [`notes/01_lora_theory_setup.md`](./sprints/13_peft_lora/notes/01_lora_theory_setup.md).
- **Hugging Face `peft` Library Usage:**
  - Installed and imported the `peft` library.
  - Configured `LoraConfig` specifying rank (`r`), alpha (`lora_alpha`), dropout (`lora_dropout`), and `target_modules` (e.g., `["c_attn"]` for GPT-2 attention).
  - Set the appropriate `task_type` (`TaskType.CAUSAL_LM`).
  - Used `get_peft_model` to wrap a base `transformers` model (`AutoModelForCausalLM`) and apply the LoRA configuration.
- **LoRA Fine-tuning Implementation:**
  - Adapted a generative fine-tuning script (`finetune_lora.py`) to use the PEFT model ([`results/finetune_lora.py`](./sprints/13_peft_lora/results/finetune_lora.py)).
  - Ensured the optimizer (`AdamW`) only targeted the trainable adapter parameters.
  - Implemented the training loop, evaluation (calculating perplexity), and learning rate scheduling (`get_scheduler`).
- **Parameter Efficiency Analysis:**
  - Implemented `print_trainable_parameters` function to calculate and display the number and percentage of trainable parameters vs. total parameters.
  - Observed a drastic reduction in trainable parameters (~0.24% for `r=8` on GPT-2).
- **Comparison & Evaluation:**
  - Compared the validation perplexity achieved with LoRA (1.2537) against the baseline full fine-tuning perplexity (1.1211 from Sprint 12).
  - Analyzed the trade-off between performance and parameter efficiency.
  - Documented results and comparison in [`notes/04_results_comparison.md`](./sprints/13_peft_lora/notes/04_results_comparison.md).
- **LoRA Model Saving:**
  - Used `model.save_pretrained()` on the PEFT model to save only the adapter weights (`adapter_model.safetensors` or `.bin`) and the adapter configuration (`adapter_config.json`).

_(Self-assessment: Gained practical understanding and implementation skills for LoRA, a key PEFT technique, including its configuration, application, and evaluation.)_

## Sprint 14 Progress (Post-Training Quantization - PoC)

- **Quantization Concepts:**
  - Documented core concepts, benefits (size, speed, power), and drawbacks (potential accuracy loss) of model quantization ([`notes/01...`](./sprints/14_quantization/notes/01_quantization_concepts.md)).
  - Differentiated between Post-Training Dynamic, Post-Training Static, and Quantization-Aware Training (QAT) ([`notes/02...`](./sprints/14_quantization/notes/02_quantization_types.md)).
  - Understood common data types used (`INT8`, `FP16`, `BF16`) ([`notes/03...`](./sprints/14_quantization/notes/03_data_types.md)).
- **PyTorch Eager Mode Quantization API:**
  - Explored the `torch.ao.quantization` namespace ([`notes/04...`](./sprints/14_quantization/notes/04_pytorch_quantization_api.md)).
  - Implemented **Dynamic Quantization** using `torch.ao.quantization.quantize_dynamic`, targeting specific layer types (`nn.Linear`).
  - Implemented **Static Quantization** workflow:
    - Used a wrapper module (`GPT2QuantWrapper`) with `QuantStub`/`DeQuantStub`.
    - Set quantization backend (`torch.backends.quantized.engine = 'qnnpack'`).
    - Applied `QConfig` directly to target submodules (`Conv1D`).
    - Used `torch.ao.quantization.prepare` to insert observers.
    - Performed calibration using data from `datasets` library (Wikitext).
    - Used `torch.ao.quantization.convert` to finalize the quantized model.
- **Debugging & Analysis:**
  - Resolved quantization backend errors (`NoQEngine`).
  - Debugged issues with applying QConfigs (`AssertionError`, `AttributeError`, `TypeError`) by switching from direct assignment/mapping to submodule targeting.
  - Identified correct target layer type (`Conv1D`) in the HF GPT-2 model for effective quantization.
  - Compared inference speed (CPU) between FP32, dynamic INT8, and static INT8 models, observing expected speedups (~1.2x).
  - Understood why `state_dict` size might not decrease (or even increase) with PTQ due to overhead, emphasizing runtime metrics.

## Sprint 15 Progress (Encoder-Decoder Architecture & Training Challenges)

- **Encoder-Decoder Assembly:**
  - [x] Reused `EncoderBlock` and `DecoderBlock` from previous sprints.
  - [x] Built `Encoder` module stacking `EncoderBlock`s, including embeddings and positional encoding (initially Learned, later Sinusoidal).
  - [x] Built `Decoder` module stacking `DecoderBlock`s, handling target embeddings, PE, and encoder output.
  - [x] Assembled the full `EncoderDecoder` model with a final linear projection layer.
  - [x] Implemented helper methods for creating padding and look-ahead masks.
  - [x] Documented the training and inference data flow ([notes/01_encoder_decoder_flow.md](./sprints/15_encoder_decoder/notes/01_encoder_decoder_flow.md)).
- **Seq2Seq Training & Evaluation:**
  - [x] Implemented a toy sequence reversal task (`SequenceReversalDataset`, `train_seq_reversal.py`).
  - [x] Implemented standard teacher-forcing training loop (`CrossEntropyLoss` with padding ignored).
  - [x] Implemented autoregressive (free-running) evaluation for sequence generation.
- **Addressing Exposure Bias:**
  - [x] Identified severe exposure bias: high teacher-forcing accuracy (100%) but near-zero free-running accuracy (model collapsing to simple loops).
  - [x] Experimented with hyperparameter tuning (epochs, LR, capacity, heads) - insufficient.
  - [x] Experimented with regularization (Dropout, Label Smoothing) - insufficient.
  - [x] Experimented with architectural changes (Pre-Layer Normalization) - insufficient.
  - [x] Implemented and tested Scheduled Sampling (linear & inverse sigmoid schedules) - insufficient.
  - [x] Implemented and tested Professor Forcing (MSE on hidden states) - insufficient.
  - [x] Implemented and tested REINFORCE (policy gradient on token accuracy) - insufficient for full reversal.
- **Problem/Tool Mismatch Analysis:**
  - [x] Concluded that standard MLE/Seq2Seq training is ill-suited for deterministic, brittle tasks like sequence reversal due to inherent exposure bias.
  - [x] Recognized that sequence-level RL (e.g., 0/1 reward for perfect reversal) or different architectures (Pointer Networks) might be needed.
  - [x] Implemented a direct, non-ML solution using `torch.flip` as the correct tool for the specific reversal task ([results/direct_reverse_demo.py](./sprints/15_encoder_decoder/results/direct_reverse_demo.py)).

## Sprint 16 Progress (GRPO Fine-Tuning - Part 1: Setup & Data Prep)

- **RLHF Data Preparation:**
  - Understood the purpose of `prompt`, `chosen`, `rejected` data format for reward modeling.
  - Implemented logic (`prepare_dataset.py`) to transform a source dataset (`moremilk/CoT_Reasoning_Cooking`) into this format.
  - Correctly applied model-specific chat templates (Llama 3.2 Instruct) using special tokens and structure.
- **LLM Generation (Batching for Data Prep):**
  - Used a base model (`unsloth/Llama-3.2-3B-Instruct`) to generate synthetic 'rejected' responses.
  - Implemented efficient batch generation using `transformers` tokenizer (with padding) and `model.generate()` within the data preparation script.
  - Optimized batch size (`batch_size=16`) based on available VRAM for significant speedup (~20 min processing time).
- **Hugging Face Hub Interaction (Datasets):**
  - Programmatically uploaded the processed dataset to the Hugging Face Hub using `datasets.Dataset.push_to_hub()`.
  - Utilized `python-dotenv` to load API keys/tokens from `.env` for Hub authentication.
- **Python Environment (`uv`):**
  - Successfully used `uv add` and `uv sync` to install required libraries (`transformers`, `datasets`, `accelerate`, `trl`, `python-dotenv`) after encountering `pip` environment restrictions.
- **Model Selection (RLHF):** Demonstrated ability to adapt model choice (3B -> 1B) based on practical resource constraints and training time considerations during RLHF setup.

## Sprint 17 Progress (Graph Neural Networks)

- **GNN Concepts & Message Passing (GCN, GraphSAGE):** Learned the core paradigm of aggregating and updating node features via neighbor messages.
- **PyTorch Geometric:** Used `GCNConv` and `SAGEConv` for building graph convolution layers and GraphSAGE models.
- **Semi-Supervised Node Classification:** Implemented a GCN on the Zachary's Karate Club dataset to predict community membership with few labeled nodes.
- **Dimensionality Reduction & Visualization:** Applied t-SNE to node embeddings and saved visualizations.
- **Dynamic Knowledge Graph Construction:** Integrated LLM-based triple extraction, `networkx` graph updates, and custom entity canonicalization.
- **Faiss Vector Store:** Indexed node embeddings for similarity search, enabling retrieval of related entities.
- **LLM Integration for Structured Data Extraction:** Used GPT-4.1-nano via `litellm` to extract SVO triples from text with prompt engineering and fallback logic.

## Sprint 16 Progress (GRPO Fine-Tuning - Part 2: Training & Analysis - Ongoing)

- **RLHF Concepts:** Understanding Reward Modeling, Reinforcement Learning from Human Feedback (RLHF) workflow.
- **GRPO Algorithm:** Understanding Group Relative Policy Optimization concepts (group size `k`, KL divergence, `beta` coefficient).
- **TRL Library:** Using `RewardTrainer` for RM training, `GRPOConfig` and `GRPOTrainer` for policy fine-tuning.
- **PEFT Integration:** Applying LoRA (`PeftConfig`) within `RewardTrainer` and `GRPOTrainer`.
- **Dataset Formatting:** Preparing datasets in `prompt`/`chosen`/`rejected` format for RM/RLHF.
- **Hyperparameter Tuning (RLHF):** Understanding the roles and interplay of `batch_size`, `gradient_accumulation_steps`, `num_generations` (`k`), and `beta` in GRPO.
- **Training Monitoring (RLHF):** Analyzing key metrics like loss, reward scores, reward standard deviation, KL divergence, gradient norm, and clip ratio during training. Initial observations: stable gradients, low KL, fluctuating rewards without strong upward trend yet.
- **Troubleshooting RLHF:** Diagnosing potential issues like reward stagnation, KL divergence behavior, batch size/sampler interactions, and interpreting library warnings (e.g., `score.weight` message).
- **Model Selection/Adaptation:** Adapting training scripts and configurations for different base model sizes (3B vs 1B) based on resource constraints.
- **Documentation:** Maintaining clear documentation of RLHF setup, experiments, and results.
