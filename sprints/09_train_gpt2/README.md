# Sprint 9: Training the GPT-2 Model

**Goal:** Implement the training pipeline for the assembled GPT-2 model, train it on a dataset (e.g., TinyShakespeare), and establish basic evaluation.

## Tasks / Learning Objectives

1.  **Data Preparation:**
    - [x] Download and prepare a suitable text dataset (e.g., TinyShakespeare).
    - [x] Integrate the GPT-2 tokenizer (`sprints/08_gpt2_assembly/results/tokenizer.py`) for encoding.
    - [x] Create a PyTorch `Dataset` for handling sequential text data (input sequences and target sequences).
    - [x] Set up a `DataLoader` for efficient batching of the text data.
2.  **Training Loop Implementation:**
    - [x] Define the training loop function for language modeling.
    - [x] Implement the core logic: forward pass, loss calculation (using `nn.CrossEntropyLoss`), backpropagation, optimizer step.
    - [x] Integrate the `GPT` model and `GPTConfig` from Sprint 8.
    - [x] Select and configure an optimizer (e.g., `torch.optim.AdamW`).
3.  **Evaluation Metrics:**
    - [x] Implement a function to calculate perplexity as the primary evaluation metric.
    - [x] Set up a basic evaluation loop (using `torch.no_grad()`, `model.eval()`).
4.  **Checkpointing Enhancement:**
    - [x] Modify the checkpointing functions (`utils.py` from Sprint 8) to save/load optimizer state, training epoch/step, and potentially validation loss.
    - [x] Implement logic to resume training from a checkpoint.
5.  **Learning Rate Scheduling:**
    - [x] Implement a learning rate scheduler (e.g., cosine decay with warmup).
    - [x] Integrate the scheduler step into the training loop.
6.  **TensorBoard Integration:**
    - [x] Add `SummaryWriter` logging for training loss, validation loss/perplexity, and learning rate.
7.  **Putting it Together:**
    - [x] Create the main training script (`train_gpt2.py` or similar) that combines data loading, model instantiation, training loop, evaluation, checkpointing, and LR scheduling.
    - [x] Run a small training experiment to ensure the pipeline works end-to-end.

## Definition of Done / Key Questions Answered

- [x] A working text data pipeline for the GPT-2 model is implemented.
- [x] The training loop successfully trains the model and logs loss.
- [x] Perplexity can be calculated on a validation set.
- [x] Checkpointing saves and loads model, optimizer, and training state.
- [x] Learning rate scheduling is functional.
- [x] TensorBoard logs basic training metrics.
- [x] A complete training script exists and can run a basic training session.
- What are the challenges in preparing sequential text data for a Transformer? (Answered in notes)
- How does the language modeling training objective work in practice? (Answered in notes)
- How can optimizer state and LR scheduling be managed during checkpointing? (Answered in notes)

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Hugging Face Tokenizers Documentation](https://huggingface.co/docs/tokenizers/index)
- Previous Sprint Notes/Results (Sprint 8 mainly)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) (Review)
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) (Reference Implementation)

## Notes

_(Link to detailed notes created during the sprint)_

- [notes/01_data_pipeline.md](./notes/01_data_pipeline.md)
- [notes/02_training_loop_lm.md](./notes/02_training_loop_lm.md)
- [notes/03_perplexity.md](./notes/03_perplexity.md)
- [notes/04_checkpointing_refinement.md](./notes/04_checkpointing_refinement.md)
- [notes/05_lr_scheduling_warmup.md](./notes/05_lr_scheduling_warmup.md)

## Results

_(Link to code, scripts, or outputs generated)_

- [results/prepare_data.py](./results/prepare_data.py)
- [results/dataset.py](./results/dataset.py)
- [results/train_gpt2.py](./results/train_gpt2.py)
- [results/evaluate.py](./results/evaluate.py) (Evaluation logic within `train_gpt2.py`)

## Retrospective

_(To be filled out after the sprint)_

**What went well?**

- The implementation of the full training pipeline, including data loading, training loop, evaluation, checkpointing, LR scheduling, and logging, was successful.
- The training script ran successfully on the target hardware (RTX 4090).
- Integration of components from previous sprints (tokenizer, model structure) worked as expected.

**What could be improved?**

- The initial setup and ensuring correct dependencies (CUDA, PyTorch version) always requires careful attention.
- Documentation (notes) could be written more concurrently during the implementation phase rather than afterwards.

**Key Learnings:**

- Gained a concrete understanding of the compute resources (VRAM, time) required to train even a relatively small transformer model like GPT-2 from scratch.
- Reinforced the practical steps involved in building and debugging a complete deep learning training pipeline in PyTorch.
- The exercise highlighted the significant advantage and practicality of using pre-trained models for many applications, given the high cost of training from scratch. This informs the decision to use a pre-trained model in the next sprint.
