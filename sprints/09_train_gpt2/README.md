# Sprint 9: Training the GPT-2 Model

**Goal:** Implement the training pipeline for the assembled GPT-2 model, train it on a dataset (e.g., TinyShakespeare), and establish basic evaluation.

## Tasks / Learning Objectives

1.  **Data Preparation:**
    - [ ] Download and prepare a suitable text dataset (e.g., TinyShakespeare).
    - [ ] Integrate the GPT-2 tokenizer (`sprints/08_gpt2_assembly/results/tokenizer.py`) for encoding.
    - [ ] Create a PyTorch `Dataset` for handling sequential text data (input sequences and target sequences).
    - [ ] Set up a `DataLoader` for efficient batching of the text data.
2.  **Training Loop Implementation:**
    - [ ] Define the training loop function for language modeling.
    - [ ] Implement the core logic: forward pass, loss calculation (using `nn.CrossEntropyLoss`), backpropagation, optimizer step.
    - [ ] Integrate the `GPT` model and `GPTConfig` from Sprint 8.
    - [ ] Select and configure an optimizer (e.g., `torch.optim.AdamW`).
3.  **Evaluation Metrics:**
    - [ ] Implement a function to calculate perplexity as the primary evaluation metric.
    - [ ] Set up a basic evaluation loop (using `torch.no_grad()`, `model.eval()`).
4.  **Checkpointing Enhancement:**
    - [ ] Modify the checkpointing functions (`utils.py` from Sprint 8) to save/load optimizer state, training epoch/step, and potentially validation loss.
    - [ ] Implement logic to resume training from a checkpoint.
5.  **Learning Rate Scheduling:**
    - [ ] Implement a learning rate scheduler (e.g., cosine decay with warmup).
    - [ ] Integrate the scheduler step into the training loop.
6.  **TensorBoard Integration:**
    - [ ] Add `SummaryWriter` logging for training loss, validation loss/perplexity, and learning rate.
7.  **Putting it Together:**
    - [ ] Create the main training script (`train_gpt2.py` or similar) that combines data loading, model instantiation, training loop, evaluation, checkpointing, and LR scheduling.
    - [ ] Run a small training experiment to ensure the pipeline works end-to-end.

## Definition of Done / Key Questions Answered

- [ ] A working text data pipeline for the GPT-2 model is implemented.
- [ ] The training loop successfully trains the model and logs loss.
- [ ] Perplexity can be calculated on a validation set.
- [ ] Checkpointing saves and loads model, optimizer, and training state.
- [ ] Learning rate scheduling is functional.
- [ ] TensorBoard logs basic training metrics.
- [ ] A complete training script exists and can run a basic training session.
- What are the challenges in preparing sequential text data for a Transformer?
- How does the language modeling training objective work in practice?
- How can optimizer state and LR scheduling be managed during checkpointing?

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Hugging Face Tokenizers Documentation](https://huggingface.co/docs/tokenizers/index)
- Previous Sprint Notes/Results (Sprint 8 mainly)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) (Review)
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) (Reference Implementation)

## Notes

_(Link to detailed notes created during the sprint)_

- [notes/01_data_pipeline.md](./notes/01_data_pipeline.md) (Placeholder)
- [notes/02_training_loop_lm.md](./notes/02_training_loop_lm.md) (Placeholder)
- [notes/03_perplexity.md](./notes/03_perplexity.md) (Placeholder)
- [notes/04_checkpointing_refinement.md](./notes/04_checkpointing_refinement.md) (Placeholder)
- [notes/05_lr_scheduling_warmup.md](./notes/05_lr_scheduling_warmup.md) (Placeholder)

## Results

_(Link to code, scripts, or outputs generated)_

- [results/prepare_data.py](./results/prepare_data.py) (Placeholder)
- [results/dataset.py](./results/dataset.py) (Placeholder)
- [results/train_gpt2.py](./results/train_gpt2.py) (Placeholder)
- [results/evaluate.py](./results/evaluate.py) (Placeholder)

## Retrospective

_(To be filled out after the sprint)_

**What went well?**

- ...

**What could be improved?**

- ...

**Key Learnings:**

- ...
