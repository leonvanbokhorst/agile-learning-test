# Sprint 12: Fine-tuning GPT-2 for Generative Tasks

**Sprint Goal:** Fine-tune a pre-trained GPT-2 model to generate text in a specific style or domain, deepening the understanding of generative fine-tuning techniques.

**Status:** Training Completed ✅

## Overview

In Sprint 10, we learned how to use pre-trained models like GPT-2 and generate text using various sampling strategies. In Sprint 11, we successfully fine-tuned GPT-2 for a _classification_ task. Now, we return to GPT-2's core strength – generation – but aim to _specialize_ its output through fine-tuning.

We will select a dataset representing a particular style (e.g., poetry, dialogues, code) and fine-tune the standard pre-trained GPT-2 model on it. The goal is to make the model's generated text more closely resemble the style of the fine-tuning dataset compared to the original, general-purpose GPT-2.

This involves adapting the data loading pipeline for causal language modeling (similar to Sprint 9's training from scratch, but using the pre-trained model as a starting point) and implementing a suitable training loop.

## Learning Objectives

- Understand the process and rationale for generative fine-tuning of large language models.
- Adapt data preparation techniques for fine-tuning pre-trained generative models.
- Implement a fine-tuning loop using Hugging Face `transformers` for a generative task (causal language modeling).
- Evaluate the results qualitatively by comparing text generated before and after fine-tuning.
- Gain experience selecting appropriate datasets for specific generative fine-tuning goals.

## Tasks

- [ ] **1. Dataset Selection & Preparation:**
  - [x] Research and select a suitable dataset for generative fine-tuning (e.g., TinyShakespeare, a dataset of dialogues, code snippets, etc.). -> _Using provided `book.txt`_
  - [x] Download and inspect the chosen dataset. -> _Provided in `results/data/book.txt`_
  - [x] Implement/adapt a script (`prepare_data.py`?) to preprocess the dataset (tokenization, splitting if necessary).
  - [x] Create a PyTorch `Dataset` and `DataLoader` suitable for causal language modeling on the chosen dataset.
  - [x] Document dataset choice and preparation steps in `notes/01_dataset_prep.md`.
- [ ] **2. Fine-tuning Setup:**
  - [x] Load the pre-trained `gpt2` model and tokenizer using `AutoModelForCausalLM` and `AutoTokenizer`.
  - [x] Configure the model for training (ensure correct device placement).
  - [x] Set up the optimizer (`AdamW`) and potentially a learning rate scheduler.
  - [x] Document the setup process in `notes/02_finetuning_setup.md`.
- [ ] **3. Implement Fine-tuning Loop:**
  - [x] Implement the training loop (`finetune_generative.py`?) that feeds batches from the `DataLoader` to the model.
  - [x] Ensure the model receives `labels` for causal LM loss calculation.
  - [x] Include basic logging (e.g., training loss).
  - [x] Implement periodic saving of model checkpoints (fine-tuned weights).
  - [x] Document the training loop logic in `notes/03_training_loop.md`.
- [ ] **4. Run Fine-tuning:**
  - [x] Execute the fine-tuning script for a chosen number of steps/epochs.
  - [x] Monitor training progress and loss.

**Training Summary:**
- Completed 3 epochs with hyperparameters (block size: 128, batch size: 64, learning rate: 3e-5).
- Best validation loss achieved: 0.1143 (epoch 3).
- Final perplexity: 1.1211.
- Best model checkpoint saved to `results/checkpoints/finetuned_model`.

- [ ] **5. Evaluation & Comparison:**
  - [ ] Implement a script (`generate_text.py`?) to generate text using:
    - The original pre-trained GPT-2.
    - The fine-tuned GPT-2 checkpoint.
  - [ ] Use the same prompts and generation parameters (e.g., temperature, top-k) for fair comparison.
  - [ ] Qualitatively compare the outputs. Does the fine-tuned model adopt the target style/domain?
  - [ ] Document the generation process and comparison results in `notes/04_evaluation.md`.
- [ ] **6. Documentation & Retrospective:**
  - [ ] Ensure all code is well-commented and follows project standards.
  - [ ] Update `skills_competencies.md` and `milestones.md`.
  - [ ] Update this `README.md` with results, links to notes/code, and a retrospective.
  - [ ] Update `sprints/backlog.md`.

## Resources & Links

- **Notes:**
  - [`notes/01_dataset_prep.md`](./notes/01_dataset_prep.md)
  - [`notes/02_finetuning_setup.md`](./notes/02_finetuning_setup.md)
  - [`notes/03_training_loop.md`](./notes/03_training_loop.md)
  - [`notes/04_evaluation.md`](./notes/04_evaluation.md)
- **Results:**
  - [`results/prepare_data.py`](./results/prepare_data.py) (or similar)
  - [`results/finetune_generative.py`](./results/finetune_generative.py) (or similar)
  - [`results/generate_text.py`](./results/generate_text.py) (or similar)
  - Checkpoints directory (e.g., `results/checkpoints/`)
- **Dependencies:**
  - `torch`, `transformers`, `datasets`, `tokenizers` (Add others as needed to `pyproject.toml`)

## Retrospective

_(To be filled out at the end of the sprint)_

- **What went well?**
- **What could be improved?**
- **Key learnings?**
- **Blockers encountered?**
