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

- [x] **5. Evaluation & Comparison:**
  - [x] Implemented `generate_text.py` to compare original vs fine-tuned models.
  - [x] Used same prompts and generation parameters (including top-p and attention mask).
  - [x] Qualitatively compared outputs for prompts: "The old house", "Lorem ipsum", and empty string.
  - [ ] Document the generation process and comparison results in `notes/04_evaluation.md`.

## Evaluation & Comparison Results

**Usage Examples:**
```bash
cd sprints/12_finetune_gpt2_generative/results
python generate_text.py --do-sample False --top-k 1
python generate_text.py --do-sample True --temperature 0.3 --top-p 0.9
```

**Summary:**
- **Prompt 1 ("The old house"):**
  - Original: coherent, human-like sentence repetition.
  - Fine-tuned: Latin-like gibberish persists.
- **Prompt 2 ("Lorem ipsum"):**
  - Original: classic "Lorem ipsum ipsum..." repetition.
  - Fine-tuned: fragmented pseudo-Latin, no true repetition.
- **Prompt 3 (empty prompt):**
  - Original: sensible code generation advice.
  - Fine-tuned: repeated Latin-like tokens.

Despite greedy decoding (do-sample=False, top-k=1) and nucleus sampling (do-sample=True, temp=0.3, top-p=0.9), the fine-tuned model did not replicate the expected repeating pattern.
**Next Steps:** augment training data with repeated "Lorem ipsum" lines or run additional epochs to reinforce this motif.

- [ ] **6. Documentation & Retrospective:**
  - [ ] Ensure all code is well-commented and follows project standards.
  - [ ] Update `skills_competencies.md` and `milestones.md`.
  - [ ] Update this `