# Sprint 11: Fine-tuning GPT-2 for Classification

**Goal:** Adapt the pre-trained GPT-2 model for a sequence classification task.

**Status:** Completed

## Context

Following Sprint 10 where we explored loading a pre-trained GPT-2 and building a basic generation demo, this sprint focused on adapting that knowledge for a different downstream task: classification (Fake vs. Real News). We leveraged the `transformers` and `datasets` libraries heavily.

## Tasks / Learning Objectives:

1.  [x] **Dataset Selection & Preparation:** ([Notes](./notes/01_dataset_selection.md), [Script](./results/01_load_dataset.py))

    - Chose the `Pulk17/Fake-News-Detection-dataset` after initial attempts with Onion datasets failed.
    - Explored the dataset structure and content.
    - Loaded and preprocessed using `datasets` library.

2.  [x] **Tokenizer Adaptation & Data Prep:** ([Tokenize Notes](./notes/02_tokenization.md), [Split Notes](./notes/03_data_splitting.md), [DataLoader Notes](./notes/04_dataloaders.md), [Tokenize Script](./results/02_tokenize_data.py), [Split Script](./results/03_split_data.py), [DataLoader Script](./results/04_create_dataloaders.py))

    - Loaded the `gpt2` tokenizer.
    - Applied padding and truncation (max_length=512).
    - Split data into train/validation/test sets (80/10/10 stratified).
    - Prepared `DataLoader`s for batching.

3.  [x] **Model Modification:** ([Notes](./notes/05_model_loading.md), [Script](./results/05_load_model.py), [Explanation](./notes/06_finetuning_explained.md))

    - Loaded `GPT2ForSequenceClassification` from `transformers`, specifying `num_labels=2`.
    - Understood that this adds a new classification head on top of the base model.

4.  [x] **Fine-tuning Loop:** ([Notes](./notes/07_finetuning_run_1.md), [Script](./results/06_finetune_loop.py))

    - Implemented a training loop using PyTorch.
    - Used `AdamW` optimizer and `CrossEntropyLoss` (implicitly handled by the model).
    - Implemented periodic validation (every 250 steps).
    - Ran for 1 epoch.

5.  [x] **Evaluation:** ([Notes](./notes/08_evaluation_results.md), [Script](./results/07_evaluate_model.py))

    - Implemented evaluation on the test set.
    - Calculated accuracy (99.83%) and classification report (perfect precision/recall/F1).

6.  [x] **Documentation & Reflection:**
    - Documented findings, code, and experiments in [`notes/`](./notes/).
    - Saved key results (scripts, saved model, saved dataset splits) in [`results/`](./results/).

## Definition of Done / Key Questions Answered:

- [x] Successfully loaded and preprocess a chosen classification dataset.
- [x] Configured the GPT-2 tokenizer and `DataLoaders` for the task.
- [x] Loaded a `GPT2ForSequenceClassification` model.
- [x] Implemented and ran a fine-tuning loop.
- [x] Evaluated the fine-tuned model on a test set and reported metrics (Excellent results!).
- [x] Understood the key differences between generative pre-training and discriminative fine-tuning for Transformers.

## Notes & Results

- Working notes, observations, and code snippets: [`notes/`](./notes/) (Detailed notes cover each step)
- Final code, saved datasets, and saved model: [`results/`](./results/)
