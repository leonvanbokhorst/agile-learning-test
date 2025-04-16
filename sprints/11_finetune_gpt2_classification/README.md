# Sprint 11: Fine-tuning GPT-2 for Classification

**Goal:** Adapt the pre-trained GPT-2 model for a sequence classification task.

## Context

Following Sprint 10 where we explored loading a pre-trained GPT-2 and building a basic generation demo, this sprint focuses on adapting that knowledge for a different downstream task: classification. We'll leverage the `transformers` library heavily.

## Tasks / Learning Objectives:

1.  [ ] **Dataset Selection & Preparation:**

    - Choose a suitable classification dataset (e.g., IMDB sentiment, GLUE subset like SST-2).
    - Explore the dataset structure and content.
    - Understand how to load and preprocess it using `datasets` library or standard PyTorch methods.

2.  [ ] **Tokenizer Adaptation:**

    - Load the appropriate GPT-2 tokenizer.
    - Understand and apply padding and truncation strategies suitable for classification inputs.
    - Prepare `DataLoader`s, ensuring correct batching and tokenization.

3.  [ ] **Model Modification:**

    - Load a pre-trained GPT-2 model (`GPT2ForSequenceClassification` from `transformers`).
    - Understand how a classification head is added to the base GPT-2 architecture.
    - Investigate the configuration options (e.g., number of labels).

4.  [ ] **Fine-tuning Loop:**

    - Implement a training loop specifically for fine-tuning.
    - Choose an appropriate loss function (e.g., `CrossEntropyLoss`).
    - Select an optimizer (e.g., `AdamW`).
    - Consider strategies like freezing base model layers initially vs. full fine-tuning.
    - Integrate learning rate scheduling if necessary.

5.  [ ] **Evaluation:**

    - Implement an evaluation loop.
    - Calculate relevant metrics (e.g., accuracy, F1-score).
    - Run evaluation on a validation/test set.

6.  [ ] **Documentation & Reflection:**
    - Document findings, code, and experiments in `notes/`.
    - Save key results (e.g., trained model checkpoint, metrics) in `results/`.
    - Reflect on the process, challenges, and learned concepts.

## Definition of Done / Key Questions Answered:

- [ ] Successfully load and preprocess a chosen classification dataset.
- [ ] Configure the GPT-2 tokenizer and `DataLoaders` for the task.
- [ ] Load a `GPT2ForSequenceClassification` model.
- [ ] Implement and run a fine-tuning loop.
- [ ] Evaluate the fine-tuned model on a test set and report metrics.
- [ ] Understand the key differences between generative pre-training and discriminative fine-tuning for Transformers.

## Notes & Results

- Working notes, observations, and code snippets: [`notes/`](./notes/)
- Final code, model checkpoints, and evaluation results: [`results/`](./results/)
