# Task & Baseline Selection Notes

## 1. Chosen Task for LoRA Adaptation

- We will apply LoRA to the **Generative Fine-tuning Task** from **Sprint 12**.
- **Model:** `gpt2` (using `AutoModelForCausalLM` from Hugging Face `transformers`).
- **Dataset:** Custom text data from `book.txt`.
- **Objective:** Fine-tune the model for Causal Language Modeling (predicting the next token) on the `book.txt` data.

## 2. Baseline from Full Fine-tuning (Sprint 12)

- **Method:** Full fine-tuning of all parameters of the `gpt2` model.
- **Training:** Completed 3 epochs (block size: 128, batch size: 64, learning rate: 3e-5).
- **Performance Metric:** Final validation **perplexity** = **1.1211**.
- **Trainable Parameters:** Approximately **124 Million** (standard `gpt2` size). This is the number we aim to drastically reduce with LoRA while maintaining comparable perplexity.

_(This provides the reference point against which we will compare the results of LoRA fine-tuning in Task 4)_
