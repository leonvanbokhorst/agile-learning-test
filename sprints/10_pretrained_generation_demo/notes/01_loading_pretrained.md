# Sprint 10 - Task 1: Loading Pre-trained Model

## Goal

Load a pre-trained GPT-2 model and its corresponding tokenizer using the Hugging Face `transformers` library.

## Steps

### 1. Install Libraries

We need the `transformers` library from Hugging Face. Since we might want to load datasets easily later and build a demo, we'll also install `datasets` and `gradio`.

```bash
# Using uv (as seems standard for this project)
uv pip install transformers datasets gradio
```

_(Note: `torch` is also required but should already be installed from previous sprints. `tokenizers` was also installed previously)._

### 2. Load Model and Tokenizer

We can use the `AutoModelForCausalLM` and `AutoTokenizer` classes from `transformers`. These classes automatically detect the correct architecture and configuration based on the model name (e.g., "gpt2").

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2" # Or "gpt2-medium", "gpt2-large", "gpt2-xl"

print(f"Loading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Loading model for {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name)

# The model and tokenizer are now ready to use.
print("Model and tokenizer loaded successfully!")

# You can inspect the model config:
# print(model.config)
```

This code will download the model weights and tokenizer files (if not already cached) and load them into memory.

### 3. Understanding the Configuration

The `model.config` attribute holds the configuration of the loaded model (e.g., number of layers, hidden size, vocabulary size). Exploring this helps understand the model's architecture.

```python
# Example:
# print(f"Model Config: {model.config}")
# print(f"Vocab size: {model.config.vocab_size}")
# print(f"Number of layers: {model.config.n_layer}")
```

Next step is to implement the text generation logic in `results/01_text_generation.py`.
