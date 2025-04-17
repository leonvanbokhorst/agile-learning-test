# Sprint 12: Evaluation & Comparison Notes

## 1. Generation Script (`generate_text.py`)

- **Location:** [`results/generate_text.py`](../results/generate_text.py)
- **Functionality:**
  - Loads the original pre-trained model (e.g., `gpt2`) and tokenizer.
  - Loads the fine-tuned model and tokenizer from the specified checkpoint directory (e.g., `results/checkpoints/finetuned_model`).
  - Defines a list of prompts for testing.
  - Contains a `generate_text` helper function that takes a model, tokenizer, prompt, and generation parameters (max tokens, temperature, top-k, top-p, do_sample).
    - Sets the model to evaluation mode (`model.eval()`).
    - Tokenizes the prompt and sends it to the device.
    - Calls `model.generate()` with the specified parameters, ensuring `pad_token_id` is handled.
    - Decodes the output tokens back into text.
  - Iterates through the prompts, generating output from both the original and fine-tuned models using the same parameters.
  - Prints the prompt and the two outputs side-by-side for comparison.

## 2. Comparison Results

_(Summary copied from Sprint 12 README)_

**Usage Examples:**

```bash
# Assuming CWD is project root
cd sprints/12_finetune_gpt2_generative/results
# Example 1: Greedy Decoding
python generate_text.py --do-sample False --top-k 1
# Example 2: Sampling with low temperature
python generate_text.py --do-sample True --temperature 0.3 --top-p 0.9
```

**Summary of Observations:**

- **Prompt 1 ("The old house "):**
  - Original: coherent, human-like sentence repetition.
  - Fine-tuned: Latin-like gibberish persists.
- **Prompt 2 ("Lorem ipsum "):**
  - Original: classic "Lorem ipsum ipsum..." repetition.
  - Fine-tuned: fragmented pseudo-Latin, no true repetition.
- **Prompt 3 (" "):** _(Note: Changed from empty string based on user file)_
  - Original: sensible code generation advice.
  - Fine-tuned: repeated Latin-like tokens.

**Conclusion:**
Despite trying both greedy decoding (`do-sample=False, top-k=1`) and nucleus sampling with low temperature (`do-sample=True, temp=0.3, top-p=0.9`), the fine-tuned model did not replicate the expected repeating pattern observed in the original `book.txt` (like "Lorem ipsum"). The fine-tuning appears to have shifted the model's distribution, but not strongly enough or in the specific way needed to mimic that particular stylistic element after only one epoch.

**Potential Next Steps (if goal was style replication):**

- Augment training data with more explicit examples of the desired repeating patterns.
- Fine-tune for more epochs.
- Experiment with different hyperparameters (learning rate, batch size).
