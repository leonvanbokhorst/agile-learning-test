# Sprint 12: Fine-tuning Setup Notes

## 1. Fine-tuning: Generative vs. Classification (for Novices)

Imagine we have a powerful language model like GPT-2 that we downloaded. It's already been trained on tons of text from the internet, so it "knows" English pretty well. We call this the **pre-trained model**.

Fine-tuning is like giving this smart model some specialized coaching for a specific job.

**A. Fine-tuning for Classification (Like Sprint 11):**

- **Goal:** Teach the model to _categorize_ text. For example, is this news article "Real" or "Fake"? Is this movie review "Positive" or "Negative"?
- **How it works:**
  1.  We take the pre-trained model (which understands language).
  2.  We _add a new, small part_ on top – a "classification head". Think of it like adding a specific decision-making module.
  3.  We show the model examples of text _and_ their correct category (e.g., "This news is fake", "This review is positive").
  4.  We train _mostly just the new classification head_ (and maybe gently tweak the rest of the model) to get good at assigning the right category.
- **Analogy:** You have someone who understands English perfectly. You give them a stack of documents and just ask them to put each one into either a "YES" pile or a "NO" pile based on some criteria.
- **Output:** The model gives you a category label (like "Fake News" or "Positive Sentiment").
- **Hugging Face Model:** We typically use `AutoModelForSequenceClassification`.

**B. Fine-tuning for Generation (This Sprint!):**

- **Goal:** Teach the model to _generate text_ that sounds like a specific style or comes from a specific domain. For example, make it write poetry like Shakespeare, or answer questions like a specific character, or write code documentation like in our `book.txt`.
- **How it works:**
  1.  We take the same pre-trained model.
  2.  We _don't_ usually add a new head. We use the model's original ability to predict the next word.
  3.  We show the model examples of the _specific style_ we want it to learn (like our `book.txt`).
  4.  We continue the model's original training process (predicting the next word) but _only using our specialized dataset_. This adjusts the model's internal wiring to make it prefer generating text similar to what it's seeing now.
- **Analogy:** You have someone who understands English perfectly. You make them read _only_ Shakespeare for a while. After some time, when you ask them to write something, their writing style will start sounding more like Shakespeare.
- **Output:** The model generates new text that (hopefully) mimics the style/content of the fine-tuning data.
- **Hugging Face Model:** We typically use `AutoModelForCausalLM` (for models like GPT-2) or `AutoModelForSeq2SeqLM` (for models like T5/BART).

**In simple terms:**

| Feature           | Classification Fine-tuning | Generative Fine-tuning (This Sprint) |
| :---------------- | :------------------------- | :----------------------------------- |
| **Main Goal**     | Categorize Text            | Generate Text in a Specific Style    |
| **Training Data** | Text + Category Labels     | Text in the Target Style             |
| **Training Task** | Learn to Assign Labels     | Continue Predicting Next Word        |
| **Model Head**    | Usually Adds a New Head    | Uses Original Language Model Head    |
| **Final Output**  | A Category/Label           | New Text                             |

This sprint, we are doing **generative fine-tuning** on `book.txt` using `AutoModelForCausalLM`.

## 2. Code Setup (`finetune_generative.py`)

- **Location:** [`results/finetune_generative.py`](../results/finetune_generative.py)
- **Key Setup Steps Implemented:**
  - **Argument Parsing:** Uses `argparse` to define command-line arguments for model name, data paths, hyperparameters (epochs, batch size, learning rate, etc.), and runtime settings (output directory, number of workers).
  - **Logging:** Basic logging is configured to show progress and information.
  - **Device Selection:** A `get_device()` function checks for CUDA (NVIDIA GPU), MPS (Apple Silicon GPU), or defaults to CPU, ensuring the model runs on the best available hardware.
  - **Tokenizer Loading:** Loads the specified tokenizer (default `gpt2`) using `AutoTokenizer.from_pretrained()`. It also sets the `pad_token` to the `eos_token` if it's not already defined, which is a common practice for GPT-2 generation/fine-tuning.
  - **Model Loading:** Loads the specified pre-trained causal language model (default `gpt2`) using `AutoModelForCausalLM.from_pretrained()`. The model is then moved to the selected device.
  - **Dataset/DataLoader:** Initializes `TextDataset` (from `dataset.py`) for both training (`train.bin`) and validation (`val.bin`) data. Wraps these datasets in `DataLoader` instances for efficient batching, shuffling (for training), and potential parallel loading (`num_workers`).
  - **Optimizer:** Initializes the `AdamW` optimizer, a standard choice for transformer models, passing the model's parameters and configured learning rate/weight decay.
  - **Scheduler:** Sets up a learning rate scheduler (default `linear` warmup and decay) using `transformers.get_scheduler`. This adjusts the learning rate during training, which often improves convergence. The total number of training steps and warmup steps are calculated based on the number of epochs and the size of the training DataLoader.
  - **Output Directory:** Ensures the specified output directory for checkpoints (`--output-dir`) exists, creating it if necessary.
