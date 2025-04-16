# Sprint 9: Data Pipeline Notes

This document covers the steps involved in preparing the data for training the GPT-2 model in Sprint 9.

## 1. Data Source

- **Dataset:** TinyShakespeare
- **Source URL:** `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
- **Script:** `sprints/09_train_gpt2/results/prepare_data.py`

## 2. Data Download & Preparation

- The `prepare_data.py` script handles downloading the raw `input.txt` file if it doesn't exist locally.
- It saves the raw data to `sprints/09_train_gpt2/data/tinyshakespeare.txt`.
- Error handling is included for download issues.

## 3. Train/Validation Split

- The script splits the raw text data into training and validation sets.
- **Split Ratio:** 90% Training, 10% Validation (defined by `TRAIN_SPLIT_RATIO` in `prepare_data.py`).
- **Output Files:**
  - Training data: `sprints/09_train_gpt2/data/train.txt`
  - Validation data: `sprints/09_train_gpt2/data/val.txt`

## 4. Tokenization

- **Tokenizer:** Standard GPT-2 tokenizer loaded using the `tokenizers` library via `sprints/09_train_gpt2/results/tokenizer.py` (copied from Sprint 8).
- **Vocabulary Size:** 50257 (as reported by the tokenizer).
- Tokenization is performed within the `TextDataset` class.

## 5. PyTorch Dataset (`TextDataset`)

- **Script:** `sprints/09_train_gpt2/results/dataset.py`
- **Class:** `TextDataset(Dataset)`
- **Initialization:**
  - Takes the path to the data file (`train.txt` or `val.txt`) and the desired `sequence_length`.
  - Loads the tokenizer.
  - Reads the entire text file content.
  - Tokenizes the text into a single `torch.long` tensor (`self.tokens`).
- **`__len__`:** Returns the total number of possible starting positions for a sequence of `sequence_length + 1` tokens (`len(self.tokens) - self.sequence_length`).
- **`__getitem__`:**
  - Takes an index `idx`.
  - Returns a tuple `(x, y)`:
    - `x`: Input sequence tensor (`self.tokens[idx : idx + self.sequence_length]`).
    - `y`: Target sequence tensor (`self.tokens[idx + 1 : idx + self.sequence_length + 1]`), shifted by one token relative to `x`.
  - This structure is standard for autoregressive language model training.
- **Considerations:**
  - The current implementation loads the entire tokenized dataset into memory. For very large datasets, this might be inefficient. Alternatives include memory-mapping or streaming data loading.

## 6. DataLoader Setup

- The `DataLoader` will wrap the `TextDataset` instances (train and validation).
- Key parameters will include `batch_size`, `shuffle=True` (for training), and potentially `num_workers` for parallel loading.
- This setup will occur in the main training script (`train_gpt2.py`).

# Data Pipeline Implementation Notes

## Overview

We implemented a text data pipeline for training our GPT-2 model, focusing on efficient data loading and preprocessing.

## Key Components

### TextDataset

- Implemented a custom PyTorch Dataset class for handling sequential text data
- Handles tokenization, sequence creation, and padding
- Supports both training and validation data
- Efficient memory usage through lazy loading

### DataLoader Configuration

- Batch size: Configurable (default 32)
- Shuffling: Enabled for training, disabled for validation
- Pin memory: Enabled for CUDA devices
- Number of workers: Configurable (default 0 for simplicity)

## Implementation Decisions

### Sequence Length

- Configurable sequence length (default 256)
- Shorter sequences (64) used for testing
- Longer sequences possible for better context

### Tokenization

- Using GPT-2 tokenizer from previous sprint
- Handles special tokens (BOS, EOS, PAD)
- Efficient vocabulary management

### Memory Management

- Lazy loading of text data
- Efficient batching to minimize memory usage
- Proper cleanup of unused tensors

## Challenges & Solutions

1. **Memory Usage**

   - Problem: Large datasets could consume too much memory
   - Solution: Implemented lazy loading and efficient batching

2. **Sequence Padding**

   - Problem: Variable length sequences need padding
   - Solution: Implemented proper padding with attention masks

3. **Data Loading Speed**
   - Problem: Slow data loading could bottleneck training
   - Solution: Added pin_memory for CUDA and configurable workers

## Future Improvements

1. **Data Augmentation**

   - Could add text augmentation techniques
   - Implement dynamic masking for better training

2. **Streaming Support**

   - Add support for streaming very large datasets
   - Implement proper checkpointing for streaming

3. **Multi-GPU Support**
   - Add distributed data loading
   - Implement proper sharding

## Code References

- `dataset.py`: Main dataset implementation
- `train_gpt2.py`: Data loading configuration
