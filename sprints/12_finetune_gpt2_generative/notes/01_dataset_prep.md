# Sprint 12: Dataset Selection & Preparation Notes

## 1. Dataset Choice

- **Dataset:** The user provided a custom text file named `book.txt` located in `sprints/12_finetune_gpt2_generative/results/data/`.
- **Goal:** Fine-tune a pre-trained GPT-2 model to potentially adopt the writing style or domain-specific language present in `book.txt`.
- **Initial Inspection:** The file contains text, including numerous newline characters (`\n`). After discussion, the decision was made **not** to normalize or remove excessive newlines during preprocessing for the initial fine-tuning attempt. We will observe the generation output later and revisit if necessary.

## 2. Preprocessing Script (`prepare_data.py`)

- **Location:** [`results/prepare_data.py`](../results/prepare_data.py)
- **Functionality:**
  - Takes the input text file path (`--input-path`), output directory (`--output-dir`), validation split fraction (`--val-split`), and tokenizer name (`--tokenizer-name`) as arguments.
  - Reads the entire input text file (`book.txt`).
  - Loads the specified Hugging Face tokenizer (defaulting to `gpt2`).
  - Encodes the entire text into a single sequence of token IDs using `tokenizer.encode()`.
  - Splits the token sequence into training and validation sets based on the `--val-split` fraction (default 0.1, meaning 90% train, 10% validation). The split is done sequentially (first 90% for train, last 10% for val).
  - Saves the training token IDs to `<output_dir>/train.bin` and validation token IDs to `<output_dir>/val.bin`.
  - The token IDs are saved as `numpy` arrays with `dtype=np.uint16`, as the GPT-2 vocabulary size (50257) fits within this data type, saving disk space compared to `int32` or `int64`.
- **Execution:**
  ```bash
  python sprints/12_finetune_gpt2_generative/results/prepare_data.py \
      --input-path sprints/12_finetune_gpt2_generative/results/data/book.txt \
      --output-dir sprints/12_finetune_gpt2_generative/results/data
  ```
- **Output Files:**
  - `sprints/12_finetune_gpt2_generative/results/data/train.bin`
  - `sprints/12_finetune_gpt2_generative/results/data/val.bin`

## 3. Dataset Loading (`dataset.py`)

- **Location:** [`results/dataset.py`](../results/dataset.py)
- **Class:** `TextDataset(Dataset)`
  - **Initialization (`__init__`)**:
    - Takes the path to a `.bin` data file (e.g., `train.bin` or `val.bin`) and the desired `block_size` (sequence length for the model).
    - Uses `numpy.memmap` to load the token IDs from the `.bin` file. Memory mapping is efficient, especially for large datasets, as it doesn't load the entire file into RAM at once.
  - **Length (`__len__`)**:
    - Returns the total number of possible starting positions for a sequence chunk in the dataset. Calculated as `len(data) - block_size`.
  - **Get Item (`__getitem__`)**:
    - Given an index `idx`, it retrieves a chunk of `block_size + 1` tokens starting from `idx`.
    - Creates the input sequence `x` by taking the first `block_size` tokens (`chunk[:-1]`).
    - Creates the target sequence `y` by taking the last `block_size` tokens, shifted one position (`chunk[1:]`). This sets up the standard causal language modeling objective (predict the next token).
    - Converts `x` and `y` to `torch.LongTensor` (`int64`), which is commonly required by embedding layers.
- **DataLoader:** The `Dataset` is designed to be used with `torch.utils.data.DataLoader` for batching and shuffling during training.
