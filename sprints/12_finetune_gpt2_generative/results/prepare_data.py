import os
import argparse
import numpy as np
from transformers import AutoTokenizer
import logging
from typing import List

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def prepare_data(
    input_path: str,
    output_dir: str,
    val_split: float = 0.1,
    tokenizer_name: str = "gpt2",
) -> None:
    """
    Reads a text file, tokenizes it using a Hugging Face tokenizer,
    splits it into training and validation sets, and saves the token IDs
    as numpy binary files.

    Args:
        input_path: Path to the input text file (e.g., 'book.txt').
        output_dir: Directory to save the tokenized 'train.bin' and 'val.bin' files.
        val_split: Fraction of the data to use for the validation set (default: 0.1).
        tokenizer_name: Name of the Hugging Face tokenizer to use (default: 'gpt2').
    """
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

    # Load tokenizer
    logging.info(f"Loading tokenizer: {tokenizer_name}...")
    # Use trust_remote_code=True if necessary for certain tokenizers, but generally okay for gpt2
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    logging.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    # Read the input file
    logging.info(f"Reading input file: {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    logging.info(f"Read {len(text):,} characters from input file.")

    # Tokenize the entire text
    logging.info("Tokenizing text...")
    # We encode directly to IDs. Add special tokens if needed, but usually not for pre-training/fine-tuning LM.
    all_tokens: List[int] = tokenizer.encode(text)
    logging.info(f"Tokenized into {len(all_tokens):,} tokens.")

    # Convert to numpy array
    all_tokens_np = np.array(all_tokens, dtype=np.uint16)  # GPT-2 vocab fits in uint16

    # Split data
    n = len(all_tokens_np)
    n_val = int(n * val_split)
    n_train = n - n_val

    train_data = all_tokens_np[:n_train]
    val_data = all_tokens_np[n_train:]

    logging.info(
        f"Splitting data: {n_train:,} tokens for training, {n_val:,} tokens for validation."
    )

    # Save to file
    train_output_path = os.path.join(output_dir, "train.bin")
    val_output_path = os.path.join(output_dir, "val.bin")

    logging.info(f"Saving training tokens to {train_output_path}...")
    train_data.tofile(train_output_path)

    logging.info(f"Saving validation tokens to {val_output_path}...")
    val_data.tofile(val_output_path)

    logging.info("Data preparation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare text data for fine-tuning GPT-2."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/book.txt",
        help="Path to the input text file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save the tokenized train.bin and val.bin files.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data for validation set (default: 0.1).",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="gpt2",
        help="Hugging Face tokenizer name (default: gpt2).",
    )

    args = parser.parse_args()

    # Adjust relative paths to be relative to the script location if needed,
    # or assume they are relative to the CWD where the script is run.
    # For simplicity, let's assume relative to CWD.

    prepare_data(args.input_path, args.output_dir, args.val_split, args.tokenizer_name)
