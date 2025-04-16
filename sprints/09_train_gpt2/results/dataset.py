import torch
from torch.utils.data import Dataset
import logging
from pathlib import Path

# Assuming tokenizer.py is copied to the same directory or accessible
# Adjust the import path if necessary
from tokenizer import get_gpt2_tokenizer  # Relative import

# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class TextDataset(Dataset):
    """PyTorch Dataset for loading and tokenizing text data for language modeling."""

    def __init__(
        self, file_path: str | Path, sequence_length: int, tokenizer_name: str = "gpt2"
    ):
        """
        Args:
            file_path (str | Path): Path to the text file.
            sequence_length (int): The length of token sequences to return.
            tokenizer_name (str): The name of the tokenizer to use (default: 'gpt2').
        """
        self.file_path = Path(file_path)
        self.sequence_length = sequence_length
        self.tokenizer_name = tokenizer_name

        logging.info(f"Initializing tokenizer: {self.tokenizer_name}")
        self.tokenizer = get_gpt2_tokenizer()
        self.vocab_size = self.tokenizer.get_vocab_size()
        logging.info(f"Tokenizer vocabulary size: {self.vocab_size}")

        logging.info(f"Loading and tokenizing data from {self.file_path}...")
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                text = f.read()
            # Tokenize the entire text
            # Note: For very large datasets, consider memory-mapped files or streaming
            self.tokens = torch.tensor(
                self.tokenizer.encode(text).ids, dtype=torch.long
            )
            logging.info(f"Loaded {len(self.tokens)} tokens.")
        except FileNotFoundError:
            logging.error(f"Data file not found: {self.file_path}")
            raise
        except Exception as e:
            logging.error(f"Error processing data file {self.file_path}: {e}")
            raise

    def __len__(self):
        """Returns the number of sequences available in the dataset."""
        # We can create a sequence starting at any token index up to the point
        # where a full sequence (input + target) fits.
        return len(self.tokens) - self.sequence_length

    def __getitem__(self, idx):
        """
        Returns a tuple (input_sequence, target_sequence).
        Target sequence is the input sequence shifted by one token.
        """
        # Extract the chunk of tokens for the input sequence
        input_chunk = self.tokens[idx : idx + self.sequence_length]
        # The target sequence is the next token for each input token
        target_chunk = self.tokens[idx + 1 : idx + self.sequence_length + 1]
        return input_chunk, target_chunk


# Example Usage (for testing)
if __name__ == "__main__":
    logging.info("--- Testing TextDataset ---")

    # Assuming prepare_data.py has been run and created the files
    # Adjust path relative to the project root if running this file directly
    # For `python -m sprints.09...` the paths relative to this file are correct
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent / "data"
    train_file_path = data_dir / "train.txt"
    val_file_path = data_dir / "val.txt"
    SEQUENCE_LENGTH = 128  # Example sequence length

    if not train_file_path.exists() or not val_file_path.exists():
        logging.warning("train.txt or val.txt not found. Run prepare_data.py first.")
        logging.warning("Attempting to run prepare_data.py...")
        try:
            # Attempt to run prepare_data relative to the current script
            from . import prepare_data

            prepare_data.download_data(
                prepare_data.DATA_URL, prepare_data.RAW_DATA_FILE
            )
            if prepare_data.RAW_DATA_FILE.exists():
                prepare_data.split_data(
                    prepare_data.RAW_DATA_FILE,
                    prepare_data.TRAIN_FILE,
                    prepare_data.VAL_FILE,
                    prepare_data.TRAIN_SPLIT_RATIO,
                )
            else:
                raise FileNotFoundError("Raw data failed to download.")
        except Exception as e:
            logging.error(f"Failed to run prepare_data automatically: {e}")
            exit(1)

    logging.info(f"Creating training dataset from: {train_file_path}")
    train_dataset = TextDataset(train_file_path, SEQUENCE_LENGTH)

    logging.info(f"Creating validation dataset from: {val_file_path}")
    val_dataset = TextDataset(val_file_path, SEQUENCE_LENGTH)

    logging.info(f"Number of training sequences: {len(train_dataset)}")
    logging.info(f"Number of validation sequences: {len(val_dataset)}")

    # Check a sample
    if len(train_dataset) > 0:
        sample_idx = 0
        x, y = train_dataset[sample_idx]
        logging.info(f"Sample {sample_idx} from training set:")
        logging.info(f"  Input shape: {x.shape}, dtype: {x.dtype}")
        logging.info(f"  Target shape: {y.shape}, dtype: {y.dtype}")
        # Decode for verification (optional, can be slow)
        # logging.info(f"  Input tokens: {x.tolist()}")
        # logging.info(f"  Target tokens: {y.tolist()}")
        # logging.info(f"  Decoded Input: {train_dataset.tokenizer.decode(x.tolist())}")
        # logging.info(f"  Decoded Target: {train_dataset.tokenizer.decode(y.tolist())}")

        # Verify the shift
        assert torch.equal(x[1:], y[:-1]), "Input/Target shift is incorrect!"
        logging.info("Input/Target shift verified.")

    logging.info("--- TextDataset Test Finished ---")
