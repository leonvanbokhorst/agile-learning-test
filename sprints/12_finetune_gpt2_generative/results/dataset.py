import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import logging
from typing import Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class TextDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset for loading tokenized text data from a binary file.

    Reads a binary file containing token IDs (uint16) and serves
    sequences of a specified block size for causal language modeling.
    """

    def __init__(self, data_file: str, block_size: int):
        """
        Args:
            data_file (str): Path to the .bin file containing token IDs.
            block_size (int): The length of the token sequences to return.
                             Input and target sequences will have this length.
        """
        super().__init__()
        self.block_size = block_size
        logging.info(f"Loading data from: {data_file}")

        if not os.path.exists(data_file):
            logging.error(f"Data file not found: {data_file}")
            raise FileNotFoundError(f"Data file not found: {data_file}")

        # Load the data using memory mapping for efficiency with large files
        # The data is expected to be uint16 tokens as saved by prepare_data.py
        self.data = np.memmap(data_file, dtype=np.uint16, mode="r")
        logging.info(f"Loaded {len(self.data):,} tokens.")

        if len(self.data) <= block_size:
            logging.warning(
                f"Dataset size ({len(self.data)}) is not greater than block_size ({block_size}). "
                f"This dataset will yield 0 examples."
            )

    def __len__(self) -> int:
        """Returns the number of sequences of block_size that can be generated.

        We subtract block_size because the last block needs a subsequent token
        to form the target.
        """
        # Ensure length is at least 0
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gets a sequence chunk for training.

        Args:
            idx (int): The starting index of the sequence chunk.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - x: The input sequence (indices from idx to idx + block_size - 1).
                - y: The target sequence (indices from idx + 1 to idx + block_size).
        """
        # Grab a chunk of block_size + 1 tokens starting at idx
        # We need block_size + 1 because the target (y) is shifted by one
        chunk = self.data[idx : idx + self.block_size + 1]

        # Convert chunk to PyTorch tensors
        # We use LongTensor because token IDs are usually treated as categorical indices
        # which often require 64-bit integers (long) for embedding layers etc.
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))

        return x, y


# Example Usage & Testing
if __name__ == "__main__":
    # Assume prepare_data.py has run and created these files
    # Create dummy data for testing if files don't exist
    DUMMY_DATA_DIR = "data_test"
    DUMMY_TRAIN_FILE = os.path.join(DUMMY_DATA_DIR, "train.bin")
    DUMMY_VAL_FILE = os.path.join(DUMMY_DATA_DIR, "val.bin")
    BLOCK_SIZE = 128
    BATCH_SIZE = 4

    if not os.path.exists(DUMMY_TRAIN_FILE) or not os.path.exists(DUMMY_VAL_FILE):
        logging.info("Dummy data files not found, creating them...")
        os.makedirs(DUMMY_DATA_DIR, exist_ok=True)
        # Create ~1000 tokens for train, ~100 for val
        dummy_train_tokens = np.random.randint(
            0, 50257, size=(BLOCK_SIZE * 8 + 1), dtype=np.uint16
        )
        dummy_val_tokens = np.random.randint(
            0, 50257, size=(BLOCK_SIZE + 1), dtype=np.uint16
        )
        dummy_train_tokens.tofile(DUMMY_TRAIN_FILE)
        dummy_val_tokens.tofile(DUMMY_VAL_FILE)
        logging.info(f"Created dummy files: {DUMMY_TRAIN_FILE}, {DUMMY_VAL_FILE}")
        data_file_to_test = DUMMY_TRAIN_FILE
    else:
        # If the actual data exists from prepare_data, use it for a quick test
        # Make sure these paths match where prepare_data.py saved the files!
        ACTUAL_DATA_DIR = "sprints/12_finetune_gpt2_generative/results/data"
        ACTUAL_TRAIN_FILE = os.path.join(ACTUAL_DATA_DIR, "train.bin")
        if os.path.exists(ACTUAL_TRAIN_FILE):
            data_file_to_test = ACTUAL_TRAIN_FILE
            logging.info(f"Using actual data file for testing: {data_file_to_test}")
        else:
            logging.warning(
                f"Actual data file {ACTUAL_TRAIN_FILE} not found, using dummy data."
            )
            data_file_to_test = DUMMY_TRAIN_FILE

    logging.info("--- Testing TextDataset ---")
    try:
        dataset = TextDataset(data_file=data_file_to_test, block_size=BLOCK_SIZE)
        logging.info(f"Dataset length: {len(dataset)}")

        if len(dataset) > 0:
            x, y = dataset[0]  # Get the first item
            logging.info(f"First item shapes: x={x.shape}, y={y.shape}")
            logging.info(f"First item types: x={x.dtype}, y={y.dtype}")
            assert x.shape == (BLOCK_SIZE,)
            assert y.shape == (BLOCK_SIZE,)
            assert x.dtype == torch.int64
            assert y.dtype == torch.int64

            logging.info("--- Testing DataLoader ---")
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            logging.info(f"Number of batches: {len(dataloader)}")

            # Get the first batch
            first_batch = next(iter(dataloader))
            xb, yb = first_batch
            logging.info(f"First batch shapes: xb={xb.shape}, yb={yb.shape}")
            logging.info(f"First batch types: xb={xb.dtype}, yb={yb.dtype}")
            assert xb.shape == (BATCH_SIZE, BLOCK_SIZE)
            assert yb.shape == (BATCH_SIZE, BLOCK_SIZE)
            assert xb.dtype == torch.int64
            assert yb.dtype == torch.int64

            logging.info("Dataset and DataLoader tests passed (basic checks).")
        else:
            logging.warning("Dataset is empty, skipping DataLoader tests.")

    except FileNotFoundError as e:
        logging.error(f"Test failed: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during testing: {e}")
        # Clean up dummy files if they were created
        if os.path.exists(DUMMY_TRAIN_FILE):
            os.remove(DUMMY_TRAIN_FILE)
        if os.path.exists(DUMMY_VAL_FILE):
            os.remove(DUMMY_VAL_FILE)
        if os.path.exists(DUMMY_DATA_DIR):
            try:
                os.rmdir(DUMMY_DATA_DIR)  # Only removes if empty
            except OSError:
                pass  # Directory might not be empty if other files exist
        raise e

    # Optional: Clean up dummy files after successful test run
    # if os.path.exists(DUMMY_TRAIN_FILE):
    #     os.remove(DUMMY_TRAIN_FILE)
    # if os.path.exists(DUMMY_VAL_FILE):
    #     os.remove(DUMMY_VAL_FILE)
    # if os.path.exists(DUMMY_DATA_DIR):
    #     try:
    #         os.rmdir(DUMMY_DATA_DIR)
    #     except OSError:
    #         pass
