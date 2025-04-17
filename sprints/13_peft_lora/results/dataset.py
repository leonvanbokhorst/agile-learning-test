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
    Used for PEFT/LoRA fine-tuning in Sprint 13.
    """

    def __init__(self, data_file: str, block_size: int):
        """
        Args:
            data_file (str): Path to the .bin file containing token IDs.
                             (Should point to the data prepared in Sprint 12)
            block_size (int): The length of the token sequences to return.
                             Input and target sequences will have this length.
        """
        super().__init__()
        self.block_size = block_size
        logging.info(f"[Sprint 13 Dataset] Loading data from: {data_file}")

        if not os.path.exists(data_file):
            logging.error(f"[Sprint 13 Dataset] Data file not found: {data_file}")
            # Provide hint about where the data should be from Sprint 12
            s12_data_path = "sprints/12_finetune_gpt2_generative/results/data"
            logging.error(
                f"Ensure the data was prepared in Sprint 12 ({s12_data_path})"
            )
            raise FileNotFoundError(f"Data file not found: {data_file}")

        # Load the data using memory mapping for efficiency with large files
        # The data is expected to be uint16 tokens as saved by prepare_data.py in Sprint 12
        try:
            self.data = np.memmap(data_file, dtype=np.uint16, mode="r")
            logging.info(f"[Sprint 13 Dataset] Loaded {len(self.data):,} tokens.")
        except Exception as e:
            logging.error(
                f"[Sprint 13 Dataset] Failed to load data from {data_file}: {e}"
            )
            raise

        if len(self.data) <= block_size:
            logging.warning(
                f"[Sprint 13 Dataset] Dataset size ({len(self.data)}) is not greater "
                f"than block_size ({block_size}). This dataset will yield 0 examples."
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
        try:
            x = torch.from_numpy(chunk[:-1].astype(np.int64))
            y = torch.from_numpy(chunk[1:].astype(np.int64))
        except Exception as e:
            logging.error(
                f"[Sprint 13 Dataset] Error converting chunk to tensor at index {idx}: {e}"
            )
            # Return dummy tensors or raise an error, depending on desired robustness
            # For simplicity, we'll raise here
            raise RuntimeError(f"Failed to process data chunk at index {idx}") from e

        return x, y


# Example Usage & Testing (Self-contained within Sprint 13)
if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: Point this to the data generated in Sprint 12
    # Adjust the relative path if needed based on where you run this script
    # Assuming running from the root of the project
    SPRINT12_DATA_DIR = "sprints/12_finetune_gpt2_generative/results/data"
    DEFAULT_TRAIN_FILE = os.path.join(SPRINT12_DATA_DIR, "train.bin")
    DEFAULT_VAL_FILE = os.path.join(SPRINT12_DATA_DIR, "val.bin")

    # Use environment variables or command-line args for flexibility? For now, hardcode.
    DATA_FILE_TO_TEST = DEFAULT_TRAIN_FILE  # or DEFAULT_VAL_FILE
    BLOCK_SIZE = 128  # Should match the intended block size for LoRA training
    BATCH_SIZE = 4  # Example batch size for DataLoader test

    logging.info("--- [Sprint 13] Testing TextDataset ---")

    if not os.path.exists(DATA_FILE_TO_TEST):
        logging.error(f"TEST FAILED: Data file '{DATA_FILE_TO_TEST}' not found.")
        logging.error("Please ensure you have run the data preparation in Sprint 12")
        logging.error(f"Expected location: {SPRINT12_DATA_DIR}")
    else:
        try:
            dataset = TextDataset(data_file=DATA_FILE_TO_TEST, block_size=BLOCK_SIZE)
            logging.info(f"Dataset length: {len(dataset)}")

            if len(dataset) > 0:
                # Test __getitem__
                x, y = dataset[0]  # Get the first item
                logging.info(f"First item shapes: x={x.shape}, y={y.shape}")
                logging.info(f"First item types: x={x.dtype}, y={y.dtype}")
                assert x.shape == (BLOCK_SIZE,)
                assert y.shape == (BLOCK_SIZE,)
                assert x.dtype == torch.int64
                assert y.dtype == torch.int64

                # Test DataLoader
                logging.info("--- Testing DataLoader ---")
                # Use num_workers=0 for simplicity in testing, adjust for real training
                dataloader = DataLoader(
                    dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
                )
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

                logging.info(
                    "[Sprint 13] Dataset and DataLoader tests passed (basic checks)."
                )
            else:
                logging.warning("Dataset is empty, skipping DataLoader tests.")

        except FileNotFoundError as e:
            logging.error(f"Test failed during dataset instantiation: {e}")
        except Exception as e:
            logging.error(
                f"An unexpected error occurred during testing: {e}", exc_info=True
            )
