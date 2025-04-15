import os
import requests
import logging
from pathlib import Path

# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DATA_FILE = DATA_DIR / "tinyshakespeare.txt"
TRAIN_FILE = DATA_DIR / "train.txt"
VAL_FILE = DATA_DIR / "val.txt"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
TRAIN_SPLIT_RATIO = 0.9


def download_data(url: str, dest_path: Path):
    """Downloads data from url to dest_path if it doesn't exist."""
    if dest_path.exists():
        logging.info(f"Data already exists at {dest_path}. Skipping download.")
        return

    logging.info(f"Downloading data from {url} to {dest_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        dest_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info("Download complete.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading data: {e}")
        # Clean up incomplete download if it exists
        if dest_path.exists():
            os.remove(dest_path)
        raise  # Re-raise the exception after logging


def split_data(raw_file: Path, train_file: Path, val_file: Path, ratio: float):
    """Reads raw data, splits it, and saves train/validation sets."""
    logging.info(f"Reading raw data from {raw_file}...")
    try:
        with open(raw_file, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        logging.error(f"Raw data file not found at {raw_file}. Please download first.")
        return
    except Exception as e:
        logging.error(f"Error reading raw data file: {e}")
        return

    n = len(text)
    train_size = int(n * ratio)
    train_data = text[:train_size]
    val_data = text[train_size:]

    logging.info(
        f"Splitting data: {len(train_data)} training characters, {len(val_data)} validation characters."
    )

    try:
        logging.info(f"Writing training data to {train_file}...")
        with open(train_file, "w", encoding="utf-8") as f:
            f.write(train_data)
        logging.info(f"Writing validation data to {val_file}...")
        with open(val_file, "w", encoding="utf-8") as f:
            f.write(val_data)
        logging.info("Data splitting complete.")
    except IOError as e:
        logging.error(f"Error writing split data files: {e}")


if __name__ == "__main__":
    logging.info("--- Starting Data Preparation ---")

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download data
    try:
        download_data(DATA_URL, RAW_DATA_FILE)
    except Exception:
        logging.error("Failed to download data. Exiting.")
        exit(1)  # Exit if download fails

    # Step 2: Split data
    if RAW_DATA_FILE.exists():
        split_data(RAW_DATA_FILE, TRAIN_FILE, VAL_FILE, TRAIN_SPLIT_RATIO)
    else:
        logging.warning("Raw data file does not exist, cannot split data.")

    logging.info("--- Data Preparation Finished ---")
