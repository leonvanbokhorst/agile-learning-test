from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

import os
from pathlib import Path

# Default cache directory for Hugging Face models/tokenizers
DEFAULT_CACHE_DIR = Path(os.getenv("HF_HOME", Path.home() / ".cache/huggingface"))
TOKENIZER_CACHE_DIR = DEFAULT_CACHE_DIR / "hub"

# Standard GPT-2 tokenizer files (usually found in cache after download)
# We'll try to load it directly first, which handles caching.
GPT2_TOKENIZER_ID = "gpt2"


def get_gpt2_tokenizer() -> Tokenizer:
    """Loads the standard GPT-2 tokenizer from Hugging Face Hub (or cache).

    Returns:
        A Tokenizer instance configured for GPT-2.

    Raises:
        RuntimeError: If the tokenizer cannot be loaded.
    """
    try:
        # Tokenizer.from_pretrained handles downloading and caching
        tokenizer = Tokenizer.from_pretrained(GPT2_TOKENIZER_ID)
        print(
            f"GPT-2 Tokenizer loaded successfully (vocab size: {tokenizer.get_vocab_size()})"
        )

        # Optional: Configure template processing if needed for specific tasks,
        # but the base tokenizer usually works well for generation.
        # tokenizer.post_processor = TemplateProcessing(...)

        return tokenizer

    except Exception as e:
        print(f"Error loading GPT-2 tokenizer '{GPT2_TOKENIZER_ID}': {e}")
        print("Please ensure you have internet connectivity or the model is cached.")
        # You might need to run `huggingface-cli login` or set environment variables
        # if you encounter authentication issues.
        raise RuntimeError(f"Failed to load tokenizer '{GPT2_TOKENIZER_ID}'") from e


# --- Basic Test --- #
if __name__ == "__main__":
    print("--- Testing GPT-2 Tokenizer Loading ---")
    try:
        gpt2_tokenizer = get_gpt2_tokenizer()

        # Test encoding
        text_sample = "Hello, world! This is a test."
        print(f"\nEncoding text: '{text_sample}'")
        encoding = gpt2_tokenizer.encode(text_sample)

        print(f"  -> Tokens: {encoding.tokens}")
        print(f"  -> IDs: {encoding.ids}")
        print(f"  -> Attention Mask: {encoding.attention_mask}")

        # Test decoding
        encoded_ids = encoding.ids
        print(f"\nDecoding IDs: {encoded_ids}")
        decoded_text = gpt2_tokenizer.decode(encoded_ids)
        print(f"  -> Decoded Text: '{decoded_text}'")

        # Verify vocabulary size matches config default if possible
        # from .config import GPTConfig # Assuming config.py is available
        # default_config = GPTConfig()
        # if gpt2_tokenizer.get_vocab_size() != default_config.vocab_size:
        #     print(f"\nWarning: Tokenizer vocab size ({gpt2_tokenizer.get_vocab_size()}) "
        #           f"does not match default config ({default_config.vocab_size}). "
        #           f"Ensure model config uses the correct vocab size.")
        # else:
        #     print("\nTokenizer vocab size matches default config.")

        print("\nTokenizer tests passed.")

    except RuntimeError as e:
        print(f"\nError during tokenizer test: {e}")
    except ImportError:
        print("Could not import GPTConfig, skipping vocab size comparison.")
