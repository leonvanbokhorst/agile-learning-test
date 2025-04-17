import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_device() -> torch.device:
    """Gets the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        logging.info("CUDA detected, using GPU.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        logging.info("MPS detected, using Apple Silicon GPU.")
        return torch.device("mps")
    else:
        logging.info("No GPU detected, using CPU.")
        return torch.device("cpu")


def generate_text(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=50,
    temperature=0.7,
    top_k=50,
    do_sample=True,
):
    """Generates text from a given model and tokenizer."""
    model.eval()  # Ensure model is in eval mode
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids

    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=(
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            ),  # Ensure pad token is set
        )

    # Decode the generated tokens
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def main(args):
    """Loads models and generates text for comparison."""
    device = get_device()

    # --- Load Original Model --- (Make sure this matches the base model used for fine-tuning)
    logging.info(f"Loading original tokenizer: {args.original_model_name}")
    original_tokenizer = AutoTokenizer.from_pretrained(args.original_model_name)
    if original_tokenizer.pad_token is None:
        original_tokenizer.pad_token = original_tokenizer.eos_token

    logging.info(f"Loading original model: {args.original_model_name}")
    original_model = AutoModelForCausalLM.from_pretrained(args.original_model_name)
    original_model.to(device)
    logging.info("Original model loaded.")

    # --- Load Fine-tuned Model ---
    if not os.path.exists(args.finetuned_model_path) or not os.listdir(
        args.finetuned_model_path
    ):
        logging.error(
            f"Fine-tuned model directory not found or empty: {args.finetuned_model_path}"
        )
        logging.error(
            "Please ensure the fine-tuning script ran successfully and saved the model."
        )
        return

    logging.info(f"Loading fine-tuned tokenizer from: {args.finetuned_model_path}")
    try:
        finetuned_tokenizer = AutoTokenizer.from_pretrained(args.finetuned_model_path)
    except Exception as e:
        logging.error(f"Error loading fine-tuned tokenizer: {e}")
        return

    logging.info(f"Loading fine-tuned model from: {args.finetuned_model_path}")
    try:
        finetuned_model = AutoModelForCausalLM.from_pretrained(
            args.finetuned_model_path
        )
        finetuned_model.to(device)
    except Exception as e:
        logging.error(f"Error loading fine-tuned model: {e}")
        return
    logging.info("Fine-tuned model loaded.")

    # --- Define Prompts ---
    prompts = [
        "The old house stood on a hill overlooking the town. ",
        "Chapter 1: It was a dark and stormy night",
        "To be, or not to be, that is the",
        # Add more prompts relevant to your book.txt content if possible
    ]

    # --- Generate and Compare ---
    print("\n--- Text Generation Comparison ---")
    for i, prompt in enumerate(prompts):
        print(f"\n--- Prompt {i+1} ---")
        print(f"Prompt: {prompt}")

        # Original Model Generation
        print("\nOriginal Model Output:")
        try:
            original_output = generate_text(
                original_model,
                original_tokenizer,
                prompt,
                device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                do_sample=args.do_sample,
            )
            print(original_output)
        except Exception as e:
            print(f"Error generating from original model: {e}")

        # Fine-tuned Model Generation
        print("\nFine-tuned Model Output:")
        try:
            finetuned_output = generate_text(
                finetuned_model,
                finetuned_tokenizer,
                prompt,
                device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                do_sample=args.do_sample,
            )
            print(finetuned_output)
        except Exception as e:
            print(f"Error generating from fine-tuned model: {e}")

        print("-" * 40)

    logging.info("Generation comparison complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text from original and fine-tuned models for comparison."
    )

    parser.add_argument(
        "--original-model-name",
        type=str,
        default="gpt2",
        help="Name of the original pre-trained model used as base.",
    )
    parser.add_argument(
        "--finetuned-model-path",
        type=str,
        default="checkpoints/finetuned_model",
        help="Path to the directory containing the fine-tuned model checkpoint.",
    )

    # Generation parameters
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=75,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature."
    )
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling.")
    parser.add_argument(
        "--do-sample",
        type=bool,
        default=True,
        help="Whether to use sampling; set to False for greedy decoding.",
    )

    args = parser.parse_args()

    # Adjust fine-tuned path relative to the sprint directory
    base_dir = os.path.join("sprints", "12_finetune_gpt2_generative", "results")
    args.finetuned_model_path = os.path.join(base_dir, args.finetuned_model_path)

    main(args)
