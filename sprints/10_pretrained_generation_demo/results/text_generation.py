import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device():
    """Gets the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Note: MPS support might vary depending on torch version and hardware.
        print("MPS backend detected. NOTE: Ensure you have a compatible torch version.")
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_model_and_tokenizer(model_name: str = "gpt2"):
    """Loads the specified pre-trained model and tokenizer."""
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading model for {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("Model and tokenizer loaded successfully!")

    # Add special tokens if they don't exist (GPT2 often doesn't have a PAD token)
    if tokenizer.pad_token is None:
        print("Adding PAD token to tokenizer.")
        # Option 1: Use EOS token as PAD token
        # tokenizer.pad_token = tokenizer.eos_token
        # Option 2: Add a new PAD token (might require resizing model embeddings)
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        # Important: Resize model embeddings if adding new tokens
        model.resize_token_embeddings(len(tokenizer))
        print(f"Tokenizer vocab size after adding PAD: {len(tokenizer)}")

    # Move model to the appropriate device
    device = get_device()
    print(f"Moving model to device: {device}")
    model.to(device)

    # Set model to evaluation mode
    model.eval()
    print("Model set to evaluation mode.")

    return model, tokenizer, device


# --- Generation Functions ---


def generate_text(
    model,
    tokenizer,
    device,
    prompt: str,
    max_new_tokens: int = 50,
    method: str = "greedy",  # greedy, top-k, top-p
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
):
    """Generates text using the specified method."""
    print(f"\n--- Generating ({method}) ---")
    print(f"Prompt: '{prompt}'")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,  # Use EOS for open-ended generation
        "eos_token_id": tokenizer.eos_token_id,
    }

    if method == "greedy":
        generation_kwargs["do_sample"] = False
    elif method == "top-k":
        generation_kwargs["do_sample"] = True
        generation_kwargs["top_k"] = top_k
        generation_kwargs["temperature"] = temperature
    elif method == "top-p":
        generation_kwargs["do_sample"] = True
        generation_kwargs["top_p"] = top_p
        generation_kwargs["temperature"] = temperature
    else:
        raise ValueError(f"Unknown generation method: {method}")

    print(f"Params: {generation_kwargs}")

    with torch.no_grad():
        outputs = model.generate(input_ids, **generation_kwargs)

    # Decode the generated tokens, excluding the prompt
    # output shape is [batch_size, sequence_length]
    prompt_length = input_ids.shape[1]
    generated_ids = outputs[0, prompt_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"Generated: {generated_text}")
    return generated_text


# --- Main Execution --- (for testing)
if __name__ == "__main__":
    model_name_to_load = "gpt2"  # Can change to gpt2-medium etc.
    model, tokenizer, device = load_model_and_tokenizer(model_name_to_load)

    print("\n--- Model Config ---")
    print(model.config)
    print(f"\nModel is on device: {model.device}")
    print(f"Tokenizer pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Simple test: Encode and decode
    prompt = "Hello, world!"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # print(f"\nEncoded prompt: {inputs}") # Commented out for cleaner generation test

    # outputs = model.generate(**inputs, max_length=10) # Basic generation test
    # decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(f"\nGenerated output (basic): {decoded_output}") # Commented out

    # --- Test Generation Methods ---
    test_prompt = "The meaning of life is"

    # Greedy
    generate_text(model, tokenizer, device, test_prompt, method="greedy")

    # Top-k (k=50, temp=0.7)
    generate_text(
        model, tokenizer, device, test_prompt, method="top-k", top_k=50, temperature=0.7
    )

    # Top-p (p=0.9, temp=0.7)
    generate_text(
        model,
        tokenizer,
        device,
        test_prompt,
        method="top-p",
        top_p=0.9,
        temperature=0.7,
    )

    # Top-p (p=0.9, temp=1.0 - more random)
    generate_text(
        model,
        tokenizer,
        device,
        test_prompt,
        method="top-p",
        top_p=0.9,
        temperature=1.0,
    )

    # Example with longer generation
    # generate_text(model, tokenizer, device, "Once upon a time", method="top-p", max_new_tokens=100)
