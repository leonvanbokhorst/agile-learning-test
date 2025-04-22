import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Specify the model ID
model_id = "unsloth/Llama-3.2-3B-Instruct"

print(f"Loading model: {model_id}")

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit()

# Load the model
# Use bfloat16 for potential speedup and memory saving if supported
# Use device_map='auto' to automatically place the model on available devices (GPU if possible)
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16, # Use bfloat16 for efficiency
        device_map="auto", # Automatically uses CUDA if available
    )
    print("Model loaded successfully.")
    print(f"Model memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")

    # Optional: Test encoding a simple prompt
    prompt = "What is the recipe for chocolate chip cookies?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(f"Successfully encoded test prompt on device: {inputs.input_ids.device}")

except Exception as e:
    print(f"Error loading model: {e}") 