import torch
import torch.ao.quantization
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os
import psutil  # For memory usage (optional, might need pip install psutil)

# --- Configuration ---
MODEL_ID = "gpt2"  # Standard GPT-2 model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
# Note: Dynamic quantization primarily benefits CPU inference.
# While it runs on GPU, the speed benefits might not be as pronounced
# as INT8 ops are often highly optimized for CPU execution paths in PyTorch's backend.


# --- Utility Functions ---
def print_model_size(model, label=""):
    """Prints the size of the model's state_dict in MB."""
    torch.save(model.state_dict(), "temp_weights.pt")
    size_mb = os.path.getsize("temp_weights.pt") / (1024 * 1024)
    print(f"{label} Model size: {size_mb:.2f} MB")
    os.remove("temp_weights.pt")


def get_memory_usage(label=""):
    """Gets current process memory usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"{label} Memory Usage: {mem_info.rss / (1024 * 1024):.2f} MB (RSS)")


# --- Load Original Model ---
print(f"\n--- Loading Original FP32 Model ({MODEL_ID}) ---")
start_mem = get_memory_usage("Initial")
load_start_time = time.time()

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model_fp32 = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model_fp32.eval()  # Set to evaluation mode
model_fp32.to(DEVICE)

load_end_time = time.time()
get_memory_usage("After loading FP32 model")
print(f"Time to load FP32 model: {load_end_time - load_start_time:.2f} seconds")
print_model_size(model_fp32, "Original FP32")
# print("\nOriginal FP32 Model Structure (Transformer Block Excerpt):")
# try:
#     print(model_fp32.transformer.h[0])
# except AttributeError:
#     print("Could not access model_fp32.transformer.h[0] to show structure.")


# --- Set Quantization Backend ---
# For dynamic quantization on CPU, especially on ARM architectures (like macOS M1/M2),
# 'qnnpack' is often required. The default might not support all operations.
# Other options could be 'fbgemm' or 'x86', but 'qnnpack' is typical for mobile/ARM.
print(f"\n--- Setting Quantization Backend ---")
# Make sure this is done *before* quantization calls if issues arise.
# Note: This might change the backend for the entire process.
original_engine = torch.backends.quantized.engine
print(f"Original quantized engine: {original_engine}")
try:
    # Recommended backend for ARM/mobile/macOS often
    torch.backends.quantized.engine = "qnnpack"
    print(f"Set quantized engine to: {torch.backends.quantized.engine}")
except RuntimeError as e:
    print(f"Failed to set engine to 'qnnpack': {e}. Trying 'x86'.")
    try:
        # Fallback or alternative for some systems
        torch.backends.quantized.engine = "x86"
        print(f"Set quantized engine to: {torch.backends.quantized.engine}")
    except RuntimeError as e2:
        print(f"Failed to set engine to 'x86': {e2}. Using original: {original_engine}")
        torch.backends.quantized.engine = original_engine


# --- Apply Dynamic Quantization ---
print(f"\n--- Applying Dynamic Quantization (INT8 weights for Linear layers) ---")
quantize_start_time = time.time()

# Specify the layers to quantize dynamically
# For GPT-2, Linear layers are the primary targets within the transformer blocks
# and the final output layer.
# We use a set of the types of layers we want to target.
quantization_config = {torch.nn.Linear}

# Apply dynamic quantization
# Note: quantize_dynamic works best on CPU.
# If the model is on GPU, it might need to be moved back for the quantization
# function itself, depending on the backend support, but inference can
# sometimes still happen on GPU with quantized weights (performance may vary).
# For simplicity here, we'll assume CPU or handle potential device mismatches implicitly.
# If running on CPU, this is straightforward. If on GPU, PyTorch handles some cases.
# For explicit control, move model_fp32 to CPU first if needed.
# model_fp32.to('cpu') # Optional: Move to CPU if GPU causes issues with quantize_dynamic

model_int8_dynamic = torch.ao.quantization.quantize_dynamic(
    model=model_fp32,
    qconfig_spec=quantization_config,  # Specify which layer types to quantize
    dtype=torch.qint8,  # Target data type for weights
)
# If moved to CPU: model_int8_dynamic.to(DEVICE) # Move back if needed

quantize_end_time = time.time()
get_memory_usage("After dynamic quantization")
print(
    f"Time to apply dynamic quantization: {quantize_end_time - quantize_start_time:.2f} seconds"
)
print_model_size(model_int8_dynamic, "Dynamically Quantized (INT8 Weights)")

print("\nDynamically Quantized Model Structure (Transformer Block Excerpt):")
# Check if the layers have changed type (e.g., Linear to QuantizedLinear)
# Note: The exact type might depend on the PyTorch version and backend.
# Common types include DynamicQuantizedLinear or similar.
try:
    # Accessing the specific layer might change if the structure is altered.
    # We inspect the *type* of the layers rather than printing the whole module.
    print("Sample Linear layers after dynamic quantization:")
    for name, module in model_int8_dynamic.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Check if it's one of the dynamically quantized types
            if hasattr(torch.ao.nn.quantized.dynamic, "Linear") and isinstance(
                module, torch.ao.nn.quantized.dynamic.Linear
            ):
                print(f"  - {name}: {type(module)}")
            elif "quantized" in str(type(module)).lower():  # More general check
                print(f"  - {name}: {type(module)}")
            # Limit the output
            if name.count(".") > 3:  # Stop after inspecting a few nested layers
                break
        # Print the first encountered original Linear if any remain (shouldn't for targeted types)
        # elif isinstance(module, torch.nn.Linear) and 'transformer.h.0.mlp.c_fc' in name:
        #      print(f"  - {name}: {type(module)} (Original - Should not happen if targeted)")

    # Let's try printing the first block again to see overall structure
    # print(model_int8_dynamic.transformer.h[0])

except AttributeError:
    print("Could not access model_int8_dynamic.transformer.h[0] or submodules.")
except Exception as e:
    print(f"An error occurred while inspecting the quantized model structure: {e}")


print("\n--- Dynamic Quantization Applied ---")
print("Next steps could involve: ")
print("1. Comparing inference speed (e.g., text generation time).")
print("2. Evaluating generation quality or perplexity.")
print("3. Trying Static Quantization.")

# --- Inference Comparison ---
print("\n--- Comparing Inference Time (CPU might show more benefit) ---")

# Ensure models are on the target device (CPU likely best for dynamic quant benefit)
# Move models to CPU for a clearer comparison if DEVICE was GPU
if DEVICE == torch.device("cuda"):
    print("Moving models to CPU for inference comparison...")
    try:
        model_fp32.to("cpu")
        model_int8_dynamic.to("cpu")
        comparison_device = torch.device("cpu")
    except Exception as e:
        print(f"Could not move models to CPU: {e}. Comparison will run on {DEVICE}")
        comparison_device = DEVICE
else:
    comparison_device = DEVICE
print(f"Performing comparison on: {comparison_device}")


prompt = "May the Force be with"
# Increase max_length slightly for a bit more work
max_length = 50
num_runs = 3  # Run a few times for averaging

print(f"\nGenerating with FP32 model ({num_runs} runs)...")
fp32_times = []
for i in range(num_runs):
    inputs = tokenizer(prompt + f" {i}", return_tensors="pt").to(comparison_device)
    gen_start_time = time.time()
    # Use torch.no_grad() for fair comparison (inference mode)
    with torch.no_grad():
        generated_ids = model_fp32.generate(
            inputs.input_ids,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id,  # Suppress warning
        )
    gen_end_time = time.time()
    fp32_times.append(gen_end_time - gen_start_time)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(
        f"  Run {i+1}/{num_runs} Time: {fp32_times[-1]:.2f}s | Output: '{generated_text[:80]}...'"
    )
fp32_avg_time = sum(fp32_times) / num_runs
print(f"-> Average FP32 generation time: {fp32_avg_time:.2f} seconds")


print(f"\nGenerating with Dynamically Quantized model ({num_runs} runs)...")
int8_times = []
for i in range(num_runs):
    inputs = tokenizer(prompt + f" {i}", return_tensors="pt").to(comparison_device)
    gen_start_time = time.time()
    with torch.no_grad():
        generated_ids = model_int8_dynamic.generate(
            inputs.input_ids,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id,  # Suppress warning
        )
    gen_end_time = time.time()
    int8_times.append(gen_end_time - gen_start_time)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(
        f"  Run {i+1}/{num_runs} Time: {int8_times[-1]:.2f}s | Output: '{generated_text[:80]}...'"
    )
int8_avg_time = sum(int8_times) / num_runs
print(f"-> Average INT8 (Dynamic) generation time: {int8_avg_time:.2f} seconds")

# --- Comparison Summary ---
print("\n--- Comparison Summary ---")
print(f"Average FP32 Time: {fp32_avg_time:.2f}s")
print(f"Average INT8 Time: {int8_avg_time:.2f}s")
if int8_avg_time < fp32_avg_time:
    speedup = fp32_avg_time / int8_avg_time
    print(f"Dynamic Quantization Speedup: {speedup:.2f}x")
else:
    print("Dynamic Quantization did not result in a speedup in this test.")

# --- End of Script ---
