# Task 2: Model Loading Notes

**Goal:** Select, access, and configure the Llama 3.2 3B-Instruct model as the base policy model (πθ).

**Steps Taken:**

1.  **Model Identification:**
    *   Confirmed the specific model ID to use: `unsloth/Llama-3.2-3B-Instruct`.

2.  **Script Creation (`load_model.py`):**
    *   Created a Python script (`sprints/16_grpo_cooking/load_model.py`) to handle model loading.

3.  **Loading Logic:**
    *   Used `transformers.AutoTokenizer.from_pretrained(model_id)` to load the tokenizer.
    *   Used `transformers.AutoModelForCausalLM.from_pretrained(...)` to load the model.
    *   Specified `torch_dtype=torch.bfloat16` for potential efficiency gains.
    *   Used `device_map="auto"` to automatically utilize the available CUDA GPU.
    *   Included basic checks for tokenizer/model loading success and printed device usage and memory footprint.

4.  **Verification:**
    *   Executed the script (`python sprints/16_grpo_cooking/load_model.py`).
    *   Confirmed the model and tokenizer loaded successfully onto the CUDA device (~6.43 GB VRAM reported).

**Outcome:** The base policy model (`unsloth/Llama-3.2-3B-Instruct`) was successfully loaded into memory and placed on the GPU, ready for use. 