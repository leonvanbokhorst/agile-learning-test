import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm # Import tqdm for progress bar
import os # Import os for environment variables
from dotenv import load_dotenv # Import dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
dataset_id = "moremilk/CoT_Reasoning_Cooking"
model_id = "unsloth/Llama-3.2-3B-Instruct"
# Get HF username from environment variable
hf_username = os.getenv("HF_USERNAME")
if not hf_username:
    print("Error: HF_USERNAME not found in .env file.")
    print("Please add HF_USERNAME=<your_hf_username> to your .env file.")
    exit()
hub_dataset_id = f"{hf_username}/CoT_Reasoning_Cooking_GRPO_Formatted"

# Define the chat template components for Llama 3 Instruct
# Correct format based on https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
prompt_template = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are a helpful assistant skilled in cooking and explaining your reasoning step-by-step."
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "{question}"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    # NO assistant response here, model generates from this point
)
# System prompt for generating the *direct* answer (without CoT instruction)
direct_answer_system_prompt = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are a helpful assistant skilled in cooking. Provide clear and concise answers."
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "{question}"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    # NO assistant response here, model generates from this point
)
max_new_tokens_rejected = 256 # Max tokens for the generated rejected response
# --- --- --- ---

# --- Load Model and Tokenizer ---
print(f"Loading base model for generation: {model_id}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Set pad token if not set (common for Llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("Base model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading base model/tokenizer: {e}")
    exit()
# --- --- --- ---

# --- Function to Generate Rejected Responses (Batched) ---
def generate_batch_direct_answers(prompt_texts: list[str]):
    """Generates direct answers for a batch of prompts using the base model."""
    # Tokenize the batch of prompts. Enable padding and truncation.
    # The tokenizer handles padding to the longest sequence in the batch.
    inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True).to(device)

    # Ensure generation config handles pad token ID
    generation_kwargs = {
        "max_new_tokens": max_new_tokens_rejected,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id, # Use EOS if PAD is not set
        "do_sample": True, # Add some sampling
        "temperature": 0.7,
        "top_p": 0.9,
    }

    outputs = model.generate(**inputs, **generation_kwargs)

    # Decode the generated sequences
    # We need to decode each sequence in the batch separately, skipping the prompt part.
    input_token_len = inputs.input_ids.shape[1]
    generated_texts = tokenizer.batch_decode(outputs[:, input_token_len:], skip_special_tokens=True)

    # Add the EOT token manually to each generated text
    results = [text.strip() + "<|eot_id|>" for text in generated_texts]
    return results
# --- --- --- ---

# --- Preprocessing (No Generation Here) ---
def format_chosen_and_prompts(example):
    """Formats the chosen response and the prompts needed for generation."""
    question = example['question']
    reasoning = example['metadata']['reasoning']
    answer = example['answer']

    # Format the base prompt (system + user message)
    formatted_prompt = prompt_template.format(question=question)

    # Construct the chosen response (CoT + Answer + EOT token)
    chosen_completion = f"""<think>
{reasoning}
</think>
{answer}<|eot_id|>"""

    # Format the prompt specifically for generating the direct/rejected answer
    direct_answer_prompt_for_generation = direct_answer_system_prompt.format(question=question)

    return {
        "prompt": formatted_prompt, # Base prompt for training data pair
        "chosen": chosen_completion, # Desired assistant response
        "direct_answer_prompt": direct_answer_prompt_for_generation # Prompt for generating rejected response
    }
# --- --- --- ---

# --- Load and Process Dataset (Batched) ---
print(f"Loading dataset: {dataset_id}")

# Define batch size (adjust based on VRAM)
batch_size = 16 # Let's start with 8
print(f"Using batch size: {batch_size}")

try:
    dataset = load_dataset(dataset_id)
    print("Dataset loaded successfully.")

    # --- Batch Processing Loop ---
    print("Processing dataset in batches (FULL RUN - this may take a while)...")
    processed_data_list = [] # Store final dictionaries here
    # Use ceil division for the number of batches
    num_batches = (len(dataset['train']) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Processing Batches"):
        # Get batch indices
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(dataset['train']))
        batch_slice = dataset['train'][start_idx:end_idx] # This is a dict of lists

        # 1. Format prompts and chosen responses for the batch
        # Reconstruct individual example dicts from the batch_slice
        num_in_batch = len(batch_slice['question']) # Get number of examples in this batch
        reconstructed_examples = [{key: batch_slice[key][idx] for key in batch_slice.keys()} for idx in range(num_in_batch)]
        formatted_batch = [format_chosen_and_prompts(example) for example in reconstructed_examples]

        # 2. Extract prompts needed for generating rejected responses
        prompts_for_rejected_gen = [item["direct_answer_prompt"] for item in formatted_batch]

        # 3. Generate rejected responses for the batch
        rejected_completions = generate_batch_direct_answers(prompts_for_rejected_gen)

        # 4. Combine results into final dictionaries
        for j, item in enumerate(formatted_batch):
            processed_data_list.append({
                "prompt": item["prompt"],
                "chosen": item["chosen"],
                "rejected": rejected_completions[j]
            })
    # --- End Batch Loop ---

    # Convert back to Hugging Face Dataset object
    if processed_data_list:
        processed_dataset = Dataset.from_list(processed_data_list)
        print("\nDataset processed into prompt/chosen/rejected format.")
        print(processed_dataset)

        # Show the first processed example (still inside the try block)
        print("\nFirst processed example:")
        print(processed_dataset[0])

        # Push to Hugging Face Hub
        print(f"\nAttempting to push dataset to Hub: {hub_dataset_id}")
        try:
            processed_dataset.push_to_hub(hub_dataset_id)
            print("Dataset successfully pushed to Hub!")
        except Exception as push_error:
            print(f"\nError pushing dataset to Hub: {push_error}")
            print("Please ensure you have the correct permissions or try 'huggingface-cli login'.")

    else:
        print("No data was processed.")

except Exception as e:
    print(f"Error loading or processing dataset: {e}")
# --- --- --- ---