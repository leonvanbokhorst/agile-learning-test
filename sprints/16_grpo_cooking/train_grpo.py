import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from trl import GRPOTrainer, GRPOConfig, RewardConfig # Import GRPO specific classes
from peft import LoraConfig, TaskType
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
policy_model_id = "unsloth/Llama-3.2-3B-Instruct" # Base model for policy
reward_model_path = "sprints/16_grpo_cooking/results/reward_model" # Path to our trained RM
hub_dataset_id = "leonvanbokhorst/CoT_Reasoning_Cooking_GRPO_Formatted" # Dataset with prompts
output_dir = "sprints/16_grpo_cooking/results/grpo_adapter" # Where to save the PEFT adapter

# Training Hyperparameters (adjust as needed)
num_train_epochs = 1
batch_size = 4 # Number of prompts per batch
gradient_accumulation_steps = 4 # Accumulate gradients for effective batch size
learning_rate = 1e-5 # Learning rate for the policy model adapters
logging_steps = 10
save_steps = 100 # Save adapter checkpoints periodically
max_prompt_length = 512 # Max length for the prompt part
max_completion_length = 512 # Max tokens to generate during training rollouts (generation_max_new_tokens)

# GRPO specific parameters
beta = 0.1  # KL divergence coefficient (controls how much policy deviates from reference)
k = 4       # Group size (number of responses generated per prompt)
# --- --- --- ---

# --- Load Tokenizer, Policy Model, Reward Model ---
print(f"Loading tokenizer for: {policy_model_id}")
tokenizer = AutoTokenizer.from_pretrained(policy_model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set tokenizer pad token to EOS token.")

print(f"Loading policy model: {policy_model_id}")
policy_model = AutoModelForCausalLM.from_pretrained(
    policy_model_id,
    torch_dtype=torch.bfloat16,
    # No device_map here, PEFT/Trainer handles it
)
print("Policy model loaded.")

print(f"Loading reward model from: {reward_model_path}")
reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_path,
    num_labels=1, # Explicitly set num_labels to match trained model
    torch_dtype=torch.bfloat16,
    device_map="auto" # Let trainer handle device placement
)
print("Reward model loaded.")
# --- --- --- ---

# --- Load Dataset (only need prompts) ---
print(f"Loading dataset from Hub: {hub_dataset_id}")
dataset = load_dataset(hub_dataset_id, split="train") # Load only the train split
# Define columns to remove (keep 'prompt', remove others)
columns_to_remove = [col for col in dataset.column_names if col != 'prompt']
print(f"Dataset loaded. Columns to remove: {columns_to_remove}")

# Map function to just keep prompts and tokenize them
def tokenize_prompt(examples):
    # Tokenize the prompt, ensure max_length allows for generation later
    return tokenizer(examples["prompt"], truncation=True, max_length=max_prompt_length)

print("Tokenizing prompts...")
dataset = dataset.map(
    tokenize_prompt,
    batched=True,
    remove_columns=columns_to_remove, # Keep 'prompt', remove 'chosen' and 'rejected'
    num_proc=1 # Avoid multiprocessing issues
)
print(f"Dataset prepared with {len(dataset)} prompts. Columns: {dataset.column_names}")
# --- --- --- ---

# --- PEFT Configuration for Policy Model ---
policy_peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, # Important: Set to CAUSAL_LM for the policy model
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
)
print("Policy PEFT LoRA config created.")
# --- --- --- ---

# --- GRPO Trainer Configuration ---
print("Configuring GRPO trainer...")
# Use GRPOConfig, inheriting from TrainingArguments but adding GRPO specifics
grpc_config = GRPOConfig(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size, # Note: This is prompts per device
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    logging_steps=logging_steps,
    save_steps=save_steps,
    save_strategy="steps",
    save_total_limit=2, # Keep latest 2 adapter checkpoints
    bf16=torch.cuda.is_bf16_supported(),
    remove_unused_columns=False,
    report_to="tensorboard",
    optim="adamw_torch_fused", # Try fused optimizer
    gradient_checkpointing=True, # Enable gradient checkpointing
    gradient_checkpointing_kwargs={"use_reentrant": False}, # Recommended for newer PyTorch/PEFT
    # GRPO specific args:
    beta=beta,
    num_generations=k, # Correct parameter name for group size
    max_prompt_length=max_prompt_length,        # Correct parameter for prompt length
    max_completion_length=max_completion_length,# Correct parameter for completion length
    eval_strategy="no",
)

# --- Initialize GRPOTrainer ---
print("Initializing GRPOTrainer...")
trainer = GRPOTrainer(
    model=policy_model,         # Base policy model (PEFT applied internally)
    # ref_model=None,             # Trainer creates reference model internally
    reward_funcs=reward_model,    # Correct argument name (can be model, path, func, or list)
    processing_class=tokenizer,
    args=grpc_config,             # Pass GRPOConfig
    train_dataset=dataset,        # Dataset containing tokenized prompts
    peft_config=policy_peft_config, # Apply PEFT to the policy model
)
# --- --- --- ---

# --- Train ---
print("Starting GRPO training...")
trainer.train()
print("Training finished.")
# --- --- --- ---

# --- Save Final Adapter ---
print(f"Saving final GRPO adapter to {output_dir}")
trainer.save_model(output_dir) # Saves the adapter weights
print("Adapter saved.")
# --- --- --- --- 