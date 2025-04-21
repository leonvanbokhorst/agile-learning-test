import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from trl import RewardTrainer, RewardConfig
from peft import LoraConfig, TaskType # Import PEFT components
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
# Use the same base model we prepared the dataset with
model_id = "unsloth/Llama-3.2-3B-Instruct"
# Load our pre-processed dataset from the Hub
hub_dataset_id = "leonvanbokhorst/CoT_Reasoning_Cooking_GRPO_Formatted"
# Where to save the trained reward model
output_dir = "sprints/16_grpo_cooking/results/reward_model"
per_device_train_batch_size = 2 # Adjust based on VRAM
per_device_eval_batch_size = 2 # Adjust based on VRAM
learning_rate = 1e-5
gradient_accumulation_steps = 4
logging_steps = 10
eval_steps = 50 # Evaluate periodically
save_steps = 100 # Save checkpoints periodically
max_length = 1024 # Max sequence length for tokenizer
num_train_epochs = 1
# --- --- --- ---

# --- Load Tokenizer and Model ---
print(f"Loading tokenizer for: {model_id}")
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Set pad token if necessary (Llama often requires this)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set tokenizer pad token to EOS token.")

print(f"Loading base model for reward modeling: {model_id}")
# Load the base model - TRL's RewardTrainer expects a SequenceClassification model head
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=1,
    torch_dtype=torch.bfloat16,
    # device_map="auto", # Remove device_map when using PEFT with Trainer
)
# Ensure pad token id is set in model config (important for RewardTrainer)
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id
    print("Set model config pad token id.")

print("Model and tokenizer loaded.")
# --- --- --- ---

# --- PEFT Configuration ---
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8, # LoRA rank
    lora_alpha=32, # LoRA alpha
    lora_dropout=0.1, # LoRA dropout
    # Target modules based on common Llama architectures (might need adjustment)
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        # "gate_proj", # Sometimes included
        # "up_proj",   # Sometimes included
        # "down_proj", # Sometimes included
    ],
)
print("PEFT LoRA config created.")
# --- --- --- ---

# --- Load and Prepare Dataset ---
print(f"Loading dataset from Hub: {hub_dataset_id}")
dataset = load_dataset(hub_dataset_id)
print("Dataset loaded.")

# Optional: Split dataset if it doesn't have train/test splits already
# For simplicity, assuming 'train' split contains all data and we'll use a portion for eval
if "test" not in dataset:
    print("Creating train/test split (90/10)")
    dataset_split = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
else:
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)}")

# Preprocess function to tokenize pairs for RewardTrainer
def preprocess_function(examples):
    """Tokenizes chosen and rejected responses for the RewardTrainer."""
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    # The RewardTrainer expects tokenized chosen/rejected fields
    # It internally combines prompt + chosen and prompt + rejected
    for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer(prompt + chosen, truncation=True, max_length=max_length)
        tokenized_rejected = tokenizer(prompt + rejected, truncation=True, max_length=max_length)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples

print("Preprocessing dataset...")
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1 # Force single process to avoid WSL multiprocessing issues
)
eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1 # Force single process
)

# Filter out examples that might be too long after tokenization (optional but good practice)
train_dataset = train_dataset.filter(lambda x: len(x['input_ids_chosen']) <= max_length and len(x['input_ids_rejected']) <= max_length)
eval_dataset = eval_dataset.filter(lambda x: len(x['input_ids_chosen']) <= max_length and len(x['input_ids_rejected']) <= max_length)
print("Dataset preprocessed.")
# --- --- --- ---

# --- Configure Training ---
print("Configuring training arguments...")
training_args = RewardConfig(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    learning_rate=learning_rate,
    gradient_accumulation_steps=gradient_accumulation_steps,
    logging_steps=logging_steps,
    eval_steps=eval_steps,
    save_steps=save_steps,
    eval_strategy="steps",
    save_strategy="steps",
    save_total_limit=2,
    load_best_model_at_end=True,
    remove_unused_columns=False,
    label_names=[],
    bf16=torch.cuda.is_bf16_supported(),
    report_to="tensorboard",
    max_length=max_length,
    
)
# --- --- --- ---

# --- Initialize Trainer ---
print("Initializing RewardTrainer with PEFT...")
trainer = RewardTrainer(
    model=model, # Pass the base model
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config, # Pass the PEFT config here!
    processing_class=tokenizer,
    # Tokenizer/processing class is not needed if dataset is pre-tokenized
)
# --- --- --- ---

# --- Train ---
print("Starting training...")
trainer.train()
print("Training finished.")
# --- --- --- ---

# --- Save Final Model ---
print(f"Saving final reward model to {output_dir}")
trainer.save_model(output_dir)
print("Model saved.")
# --- --- --- --- 