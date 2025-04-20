import torch
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

def main():
    # ---------- 1. Configuration ----------
    rm_base_model_name = "distilroberta-base"
    dataset_name = "ieuniversity/flirty_or_not"
    output_dir = "flirty_reward_model_adapter"
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.1
    target_modules = ["query", "value"] # Check module names for RoBERTa
    num_train_epochs = 1
    per_device_train_batch_size = 16
    learning_rate = 2e-5
    max_seq_length = 64 # As suggested in ideation

    print("--- Configuration ---")
    print(f"Base Model: {rm_base_model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Output Dir: {output_dir}")
    print(f"Epochs: {num_train_epochs}")
    print(f"Batch Size: {per_device_train_batch_size}")
    print("---------------------")

    # ---------- 2. Load Tokenizer and Model ----------
    print("\n--- Loading Tokenizer & Model ---")
    tokenizer = AutoTokenizer.from_pretrained(rm_base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(rm_base_model_name, num_labels=2)

    # Add padding token if missing (like for GPT-2)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # ---------- 3. Load and Prepare Dataset ----------
    print("\n--- Loading & Preparing Dataset ---")
    dataset = load_dataset(dataset_name)
    print(f"Dataset loaded: {dataset}")

    def tokenize_function(examples):
        return tokenizer(examples["texts"], padding="max_length", truncation=True, max_length=max_seq_length)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    # Remove columns not needed by the model
    tokenized_datasets = tokenized_datasets.remove_columns(["texts", "id"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    train_dataset = tokenized_datasets["train"]
    # eval_dataset = tokenized_datasets["test"] # Optional: Use test set for evaluation

    print(f"Example Train Sample: {train_dataset[0]}")

    # ---------- 4. PEFT/LoRA Configuration ----------
    print("\n--- Configuring PEFT (LoRA) ---")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, # Specify Sequence Classification
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none" # Usually set to 'none' for LoRA
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ---------- 5. Training ----------
    print("\n--- Setting up Training ---")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        # evaluation_strategy="epoch", # Uncomment if using eval_dataset
        # save_strategy="epoch",      # Uncomment if using eval_dataset
        # load_best_model_at_end=True, # Uncomment if using eval_dataset
        remove_unused_columns=False, # PEFT needs the original columns
        fp16=torch.cuda.is_available(), # Enable mixed precision if GPU available
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset, # Uncomment if using eval_dataset
        tokenizer=tokenizer,
        # data_collator=DataCollatorWithPadding(tokenizer=tokenizer) # Usually handled by Trainer
    )

    print("\n--- Starting Training ---")
    trainer.train()
    print("--- Training Complete ---")

    # ---------- 6. Save Adapter ----------
    print(f"\n--- Saving LoRA adapter to {output_dir} ---")
    # Saving the adapter will only save the trained LoRA weights
    trainer.save_model(output_dir)
    # Also save the tokenizer
    tokenizer.save_pretrained(output_dir)
    print("--- Adapter Saved ---")

if __name__ == "__main__":
    main() 