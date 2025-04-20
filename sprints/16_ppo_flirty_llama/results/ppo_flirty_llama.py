import torch
import random
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from peft import PeftModel, PeftConfig # Need PeftModel to load adapter
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, setup_chat_format
import warnings
from tqdm import tqdm # Import tqdm for progress bar

# Suppress UserWarnings from TRL setup_chat_format about padding side
warnings.filterwarnings("ignore", category=UserWarning, message=r".*Padding side is set to left.*")

def main():
    # ---------- 1. Configuration ----------
    # Reward Model Config
    rm_adapter_path = "../reward_model/flirty_reward_model_adapter" # Path to the trained adapter
    rm_base_model_name = "distilroberta-base" # Base model used for the reward model

    # Policy Model Config
    policy_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    # policy_model_name = "gpt2" # Smaller model for faster testing if needed

    # PPO Config
    ppo_output_dir = "flirty_llama_ppo_model"
    ppo_batch_size = 32 # Adjust based on GPU memory
    ppo_epochs = 4 # Number of optimization epochs per batch
    learning_rate = 1.41e-5 # TRL default / common starting point
    target_kl = 0.1 # Target KL divergence
    log_with = "tensorboard" # Log metrics for TensorBoard
    gradient_accumulation_steps = 1
    seed = 42

    # Generation Config
    max_new_tokens = 80
    do_sample = True
    top_k = 0 # 0 means disabled
    temperature = 1.0 # Higher temperature means more randomness
    use_chat_template = True # Use Llama 3 Instruct template
    num_ppo_steps = 500 # Number of PPO steps (batches) to run
    prompt_max_length = 128 # Max length for prompt tokenization

    print("--- Configuration ---")
    print(f"Reward Adapter: {rm_adapter_path}")
    print(f"Policy Model: {policy_model_name}")
    print(f"PPO Output Dir: {ppo_output_dir}")
    print(f"PPO Batch Size: {ppo_batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Use Chat Template: {use_chat_template}")
    print("---------------------")

    # ---------- 2. Load Reward Model Components ----------
    print("\n--- Loading Reward Model Components ---")
    # Load the base tokenizer and model used for reward calculation
    reward_tokenizer = AutoTokenizer.from_pretrained(rm_base_model_name)
    # Load the base model
    reward_base_model = AutoModelForSequenceClassification.from_pretrained(rm_base_model_name, num_labels=2)

    # Load the PEFT config and combine with base model to get the trained reward model
    try:
        reward_model = PeftModel.from_pretrained(reward_base_model, rm_adapter_path)
        print("Successfully loaded PEFT adapter for reward model.")
    except Exception as e:
        print(f"Error loading PEFT adapter: {e}")
        print("Ensure the adapter path is correct and files exist:", rm_adapter_path)
        return # Exit if adapter loading fails

    # Move reward model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_model.to(device)
    reward_model.eval() # Set to evaluation mode
    print(f"Reward model loaded on device: {device}")

    # ---------- 3. Define Reward Function ----------
    print("\n--- Defining Reward Function ---")
    def reward_fn(samples: list[str]) -> torch.Tensor:
        """Calculates reward scores for generated samples based on the reward model."""
        rewards = []
        with torch.no_grad():
            # Process samples in batches to avoid OOM for large generations/batches
            for i in range(0, len(samples), ppo_batch_size):
                batch_samples = samples[i : i + ppo_batch_size]
                try:
                    inputs = reward_tokenizer(batch_samples, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
                    outputs = reward_model(**inputs)
                    logits = outputs.logits
                    # Reward is the positive class (flirty) logit - can adjust scoring logic here
                    # Example: score = logits[:, 1] # Raw logit for positive class
                    score = logits[:, 1] - logits[:, 0] # Difference between positive and negative logits
                    rewards.extend(score.cpu().tolist()) # Move rewards to CPU and convert to list
                except Exception as e:
                    print(f"Error during reward calculation for batch: {e}")
                    # Append a default low reward or handle error as appropriate
                    rewards.extend([0.0] * len(batch_samples))

        return torch.tensor(rewards, dtype=torch.float32) # Return as a tensor

    print("Reward function defined.")

    # ---------- 4. Load Policy Model and Tokenizer ----------
    print("\n--- Loading Policy Model & Tokenizer ---")
    policy_tokenizer = AutoTokenizer.from_pretrained(policy_model_name)
    # Set pad token if missing (Llama 3 usually has it, but good practice)
    if policy_tokenizer.pad_token is None:
        print("Policy tokenizer missing pad token, setting to eos_token.")
        policy_tokenizer.pad_token = policy_tokenizer.eos_token

    # Load the model with a value head for PPO
    # Use `low_cpu_mem_usage=True` and `torch_dtype=torch.bfloat16` for large models
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        policy_model_name,
        torch_dtype=torch.bfloat16, # Use bfloat16 for efficiency if supported
        low_cpu_mem_usage=True, # Optimize memory usage during loading
        # load_in_8bit=True, # Uncomment to use bitsandbytes 8-bit quantization
        # load_in_4bit=True, # Uncomment to use bitsandbytes 4-bit quantization
        trust_remote_code=True # Needed for some models like Llama 3
    )

    # Setup chat format if needed (Llama 3 Instruct)
    if use_chat_template:
        print("Setting up chat format for Llama 3 Instruct model.")
        policy_model, policy_tokenizer = setup_chat_format(
            policy_model,
            policy_tokenizer,
            format='chatml', # Llama 3 uses ChatML format
            resize_to_multiple_of=8, # Optimize embedding/padding
        )
        # Workaround TRL/HF bug where `pad_token_id` gets unset by setup_chat_format
        if policy_tokenizer.pad_token_id is None:
            print("Re-setting pad_token_id after setup_chat_format.")
            policy_tokenizer.pad_token_id = policy_tokenizer.eos_token_id

    print(f"Policy model loaded. Device: {policy_model.device}")

    # ---------- 5. Configure PPO ----------
    print("\n--- Configuring PPO Trainer ---")
    ppo_config = PPOConfig(
        model_name=policy_model_name,
        log_with=log_with,
        learning_rate=learning_rate,
        batch_size=ppo_batch_size,
        mini_batch_size=ppo_batch_size // gradient_accumulation_steps, # Adjust if using accumulation
        gradient_accumulation_steps=gradient_accumulation_steps,
        optimize_cuda_cache=True,
        early_stopping=False,
        target_kl=target_kl,
        ppo_epochs=ppo_epochs,
        seed=seed,
        init_kl_coef=0.2, # TRL default
        adap_kl_ctrl=True, # TRL default
        accelerator_kwargs={'project_dir': ppo_output_dir} # Ensure logs go to the right place
    )

    # ---------- 6. Instantiate PPOTrainer ----------
    # We don't use a reference model (`ref_model=None`) for simplicity here,
    # which means KL divergence is calculated against the initial policy model state.
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=None,
        tokenizer=policy_tokenizer,
        # dataset=None, # We provide prompts directly in the loop
        # data_collator=None, # Not needed when providing prompts directly
    )
    print("PPOTrainer instantiated.")

    print("\n--- Setup Complete - Starting PPO Training Loop (Task 4) ---")

    # ---------- 7. Define Prompts ----------
    # Use prompts relevant to the target task (flirty conversation starters)
    prompts = [
        "Write a playful but not cheesy opening line to someone you just matched with.",
        "Send a flirty yet respectful good-morning message:",
        "How would you start a conversation with someone cute at a coffee shop?",
        "Give me a witty compliment I can use.",
        "Text a crush after meeting at a conference.",
        "Break the ice on Bumble.",
        "What's a fun way to ask for someone's number?",
        "Compose a short, intriguing message for a dating app profile visit."
        # Add more diverse prompts if needed
    ]

    # Generation kwargs setup
    generation_kwargs = {
        # "min_length": -1, # don't ignore the EOS token (needed for PPO)
        "top_k": top_k,
        "top_p": 1.0, # Default, top_p disabled if top_k > 0
        "temperature": temperature,
        "do_sample": do_sample,
        "pad_token_id": policy_tokenizer.pad_token_id,
        "eos_token_id": policy_tokenizer.eos_token_id,
        "max_new_tokens": max_new_tokens,
    }

    # ---------- 8. PPO Training Loop ----------
    for step in tqdm(range(num_ppo_steps), desc="PPO Steps"):
        batch_prompts_text = random.choices(prompts, k=ppo_config.batch_size)

        # Prepare prompts for the policy model
        if use_chat_template:
            # Format prompts using the chat template
            batch_prompts_chat = [[{"role": "user", "content": p}] for p in batch_prompts_text]
            prompt_tokens_list = [policy_tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(policy_model.device) for chat in batch_prompts_chat]
            # Find max length in the batch for padding
            max_len = max(t.shape[1] for t in prompt_tokens_list)
            # Pad each tensor manually to the max length
            query_tensors = []
            for tokens in prompt_tokens_list:
                padding_size = max_len - tokens.shape[1]
                if padding_size > 0:
                    padded_tokens = torch.nn.functional.pad(tokens, (0, padding_size), value=policy_tokenizer.pad_token_id)
                else:
                    padded_tokens = tokens
                query_tensors.append(padded_tokens)
            # Stack the padded tensors
            query_tensors = torch.cat(query_tensors, dim=0)
        else:
            # Simple tokenization for non-chat models
            query_tensors = policy_tokenizer(batch_prompts_text, return_tensors="pt", padding="max_length", truncation=True, max_length=prompt_max_length).input_ids.to(policy_model.device)

        # Generate responses
        # The generate method returns the full sequence (prompt + response)
        response_tensors_list = []
        policy_model.eval() # Set to eval for generation consistency
        with torch.no_grad():
             response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
        policy_model.train() # Set back to train for PPO step

        # Decode generated responses (including prompts)
        # Need to decode prompt + response for the reward model
        # response_tensors contains prompt + generation
        decoded_responses = policy_tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        # Calculate rewards
        # We pass the full decoded text (prompt + response) to the reward function
        rewards = reward_fn(decoded_responses)

        # Perform PPO step
        # Ensure query_tensors and response_tensors are on the same device as the model
        # The `step` method expects prompt tensors, response tensors (prompt+gen), and reward scores
        try:
            stats = ppo_trainer.step([q.squeeze() for q in query_tensors], [r.squeeze() for r in response_tensors], rewards)
            ppo_trainer.log_stats(stats, {"prompt": batch_prompts_text[0]}, rewards) # Log basic stats
        except Exception as e:
            print(f"\nError during PPO step {step}: {e}")
            print("Query tensor shapes:", [q.shape for q in query_tensors])
            print("Response tensor shapes:", [r.shape for r in response_tensors])
            print("Rewards shape:", rewards.shape)
            # Consider continuing or breaking the loop based on the error
            continue # Skip this step

        # Optional: Print a sample generation periodically
        if step % 50 == 0:
            print(f"\n--- Step {step} Sample ---")
            print(f"Prompt: {batch_prompts_text[0]}")
            print(f"Generated: {decoded_responses[0]}")
            print(f"Reward: {rewards[0].item():.4f}")
            print("-----------------------")

    print("\n--- PPO Training Loop Complete ---")

    # ---------- 9. Save Final Model ----------
    print(f"\n--- Saving final PPO model to {ppo_output_dir} ---")
    ppo_trainer.save_pretrained(ppo_output_dir)
    # The tokenizer is often saved automatically by save_pretrained, but saving explicitly is safe
    policy_tokenizer.save_pretrained(ppo_output_dir)
    print("--- Model Saved ---")

if __name__ == "__main__":
    main() 