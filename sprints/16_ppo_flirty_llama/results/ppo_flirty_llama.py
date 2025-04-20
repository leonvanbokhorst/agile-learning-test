import torch
import random
import os
import types # Import types for MethodType monkey-patching
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from peft import PeftModel, PeftConfig # Need PeftModel to load adapter
from trl import PPOTrainer, PPOConfig, setup_chat_format
import warnings
from tqdm import tqdm # Import tqdm for progress bar
import torch.nn as nn # Need nn for wrapper
import trl.trainer.utils as utils
import trl.trainer.ppo_trainer as ppo_mod
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings

# Suppress UserWarnings from TRL setup_chat_format about padding side
warnings.filterwarnings("ignore", category=UserWarning, message=r".*Padding side is set to left.*")

# --- Globals for Monkey-Patch --- (Not ideal, but simplest for patching)
policy_tokenizer_global = None
reward_tokenizer_global = None

# --- Debug: Patch get_reward globally to bypass RoBERTa embedding crash ---
def patched_get_reward(model, query_responses, pad_token_id, context_length):
    # Decode sequences to text and re-tokenize with reward tokenizer
    # print("***** DEBUG: ENTERING patched_get_reward *****") # DEBUG
    decoded_texts = policy_tokenizer_global.batch_decode(query_responses, skip_special_tokens=True)
    reward_inputs = reward_tokenizer_global(
        decoded_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    ).to(model.device)
    # Get model classification logits
    outputs = model(
        input_ids=reward_inputs['input_ids'],
        attention_mask=reward_inputs['attention_mask'],
        return_dict=True,
    )
    # Extract positive class score per example
    # Assume logits shape is (batch_size, num_labels)
    logits = outputs.logits
    if logits.dim() == 2:
        positive_scores = logits[:, 1]
    else:
        # Collapse extra dims to get per-example score
        positive_scores = logits.view(logits.size(0), -1)[:, 1]
    batch_size = positive_scores.size(0)
    # Dummy token-level reward logits (not used by PPOTrainer.step)
    seq_len = query_responses.size(1)
    reward_logits = torch.zeros(batch_size, seq_len, device=positive_scores.device)
    final_rewards = positive_scores
    # Sequence lengths: assume last token index
    seq_lengths = torch.full((batch_size,), seq_len - 1, dtype=torch.long, device=positive_scores.device)
    # Move outputs to the same device as the input query_responses
    device = query_responses.device
    reward_logits = reward_logits.to(device)
    final_rewards = final_rewards.to(device)
    seq_lengths = seq_lengths.to(device)
    return reward_logits, final_rewards, seq_lengths

utils.get_reward = patched_get_reward
ppo_mod.get_reward = patched_get_reward
# end of global patch

# --- Define Monkey-Patch Forward Function ---
def patched_reward_forward(self, input_ids, attention_mask=None, **kwargs):
    # --- !! VERY FIRST LINE DEBUG !! --- (COMMENTED OUT)
    # print("***** ENTERING PATCHED REWARD FORWARD *****") 
    # -------------------------------------

    # self is the base RoBERTa model instance
    global policy_tokenizer_global, reward_tokenizer_global

    # --- DEBUG: Print incoming input_ids --- (COMMENTED OUT)
    # print("--- DEBUG MONKEY PATCH ---")
    # print(f"Received input_ids shape: {input_ids.shape if isinstance(input_ids, torch.Tensor) else type(input_ids)}")
    # print(f"Received input_ids dtype: {input_ids.dtype if isinstance(input_ids, torch.Tensor) else 'N/A'}")
    # print(f"Received input_ids device: {input_ids.device if isinstance(input_ids, torch.Tensor) else 'N/A'}")
    # if isinstance(input_ids, torch.Tensor) and input_ids.numel() > 0:
    #     print(f"Received input_ids sample: {input_ids.view(-1)[:10]}...") 
    # print("--------------------------")
    # --------------------------------------

    if policy_tokenizer_global is None or reward_tokenizer_global is None:
        print("ERROR: Tokenizers not set globally for monkey patch!")
        raise ValueError("Tokenizers not available for reward model monkey patch.")

    # 1. Decode Llama3 input_ids
    try:
        decoded_texts = policy_tokenizer_global.batch_decode(input_ids, skip_special_tokens=True)
        # --- DEBUG: Print decoded texts --- (COMMENTED OUT)
        # print("--- DEBUG MONKEY PATCH ---")
        # print(f"Decoded texts sample: {decoded_texts[0] if decoded_texts else 'N/A'}...")
        # print("--------------------------")
        # --------------------------------------
    except Exception as e:
        print(f"Error decoding in patch: {e}")
        print(f"Input IDs type: {type(input_ids)}, shape: {input_ids.shape if isinstance(input_ids, torch.Tensor) else 'N/A'}")
        raise e

    # 2. Re-tokenize using reward tokenizer
    try:
        reward_inputs = reward_tokenizer_global(
            decoded_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64, # Match reward model training
        ).to(self.device) # Use the RoBERTa model's own device
        # --- DEBUG: Print re-tokenized inputs --- (COMMENTED OUT)
        # print("--- DEBUG MONKEY PATCH ---")
        # print(f"Re-tokenized input_ids shape: {reward_inputs['input_ids'].shape}")
        # print(f"Re-tokenized input_ids sample: {reward_inputs['input_ids'].view(-1)[:10]}...") 
        # print("--------------------------")
        # --------------------------------------
    except Exception as e:
        print(f"Error re-tokenizing in patch: {e}")
        print(f"Decoded texts sample: {decoded_texts[0] if decoded_texts else 'N/A'}")
        raise e

    # 3. Prepare args for original forward, removing incompatible ones
    original_kwargs = kwargs.copy()
    original_kwargs.pop("use_cache", None)

    # 4. Call the *original* forward method with *correct* inputs
    # --- DEBUG: Call original forward --- (COMMENTED OUT)
    # print("--- DEBUG MONKEY PATCH ---")
    # print("Calling original forward...")
    # --- DEBUG: Print kwargs being passed --- 
    # print(f"Passing kwargs: {original_kwargs.keys()}") 
    # --- DEBUG: Print input details and config --- 
    # print(f"Input IDs shape: {reward_inputs['input_ids'].shape}, dtype: {reward_inputs['input_ids'].dtype}, device: {reward_inputs['input_ids'].device}")
    # print(f"Attention Mask shape: {reward_inputs['attention_mask'].shape}, dtype: {reward_inputs['attention_mask'].dtype}, device: {reward_inputs['attention_mask'].device}")
    # print(f"Model max_position_embeddings: {self.config.max_position_embeddings}")
    # print("--------------------------")
    output = self._original_forward(
        input_ids=reward_inputs['input_ids'],
        attention_mask=reward_inputs['attention_mask'],
        **original_kwargs # Pass only compatible kwargs
    )
    # --- DEBUG: Original forward returned --- (COMMENTED OUT)
    # print("--- DEBUG MONKEY PATCH ---")
    # print("Original forward returned.")
    # print("--------------------------")
    return output

# Debug monkey-patch get_reward to inspect which backbone and forward is called
old_get_reward = utils.get_reward

def debug_get_reward(model, query_responses, pad_token_id, context_length):
    # print("***** DEBUG GET_REWARD *****") # DEBUG
    # backbone = getattr(model, model.base_model_prefix)
    # print("Backbone type:", type(backbone)) # DEBUG
    # print("Backbone forward method:", backbone.forward) # DEBUG
    return old_get_reward(model, query_responses, pad_token_id, context_length)

utils.get_reward = debug_get_reward

# After imports at top of file, patch RobertaEmbeddings
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings

# Save original forward
orig_roberta_embeddings_forward = RobertaEmbeddings.forward

def safe_roberta_embeddings_forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, **kwargs):
    # if input_ids is not None: # DEBUG - Temporarily disabled clamp
    #     # Clamp any out-of-range token IDs to the max valid index to prevent asserts
    #     # print("Clamping input_ids in safe_roberta_embeddings_forward") # DEBUG
    #     max_id = self.word_embeddings.num_embeddings - 1
    #     input_ids = input_ids.clamp(0, max_id)
    return orig_roberta_embeddings_forward(
        self,
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        inputs_embeds=inputs_embeds,
        **kwargs
    )

# Apply the patch
RobertaEmbeddings.forward = safe_roberta_embeddings_forward

def main():
    global policy_tokenizer_global, reward_tokenizer_global
    # ---------- 1. Configuration ----------
    # Reward Model Config
    rm_adapter_path = "flirty_reward_model_adapter" # Path to the trained adapter
    rm_base_model_name = "distilroberta-base" # Base model used for the reward model

    # Policy Model Config
    policy_model_name = "unsloth/Llama-3.2-3B-Instruct"
    # policy_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    # policy_model_name = "gpt2" # Smaller model for faster testing if needed

    # PPO Config
    ppo_output_dir = "flirty_llama_ppo_model"
    ppo_batch_size = 4 # Adjust based on GPU memory
    ppo_epochs = 4 # Number of optimization epochs per batch
    learning_rate = 1.41e-5 # TRL default / common starting point
    target_kl = 0.1 # Target KL divergence
    log_with = "tensorboard" # Log metrics for TensorBoard
    gradient_accumulation_steps = 2
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

    # ---------- 2. Load Reward Model Components & Apply Patch ----------
    print("\n--- Loading Reward Model Components ---")
    reward_tokenizer = AutoTokenizer.from_pretrained(rm_base_model_name)
    reward_tokenizer_global = reward_tokenizer # Set global
    reward_base_model = AutoModelForSequenceClassification.from_pretrained(rm_base_model_name, num_labels=2)

    try:
        reward_model_peft = PeftModel.from_pretrained(reward_base_model, rm_adapter_path)
        print("Successfully loaded PEFT adapter for reward model.")
    except Exception as e:
        print(f"Error loading PEFT adapter: {e}")
        print("Ensure the adapter path is correct and files exist:", rm_adapter_path)
        return

    # --- Apply Monkey-Patch ---
    try:
        actual_base_reward_model = reward_model_peft.base_model
        actual_base_reward_model._original_forward = actual_base_reward_model.forward
        actual_base_reward_model.forward = types.MethodType(patched_reward_forward, actual_base_reward_model)
        print("Monkey-patched the forward method of the base reward model TO RE-TOKENIZE and ignore 'use_cache'.")
    except Exception as e:
        print(f"Error during monkey-patching reward model: {e}")
        # Decide how to handle - maybe proceed cautiously or exit
        # return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move reward model to CPU to avoid CUDA embedding crash in LayerNorm
    reward_model_peft.to("cpu")
    reward_model_peft.eval()
    print("Warning: Reward model moved to CPU to avoid CUDA embedding crash. Reward computation will run on CPU.")

    # ---------- 3. Load Policy Model and Tokenizer ----------
    print("\n--- Loading Policy Model & Tokenizer (Quantized) ---")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    policy_tokenizer = AutoTokenizer.from_pretrained(policy_model_name)
    policy_tokenizer_global = policy_tokenizer # Set global
    if policy_tokenizer.pad_token is None:
        print("Policy tokenizer missing pad token, setting to eos_token.")
        policy_tokenizer.pad_token = policy_tokenizer.eos_token

    # Monkey-patch the utils.get_reward function to bypass embedding crash
    orig_utils_get_reward = utils.get_reward
    def patched_utils_get_reward(model, query_responses, pad_token_id, context_length):
        # Decode to text and re-tokenize with the reward tokenizer
        decoded_texts = policy_tokenizer_global.batch_decode(query_responses, skip_special_tokens=True)
        reward_inputs = reward_tokenizer_global(
            decoded_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        ).to(model.device)
        # Use classification wrapper to score
        outputs = model(
            input_ids=reward_inputs['input_ids'],
            attention_mask=reward_inputs['attention_mask'],
            return_dict=True,
        )
        logits = outputs.logits  # shape (batch_size, num_labels)
        positive_scores = logits[:, 1]
        batch_size, seq_len = reward_inputs['input_ids'].shape
        # Expand to token-level logits
        reward_logits = positive_scores.unsqueeze(1).expand(batch_size, seq_len).contiguous()
        final_rewards = positive_scores
        seq_lengths = torch.full((batch_size,), seq_len - 1, dtype=torch.long, device=positive_scores.device)
        # Move outputs to the same device as the input query_responses
        device = query_responses.device
        reward_logits = reward_logits.to(device)
        final_rewards = final_rewards.to(device)
        seq_lengths = seq_lengths.to(device)
        return reward_logits, final_rewards, seq_lengths
    utils.get_reward = patched_utils_get_reward

    print("\n--- Loading Policy Model & Tokenizer (Quantized) ---")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    policy_model = AutoModelForCausalLM.from_pretrained(
        policy_model_name,
        quantization_config=bnb_config, # Pass the config here!
        # torch_dtype is inferred from compute_dtype, no need to specify both
        low_cpu_mem_usage=True, # Keep this
        trust_remote_code=True,
        device_map="auto" # Add this for automatic device placement with quantization
    )
    # Add a value head for PPO AFTER loading the quantized model
    policy_model.score = nn.Linear(policy_model.config.hidden_size, 1, bias=False)
    # Ensure the head matches compute dtype (though accelerate might handle this)
    policy_model.score.to(dtype=torch.bfloat16)
    # Explicitly initialize value head weights
    torch.nn.init.xavier_uniform_(policy_model.score.weight)

    print("Policy model loaded (quantized) with value head initialized.")
    # Use the same model for both policy & value
    value_model = policy_model
    # Enable gradient checkpointing for further memory savings
    policy_model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled.")

    # --- Patch PolicyAndValueWrapper forward for dtype consistency ---
    # PolicyAndValueWrapper class is defined within ppo_trainer module
    PolicyAndValueWrapper = ppo_mod.PolicyAndValueWrapper # Get the class
    orig_wrapper_forward = PolicyAndValueWrapper.forward

    def patched_wrapper_forward(self, **kwargs):
        # print("***** DEBUG PolicyAndValueWrapper.forward *****") # DEBUG
        # Call the policy model (handles potential quantization/checkpointing internally)
        policy_output = self.policy(**kwargs)

        # Ensure hidden state passed to value head matches its dtype (bfloat16)
        value_head_input = policy_output.hidden_states[-1].to(torch.bfloat16)
        value_logits = self.value_model.score(value_head_input)

        # Return original policy output structure and the value logits
        return policy_output, value_logits

    PolicyAndValueWrapper.forward = patched_wrapper_forward
    # print("Patched PolicyAndValueWrapper.forward to ensure bfloat16 consistency for value head.") # DEBUG
    # --- End Patch ---

    # Setup chat format if needed (apply to policy_tokenizer only)
    if use_chat_template:
        if policy_tokenizer.chat_template is None:
            print("Setting up chat format for tokenizer.")
            # Only modify the tokenizer needed for data processing
            policy_tokenizer.chat_template = processor.tokenizer.chat_template # Use a default chatml one if needed
            # Resize embeddings if necessary ONLY for policy model
            # policy_model.resize_token_embeddings(len(policy_tokenizer))
        else:
            print("Tokenizer already has a chat template. Skipping setup.")

    # --- Note: device placement will be handled by accelerate.prepare() inside PPOTrainer ---
    # print(f"Policy model loaded. Device: {policy_model.device}") # Can't check device before prepare

    # ---------- 5. Configure PPO ----------
    print("\n--- Configuring PPO Trainer ---")
    ppo_config = PPOConfig(
        # Inherited TrainingArguments
        output_dir=ppo_output_dir,
        learning_rate=1e-6, # Even lower LR
        per_device_train_batch_size=ppo_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        report_to=log_with,  # Use 'report_to' for logging
        save_strategy="steps", # Add save strategy
        save_steps=100,        # Save every 100 steps
        max_grad_norm=1.0, # Added gradient clipping
        adam_epsilon=1e-7, # Increase Adam epsilon
        # PPO-specific arguments from 0.16.1/matching 0.17 structure
        exp_name=ppo_output_dir,
        reward_model_path=rm_adapter_path,
        num_ppo_epochs=ppo_epochs,
        whiten_rewards=False, # Default
        kl_coef=0.0,       # Temporarily disable KL penalty
        cliprange=0.2,        # Default
        vf_coef=0.1,        # Default
        cliprange_value=0.2,  # Default
        gamma=1.0,          # Default
        lam=0.95,           # Default
        ds3_gather_for_generation=True # Default
        # Removed args not in the provided list: model_adapter_name, ref_adapter_name,
        # total_episodes, model_name_or_path, missing_eos_penalty, num_mini_batches
    )

    # --- Define Prompts Here (Moved Before Usage) ---
    print("\n--- Define Prompts ---")
    prompts = [
        "Write a playful but not cheesy opening line to someone you just matched with.",
        "Send a flirty yet respectful good-morning message:",
        "How would you start a conversation with someone cute at a coffee shop?",
        "Give me a witty compliment I can use.",
        "Text a crush after meeting at a conference.",
        "Break the ice on Bumble.",
        "What's a fun way to ask for someone's number?",
        "Compose a short, intriguing message for a dating app profile visit."
    ]

    # ---------- 6. Tokenize Prompts and Create Dataset ----------
    # Moved dataset tokenization here as policy_tokenizer is now available
    print("\n--- Tokenize Prompts and Create Dataset for PPOTrainer ---")
    def tokenize_prompts(examples):
        return policy_tokenizer(examples["query"], padding=False, truncation=True, max_length=prompt_max_length)
    temp_prompts_dict = {"query": prompts * 5}
    temp_dataset = Dataset.from_dict(temp_prompts_dict)
    tokenized_dataset = temp_dataset.map(tokenize_prompts, batched=True, remove_columns=["query"])
    train_hf_dataset = tokenized_dataset
    print(f"Created tokenized dataset with columns: {train_hf_dataset.column_names}")

    # ---------- 7. Instantiate PPOTrainer ----------
    print("\n--- Instantiate PPOTrainer (Patched Reward Model) ---")
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        model=policy_model,
        value_model=value_model,
        ref_model=None,
        processing_class=policy_tokenizer,
        reward_model=reward_model_peft, # Pass the PeftModel containing the patched base
        train_dataset=train_hf_dataset,
        eval_dataset=train_hf_dataset, # Reuse train dataset for sampling generation
    )
    print("PPOTrainer instantiated.")
    # Move reward_model to CPU again to avoid CUDA embedding asserts
    ppo_trainer.reward_model = ppo_trainer.reward_model.to("cpu")
    print("Warning: reward_model moved to CPU after PPOTrainer init to avoid CUDA device-side asserts.")

    # ---------- 8. Start Training ----------
    print("\n--- Starting PPO Training using ppo_trainer.train() ---")
    ppo_trainer.train()
    print("--- Training Complete ---")

    # ---------- 9. Save Final Model (Implicitly handled by Trainer) ----------
    print(f"\n--- Final model should be saved in {ppo_output_dir} by the Trainer ---")

    # --- REMOVED Sections 7, 8 (Manual Loop), and 9 (Manual Save) ---

    # After loading reward_tokenizer_global and later policy_tokenizer_global, override get_reward to handle re-tokenization
    orig_get_reward = ppo_mod.get_reward
    def patched_get_reward(model, query_responses, pad_token_id, context_length):
        # Decode full sequences into text and re-tokenize with reward tokenizer
        decoded_texts = policy_tokenizer_global.batch_decode(query_responses, skip_special_tokens=True)
        reward_inputs = reward_tokenizer_global(
            decoded_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        ).to(model.device)
        # Use the PeftModel wrapper (SequenceClassification) to compute logits
        outputs = model(
            input_ids=reward_inputs['input_ids'],
            attention_mask=reward_inputs['attention_mask'],
            return_dict=True,
        )
        logits = outputs.logits  # shape (batch_size, num_labels)
        # Assume label 1 is the positive (flirty) score
        positive_scores = logits[:, 1]
        batch_size, seq_len = reward_inputs['input_ids'].size()
        # Construct token-level reward logits (same score for all tokens)
        reward_logits = positive_scores.unsqueeze(1).expand(batch_size, seq_len).contiguous()
        final_rewards = positive_scores
        seq_lengths = torch.full((batch_size,), seq_len - 1, dtype=torch.long, device=positive_scores.device)
        # Move outputs to the same device as the input query_responses
        device = query_responses.device
        reward_logits = reward_logits.to(device)
        final_rewards = final_rewards.to(device)
        seq_lengths = seq_lengths.to(device)
        return reward_logits, final_rewards, seq_lengths
    ppo_mod.get_reward = patched_get_reward

if __name__ == "__main__":
    main() 