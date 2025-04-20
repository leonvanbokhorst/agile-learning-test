# Sprint 16: PPO Flirty Llama

**Goal:** Use the TRL library to implement Proximal Policy Optimization (PPO) for fine-tuning a Llama 3 model to generate flirty responses.

**Status:** Planning

## Learning Objectives

- [ ] Understand the core concepts of Reinforcement Learning from Human Feedback (RLHF) and PPO.
- [ ] Learn how to use the Hugging Face TRL (Transformer Reinforcement Learning) library.
- [ ] Define a suitable reward function for "flirtiness".
- [ ] Prepare a dataset suitable for PPO fine-tuning (prompts).
- [ ] Configure and run a PPO training job using TRL.
- [ ] Evaluate the "flirtiness" of the fine-tuned model.
- [ ] Compare PPO fine-tuning results with standard Supervised Fine-Tuning (SFT) if applicable.

## Tasks

- [ ] **1. Setup & Environment:**
    - [ ] Ensure `trl`, `peft`, `accelerate`, `transformers`, `datasets`, and `torch` are installed in the project environment (`uv`).
    - [ ] Verify GPU availability and setup for `accelerate`.
    - [ ] (Optional) Add `bitsandbytes` if quantization is needed for Llama 3.
- [ ] **2. Reward Model Implementation:**
    - [ ] Load the `ieuniversity/flirty_or_not` dataset.
    - [ ] Choose and load a base model for the reward classifier (e.g., `distilroberta-base`).
    - [ ] Choose and load the corresponding tokenizer.
    - [ ] Preprocess the dataset (tokenize, format for PyTorch).
    - [ ] Fine-tune the base model on the "flirty" dataset (consider PEFT/LoRA for efficiency).
        - Implement the training loop (e.g., 1 epoch).
        - Save the trained reward model/adapter.
    - [ ] Create the `reward_fn(samples: list[str]) -> torch.Tensor` that takes generated text samples and returns scalar reward scores based on the classifier's positive logit.
- [ ] **3. PPO Agent & Training Setup:**
    - [ ] Load the base policy model (`meta-llama/Meta-Llama-3-8B-Instruct`) using `AutoModelForCausalLMWithValueHead`.
    - [ ] Load the corresponding tokenizer (`AutoTokenizer`).
    - [ ] Configure `PPOConfig` (batch size, learning rate, KL target ~0.1, logging `tensorboard`, etc.).
    - [ ] Instantiate the `PPOTrainer` with the policy model, tokenizer, and PPO config (using `ref_model=None`).
- [ ] **4. PPO Training Loop:**
    - [ ] Define a set of initial prompts relevant to flirty conversation starters (e.g., 20 curated prompts).
    - [ ] Implement the PPO training loop:
        - Sample prompts randomly for each batch.
        - Generate responses using `ppo_trainer.generate()` (with sampling kwargs like `max_new_tokens`, `do_sample`, `top_k`, `temperature`).
        - Decode responses.
        - Compute rewards using the `reward_fn`.
        - Perform PPO step using `ppo_trainer.step(prompts, responses, rewards)`.
    - [ ] Integrate TensorBoard logging to monitor `objective/rlhf_reward`, `policy/approxkl_avg`, etc.
    - [ ] Run the training loop for a sufficient number of steps (e.g., ~500 steps).
- [ ] **5. Evaluation & Saving:**
    - [ ] Periodically sample generations during training to qualitatively assess "flirtiness".
    - [ ] Save the final fine-tuned policy model and tokenizer using `save_pretrained`.
    - [ ] (Optional) Create a simple script or use a Gradio interface to interactively test the final model.
    - [ ] (Optional Stretch) Implement a toxicity filter alongside the flirtiness reward to penalize undesirable outputs.
- [ ] **6. Documentation & Retrospective:**
    - [ ] Document the process, code, and key findings in `notes/`.
    - [ ] Update this `README.md` with results, links to code/models, and learnings.
    - [ ] Update `skills_competencies.md` and `milestones.md`.

## Notes

- Link to notes files will be added here.

## Results

- Link to results files (scripts, model checkpoints) will be added here.
