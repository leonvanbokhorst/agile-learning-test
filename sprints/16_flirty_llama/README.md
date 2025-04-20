# Sprint 16: Fine-tuning Llama-3 for Flirtiness with PPO

**Goal:** Use the `trl` library to implement Proximal Policy Optimization (PPO) to fine-tune the `meta-llama/Meta-Llama-3-8B-Instruct` model to generate flirty, yet appropriate, dialogue.

**Reference:** [Ideation Notes](ideation.md)

## Tasks

### 1. Setup & Environment

- [ ] Ensure `trl`, `peft`, `accelerate`, `transformers`, `datasets`, and `torch` are installed in the project environment (`uv`).
- [ ] Verify GPU availability and setup for `accelerate`.

### 2. Reward Model Implementation

- [ ] Load the `ieuniversity/flirty_or_not` dataset.
- [ ] Choose a base model for the reward classifier (e.g., `distilroberta-base` as suggested).
- [ ] Preprocess the dataset (tokenize, format for PyTorch).
- [ ] Fine-tune the base model (potentially using PEFT/LoRA for efficiency) on the flirty dataset to act as a reward model.
  - Implement the training loop.
  - Save the trained reward model/adapter.
- [ ] Create the `reward_fn(samples: list[str]) -> torch.Tensor` that takes generated text samples and returns scalar reward scores based on the classifier's logits.

### 3. PPO Agent & Training Setup

- [ ] Load the base policy model (`meta-llama/Meta-Llama-3-8B-Instruct`) using `AutoModelForCausalLMWithValueHead`.
- [ ] Load the corresponding tokenizer (`AutoTokenizer`).
- [ ] Configure `PPOConfig` (batch size, learning rate, KL target, logging, etc.).
- [ ] Instantiate the `PPOTrainer` with the policy model, (optional) reference model, tokenizer, and PPO config.

### 4. PPO Training Loop

- [ ] Define a set of initial prompts relevant to flirty conversation starters.
- [ ] Implement the PPO training loop:
  - Sample prompts.
  - Generate responses using `ppo_trainer.generate()`.
  - Decode responses.
  - Compute rewards using the `reward_fn`.
  - Perform PPO step using `ppo_trainer.step(prompts, responses, rewards)`.
- [ ] Integrate TensorBoard logging (`log_with="tensorboard"` in `PPOConfig`) to monitor `objective/rlhf_reward`, `policy/approxkl_avg`, etc.
- [ ] Run the training loop for a sufficient number of steps (e.g., 500 as suggested, adjust based on compute).

### 5. Evaluation & Saving

- [ ] Periodically sample generations during training to qualitatively assess "flirtiness".
- [ ] Save the final fine-tuned policy model and tokenizer using `save_pretrained`.
- [ ] (Optional) Create a simple script or use a Gradio interface to interactively test the final model.
- [ ] (Optional Stretch) Implement a toxicity filter alongside the flirtiness reward to penalize undesirable outputs.

### 6. Documentation & Retrospective

- [ ] Document the process, code, and key findings in `notes/`.
- [ ] Update this `README.md` with results, links to code/models, and learnings.
- [ ] Update `skills_competencies.md` and `milestones.md`.
- [ ] Update `sprints/backlog.md` for the next sprint.

## Key Learnings Targeted

- Reinforcement Learning from Human Feedback (RLHF) concepts.
- Practical implementation of PPO using `trl`.
- Building and using custom reward models.
- Fine-tuning large language models for specific stylistic attributes.
- Using `accelerate` for efficient training.
- Monitoring RL training metrics (KL divergence, reward).

## Initial Code Structure

```
sprints/16_flirty_llama/
├── README.md
├── ideation.md
├── requirements.txt  # Or confirm uv environment has deps
├── reward_model/
│   ├── train_reward_model.py
│   └── flirty_reward_model/  # Saved model/adapter
├── ppo_finetune/
│   ├── flirty_ppo.py         # Main PPO script based on ideation
│   └── flirty-llama-ppo/     # Saved final model
└── notes/
    └── ppo_notes.md
```
