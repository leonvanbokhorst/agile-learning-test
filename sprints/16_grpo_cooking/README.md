# Sprint 16: GRPO Fine-Tuning of Llama3.2 3B for Chain-of-Thought Reasoning

**Epic:** Enhance LLM reasoning capabilities for specialized domains.

**Goal:** Fine-tune the Llama 3.2 3B model using Group Relative Policy Optimization (GRPO) to improve its chain-of-thought (CoT) reasoning abilities, specifically on the `moremilk/CoT_Reasoning_Cooking` dataset.

**Sprint Tasks:**

Based on the detailed plan in [ideation.md](ideation.md) and concepts in [notes/00_concepts.md](notes/00_concepts.md):

1.  [X] **Set Up RLHF Training Environment:** Prepare the local machine (WSL2, GPU, Python, PyTorch, Hugging Face libs, TRL) for RLHF training.
2.  [X] **Load and Prepare Llama3.2 3B Model:** Select, access, and configure the Llama 3.2 3B-Instruct model as the base policy model (πθ).
3.  [X] **Prepare CoT Reasoning Dataset:** Load, inspect, preprocess (`prepare_dataset.py`), and upload (`leonvanbokhorst/CoT_Reasoning_Cooking_GRPO_Formatted`) the `moremilk/CoT_Reasoning_Cooking` dataset.
4.  [X] **Build Reward Model (RMφ):** Create and train a reward model ([`train_reward_model.py`](./train_reward_model.py)) to assess the quality of generated CoT answers, saving the result to [`results/reward_model`](./results/reward_model).
5.  [WIP] **Implement GRPO Algorithm:** Set up the GRPO training logic ([`train_grpo.py`](./train_grpo.py)) using the TRL library, defining hyperparameters (group size k=4, KL coeff β=0.1, etc.).
6.  [ ] **Fine-Tune Llama3.2 with GRPO:** Execute the training loop, monitoring progress, rewards, and KL divergence, and save checkpoints.
7.  [ ] **Evaluate Pre- vs Post-GRPO Performance:** Compare the baseline and fine-tuned models quantitatively (accuracy, reward scores) and qualitatively (CoT coherence, correctness) on a test set.

**Notes & Results:**

- Conceptual Overview: [RLHF, RM, & GRPO Concepts](notes/00_concepts.md)
- Processed dataset: [leonvanbokhorst/CoT_Reasoning_Cooking_GRPO_Formatted](https://huggingface.co/datasets/leonvanbokhorst/CoT_Reasoning_Cooking_GRPO_Formatted)
- Task 1 Details: [Environment Setup Notes](notes/01_environment_setup.md)
- Task 2 Details: [Model Loading Notes](notes/02_model_loading.md)
- Task 3 Details: [Dataset Preparation Notes](notes/03_dataset_preparation.md)
- Trained Reward Model: [`results/reward_model`](./results/reward_model)
- _(Links to results, code, trained models, etc. will be added here as tasks are completed)_

**Retrospective:**

- _(Sprint outcomes and learnings will be summarized here)_
