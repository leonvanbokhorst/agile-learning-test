# Sprint 13: Parameter-Efficient Fine-Tuning (PEFT - LoRA)

**Sprint Goal:** Understand and implement Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning, comparing its effectiveness and resource usage against full fine-tuning.

**Status:** Completed ✅

## Overview

In previous sprints, we explored full fine-tuning for both classification (Sprint 11) and generative tasks (Sprint 12). While effective, fine-tuning all parameters of large models like GPT-2 can be computationally expensive and require significant memory. Parameter-Efficient Fine-Tuning (PEFT) techniques aim to address this by modifying only a small subset of the model's parameters.

This sprint focuses on LoRA (Low-Rank Adaptation), a popular PEFT method. LoRA freezes the pre-trained model weights and injects trainable low-rank decomposition matrices into specific layers (typically attention layers). This drastically reduces the number of trainable parameters, making fine-tuning faster and less memory-intensive, while often achieving performance comparable to full fine-tuning.

We will learn the theory behind LoRA, implement it (likely using the Hugging Face `peft` library), apply it to a fine-tuning task (revisiting either classification or generation from Sprints 11/12), and compare the results.

## Learning Objectives

- Understand the motivation and core concepts behind Parameter-Efficient Fine-Tuning (PEFT).
- Learn the specific mechanism of Low-Rank Adaptation (LoRA).
- Gain practical experience implementing LoRA using libraries like Hugging Face `peft`.
- Apply LoRA to a practical fine-tuning task (e.g., classification or generation).
- Compare the performance (accuracy/perplexity), training time, and number of trainable parameters between LoRA and full fine-tuning.
- Understand the trade-offs involved in using PEFT methods.

## Tasks

- [x] **1. LoRA Theory & Setup:**
  - [x] Research and understand the theoretical underpinnings of LoRA (low-rank decomposition, injecting adapter layers).
  - [x] Explore the Hugging Face `peft` library and its API for LoRA integration.
  - [x] Set up the environment with necessary libraries (e.g., `pip install peft`).
  - [x] Document LoRA concepts and `peft` library usage in `notes/01_lora_theory_setup.md`.
- [x] **2. Choose Task & Baseline:**
  - [x] Decide whether to re-run the classification task (Sprint 11) or the generative task (Sprint 12) as the target for LoRA.
  - [x] Identify the baseline performance and parameter count from the chosen previous sprint's full fine-tuning results.
  - [x] Document the chosen task and baseline metrics in `notes/02_task_baseline.md`.
- [x] **3. Implement LoRA Fine-tuning:**
  - [x] Adapt the chosen fine-tuning script (from Sprint 11 or 12) to incorporate LoRA using the `peft` library.
  - [x] Configure LoRA parameters (e.g., `r`, `lora_alpha`, `target_modules`).
  - [x] Implement the training loop, ensuring only LoRA parameters are trained.
  - [x] Document the implementation details in `notes/03_lora_implementation.md`.
- [x] **4. Run LoRA Fine-tuning & Evaluation:**
  - [x] Execute the LoRA fine-tuning script.
  - [x] Monitor training progress (loss, metrics).
  - [x] Evaluate the final LoRA-tuned model on the relevant test set.
  - [x] Compare performance (accuracy/perplexity) against the full fine-tuning baseline.
  - [x] Compare the number of trainable parameters and approximate training time difference.
  - [x] Document results and comparisons in `notes/04_results_comparison.md`.
- [x] **5. Documentation & Retrospective:**
  - [x] Ensure all code is well-commented and follows project standards.
  - [x] Update `skills_competencies.md` and `milestones.md`.
  - [x] Update this `README.md` with results, links to notes/code, and a retrospective.
  - [x] Update `sprints/backlog.md`.

## Notes & Results Links

_(To be filled in as the sprint progresses)_

- Notes:
  - [LoRA Theory & Setup](./notes/01_lora_theory_setup.md)
  - [Task & Baseline Selection](./notes/02_task_baseline.md)
  - [LoRA Implementation](./notes/03_lora_implementation.md)
  - [Results & Comparison](./notes/04_results_comparison.md)
- Results:
  - LoRA fine-tuning script: [`finetune_lora.py`](./results/finetune_lora.py)
  - Adapter model and config: [`checkpoints/lora_finetuned_model`](./results/checkpoints/lora_finetuned_model/)

## Retrospective

- **What went well?** LoRA adapters trained quickly with only 0.236% trainable parameters and achieved a best perplexity of 1.2496.
- **What could be improved?** Experiment with different LoRA ranks and target modules, enhance evaluation and analysis scripts.
- **Key learnings?** PEFT with the Hugging Face `peft` library makes integrating LoRA straightforward and efficient.
- **Blockers encountered?** No significant blockers; minor logging adjustments were made.
