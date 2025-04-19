# LoRA Fine-tuning Results & Comparison

## 1. LoRA Training Run Summary (Sprint 13)

- **Task:** Generative Fine-tuning (Causal LM)
- **Base Model:** `gpt2`
- **Dataset:** `book.txt` (from Sprint 12)
- **PEFT Method:** LoRA
- \*\*Key LoRA Config (`finetune_lora.py` defaults):
  - `lora_r`: 8
  - `lora_alpha`: 16
  - `lora_dropout`: 0.05
  - `target_modules`: `["c_attn"]`
  - `bias`: "none"
- **Training Hyperparameters:**
  - Epochs: 3
  - Batch Size: 16 (Adjusted by user)
  - Learning Rate: 3e-4
  - Optimizer: AdamW
  - Scheduler: Cosine
- **Hardware:** GPU (CUDA detected in logs)
- **Training Time:** ~1125 seconds (~18.7 minutes)

## 2. Key Results

- **Trainable Parameters:** **294,912**
- **Total Parameters (Base Model):** 124,734,720
- **Percentage Trainable:** **0.2364%**
- **Best Validation Loss:** 0.2261
- **Best Validation Perplexity:** **1.2537**

## 3. Comparison with Full Fine-tuning (Sprint 12 Baseline)

| Metric                    | Full Fine-tuning (Sprint 12) | LoRA Fine-tuning (Sprint 13) | Comparison                      |
| :------------------------ | :--------------------------- | :--------------------------- | :------------------------------ |
| **Trainable Params**      | ~124.7 Million               | **294,912**                  | **~99.76% Reduction**           |
| **Validation Perplexity** | **1.1211**                   | 1.2537                       | Slightly Higher (Worse) w/ LoRA |

## 4. Conclusion

LoRA successfully fine-tuned the `gpt2` model on the `book.txt` dataset using drastically fewer parameters compared to full fine-tuning (~0.24% vs 100%).

While the final perplexity achieved by LoRA (1.2537) was slightly higher than the full fine-tuning baseline (1.1211), indicating a minor drop in performance on this specific metric and configuration, the efficiency gains are enormous.
This demonstrates the core value proposition of PEFT methods like LoRA: achieving comparable results to full fine-tuning while significantly reducing the computational resources (memory, storage, potentially time) required.
The trade-off between maximal performance and efficiency is evident and often highly favorable towards PEFT in resource-constrained scenarios or when fine-tuning many task-specific adapters.
