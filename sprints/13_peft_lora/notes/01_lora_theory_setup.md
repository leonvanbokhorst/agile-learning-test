# LoRA Theory & Setup Notes

## 1. Motivation for PEFT/LoRA

- **Problem:** Fine-tuning large pre-trained models (like GPT-2) on all parameters is computationally expensive (time, memory, GPU requirements) and storage-intensive (saving full copies for each task).
- **Solution:** Parameter-Efficient Fine-Tuning (PEFT) methods aim to adapt models by tuning only a _small_ number of parameters.
- **Goal:** Achieve performance comparable to full fine-tuning while significantly reducing computational and storage costs, making LLM adaptation more accessible.

## 2. Core Concept of LoRA (Low-Rank Adaptation)

- **Mechanism:**
  - Freeze the original pre-trained model weights.
  - Inject _new_, smaller, trainable layers (called "adapter layers" or "LoRA layers") into the model architecture (often within attention blocks).
  - These adapter layers use **low-rank decomposition**. Instead of learning a full change (ΔW) to a large weight matrix (W), LoRA learns two smaller matrices (A and B) such that ΔW ≈ BA, where the rank `r` (inner dimension of A and B) is much smaller than the original dimensions.
  - During training, only the parameters of these low-rank matrices (A and B) are updated.
  - For inference, the learned change (BA) can be merged back with the original weights (W + BA) without adding extra latency, or kept separate.
- **Benefits:**
  - **Drastically Fewer Trainable Parameters:** Since `r` is small, the number of parameters in A and B is much less than in W.
  - **Reduced Memory Usage:** Less memory needed for gradients and optimizer states during training.
  - **Faster Training:** Fewer parameters to update.
  - **Portable Fine-tuning:** Only the small LoRA adapter weights need to be saved and shared for a specific task, not the entire model.

## 3. Hugging Face `peft` Library

- **Purpose:** Provides implementations of various PEFT methods, including LoRA.
- **Integration:** Works seamlessly with Hugging Face `transformers`, `diffusers`, and `accelerate`.
- **Usage (High-Level):** Involves:
  - Defining a `LoraConfig` specifying parameters like rank (`r`), `lora_alpha`, target modules (which layers to adapt), task type, etc.
  - Using `get_peft_model` to wrap the original `transformers` model with the LoRA adapters based on the config.
  - Training the resulting model as usual (the library handles freezing base model weights and training only adapters).

## 4. Environment Setup

- The `peft` library was added using `uv add peft`.
- Dependencies like `accelerate` were likely installed alongside `peft`.

_(Further details on specific `peft` API usage and configuration options will be added as we implement Task 3)_
