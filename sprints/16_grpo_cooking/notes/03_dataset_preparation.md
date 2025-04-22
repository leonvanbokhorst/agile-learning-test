# Task 3: Dataset Preparation Notes

**Goal:** Load, inspect, preprocess the `moremilk/CoT_Reasoning_Cooking` dataset into the required format for GRPO, and upload it for reuse.

**Script:** `sprints/16_grpo_cooking/prepare_dataset.py`

**Steps Taken:**

1.  **Initial Loading & Inspection:**
    *   Used `datasets.load_dataset("moremilk/CoT_Reasoning_Cooking")`.
    *   Inspected features (`question`, `answer`, `metadata` containing `reasoning`).

2.  **Format Definition (Prompt/Chosen/Rejected):**
    *   **Prompt:** Defined based on the Llama 3.2 Instruct template (verified via web search), including system prompt and user question, ending before the assistant response (`...assistant<|end_header_id|>\n\n`).
    *   **Chosen:** Formatted as the desired assistant output, containing the reasoning from the dataset wrapped in `<think>` tags, followed by the answer, and ending with `<|eot_id|>`. Used triple-quoted f-string for multi-line formatting.
    *   **Rejected:** Aimed for a direct answer without the CoT reasoning. Generated using the base model (`unsloth/Llama-3.2-3B-Instruct`) loaded within the script.

3.  **Chat Template Correction:**
    *   Initially used a slightly incorrect template.
    *   Performed a web search (`Llama 3.2 Instruct chat template format`).
    *   Corrected the template in the script to use the proper special tokens (`<|begin_of_text|>`, `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`) and newline structure.

4.  **Rejected Response Generation:**
    *   Loaded the base model and tokenizer within the script.
    *   Created a function (`generate_batch_direct_answers`) to generate responses for a batch of prompts.
    *   Used a different system prompt for generation to encourage direct answers.
    *   Added `<|eot_id|>` manually to the end of generated rejected responses.

5.  **Optimization (Batching):**
    *   Realized single-example generation would be very slow.
    *   Refactored the script to process the dataset in batches.
    *   Implemented a loop iterating in steps of `batch_size`.
    *   Correctly handled slicing the `Dataset` object within the loop to reconstruct individual examples for formatting.
    *   Called the batched generation function (`generate_batch_direct_answers`).
    *   Started with `batch_size=8`, user increased to `batch_size=16` based on VRAM.

6.  **Hugging Face Hub Integration:**
    *   Added `python-dotenv` to load environment variables.
    *   Read `HF_USERNAME` from `.env`.
    *   Used `datasets.Dataset.from_list()` to convert the processed Python list to a Dataset object.
    *   Used `processed_dataset.push_to_hub(hub_dataset_id)` to upload the result.
    *   Confirmed the script used the `HUGGINGFACE_API_KEY` from `.env` for authentication.

7.  **Execution & Verification:**
    *   Ran a sample run (first 10 examples) to test the full pipeline (including Hub push). Confirmed success.
    *   Ran the full dataset processing with `batch_size=16`. Took approximately 20 minutes.

**Outcome:** The `moremilk/CoT_Reasoning_Cooking` dataset was successfully processed into the `prompt`/`chosen`/`rejected` format suitable for GRPO training. The resulting dataset was uploaded to the Hugging Face Hub at [leonvanbokhorst/CoT_Reasoning_Cooking_GRPO_Formatted](https://huggingface.co/datasets/leonvanbokhorst/CoT_Reasoning_Cooking_GRPO_Formatted). 