# Sprint 11 - Task 1: Dataset Selection & Preparation Notes

## Objective

Find and load a suitable dataset for fine-tuning GPT-2 on a binary text classification task.

## Process

1.  **Initial Idea:** Use a dataset distinguishing real news from satirical news (The Onion). This seemed like a fun twist on standard sentiment analysis.
2.  **Attempt 1 (`splitgraph/onion-news-headlines`):** Tried loading this dataset using the `datasets` library via `01_load_dataset.py`.
    - **Result:** Failed. The dataset ID was not found on the Hugging Face Hub.
3.  **Attempt 2 (`onion`):** Modified the script to try the simpler ID `onion`.
    - **Result:** Failed. This ID was also not found on the Hub.
4.  **Web Search & Pivot:** Searched for alternative fake news datasets on Hugging Face. Identified several candidates, including datasets based on Kaggle sources and purpose-built ones.
5.  **Attempt 3 (`Pulk17/Fake-News-Detection-dataset`):** Selected this dataset as it seemed well-suited and directly available. Modified `01_load_dataset.py` to load this ID.
    - **Result:** Success! The dataset loaded correctly.

## Loaded Dataset Details (`Pulk17/Fake-News-Detection-dataset`)

- **Source:** Hugging Face Hub
- **Script:** `sprints/11_finetune_gpt2_classification/results/01_load_dataset.py`
- **Structure:** Contains a single `train` split with 30,000 examples.
  - _Note:_ Will need manual splitting into train/validation/test sets later.
- **Relevant Features:**
  - `text` (string): Contains the news article content.
  - `label` (int64): The classification label.
- **Label Mapping (Inferred):**
  - `0`: Fake News
  - `1`: Real News
- **Other Features:** `Unnamed: 0`, `title`, `subject`, `date` (likely ignorable for basic classification).

## Next Steps

Proceed to Task 2: Tokenizer Adaptation, using the loaded `text` and `label` columns.
