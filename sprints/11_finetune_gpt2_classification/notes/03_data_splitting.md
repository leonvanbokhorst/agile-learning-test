# Sprint 11 - Task 2: Data Splitting Notes

## Objective

Split the tokenized fake news dataset into distinct training, validation, and test sets for the fine-tuning process.

## Process

1.  **Script Used:** `sprints/11_finetune_gpt2_classification/results/03_split_data.py`
2.  **Data Loading & Tokenization:** The script reused the logic from `02_tokenize_data.py` to load the `Pulk17/Fake-News-Detection-dataset` and tokenize the `text` column using the `gpt2` tokenizer (max length 512, padding, truncation).
3.  **Label Casting (Correction):**
    - **Initial Problem:** The first attempt to split failed because `train_test_split` requires the stratification column (`label`) to be of type `ClassLabel`, but it was initially `Value` (integer).
    - **Fix:** Modified the script to explicitly cast the `label` column to `datasets.ClassLabel(num_classes=2, names=["Fake", "Real"])` _after_ tokenization and before splitting.
4.  **Splitting Strategy:**
    - **First Split:** Applied `tokenized_data.train_test_split` with `test_size=0.2`, `seed=42`, and `stratify_by_column="label"` to create an 80% training set and a 20% temporary set.
    - **Second Split:** Applied `train_test_split` again to the temporary set with `test_size=0.5`, `seed=42`, and `stratify_by_column="label"` to create the final 10% validation and 10% test sets.
5.  **Final Structure:** Combined the resulting splits into a single `DatasetDict` with keys `"train"`, `"validation"`, and `"test"`.
6.  **Saving:** Saved the final `DatasetDict` object to disk using `split_dataset.save_to_disk()`.

## Results

- The script successfully split the dataset after implementing the label casting fix.
- The resulting dataset splits have the following sizes:
  - Train: 24,000 examples
  - Validation: 3,000 examples
  - Test: 3,000 examples
- The splits are saved in the directory: `sprints/11_finetune_gpt2_classification/results/tokenized_data_split/`

## Next Steps

- Load the saved `DatasetDict` containing the train/validation/test splits.
- Set up PyTorch `DataLoader`s to handle batching and shuffling for model training and evaluation.
