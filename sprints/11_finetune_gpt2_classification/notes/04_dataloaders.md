# Sprint 11 - Task 2: DataLoader Creation Notes

## Objective

Create PyTorch `DataLoader` objects to efficiently batch and serve the preprocessed (tokenized and split) data during model training and evaluation.

## Process

1.  **Script Used:** `sprints/11_finetune_gpt2_classification/results/04_create_dataloaders.py`
2.  **Load Preprocessed Data:** Loaded the saved `DatasetDict` (containing `train`, `validation`, and `test` splits) from the `tokenized_data_split` directory using `load_from_disk`.
3.  **Set Format:** Explicitly set the format of all dataset splits to `"torch"` using `.set_format()`, ensuring that the relevant columns (`input_ids`, `attention_mask`, `label`) are returned as PyTorch tensors.
4.  **Instantiate DataLoaders:**
    - Imported `DataLoader` from `torch.utils.data`.
    - Set a `BATCH_SIZE` of 16.
    - Created `train_dataloader` with `shuffle=True`.
    - Created `val_dataloader` and `test_dataloader` with `shuffle=False`.
5.  **Verification:**
    - Retrieved one batch from the `train_dataloader` using `next(iter(train_dataloader))`.
    - Printed the keys and the shapes of the tensors within the batch.

## Results

- The script executed successfully.
- `train_dataloader`, `val_dataloader`, and `test_dataloader` objects were created.
- Verification confirmed that batches contain the expected keys (`label`, `input_ids`, `attention_mask`).
- Tensor shapes matched expectations for the chosen batch size (16) and max length (512):
  - `label`: `torch.Size([16])`
  - `input_ids`: `torch.Size([16, 512])`
  - `attention_mask`: `torch.Size([16, 512])`
- The `label` tensor dtype was confirmed as `torch.int64`.

## Conclusion

Task 2 (Tokenizer Adaptation & Data Prep) is complete. We now have the data fully processed, split, and ready to be fed into a PyTorch model via DataLoaders.

## Next Steps

Proceed to Task 3: Model Modification (Loading `GPT2ForSequenceClassification`).
