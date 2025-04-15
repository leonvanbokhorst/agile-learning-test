# Understanding Attention Masking

In Transformer models, attention masks are crucial for controlling which parts of the sequence a particular token should be allowed to attend to. They prevent the model from accessing information it shouldn't, ensuring correctness and preventing "cheating".

There are two primary types of attention masks:

## 1. Padding Mask

**Purpose:** To prevent the attention mechanism from considering padding tokens in the input sequence.

- **Why?** Input sequences in a batch often have different lengths. To process them together efficiently, shorter sequences are padded (usually with zeros) to match the length of the longest sequence. These padding tokens don't contain real information and should be ignored during attention calculations.
- **How it works:** We create a boolean mask tensor with the same shape as the input sequence (or broadcastable to the attention scores shape). It has `True` for real tokens and `False` for padding tokens.
- **Implementation:** Inside the attention calculation (like our `scaled_dot_product_attention`), before the softmax step, we use this mask to set the attention scores corresponding to padding keys to a very large negative number (like `-infinity`). When softmax is applied, these positions get effectively zero weight.

**Example (Key Padding Mask):**

```python
# Assume seq_len_k = 7
# Batch item 0 has 5 real tokens, Batch item 1 has 6 real tokens
key_padding_mask = torch.tensor([
    [True, True, True, True, True, False, False],  # 5 real, 2 pad
    [True, True, True, True, True, True, False]   # 6 real, 1 pad
], dtype=torch.bool)
# Shape: (batch_size, seq_len_k)
```

This mask tells the attention mechanism: "For the first batch item, ignore keys at index 5 and 6. For the second batch item, ignore the key at index 6."

- **Broadcasting:** When used in `scaled_dot_product_attention`, this key padding mask needs to be broadcastable to the attention scores shape (`batch_size, num_heads, seq_len_q, seq_len_k`). This is typically done by adding dimensions for the heads and query sequence length:
  `attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)` resulting in shape `(batch_size, 1, 1, seq_len_k)`.

## 2. Look-Ahead Mask (Causal Mask)

**Purpose:** To prevent positions from attending to subsequent (future) positions. This is essential in decoder architectures during training for tasks like language modeling.

- **Why?** When training a model to predict the next word in a sequence (autoregressive generation), the prediction for position `i` should only depend on the known outputs at positions less than `i`. Allowing attention to future tokens would be like giving the model the answers during training.
- **How it works:** We create a square boolean mask where the entry `mask[i, j]` is `True` if position `i` is allowed to attend to position `j`, and `False` otherwise. For causal masking, position `i` can attend to positions `0` through `i`, but not `i+1` onwards.
- **Implementation:** Similar to the padding mask, this mask is applied before the softmax, setting attention scores for future positions to `-infinity`.

**Example (Look-Ahead Mask for `seq_len = 4`):**

```python
seq_len = 4
# Create an upper triangular matrix (True on and below diagonal)
look_ahead_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
# Result:
# tensor([[ True, False, False, False],
#         [ True,  True, False, False],
#         [ True,  True,  True, False],
#         [ True,  True,  True,  True]])
```

This mask means:

- Position 0 can only attend to position 0.
- Position 1 can attend to positions 0 and 1.
- Position 2 can attend to positions 0, 1, and 2.
- Position 3 can attend to positions 0, 1, 2, and 3.

- **Combining Masks:** Often, decoders need both padding and look-ahead masking. This can be achieved by creating both masks and combining them using logical AND (`&`). The combined mask would then have `False` for both padding positions _and_ future positions.

We have already implemented the logic to handle boolean masks (where `False` indicates a position to mask out) within our `scaled_dot_product_attention` function. The key difference is how we _generate_ the mask tensor itself (padding vs. look-ahead).
