# Sprint 6: Multi-Head Self-Attention

**Focus:** Understanding and implementing the core mechanism of the Transformer architecture: Multi-Head Self-Attention.

**Goal:** By the end of this sprint, we should have a functional implementation of Multi-Head Self-Attention in PyTorch, including understanding the underlying scaled dot-product attention and masking.

## Tasks

### 1. Scaled Dot-Product Attention

- [ ] Understand the formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
- [ ] Implement a Python function for scaled dot-product attention using PyTorch tensor operations.
- [ ] Test the implementation with sample Query (Q), Key (K), and Value (V) tensors.
- [ ] Document the implementation and concepts in `notes/01_scaled_dot_product_attention.md`.
- [ ] Save the implementation in `results/01_scaled_dot_product_attention.py`.

### 2. Multi-Head Attention Layer

- [ ] Understand the concept of splitting Q, K, V into multiple heads.
- [ ] Implement an `nn.Module` for Multi-Head Attention.
  - [ ] Include linear projections for Q, K, V for each head.
  - [ ] Apply scaled dot-product attention in parallel for each head.
  - [ ] Concatenate the outputs of the heads.
  - [ ] Apply a final linear projection.
- [ ] Test the `MultiHeadAttention` module with appropriate input dimensions.
- [ ] Document the multi-head architecture and implementation in `notes/02_multi_head_attention.md`.
- [ ] Save the implementation in `results/02_multi_head_attention.py`.

### 3. Masking

- [ ] Understand the purpose of masking in self-attention (e.g., preventing attention to future tokens in decoders, ignoring padding).
- [ ] Implement look-ahead masking (causal masking) for decoder self-attention.
- [ ] Implement padding masking.
- [ ] Modify the scaled dot-product attention function to accept and apply masks.
- [ ] Test attention calculation with different masks.
- [ ] Document masking strategies in `notes/03_attention_masking.md`.
- [ ] Add masking examples/tests to `results/01_scaled_dot_product_attention.py` or create `results/03_attention_masking_examples.py`.

### 4. Integration & Refinement

- [ ] Ensure the `MultiHeadAttention` module can handle masks correctly.
- [ ] Add type hints and thorough docstrings to all implementations.
- [ ] Review code for clarity, efficiency, and adherence to PyTorch best practices.
- [ ] Add dropout to the attention mechanism (optional but common).

## Learning Resources

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- PyTorch `nn.MultiheadAttention` documentation (for comparison)

## Sprint Retrospective (To be filled out)

**What went well?**

- ...

**What could be improved?**

- ...

**Key learnings:**

- ...

**Action items for next sprint:**

- ...
