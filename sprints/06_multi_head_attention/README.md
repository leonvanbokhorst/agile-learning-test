# Sprint 6: Multi-Head Self-Attention

**Focus:** Understanding and implementing the core mechanism of the Transformer architecture: Multi-Head Self-Attention.

**Goal:** By the end of this sprint, we should have a functional implementation of Multi-Head Self-Attention in PyTorch, including understanding the underlying scaled dot-product attention and masking.

## Tasks

### 1. Scaled Dot-Product Attention

- [x] Understand the formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V ([Notes](./notes/01_scaled_dot_product_attention.md))
- [x] Implement a Python function for scaled dot-product attention using PyTorch tensor operations ([Code](./results/scaled_dot_product_attention.py))
- [x] Test the implementation with sample Query (Q), Key (K), and Value (V) tensors ([Tests](./results/scaled_dot_product_attention.py#L78-L151))
- [x] Document the implementation and concepts in [`notes/01_scaled_dot_product_attention.md`](./notes/01_scaled_dot_product_attention.md).
- [x] Save the implementation in [`results/scaled_dot_product_attention.py`](./results/scaled_dot_product_attention.py).

### 2. Multi-Head Attention Layer

- [x] Understand the concept of splitting Q, K, V into multiple heads ([Notes](./notes/02_multi_head_attention.md))
- [x] Implement an `nn.Module` for Multi-Head Attention ([Code](./results/02_multi_head_attention.py))
  - [x] Include linear projections for Q, K, V for each head.
  - [x] Apply scaled dot-product attention in parallel for each head.
  - [x] Concatenate the outputs of the heads.
  - [x] Apply a final linear projection.
- [x] Test the `MultiHeadAttention` module with appropriate input dimensions ([Tests](./results/02_multi_head_attention.py#L133-L263))
- [x] Document the multi-head architecture and implementation in [`notes/02_multi_head_attention.md`](./notes/02_multi_head_attention.md).
- [x] Save the implementation in [`results/02_multi_head_attention.py`](./results/02_multi_head_attention.py).

### 3. Masking

- [x] Understand the purpose of masking in self-attention (padding and look-ahead) ([Notes](./notes/03_attention_masking.md))
- [x] Implement look-ahead masking (causal masking) generation ([Example](./results/02_multi_head_attention.py#L219-L221))
- [x] Implement padding masking generation ([Example](./results/02_multi_head_attention.py#L178-L180))
- [x] Modify the scaled dot-product attention function to accept and apply masks ([Code](./results/scaled_dot_product_attention.py#L45-L56))
- [x] Test attention calculation with different masks ([Padding Test](./results/02_multi_head_attention.py#L174-L200), [Look-Ahead Test](./results/02_multi_head_attention.py#L208-L251))
- [x] Document masking strategies in [`notes/03_attention_masking.md`](./notes/03_attention_masking.md).
- [x] Add masking examples/tests to result files.

### 4. Integration & Refinement

- [x] Ensure the `MultiHeadAttention` module can handle masks correctly. (Verified via tests)
- [x] Add type hints and thorough docstrings to all implementations. (Verified via review)
- [x] Review code for clarity, efficiency, and adherence to PyTorch best practices. (Completed)
- [-] Add dropout to the attention mechanism (optional but common). (Skipped for this sprint)

## Learning Resources

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- PyTorch `nn.MultiheadAttention` documentation (for comparison)

## Sprint Retrospective (To be filled out)

**What went well?**

- Successfully broke down the complex Multi-Head Attention mechanism into manageable parts (Scaled Dot-Product Attention first).
- Good conceptual understanding of Q, K, V roles and the motivation for multiple heads was achieved through discussion and notes.
- Implementation of both `scaled_dot_product_attention` function and `MultiHeadAttention` module was successful and clean.
- Correctly implemented and tested both padding and look-ahead (causal) masking.
- Testing within the `if __name__ == "__main__":` block proved effective for verifying functionality.

**What could be improved?**

- Initial understanding of mask broadcasting required specific examples.
- Minor syntax errors (f-strings with newlines) popped up during testing but were quickly identified and fixed.

**Key learnings:**

- Solidified understanding of the Scaled Dot-Product Attention formula and implementation.
- Grasped how Multi-Head Attention allows the model to focus on different representation subspaces simultaneously by running scaled dot-product attention in parallel.
- Learned the practical implementation details: linear projections for Q, K, V, splitting/combining heads, and the final output projection.
- Understood the critical roles of padding masks (ignoring padding) and look-ahead masks (preventing future peeking in decoders) and how to implement/apply them.
- Reinforced tensor shape manipulation skills (`view`, `transpose`, `unsqueeze`).

**Action items for next sprint:**

- Focus on understanding Layer Normalization and its placement.
- Pay close attention to how Residual Connections (Add & Norm) are implemented around the attention and feed-forward layers.
