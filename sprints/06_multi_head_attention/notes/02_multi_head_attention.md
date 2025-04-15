# Understanding Multi-Head Attention

Scaled dot-product attention is powerful, but Multi-Head Attention takes it a step further. Instead of calculating attention just once, it allows the model to jointly attend to information from different representation subspaces at different positions.

**Motivation: Why Multiple Heads?**

Imagine you're trying to understand the sentence: "The tired cat quickly jumped onto the warm, sunny windowsill."

- One attention calculation (one "head") might focus on the subject-verb relationship ("cat jumped").
- Another head might focus on adjectives modifying nouns ("tired cat", "warm windowsill", "sunny windowsill").
- A third head might focus on the prepositional relationship ("jumped onto windowsill").

Performing attention multiple times in parallel, with different learned transformations for Q, K, and V each time, allows the model to capture these various types of relationships simultaneously. A single attention mechanism might struggle to learn all these different aspects effectively on its own; it might average them out or focus on only the most dominant one.

**The Mechanism:**

Let's say our model's hidden dimension is `d_model` and we want to use `h` attention heads.

1.  **Input Projections:** Take the input embeddings (same as for single attention, shape `(batch, seq_len, d_model)`). Project them using _separate_ linear layers (`nn.Linear(d_model, d_model)`) for Q, K, and V. Crucially, `d_model` must be divisible by `h`.

    - `Q_proj = Input @ W_Q + b_Q`
    - `K_proj = Input @ W_K + b_K`
    - `V_proj = Input @ W_V + b_V`
      _(Note: These `W_Q`, `W_K`, `W_V` project to the full `d_model`, unlike the per-head projections described next, though implementations vary. Often these initial projections are directly to the concatenated head dimension)._ A more common view combines this with the next step:

2.  **Split into Heads:** Reshape the projected Q, K, V tensors to split the last dimension (`d_model`) into `h` heads, each with dimension `d_k = d_model / h` (for Q and K) and `d_v = d_model / h` (for V). The new shape becomes `(batch, h, seq_len, d_k)` or `(batch, h, seq_len, d_v)` (often achieved by reshaping and transposing).

    - Think of having `h` different sets of Q, K, V, each of a smaller dimension (`d_k`, `d_v`).
    - Each head gets its _own learned subspace_ to work in.

3.  **Parallel Attention:** Apply the `scaled_dot_product_attention` function (the one we just built!) independently to each head. The function handles the shapes `(batch, h, seq_len_q, d_k)` etc., performing the attention calculation across all batches and heads simultaneously thanks to tensor operations.

    - `head_i_output, head_i_weights = scaled_dot_product_attention(Q_head_i, K_head_i, V_head_i, mask)`
    - This results in `h` output tensors of shape `(batch, seq_len_q, d_v)`.

4.  **Concatenate Heads:** Combine the outputs from all heads back together by concatenating them along the last dimension.

    - `concat_output = torch.cat([head_1_output, ..., head_h_output], dim=-1)`
    - The shape becomes `(batch, seq_len_q, h * d_v)`, which is `(batch, seq_len_q, d_model)` since `h * d_v = d_model`.

5.  **Final Linear Projection:** Pass the concatenated output through one final linear layer (`W_O`), `nn.Linear(d_model, d_model)`.
    - `MultiHeadOutput = concat_output @ W_O + b_O`
    - This allows the model to blend the information learned by the different heads.

**Benefits:**

- **Richer Representations:** Allows the model to capture different types of relationships and nuances in the data.
- **Focus:** Each head can specialize in attending to different parts or aspects of the sequence.
- **Robustness:** Less likely to miss important information compared to a single attention mechanism.

**In short:** Multi-Head Attention = running scaled dot-product attention multiple times in parallel with different learned projections, then combining the results.
