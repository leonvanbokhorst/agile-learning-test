# Position-wise Feed-Forward Network (FFN)

## What is it?

The Position-wise Feed-Forward Network (FFN) is the second major sub-layer within each Transformer Encoder and Decoder block, following the Multi-Head Attention sub-layer (and its Add & Norm).

It consists of a simple two-layer fully connected neural network (or Multi-Layer Perceptron - MLP) that is applied independently and identically to each position in the sequence.

## Structure

The FFN typically has the following structure:

1.  **Linear Layer 1 (Expansion):** Takes the input of dimension $d_{\text{model}}$ and projects it to a larger inner dimension $d_{ff}$.
    - `nn.Linear(d_model, d_ff)`
2.  **Non-linear Activation:** An activation function is applied element-wise. Common choices include ReLU (`nn.ReLU`) or GELU (`nn.GELU`). GELU (Gaussian Error Linear Unit) is often used in more recent Transformer variants like BERT and GPT-2/3.
3.  **Dropout (Optional but common):** Dropout can be applied after the activation function for regularization.
    - `nn.Dropout(dropout_prob)`
4.  **Linear Layer 2 (Contraction):** Takes the activated, potentially dropped-out output of dimension $d_{ff}$ and projects it back down to the original dimension $d_{\text{model}}$.
    - `nn.Linear(d_ff, d_model)`

The formula for an input vector $x$ corresponding to a single position is:

$$ \text{FFN}(x) = W_2 (\text{Activation}(W_1 x + b_1)) + b_2 $$

Where $W_1, b_1, W_2, b_2$ are the learnable weights and biases of the two linear layers.

**Key Point:** The _same_ $W_1, b_1, W_2, b_2$ parameters are used for _every position_ in the sequence. This is why it's called "position-wise". The network processes each position independently using the identical transformation.

## Configuration

- **Inner Dimension ($d_{ff}$):** In the original Transformer paper ("Attention Is All You Need"), the inner dimension $d_{ff}$ was set to $4 \times d_{\text{model}}$. For example, if $d_{\text{model}} = 512$, then $d_{ff} = 2048$. This expansion factor of 4 is a common convention, but other values can be used.
- **Activation Function:** While the original paper used ReLU, GELU has become a popular alternative, often showing slightly better performance in large language models.
- **Dropout:** A dropout layer is frequently included within the FFN, often after the activation function or after the second linear layer, to prevent overfitting.

## Purpose

While the self-attention layers are responsible for relating different positions in the sequence and aggregating information, the FFN layers provide additional computational capacity and non-linearity.

- They process the representation of each position independently after information has been mixed by the attention mechanism.
- They introduce further non-linear transformations, allowing the model to learn more complex functions.
- The expansion to $d_{ff}$ temporarily increases the model's capacity to transform the features at each position.

Like the attention sub-layer, the FFN sub-layer is also typically wrapped within an "Add & Norm" component:

$$ \text{Output} = \text{LayerNorm}(\text{input_to_ffn} + \text{Dropout}(\text{FFN}(\text{input_to_ffn}))) $$
