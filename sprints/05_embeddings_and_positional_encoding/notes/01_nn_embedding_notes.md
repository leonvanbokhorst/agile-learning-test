# Notes: nn.Embedding Basics

**Corresponding Results:** [`results/01_nn_embedding_basics.py`](../results/01_nn_embedding_basics.py)

## What are Embeddings?

- **Purpose:** To map discrete items (like words, characters, category labels) represented by integer indices to dense, continuous vectors.
- **Why:** Neural networks work with continuous numerical data. Embeddings provide a way to represent discrete inputs in a format the network can process and learn from.
- **Benefit:** These dense vectors (embeddings) can capture semantic similarities and relationships between the discrete items. Items with similar meanings or functions might end up with similar embedding vectors after training.

## `torch.nn.Embedding`

- **Core Idea:** It's essentially a learnable **lookup table** (a matrix).
- **Initialization:**

  ```python
  import torch.nn as nn

  vocab_size = 10000 # How many unique items in our dictionary
  embedding_dimension = 300 # The desired size of the vector for each item

  embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dimension)
  ```

- **Key Parameters:**
  - `num_embeddings` (int): The size of the dictionary (e.g., vocabulary size). The input indices must be in the range `[0, num_embeddings - 1]`.
  - `embedding_dim` (int): The dimension (size) of the dense vector for each item.
- **The Weight Matrix:** The layer holds a weight tensor (`embedding_layer.weight`) of shape `(num_embeddings, embedding_dim)`. This _is_ the lookup table. Each row `i` corresponds to the embedding vector for index `i`.
  - By default, these weights are initialized randomly (often from $\mathcal{N}(0, 1)$) and are treated as model parameters to be learned during training via backpropagation.

## How to Use It

1.  **Input:** You provide the `nn.Embedding` layer with a tensor of **integer indices** (`torch.LongTensor`).
    - The indices represent the specific items you want to get embeddings for.
    - Input shape can be `(sequence_length)` for a single sequence or `(batch_size, sequence_length)` for a batch of sequences.
2.  **Lookup:** The layer fetches the corresponding embedding vector (row) from its internal weight matrix for each input index.
3.  **Output:**
    - The output tensor has the same shape as the input tensor, but with an extra dimension added at the end: the `embedding_dim`.
    - Input `(sequence_length)` -> Output `(sequence_length, embedding_dim)`
    - Input `(batch_size, sequence_length)` -> Output `(batch_size, sequence_length, embedding_dim)`

```python
import torch

# Example from results/01_nn_embedding_basics.py
num_embeddings = 10
embedding_dim = 3
embedding_layer = nn.Embedding(num_embeddings, embedding_dim)

# Input: Batch of 2 sequences, each length 4
input_indices = torch.LongTensor([[1, 2, 4, 5], [4, 3, 0, 9]])
# input_indices shape: [2, 4]

# Output: Corresponding embeddings
output_embeddings = embedding_layer(input_indices)
# output_embeddings shape: [2, 4, 3]

# Verification: output_embeddings[0, 1] is the same vector as embedding_layer.weight[2]
print(torch.equal(output_embeddings[0, 1], embedding_layer.weight[2])) # --> True
```

## Learnable Nature

- The embedding vectors are **parameters** of the model.
- During training, gradients flow back to the embedding layer, and the optimizer updates the embedding vectors just like any other weight in the network.
- This allows the model to learn meaningful representations for the discrete inputs based on the task objective.

## Common Use Cases

- **Natural Language Processing (NLP):** Representing words, subwords, or characters.
- **Recommendation Systems:** Representing users and items.
- **Categorical Features:** Representing distinct categories in tabular data.

## Distinction from Pre-trained Embeddings

It's important to distinguish the `nn.Embedding` layer discussed here from the embeddings commonly used in Retrieval-Augmented Generation (RAG) or other semantic search tasks.

| Feature             | `nn.Embedding` (Learned in _this_ model)                       | Pre-trained Embeddings (e.g., for RAG)                                      |
| :------------------ | :------------------------------------------------------------- | :-------------------------------------------------------------------------- |
| **What's Embedded** | Discrete Tokens (words, subwords, IDs)                         | Text Chunks (sentences, paragraphs)                                         |
| **Origin**          | **Learned** during model training                              | **Pre-trained** by large, separate models (e.g., Sentence-BERT, OpenAI Ada) |
| **Mutability**      | **Learnable Parameters:** Changes during this model's training | Usually **Fixed/Static:** Used as-is, not trained with this model           |
| **Primary Use**     | Internal representation _within_ the model                     | Semantic search/retrieval _outside_ the main generative model               |
| **Scope**           | Optimized for _this_ specific model's task                     | General-purpose semantic meaning                                            |

In essence:

- `nn.Embedding` is a **building block** we train _inside_ our model to create representations tailored to its specific job.
- RAG embeddings are typically **outputs** from _other, pre-existing models_, used as a tool to find relevant information _before_ feeding it to our model.

## Role within a Model Architecture

Think of `nn.Embedding` as just another layer type within the `torch.nn` toolkit, like `nn.Linear` or `nn.Conv2d`. You can integrate it into your custom `nn.Module` definitions.

- **Placement:** It typically acts as the **first layer** when dealing with sequences of discrete inputs (like word indices). Its job is to convert these indices into meaningful vector representations before further processing.

- **Multiple Embeddings:** A model can have **more than one** `nn.Embedding` layer. This is useful if you have different types of discrete inputs (e.g., word tokens and user IDs) that need separate embedding spaces.

- **Connection to Other Layers:**

  - The **output** of the `nn.Embedding` layer (a sequence of dense vectors, e.g., shape `[batch_size, sequence_length, embedding_dim]`) is fed directly into subsequent layers.
  - These next layers could be RNNs (LSTM/GRU), 1D CNNs, Transformer blocks (Attention, FeedForward), or other types of layers designed to process sequences.

- **Contextual Learning:** The embedding layer doesn't learn in isolation. Error gradients from the downstream task flow back _through_ the subsequent layers and update the `nn.Embedding` layer's weights. This means the embedding vectors are optimized to be useful _for the specific task and the specific architecture_ they are part of.

- **LEGO Analogy:** Imagine `nn.Embedding` as a specialized LEGO factory. It takes specific types of raw LEGO bricks (input indices) and transforms each one into a more complex, standard-sized assembly (the embedding vector). You then use these standard assemblies as the input pieces for building your larger LEGO creation (the rest of your neural network). The factory constantly refines how it makes the assemblies based on how well the final creation works.
