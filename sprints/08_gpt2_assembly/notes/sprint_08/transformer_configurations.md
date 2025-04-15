# Note: Transformer Architectures - A Family Portrait

While we're building a specific _type_ of Transformer (GPT-style, decoder-only), it's helpful to understand the main architectural variations. Think of them as different specialists within the Transformer family.

## 1. The Encoder-Decoder Model (The Original "Transformer")

- **Structure:** Consists of two main stacks: an **Encoder** stack and a **Decoder** stack.
- **Encoder Role:** Reads the entire input sequence (like a sentence in the source language) and builds a rich representation (context vectors) capturing its meaning. It uses self-attention where tokens can attend to all other tokens in the input.
- **Decoder Role:** Generates the output sequence (like the translated sentence) one token at a time. It uses two types of attention:
  - **Masked Self-Attention:** Attends to the previously generated tokens in the output sequence (cannot see future tokens).
  - **Cross-Attention:** Attends to the _encoder's output_ representations, allowing it to base the generation on the input sequence's meaning.
- **Use Cases:** Excellent for sequence-to-sequence tasks where you map an input sequence to a different output sequence.
  - Machine Translation (e.g., English to French)
  - Summarization
  - Question Answering (where the answer is generated based on context)
- **Examples:** Original Transformer paper, T5, BART, mBART.

## 2. The Encoder-Only Model (The Understander)

- **Structure:** Consists _only_ of an **Encoder** stack.
- **Mechanism:** Processes the input sequence using self-attention, allowing every token to attend to every other token (bidirectional context). It generates contextualized representations for each input token.
- **Output:** Doesn't generate sequences autoregressively. Instead, the final representations from the encoder are typically fed into a task-specific "head" (e.g., a linear layer for classification).
- **Use Cases:** Tasks requiring strong understanding of the input sequence.
  - Text Classification (Sentiment Analysis, Topic Classification)
  - Named Entity Recognition (NER)
  - Sequence Labeling
  - Masked Language Modeling (predicting masked tokens within the input)
- **Examples:** BERT, RoBERTa, ALBERT, DistilBERT.

## 3. The Decoder-Only Model (The Generator - What We're Building!)

- **Structure:** Consists _only_ of a **Decoder** stack, but crucially, it _lacks the cross-attention_ mechanism found in the encoder-decoder model's decoder.
- **Mechanism:** Uses **Masked Self-Attention** only. Each token can only attend to itself and the preceding tokens in the sequence. This makes it inherently **autoregressive** – it predicts the next token based solely on the tokens generated so far.
- **Input/Output:** Takes a sequence prompt and generates continuations token by token.
- **Use Cases:** Primarily focused on text generation.
  - Language Modeling (predicting the next word)
  - Text Generation (stories, code, dialogue)
  - Few-shot / Zero-shot learning via prompting
- **Examples:** GPT series (GPT-2, GPT-3, GPT-4), LLaMA, PaLM, Claude.

**Our Model (`GPT` in `model.py`) fits squarely into the Decoder-Only category.** That's why our `GPTDecoderBlock` only contains masked self-attention and feed-forward layers, and no cross-attention.
