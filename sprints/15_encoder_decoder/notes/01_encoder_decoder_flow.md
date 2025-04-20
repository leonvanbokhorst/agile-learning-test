# Encoder-Decoder Data Flow

This document outlines the step-by-step data flow through the `EncoderDecoder` model implemented in `results/encoder_decoder_model.py`.

## Key Components

- **Encoder:** Processes the source sequence and generates a context-rich representation.
- **Decoder:** Takes the encoder output and the target sequence (shifted right during training) to generate the output sequence, one token at a time.
- **Masks:**
  - **Source Padding Mask:** Identifies padding tokens in the source sequence so the encoder and cross-attention ignore them.
  - **Target Padding Mask:** Identifies padding tokens in the target sequence so the decoder ignores them.
  - **Target Look-Ahead Mask:** Prevents the decoder's self-attention from "cheating" by looking at future tokens in the target sequence during training.

## Data Flow During Training (Teacher Forcing)

Let's assume we have:

- `src_tokens`: Batch of source token IDs `(batch_size, src_seq_len)`.
- `target_tokens`: Batch of target token IDs, shifted right (e.g., starts with `<BOS>`, ends before `<EOS>`). `(batch_size, target_seq_len)`.

1.  **Mask Creation (`EncoderDecoder.forward`)**

    - `src_padding_mask`: Created from `src_tokens` using `_create_padding_mask`. Shape `(batch_size, 1, 1, src_seq_len)`. `True` where `src_tokens == padding_idx`.
    - `target_padding_mask`: Created from `target_tokens` using `_create_padding_mask`. Shape `(batch_size, 1, 1, target_seq_len)`. `True` where `target_tokens == padding_idx`.
    - `target_look_ahead_mask`: Created using `_create_look_ahead_mask`. Shape `(1, 1, target_seq_len, target_seq_len)`. `True` for upper triangle (excluding diagonal).
    - `combined_target_mask`: Logical OR of `target_padding_mask` and `target_look_ahead_mask`. Shape broadcastable to `(batch_size, num_heads, target_seq_len, target_seq_len)`. `True` indicates a position should be masked (either padding or future token).

2.  **Encoder Pass (`Encoder.forward`)**

    - `src_tokens` are embedded (`nn.Embedding`).
    - Positional encodings are added.
    - Dropout is applied.
    - The resulting tensor `(batch_size, src_seq_len, d_model)` goes through the stack of `EncoderBlock`s.
    - Each `EncoderBlock` applies self-attention (using `src_padding_mask`) and feed-forward layers.
    - The final output `encoder_output` has shape `(batch_size, src_seq_len, d_model)`. This represents the contextualized source sequence.

3.  **Decoder Pass (`Decoder.forward`)**

    - `target_tokens` are embedded.
    - Positional encodings are added.
    - Dropout is applied.
    - The resulting tensor `(batch_size, target_seq_len, d_model)` goes through the stack of `DecoderBlock`s along with `encoder_output` and the masks.
    - **Inside each `DecoderBlock`:**
      - **Masked Self-Attention:** The target sequence attends to itself. The `combined_target_mask` ensures it only attends to previous positions and non-padding tokens.
      - **Cross-Attention:** The output from the masked self-attention _queries_ the `encoder_output` (which acts as _key_ and _value_). The `src_padding_mask` ensures attention isn't paid to padding in the _source_ sequence representation.
      - **Feed-Forward:** Standard FFN layer.
    - The final output `decoder_output` has shape `(batch_size, target_seq_len, d_model)`.

4.  **Final Projection (`EncoderDecoder.forward`)**

    - The `decoder_output` is passed through the `final_linear` layer (`nn.Linear(d_model, target_vocab_size)`).
    - The result `output_logits` has shape `(batch_size, target_seq_len, target_vocab_size)`. These are the raw scores for each possible next token at each position in the target sequence.

5.  **Loss Calculation (Outside the model)**
    - The `output_logits` are typically compared to the _actual_ target tokens (not shifted, possibly ending with `<EOS>`) using a Cross-Entropy Loss function, ignoring the padding index.

## Data Flow During Inference (Autoregressive Generation)

Generation typically happens one token at a time.

1.  **Encoder Pass (Once per source sequence)**

    - Process the entire `src_tokens` through the `Encoder` exactly as in training, using the `src_padding_mask`.
    - Obtain and store the `encoder_output` `(batch_size, src_seq_len, d_model)`. This context is reused for every decoding step.

2.  **Decoder Start**

    - Initialize the decoder input with a start-of-sequence token (e.g., `<BOS>`). `target_tokens` initially has shape `(batch_size, 1)`.

3.  **Autoregressive Loop (for each step `t` up to max length):**
    - **Create Masks:**
      - `target_look_ahead_mask`: For the current length `t`. Shape `(1, 1, t, t)`.
      - `target_padding_mask`: Usually not needed if generation stops before padding, but depends on implementation. Can be combined if necessary.
    - **Decoder Pass:**
      - Pass the current `target_tokens` `(batch_size, t)`, the stored `encoder_output`, the `target_look_ahead_mask`, and the `src_padding_mask` through the `Decoder`.
      - Obtain the `decoder_output` `(batch_size, t, d_model)`.
    - **Final Projection:**
      - Take the _last_ time step's output from the `decoder_output`: `last_step_output` `(batch_size, 1, d_model)`.
      - Pass it through the `final_linear` layer to get `logits` `(batch_size, 1, target_vocab_size)`.
    - **Select Next Token:**
      - Apply a sampling strategy (e.g., argmax for greedy, top-k, top-p) to the `logits` to choose the next token ID `next_token` `(batch_size, 1)`.
    - **Append and Check Stop Condition:**
      - Append `next_token` to the `target_tokens` sequence (`target_tokens` becomes `(batch_size, t+1)`).
      - Check if `next_token` is the end-of-sequence token (`<EOS>`) or if max length is reached. If so, stop generation for that sequence in the batch.

This step-by-step process, especially the reuse of `encoder_output` and the iterative generation based on previous outputs, is the core of autoregressive decoding in Encoder-Decoder models.
