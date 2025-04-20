# Sprint 15: Exploring Other Architectures (Encoder-Decoder)

## Goal

Revisit the fundamental Transformer Encoder and Decoder blocks built in Sprint 7 and assemble them into a complete Encoder-Decoder architecture. Understand the data flow for sequence-to-sequence (seq2seq) tasks and potentially implement a basic example.

## Tasks

- **Review & Refresher:**
  - [x] Re-read the code and notes for `EncoderBlock` (copied to `results/encoder_block.py`).
  - [x] Re-read the code and notes for `DecoderBlock` (copied to `results/decoder_block.py`).
  - [x] Refresh understanding of LayerNorm, Residual Connections, FFNs, and Attention Masking (Self-Attention, Cross-Attention, Padding, Look-Ahead).
- **Architecture Assembly:**
  - [x] Define a top-level `Encoder` module that stacks multiple `EncoderBlock` layers.
  - [x] Define a top-level `Decoder` module that stacks multiple `DecoderBlock` layers.
  - [x] Define a main `EncoderDecoder` model that integrates the `Encoder`, `Decoder`, input embeddings, positional encoding, and a final output layer (e.g., a linear layer followed by softmax for generation/classification).
  - [x] Ensure correct handling of source and target masks throughout the model.
- **Understanding Seq2Seq Data Flow:**
  - [x] Document the step-by-step flow of data (source sequence, target sequence) through the `EncoderDecoder` model during training and inference - See [notes/01_encoder_decoder_flow.md](./notes/01_encoder_decoder_flow.md).
  - [x] Explain how the encoder output context is used by the decoder's cross-attention mechanism - See [notes/01_encoder_decoder_flow.md](./notes/01_encoder_decoder_flow.md).
- **(Optional) Basic Seq2Seq Task:**
  - [ ] Define a simple toy sequence-to-sequence task (e.g., reversing a sequence of numbers, simple translation).
  - [ ] Create dummy data and corresponding masks for the toy task.
  - [ ] Implement a basic training loop for the `EncoderDecoder` model on the toy task.
- **Documentation:**
  - [ ] Create notes explaining the assembled architecture and data flow (`notes/`).
  - [ ] Document the implementation steps and results (`results/`).
  - [ ] Update this README with progress and findings.

## Resources

- Sprint 7 Notes & Code (`sprints/07_transformer_block/`)
- "Attention Is All You Need" paper (Vaswani et al., 2017)
- Illustrated Transformer blog post (Jay Alammar)

## Retrospective

_(To be filled out at the end of the sprint)_

- **What went well?**
- **What could be improved?**
- **What did we learn?**
