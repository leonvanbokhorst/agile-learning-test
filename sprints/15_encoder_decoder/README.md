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
  - [x] Define a simple toy sequence-to-sequence task (reversing a sequence of numbers).
  - [x] Create dummy data and corresponding masks for the toy task (`SequenceReversalDataset`).
  - [x] Implement a basic training loop for the `EncoderDecoder` model on the toy task (see `results/train_seq_reversal.py`).
  - [x] Provide a perfect direct implementation using tensor flipping (see `results/direct_reverse_demo.py`).
- **Documentation:**
  - [ ] Create notes explaining the assembled architecture and data flow (`notes/`).
  - [ ] Document the implementation steps and results (`results/`).
  - [ ] Update this README with progress and findings.

## Resources

- Sprint 7 Notes & Code (`sprints/07_transformer_block/`)
- "Attention Is All You Need" paper (Vaswani et al., 2017)
- Illustrated Transformer blog post (Jay Alammar)

## Discoveries

What we've discovered is that **standard MLE with teacher‑forcing** (i.e. Cross‑Entropy loss) is a fantastic tool for driving training‐time loss to zero—but it's the _wrong_ tool if your goal is _robust_ autoregressive generation. In toy reversal:

- Under pure teacher forcing, the model learns the mapping perfectly (100% seq‑level accuracy).
- As soon as it has to feed itself, it collapses to trivial loops or immediate EOS (0% correct).

This is exposure bias: the training objective and the inference procedure don't match. We've tried every regularization and scheduling trick—scheduled sampling, Professor Forcing, label smoothing, curriculum learning—but the root mismatch remains.

If you really want reliable free‑running sequences, you need a _training_ approach that directly optimizes for the _inference_ behavior. Two broad classes of "right" tools here are:

1. **Sequence‑level objectives / RL**  
   • Use REINFORCE or Minimum­Risk Training to optimize a sequence‐level reward (e.g. exact reversal accuracy) rather than token‑wise cross‐entropy.  
   • This aligns what you train on (the reward of the full sequence) with what you care about at inference.

2. **Stronger consistency losses**  
   • Instead of MSE on hidden states, use KL‑divergence on the _output distributions_ between teacher‑forced and free‑running modes—forcing the model's _predicted probabilities_ to look the same regardless of which input it saw.  
   • Crank that penalty way up (pf_lambda ≫ 1) to really eliminate any drift.

For the toy reversal task (where the reward is simply "did you get the exact reversed sequence?"), a **sequence‐level RL approach** is often the cleanest "right tool": you define a 0/1 reward for a perfect reversal and let the model explore its own generation path. So we try wiring up a simple REINFORCE loop to directly optimize sequence‐level accuracy (or BLEU).

## Retrospective

_(To be filled out at the end of the sprint)_

- **What went well?**
- **What could be improved?**
- **What did we learn?**

## Demos

- `python3 -u sprints/15_encoder_decoder/results/direct_reverse_demo.py`: run the direct sequence reversal demo.
