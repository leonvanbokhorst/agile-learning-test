# Sprint 7: Building the Transformer Block

## 🎯 Sprint Goal

Assemble the core components (Multi-Head Attention, Feed-Forward Network, Layer Normalization, Residual Connections) into a functional Transformer Encoder/Decoder block in PyTorch.

## 📋 Tasks / Learning Objectives

1.  [x] **Implement Layer Normalization:**
    - [x] Understand the concept and purpose of Layer Normalization (`nn.LayerNorm`).
    - [x] Implement a `LayerNorm` layer and test its behavior.
    - [x] Write notes explaining its role, especially in stabilizing Transformer training. (`notes/01_layer_norm.md`, `notes/01a_layer_norm_output_explanation.md`)
    - [x] Create a small example script. (`results/01_layer_norm_example.py`)
2.  [x] **Implement Residual Connections (Add & Norm):**
    - [x] Understand the concept of residual connections (skip connections).
    - [x] Implement the "Add & Norm" pattern (Sublayer Output + Input, followed by LayerNorm).
    - [x] Write notes explaining why residual connections are crucial for deep networks. (`notes/02_residual_connections.md`)
    - [x] Integrate this into a conceptual example. (`results/02_residual_connection_example.py`)
3.  [x] **Implement Position-wise Feed-Forward Network (FFN):**
    - [x] Understand the structure of the FFN in Transformers (two linear layers with an activation).
    - [x] Implement the FFN as an `nn.Module`.
    - [x] Write notes on its role and typical configuration. (`notes/03_feed_forward_network.md`)
    - [x] Create a small example script. (`results/positionwise_feed_forward.py` - Renamed from `03_ffn_example.py`)
4.  [x] **Build the Transformer Encoder Block:**
    - [x] Combine Multi-Head Self-Attention, Add & Norm, FFN, and another Add & Norm into a complete Encoder block `nn.Module`.
    - [x] Ensure correct data flow and masking (if necessary).
    - [x] Write notes detailing the structure. (`notes/04_encoder_block.md`)
    - [x] Implement the `EncoderBlock` module. (`results/encoder_block.py` - Renamed from `04_encoder_block.py`)
5.  [x] **Build the Transformer Decoder Block:**
    - [x] Combine Masked Multi-Head Self-Attention, Add & Norm, Multi-Head Cross-Attention (Encoder-Decoder Attention), Add & Norm, FFN, and Add & Norm into a complete Decoder block `nn.Module`.
    - [x] Pay close attention to the two attention mechanisms and their inputs/masks.
    - [x] Write notes detailing the structure and differences from the Encoder block. (`notes/05_decoder_block.md`)
    - [x] Implement the `DecoderBlock` module. (`results/decoder_block.py` - Renamed from `05_decoder_block.py`)
6.  [x] **Testing and Integration:**
    - [x] Create test cases with dummy data to ensure the Encoder and Decoder blocks run without errors and handle shapes correctly (within `main` functions).
    - [x] Test mask propagation through the blocks (basic checks via dummy masks).
    - [x] Tested stacking multiple blocks to verify shape maintenance.

## 📚 Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- PyTorch Documentation (`nn.LayerNorm`, `nn.Linear`, `nn.Module`, `nn.Dropout`, `nn.GELU`)
- Previous Sprint Notes (Sprint 6 - Multi-Head Attention)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) (Helpful conceptual diagrams)

## 🤔 Retrospective / Key Learnings

_(To be filled out at the end of the sprint)_

- **What went well?**
  - Successfully implemented all core components: LayerNorm, Add&Norm, FFN, Encoder Block, Decoder Block.
  - Debugging import issues and module interactions (like MHA tuple output) reinforced understanding.
  - Testing with dummy data and masks helped verify shapes and basic functionality.
  - Successfully tested stacking blocks, simulating part of the full Transformer structure.
  - Fixed formatting issues in notes collaboratively.
- **What challenges were faced?**
  - Initial formatting errors in notes (newlines, LaTeX delimiters).
  - Import errors due to file renaming/copying and relative paths.
  - Understanding the expected output of the `MultiHeadAttention` module (tuple vs. tensor).
  - Repeated linter errors with assert statements (backslash gremlins!).
- **Key insights into Transformer architecture?**
  - Solidified understanding of the repetitive "Add & Norm" structure around sub-layers.
  - Clarified the distinct roles and inputs/outputs of the two attention mechanisms in the Decoder block (masked self-attention vs. cross-attention).
  - Reinforced the importance of masking (padding and look-ahead) for correct attention calculation.
  - Understood how individual blocks maintain sequence dimension, allowing stacking.
- **How did LayerNorm and Residual Connections impact the implementation?**
  - LayerNorm became a standard component applied after residual addition, ensuring stable activation statistics.
  - Residual connections ($x + \text{Sublayer}(x)$) were straightforward to implement and are crucial for enabling deep stacking by preserving information flow.

## ✅ Definition of Done

- [x] All code for LayerNorm, Residual Connection example, FFN, Encoder Block, and Decoder Block is implemented and passes basic tests.
- [x] Accompanying notes explaining each component are written.
- [x] A basic understanding of how these components fit together in a Transformer is achieved.
