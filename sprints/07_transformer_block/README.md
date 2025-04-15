# Sprint 7: Building the Transformer Block

## 🎯 Sprint Goal

Assemble the core components (Multi-Head Attention, Feed-Forward Network, Layer Normalization, Residual Connections) into a functional Transformer Encoder/Decoder block in PyTorch.

## 📋 Tasks / Learning Objectives

1.  [ ] **Implement Layer Normalization:**
    - Understand the concept and purpose of Layer Normalization (`nn.LayerNorm`).
    - Implement a `LayerNorm` layer and test its behavior.
    - Write notes explaining its role, especially in stabilizing Transformer training. (`notes/01_layer_norm.md`)
    - Create a small example script. (`results/01_layer_norm_example.py`)
2.  [ ] **Implement Residual Connections (Add & Norm):**
    - Understand the concept of residual connections (skip connections).
    - Implement the "Add & Norm" pattern (Sublayer Output + Input, followed by LayerNorm).
    - Write notes explaining why residual connections are crucial for deep networks. (`notes/02_residual_connections.md`)
    - Integrate this into a conceptual example. (`results/02_residual_connection_example.py`)
3.  [ ] **Implement Position-wise Feed-Forward Network (FFN):**
    - Understand the structure of the FFN in Transformers (two linear layers with an activation).
    - Implement the FFN as an `nn.Module`.
    - Write notes on its role and typical configuration. (`notes/03_feed_forward_network.md`)
    - Create a small example script. (`results/03_ffn_example.py`)
4.  [ ] **Build the Transformer Encoder Block:**
    - Combine Multi-Head Self-Attention, Add & Norm, FFN, and another Add & Norm into a complete Encoder block `nn.Module`.
    - Ensure correct data flow and masking (if necessary).
    - Write notes detailing the structure. (`notes/04_encoder_block.md`)
    - Implement the `EncoderBlock` module. (`results/04_encoder_block.py`)
5.  [ ] **Build the Transformer Decoder Block:**
    - Combine Masked Multi-Head Self-Attention, Add & Norm, Multi-Head Cross-Attention (Encoder-Decoder Attention), Add & Norm, FFN, and Add & Norm into a complete Decoder block `nn.Module`.
    - Pay close attention to the two attention mechanisms and their inputs/masks.
    - Write notes detailing the structure and differences from the Encoder block. (`notes/05_decoder_block.md`)
    - Implement the `DecoderBlock` module. (`results/05_decoder_block.py`)
6.  [ ] **Testing and Integration:**
    - Create test cases with dummy data to ensure the Encoder and Decoder blocks run without errors and handle shapes correctly.
    - Test mask propagation through the blocks.
    - (Optional) Write integration tests if time permits.

## 📚 Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- PyTorch Documentation (`nn.LayerNorm`, `nn.Linear`, `nn.Module`)
- Previous Sprint Notes (Sprint 6 - Multi-Head Attention)

## 🤔 Retrospective / Key Learnings

_(To be filled out at the end of the sprint)_

- What went well?
- What challenges were faced?
- Key insights into Transformer architecture?
- How did LayerNorm and Residual Connections impact the implementation?

## ✅ Definition of Done

- All code for LayerNorm, Residual Connection example, FFN, Encoder Block, and Decoder Block is implemented and passes basic tests.
- Accompanying notes explaining each component are written.
- A basic understanding of how these components fit together in a Transformer is achieved.
