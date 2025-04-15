# Note: Stacking Decoder Blocks for our GPT-2 Homie

## The Grand Plan: Building the Transformer Body

Alright, mission objective one for Sprint 8: assemble the core structure of our GPT-style model. Think of it like building the torso and limbs of a magnificent language-generating robot. We're stacking `DecoderBlock`s one on top of the other.

## Why Stack 'Em High?

Why not just use one block? Because **depth** is where the magic happens!

- **Learning Power:** Each layer learns progressively more complex patterns and relationships in the text data. The first layer might spot simple word pairings, while deeper layers grasp grammar, context, and maybe even sarcasm (if we're lucky).
- **Refined Representations:** The output of one block becomes the input for the next. Each block refines the sequence's representation, making it richer and more informative for the final prediction. It's like passing a story through multiple editors, each improving it.

## Hold Up! A GPT-2 Plot Twist! 😲

Remember our `DecoderBlock` from Sprint 7? It had _masked self-attention_ AND _cross-attention_.

**Crucial point:** GPT-2 is a **decoder-only** model. It doesn't have an encoder feeding it information. Therefore, it **doesn't need cross-attention**. Its blocks _only_ need:

1.  **Masked Self-Attention:** To look at previous tokens _within its own input sequence_ (preventing spoilers from the future!).
2.  **Feed-Forward Network:** To process the information from the attention layer.

**What this means for us:** We need to either:

- Modify our existing `DecoderBlock` to _ignore_ or _remove_ the cross-attention part when we use it in the main model.
- Create a slightly simplified `GPTDecoderBlock` specifically for this purpose.

Let's lean towards modifying the existing one for now to keep things DRY (Don't Repeat Yourself), but we'll be mindful that the `encoder_output` and `encoder_mask` arguments won't be used and should probably be handled gracefully (e.g., default to `None` and skipped internally).

## The Stacking Mechanism

In our main `model.py` file, we'll create a `GPT` class. Inside this class, we'll have:

- An `nn.ModuleList` containing multiple instances of our (slightly adapted) `DecoderBlock`.
- Input embedding layers (token + position - Task 2!).
- A final output layer (Task 3!).

The forward pass will push the embedded input through each block in the stack sequentially.

## GPT-2 Stacked Decoder Blocks

GPT-2 wasn't just one single model; OpenAI released it in several sizes, each with a different number of stacked decoder blocks (which they often just call "layers"):

*   **GPT-2 Small (124M parameters):** Had **12** layers (decoder blocks).
*   **GPT-2 Medium (355M parameters):** Had **24** layers.
*   **GPT-2 Large (774M parameters):** Had **36** layers.
*   **GPT-2 XL (1.5B parameters):** Had **48** layers.

So, the number of stacked blocks varied quite a bit depending on which version of GPT-2 you were looking at! Our `GPTConfig` currently defaults to 12 layers, mimicking the "small" version.
