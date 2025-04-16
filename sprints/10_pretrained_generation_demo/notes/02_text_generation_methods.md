# Sprint 10 - Task 2: Text Generation Methods

## Goal

Implement and understand different decoding strategies for generating text from the pre-trained GPT-2 model: Greedy Search, Temperature Scaling, Top-k Sampling, and Nucleus (Top-p) Sampling.

## Background: Autoregressive Generation

GPT-2 generates text one token at a time. At each step, it takes the sequence generated so far as input and predicts the probability distribution for the _next_ token. The decoding strategy determines how we choose the next token from this distribution.

```
[Prompt Tokens] -> Model -> [Probabilities for next token]
                                    ^
                                    | (Apply Decoding Strategy)
                                    |
                                    v
                                [Chosen Next Token]
```

This chosen token is appended to the sequence, and the process repeats.

## 1. Greedy Decoding

- **Concept:** Always choose the single token with the highest probability at each step.
- **Characteristics:** Deterministic (always produces the same output for the same input), fast, but often leads to repetitive and unnatural-sounding text.
- **Implementation (`transformers`):** Use `model.generate(..., do_sample=False)` (this is often the default if `num_beams=1`).

## 2. Temperature Scaling

- **Concept:** Modify the predicted probability distribution _before_ sampling. A temperature `T` is applied:
  `new_logits = logits / T`
  The modified logits are then passed through a softmax.
- **Effect:**
  - `T > 1.0`: Flattens the distribution, increasing randomness (more likely to pick less probable words). Makes output more creative/surprising.
  - `T = 1.0`: No change to probabilities.
  - `0 < T < 1.0`: Sharpens the distribution, decreasing randomness (more likely to pick high-probability words). Makes output more focused/conservative.
  - `T -> 0`: Approaches greedy decoding.
- **Implementation (`transformers`):** Use `model.generate(..., do_sample=True, temperature=T)`.

## 3. Top-k Sampling

- **Concept:** At each step, consider only the `k` most likely next tokens. Redistribute the probability mass among these top `k` tokens and sample from this reduced set.
- **Characteristics:** Limits the pool of potential next tokens, preventing very low-probability (often weird) tokens from being chosen, while still allowing for variety.
- **Implementation (`transformers`):** Use `model.generate(..., do_sample=True, top_k=k)`. Often used with temperature.

## 4. Nucleus (Top-p) Sampling

- **Concept:** At each step, consider the smallest set of tokens whose cumulative probability is greater than or equal to `p`. Sample only from this set (the "nucleus").
- **Characteristics:** Adapts the number of tokens considered based on the probability distribution. If the model is very confident (one token has high probability), the nucleus is small. If the model is uncertain (probabilities are spread out), the nucleus is larger. Often considered to produce more coherent and less repetitive text than top-k.
- **Implementation (`transformers`):** Use `model.generate(..., do_sample=True, top_p=p)`. Often used with temperature.

## Implementation Notes

- The `transformers` library's `model.generate()` function conveniently implements all these strategies.
- Common parameters:
  - `input_ids`: Tokenized prompt.
  - `max_length` / `max_new_tokens`: Controls output length.
  - `do_sample=True`: Enables sampling methods (temperature, top-k, top-p). If `False`, defaults to greedy (or beam search if `num_beams > 1`).
  - `temperature`: Float > 0.
  - `top_k`: Integer > 0.
  - `top_p`: Float between 0 and 1.
  - `pad_token_id`: Important to set, often `tokenizer.eos_token_id` for open-ended generation.
  - `eos_token_id`: Token ID to stop generation early.

We will add functions to `results/01_text_generation.py` to demonstrate these techniques.
