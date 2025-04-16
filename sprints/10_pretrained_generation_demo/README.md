# Sprint 10: Pre-trained GPT-2, Text Generation & Basic Demo

## Sprint Goal

Load and interact with a pre-trained GPT-2 model, implement various text generation techniques, and create a simple local interactive demo as a proof-of-concept.

## Learning Objectives / Tasks

1.  **[ ] Load Pre-trained Model:**

    - [ ] Install Hugging Face `transformers` and `datasets` libraries.
    - [ ] Load a pre-trained GPT-2 model (e.g., `gpt2`, `gpt2-medium`) using `transformers`.
    - [ ] Load the corresponding GPT-2 tokenizer.
    - [ ] Understand the configuration (`config.json`) of the pre-trained model.
    - [ ] Document the process and model details in `notes/01_loading_pretrained.md`.

2.  **[ ] Implement Text Generation:**

    - [ ] Understand the basic process of autoregressive text generation.
    - [ ] Implement greedy decoding: select the token with the highest probability at each step.
    - [ ] Implement temperature scaling to control randomness.
    - [ ] Implement top-k sampling: sample from the k most likely next tokens.
    - [ ] Implement nucleus (top-p) sampling: sample from the smallest set of tokens whose cumulative probability exceeds p.
    - [ ] Create a script (`results/01_text_generation.py`) demonstrating these methods with the pre-trained model.
    - [ ] Document the algorithms and their parameters in `notes/02_text_generation_methods.md`.

3.  **[ ] Build Basic Interactive Demo:**
    - [ ] Install `gradio` or `streamlit`.
    - [ ] Create a simple UI that allows:
      - Inputting a text prompt.
      - Selecting a generation method (Greedy, Top-k, Top-p).
      - Adjusting parameters (temperature, k, p, max length).
      - Displaying the generated text.
    - [ ] Implement the backend logic connecting the UI to the generation functions from Task 2.
    - [ ] Save the demo script as `results/02_interactive_demo.py`.
    - [ ] Document the demo setup and usage in `notes/03_interactive_demo.md`.

## Definition of Done / Key Questions Answered

- [ ] Can successfully load a pre-trained GPT-2 model and tokenizer from Hugging Face?
- [ ] Can generate text using greedy, top-k, and top-p sampling methods?
- [ ] Understand how temperature, k, and p parameters influence text generation?
- [ ] Have a working interactive demo (local) for generating text with different parameters?
- [ ] Documented the loading process, generation algorithms, and demo setup?

## Notes & Resources

- [notes/01_loading_pretrained.md](notes/01_loading_pretrained.md)
- [notes/02_text_generation_methods.md](notes/02_text_generation_methods.md)
- [notes/03_interactive_demo.md](notes/03_interactive_demo.md)
- Hugging Face `transformers` Documentation
- Gradio / Streamlit Documentation

## Results & Code

- [results/01_text_generation.py](results/01_text_generation.py)
- [results/02_interactive_demo.py](results/02_interactive_demo.py)

## Retrospective / Key Learnings

- **What went well?**

  - Loading pre-trained models and tokenizers using Hugging Face `transformers` (`AutoModel...`, `AutoTokenizer...`) was straightforward.
  - The `model.generate()` method provided a convenient high-level API for implementing various decoding strategies (greedy, top-k, top-p, temperature).
  - Setting up a basic interactive UI with Gradio was quick and relatively easy.
  - Implementing streaming output using `TextIteratorStreamer` and threading worked well and significantly improved the demo's perceived responsiveness.

- **What challenges were faced?**

  - Clarifying the workflow for managing dependencies with `uv` (`uv add` vs `uv pip install` + manual edit + `uv sync`).
  - Ensuring correct Python module imports (`python -m ...`) when running scripts that depend on each other.
  - Understanding the need for threading when using `TextIteratorStreamer` with Gradio.

- **Key takeaways about using pre-trained models?**

  - Vast time and resource saving compared to training from scratch (as done in previous sprints).
  - Leveraging the HF Hub provides access to a wide range of well-tested models and tokenizers.
  - Understanding model configurations (`model.config`) is still important.
  - Need to handle tokenizer specifics (like missing PAD tokens).

- **Key takeaways about text generation techniques?**

  - Greedy search is simple but often leads to repetitive/boring text.
  - Sampling methods (top-k, top-p) introduce necessary randomness.
  - Temperature controls the level of randomness/creativity.
  - Top-p (nucleus) sampling often provides a good balance between coherence and creativity by adapting the sampling pool size dynamically.
  - Streaming output greatly enhances the user experience for interactive generation.

- **Thoughts on building simple demos?**
  - Tools like Gradio are excellent for quickly making ML models interactive and accessible without extensive web development.
  - Helps in visualizing and experimenting with model behavior and parameters.
  - Good way to showcase progress and share results (even if just locally initially).
