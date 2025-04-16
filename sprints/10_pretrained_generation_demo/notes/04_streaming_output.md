# Sprint 10 Extra: Streaming Output with Gradio

## Goal

Understand how to make the Gradio demo stream the generated text token by token (or chunk by chunk) instead of waiting for the entire sequence to complete.

## Concept: Python Generators and `yield`

The key to streaming in Gradio (and many web frameworks) is using **Python generators**. Instead of having a function compute the entire result and `return` it at the end, a generator function uses the `yield` keyword.

- When a function contains `yield`, it becomes a generator.
- Each time `yield` is encountered, the function pauses its execution and sends the yielded value back to the caller.
- When the caller requests the next value, the function resumes execution from where it left off until it hits the next `yield` or the function ends.

## How Gradio Uses Generators for Streaming

Gradio is designed to work seamlessly with generator functions for output components like `gr.Textbox`:

1.  **Backend Function:** Your Python function that connects to the Gradio interface needs to be a generator (i.e., use `yield`). Instead of returning the full generated text string at the end, it should `yield` partial results (e.g., individual tokens or small chunks of text) as they become available.
2.  **Gradio Interface:** When you link this generator function to an output component (like our `output_text` Textbox), Gradio automatically detects that the function is a generator.
3.  **Frontend Update:** As the backend function `yield`s each piece of text, Gradio incrementally updates the content of the Textbox in the web UI in real-time.

## Implementing Streaming with `transformers`

While `model.generate()` typically returns the full sequence, the `transformers` library provides utilities to facilitate streaming:

1.  **`TextStreamer` / `TextIteratorStreamer`:** These classes handle the process of decoding tokens as they are generated.
    - `TextStreamer`: Prints directly to the console (useful for debugging).
    - `TextIteratorStreamer`: Provides an iterator that you can loop over in your code to `yield` the decoded text chunks. This is what we need for Gradio.
2.  **Threading:** Because `model.generate()` is a blocking call (it waits until generation is done), you need to run it in a separate thread. The main Gradio function thread can then iterate over the `TextIteratorStreamer` (which receives tokens from the generation thread) and `yield` the results to the Gradio UI.

## Conceptual Code Structure (for `02_interactive_demo.py`)

```python
import gradio as gr
from transformers import TextIteratorStreamer
from threading import Thread
# ... other imports ...

# ... load model/tokenizer ...

# Modified Gradio function
def gradio_interface_streaming(
    prompt: str,
    # ... other params ...
):
    # 1. Create a streamer
    streamer = TextIteratorStreamer(TOKENIZER, skip_prompt=True, skip_special_tokens=True)

    # 2. Define generation kwargs (same as before)
    generation_kwargs = {
        # ... max_new_tokens, temp, top_k, top_p etc ...
        "streamer": streamer # Pass the streamer to generate
    }
    inputs = TOKENIZER([prompt], return_tensors="pt").to(DEVICE)

    # 3. Run generate in a separate thread
    thread = Thread(target=MODEL.generate, args=(inputs.input_ids,), kwargs=generation_kwargs)
    thread.start()

    # 4. Yield results from the streamer in the main thread
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        yield generated_text # Yield the *cumulative* text for Gradio textbox update

# --- Define Gradio UI Components --- #
# (Similar UI definition as before)
with gr.Blocks(...) as demo:
    # ... define inputs/outputs ...

    # Connect UI to the *streaming* function
    generate_button.click(
        fn=gradio_interface_streaming, # Use the new function
        inputs=[...],
        outputs=output_text
    )

# ... demo.launch() ...
```

This setup allows the UI to update incrementally as the model generates each new piece of text.
