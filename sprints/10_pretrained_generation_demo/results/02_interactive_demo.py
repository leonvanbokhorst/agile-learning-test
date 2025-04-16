import gradio as gr
import torch
import time
from threading import Thread
from transformers import TextIteratorStreamer

# Assuming 01_text_generation.py is in the same directory or accessible
# Use a relative import if running as a module, or adjust sys.path if needed.
try:
    from text_generation import load_model_and_tokenizer
except ImportError:
    # Fallback for running the script directly
    from text_generation import load_model_and_tokenizer

# --- Load Model Globally (Load Once) --- #
print("Loading model and tokenizer for Gradio app...")
start_time = time.time()
# Use a specific model known to work well with streaming if default causes issues
# model_name_to_load = "gpt2" # Or choose another
MODEL, TOKENIZER, DEVICE = load_model_and_tokenizer()  # Default is "gpt2"
end_time = time.time()
print(f"Model loading complete. Took {end_time - start_time:.2f} seconds.")


# --- Gradio Interface Function (Now Streaming) --- #
def gradio_interface_streaming(
    prompt: str, method: str, max_new: int, temp: float, k_val: int, p_val: float
):
    """Wrapper function for Gradio interface with streaming."""
    print(f"\n--- Generating (Streaming - {method}) ---")
    print(f"Prompt: '{prompt}'")
    if not prompt:
        yield "Error: Prompt cannot be empty."
        return  # Use return in generator context to stop iteration

    # 1. Create a streamer
    # skip_prompt=True prevents the prompt from being yielded, cleaner output
    streamer = TextIteratorStreamer(
        TOKENIZER, skip_prompt=True, skip_special_tokens=True
    )

    # 2. Prepare inputs and generation kwargs
    inputs = TOKENIZER([prompt], return_tensors="pt").to(
        DEVICE
    )  # Note: Tokenize as a batch
    input_ids = inputs.input_ids

    generation_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new,
        "streamer": streamer,
        "pad_token_id": TOKENIZER.eos_token_id,  # Use EOS for open-ended generation
        "eos_token_id": TOKENIZER.eos_token_id,
    }

    # Adjust params based on method (sampling args only used if do_sample=True)
    if method == "greedy":
        generation_kwargs["do_sample"] = False
        print(f"Params: {{'do_sample': False, 'max_new_tokens': {max_new}}}")
    elif method == "top-k":
        generation_kwargs["do_sample"] = True
        generation_kwargs["top_k"] = k_val
        generation_kwargs["temperature"] = temp
        print(
            f"Params: {{'do_sample': True, 'max_new_tokens': {max_new}, 'temperature': {temp}, 'top_k': {k_val}}}"
        )
    elif method == "top-p":
        generation_kwargs["do_sample"] = True
        generation_kwargs["top_p"] = p_val
        generation_kwargs["temperature"] = temp
        print(
            f"Params: {{'do_sample': True, 'max_new_tokens': {max_new}, 'temperature': {temp}, 'top_p': {p_val}}}"
        )
    else:
        yield f"Error: Unknown generation method: {method}"
        return

    # 3. Run generate in a separate thread
    # Need to ensure the generation runs on the correct device implicitly via model placement
    thread = Thread(target=MODEL.generate, kwargs=generation_kwargs)
    thread.start()

    # 4. Yield results from the streamer in the main thread
    cumulative_text = ""
    try:
        for new_text in streamer:
            cumulative_text += new_text
            yield cumulative_text  # Yield the *cumulative* text for Gradio textbox update
    except Exception as e:
        print(f"Error during streaming: {e}")
        yield f"An error occurred during streaming: {e}"
    finally:
        # Ensure thread is joined
        thread.join()
        print("\nGeneration thread finished.")


# --- Define Gradio UI Components --- #
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # GPT-2 Text Generation Demo
        Enter a prompt and choose generation parameters to see the output from the pre-trained GPT-2 model.
        """
    )
    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(label="Enter Prompt Here", lines=3)
            method_select = gr.Dropdown(
                label="Generation Method",
                choices=["greedy", "top-k", "top-p"],
                value="top-p",
            )
            generate_button = gr.Button("Generate Text", variant="primary")
        with gr.Column(scale=1):
            max_new_tokens_slider = gr.Slider(
                label="Max New Tokens", minimum=10, maximum=300, value=50, step=10
            )
            temperature_slider = gr.Slider(
                label="Temperature (for sampling)",
                minimum=0.1,
                maximum=2.0,
                value=0.7,
                step=0.1,
            )
            top_k_slider = gr.Slider(
                label="Top-k (for top-k method)",
                minimum=1,
                maximum=100,
                value=50,
                step=1,
            )
            top_p_slider = gr.Slider(
                label="Top-p (for top-p method)",
                minimum=0.1,
                maximum=1.0,
                value=0.9,
                step=0.05,
            )

    output_text = gr.Textbox(label="Generated Text", lines=10, interactive=False)

    # --- Connect UI Components to Function --- #
    generate_button.click(
        fn=gradio_interface_streaming,
        inputs=[
            prompt_input,
            method_select,
            max_new_tokens_slider,
            temperature_slider,
            top_k_slider,
            top_p_slider,
        ],
        outputs=output_text,
    )

# --- Launch the Gradio App --- #
if __name__ == "__main__":
    print("Launching Gradio Interface...")
    # Share=True creates a public link (useful for temporary sharing, remove if not needed)
    # Set debug=True for more detailed error messages during development
    demo.launch(debug=True)  # share=True
