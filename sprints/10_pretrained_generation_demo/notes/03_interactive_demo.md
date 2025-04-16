# Sprint 10 - Task 3: Basic Interactive Demo

## Goal

Create a simple, local web-based demo using Gradio to interact with the pre-trained GPT-2 model and the text generation functions developed in Task 2.

## Why Gradio?

Gradio is chosen for its simplicity in creating quick UIs for machine learning models directly from Python scripts.

## Steps

### 1. Installation

Gradio was installed as part of Task 1 (`uv pip install gradio` or `uv add gradio` + `uv sync`).

### 2. Create the Demo Script (`results/02_interactive_demo.py`)

This script will:

- Import necessary libraries (`gradio`, `torch`, functions from `01_text_generation.py`).
- Load the model and tokenizer _once_ when the script starts.
- Define a function that takes the user inputs (prompt, method, parameters) and calls the `generate_text` function.
- Create a Gradio `Interface` object, mapping UI components (text boxes, sliders, dropdowns) to the inputs and outputs of the defined function.
- Launch the interface.

### 3. UI Components

The Gradio interface should include:

- Input Textbox for the prompt.
- Dropdown to select the generation method (`greedy`, `top-k`, `top-p`).
- Slider/Number Input for `max_new_tokens`.
- Slider/Number Input for `temperature`.
- Slider/Number Input for `top_k` (conditionally visible for `top-k` method).
- Slider/Number Input for `top_p` (conditionally visible for `top-p` method).
- Output Textbox to display the generated text.

_(Note: Gradio allows for some basic conditional visibility, though it can sometimes be finicky. We might start simple.)_

### 4. Running the Demo

Execute the script from the terminal:

```bash
python sprints/10_pretrained_generation_demo/results/02_interactive_demo.py
# Or, if running from workspace root:
python -m sprints.10_pretrained_generation_demo.results.02_interactive_demo
```

This will typically start a local web server, and you can access the demo in your browser at an address like `http://127.0.0.1:7860`.

## Key Considerations

- **Model Loading:** Load the model outside the main Gradio function to avoid reloading it on every generation request.
- **Device:** Ensure the model and inputs are on the correct device.
- **Error Handling:** Basic error handling can be added (e.g., for invalid parameter combinations).
