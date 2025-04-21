# Task 1: Environment Setup Notes

**Goal:** Prepare the local machine (WSL2, GPU, Python, PyTorch, Hugging Face libs, TRL) for RLHF training.

**Steps Taken:**

1.  **Initial Checks:**
    *   Confirmed PyTorch installation (`2.6.0+cu124`) and CUDA availability (`torch.cuda.is_available()` returned `True`).
    *   Checked for required Hugging Face libraries (`transformers`, `datasets`, `accelerate`, `trl`) using `pip show`. Found they were not installed.

2.  **Installation Attempt (pip):**
    *   Attempted installation using `pip install transformers datasets accelerate trl`.
    *   Encountered `error: externally-managed-environment`, indicating system Python restrictions.

3.  **Installation (uv):**
    *   Switched to using the `uv` package manager.
    *   Added dependencies: `uv add transformers datasets accelerate trl`.
    *   Synced the environment: `uv sync`.
    *   This successfully installed the required libraries.

**Outcome:** The necessary Python environment with PyTorch (GPU-enabled) and Hugging Face libraries (`transformers`, `datasets`, `accelerate`, `trl`) was successfully configured using `uv`. 