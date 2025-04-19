# Sprint 14: Model Optimization (Quantization - Proof of Concept)

## Goal

Learn the basics of post-training quantization and apply it to a GPT-2 model to understand the trade-offs between model size, speed, and performance. This will be a Proof-of-Concept (PoC) exploration.

## Tasks

- **Understanding Concepts:**
  - [x] Research and document the core ideas behind model quantization (Why do it? What are the benefits and drawbacks?) - See [notes/01_quantization_concepts.md](./notes/01_quantization_concepts.md)
  - [x] Differentiate between different quantization types (e.g., dynamic quantization, static quantization, QAT). Focus on post-training methods for this sprint. - See [notes/02_quantization_types.md](./notes/02_quantization_types.md)
  - [x] Understand common data types used (INT8, FP16, BF16). - See [notes/03_data_types.md](./notes/03_data_types.md)
- **Exploring PyTorch Tools:**
  - [x] Investigate PyTorch's built-in quantization modules (`torch.quantization`). - See [notes/04_pytorch_quantization_api.md](./notes/04_pytorch_quantization_api.md)
  - [x] Identify functions for dynamic quantization (`torch.quantization.quantize_dynamic`). - Covered in [API Notes](./notes/04_pytorch_quantization_api.md)
  - [x] Identify functions/workflow for static post-training quantization (PTQ Static - involves calibration). - Covered in [API Notes](./notes/04_pytorch_quantization_api.md)
- **Applying Quantization (Proof of Concept):**
  - [x] Choose a target model: Base GPT-2 (`gpt2`).
  - [x] Implement **Dynamic Quantization** on the chosen model.
    - Result: Achieved ~1.23x speedup on CPU generation time compared to FP32 baseline in basic test ([results/01_dynamic_quantization.py](./results/01_dynamic_quantization.py)). Model size via `state_dict` appeared larger due to quantization overhead, but runtime performance improved.
  - [x] (Stretch Goal) Implement **Static Post-Training Quantization** (requires a calibration dataset).
    - Result: Achieved ~1.26x speedup on CPU generation time compared to FP32 baseline in basic test ([results/02_static_quantization.py](./results/02_static_quantization.py)) after targeting `Conv1D` layers. `state_dict` size remained similar to FP32, but runtime performance improved slightly over dynamic quantization.
- **Evaluation:**
  - [ ] Measure the model size (e.g., file size on disk) before and after quantization.
  - [x] (Optional/If feasible) Compare inference speed (e.g., time to generate text) before and after. - _Done for CPU baseline in results scripts._
  - [ ] (Optional/If feasible) Evaluate generation quality or perplexity on a small test set to see the impact of quantization.
- **Documentation:**
  - [x] Create notes explaining the concepts learned (`notes/`). - _See [notes](./notes/)_
  - [x] Document the implementation steps and results (`results/`). - _See [results](./results/)_
  - [x] Update this README with progress and findings. - _Done_

## Resources

- PyTorch Quantization Documentation: [https://pytorch.org/docs/stable/quantization.html](https://pytorch.org/docs/stable/quantization.html)
- Hugging Face Optimum (Might be relevant later): [https://huggingface.co/docs/optimum/index](https://huggingface.co/docs/optimum/index)

## Retrospective

_(To be filled out at the end of the sprint)_

- **What went well?**

  - Successfully implemented both Dynamic and Static Post-Training Quantization (PTQ) using PyTorch's Eager Mode API.
  - Achieved measurable inference speedups on CPU (~1.23x for Dynamic, ~1.26x for Static) compared to the FP32 baseline.
  - Systematically debugged and resolved several technical challenges, including quantization backend errors (`qnnpack`) and the nuances of applying QConfigs in Eager Mode (targeting `Conv1D` submodules directly).
  - Effectively used a wrapper module (`GPT2QuantWrapper`) to apply static quantization without modifying the base Hugging Face model code.
  - Prepared and utilized calibration data (Wikitext sample) successfully for static quantization.
  - Maintained clear documentation through notes and README updates.

- **What could be improved?**

  - Initial confusion caused by `state_dict` size measurements, which didn't reflect the actual benefits of quantization (runtime speed). Exploring methods like TorchScript for more accurate size reporting could be beneficial.
  - The Eager Mode API for static quantization proved somewhat brittle and required specific submodule targeting rather than more intuitive mapping approaches, highlighting why newer APIs (like FX or Torch Export) were developed.
  - Minor script errors (like code duplication) occurred during development.
  - We focused on speed; a qualitative or quantitative evaluation of generation quality/perplexity post-quantization was skipped.

- **What did we learn?**
  - Solidified understanding of core PTQ concepts (Dynamic vs. Static, INT8/FP16/BF16, calibration).
  - Practical application of PyTorch Eager Mode quantization tools (`quantize_dynamic`, `QuantStub/DeQuantStub`, `QConfig`, `prepare`, `convert`).
  - The critical importance of selecting the correct quantization backend (`torch.backends.quantized.engine`).
  - The need to inspect model implementations to identify the correct target layer types for quantization (e.g., `Conv1D` in older GPT-2).
  - Reinforced that performance metrics (latency) are more indicative of quantization success than static file size metrics like `state_dict` size for these methods.
  - Gained valuable experience debugging common quantization pitfalls.
