# PyTorch Quantization API Overview (Eager Mode Focus)

PyTorch provides tools for quantization primarily under the `torch.ao.quantization` namespace. While newer approaches like PyTorch 2 Export exist, the **Eager Mode Quantization** workflow provides a foundational understanding and is often sufficient for many post-training quantization tasks.

## Key Concepts in Eager Mode

Eager mode quantization operates directly on the standard `nn.Module` objects. It generally requires manual configuration by the user.

1.  **Manual Module Specification:** You typically need to specify which modules or types of modules you want to quantize.
2.  **`QuantStub` and `DeQuantStub`:** These are `nn.Module`s you insert into your model definition to explicitly mark the points where tensors should transition from floating-point (`FP32`) to quantized (`INT8`) format (`QuantStub`) and back (`DeQuantStub`). This is crucial for static quantization.
3.  **Fusion:** Often, sequences of operations (like Conv -> BatchNorm -> ReLU) should be fused into a single operation before quantization for better accuracy and performance. PyTorch provides `torch.ao.quantization.fuse_modules` for this.
4.  **QConfig (Quantization Configuration):** A configuration object (`torch.ao.quantization.QConfig`) specifies _how_ quantization should be done. It holds:
    - **Observer:** Determines how to collect statistics about tensor ranges during calibration (for static quantization). Examples: `MinMaxObserver`, `MovingAverageMinMaxObserver`.
    - **Quantization Scheme:** Specifies the mapping from float to int (e.g., affine or symmetric, per-tensor or per-channel).
    - **Target Data Type:** e.g., `torch.qint8`, `torch.quint8`.
    - PyTorch provides defaults like `get_default_qconfig('x86')` (recommended for server CPU) or `get_default_qconfig('qnnpack')` (for mobile).

## Core Eager Mode APIs for PTQ

### Post-Training Dynamic Quantization (PTQ Dynamic)

- **`torch.ao.quantization.quantize_dynamic(model, qconfig_spec=None, dtype=torch.qint8, mapping=None, inplace=False)`:**
  - The primary function for dynamic quantization.
  - Converts specified layers (e.g., `{torch.nn.Linear}`) to use INT8 weights.
  - Activations remain FP32 until computed on-the-fly.
  - Relatively simple to apply.

### Post-Training Static Quantization (PTQ Static)

This involves a multi-step process:

1.  **(Optional but Recommended) Fuse Modules:**
    - `model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32, [['conv', 'relu'], ...])`
2.  **Prepare the Model:**
    - Assign a `qconfig` to the model or its submodules (`model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('x86')`).
    - Insert `QuantStub` / `DeQuantStub` markers in the `forward` method.
    - Call `model_fp32_prepared = torch.ao.quantization.prepare(model_fp32_fused)`.
    - This inserts _observers_ into the model based on the `qconfig`.
3.  **Calibrate:**
    - Feed representative data through the prepared model: `model_fp32_prepared(calibration_data)`.
    - The observers collect statistics on activation ranges.
4.  **Convert:**
    - `model_int8 = torch.ao.quantization.convert(model_fp32_prepared)`.
    - This uses the collected statistics to convert weights to INT8 and prepare layers to handle INT8 activations using the calculated scales and zero-points.

## Limitations of Eager Mode

- Requires manual insertion of stubs and fusion calls.
- Doesn't automatically handle operations defined using the functional API (`torch.nn.functional`).

For this sprint's PoC, focusing on these Eager Mode APIs for Dynamic and Static PTQ should provide a solid foundation.
