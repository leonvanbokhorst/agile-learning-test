# Types of Quantization

There are several ways to apply quantization to a model. The main distinction often lies in _when_ the quantization is applied and _what_ data is needed.

## 1. Post-Training Dynamic Quantization (PTQ Dynamic)

- **Analogy:** The "Lazy Wizard" - Quick and easy, minimal setup.
- **How it Works:**
  - Model weights (typically for specific layers like Linear, LSTM, etc.) are converted to a lower-precision format (e.g., INT8) _offline_ after training.
  - Activations (the values flowing between layers) are computed in floating-point (e.g., FP32) during inference and then converted to the lower-precision format _on-the-fly_ just before being used in a quantized operation.
  - The range and zero-point needed for activation quantization are determined dynamically based on the data observed during that specific inference pass.
- **Pros:**
  - **Simplicity:** Often the easiest method to implement, sometimes just a single function call.
  - **No Calibration Data Needed:** Doesn't require a separate dataset for calibration.
  - Good starting point for exploring quantization benefits.
- **Cons:**
  - **Potentially Less Speedup:** The overhead of dynamically calculating scales and converting activations during each inference run can limit the maximum achievable speedup compared to static quantization.
  - Accuracy might be slightly lower than static quantization in some cases.
- **Use Case:** Good for models where activation computations dominate (like LSTMs/Transformers) and when getting a calibration dataset is difficult.

## 2. Post-Training Static Quantization (PTQ Static)

- **Analogy:** The "Prepared Alchemist" - Requires preparation (calibration) for potentially better results.
- **How it Works:**
  - Both model weights _and_ activations are converted to a lower-precision format _offline_.
  - Requires a **calibration step**: After training, you feed a small, representative dataset (the calibration dataset) through the model in floating-point.
  - During calibration, the model observes the range of activation values for each layer that needs to be quantized.
  - Based on these observed ranges, fixed scaling factors (scale and zero-point) are calculated for the activations.
  - These fixed scaling factors are then used during inference, avoiding the on-the-fly calculation needed in dynamic quantization.
- **Pros:**
  - **Potentially Faster Inference:** By pre-computing activation scaling factors, it avoids runtime overhead, often leading to better speedups than dynamic quantization, especially on hardware optimized for low-precision compute.
  - Can sometimes achieve slightly better accuracy than dynamic quantization.
- **Cons:**
  - **Requires Calibration Data:** Needs a dataset that accurately reflects the data the model will see in production.
  - **More Complex Workflow:** Involves an extra calibration step compared to dynamic quantization.
- **Use Case:** Often preferred for CNNs and situations where maximum inference speed is critical and a representative calibration dataset is available.

## 3. Quantization-Aware Training (QAT)

- **Analogy:** The "Master Craftsman" - Builds robustness to quantization during training.
- **How it Works:**
  - Simulates the effects of quantization _during_ the model training or fine-tuning process.
  - Special nodes (Fake Quantization modules) are inserted into the model graph to mimic the rounding and clamping effects of converting floats to integers.
  - The model learns to adjust its weights to minimize the accuracy loss caused by this simulated quantization.
- **Pros:**
  - **Highest Potential Accuracy:** Usually achieves the best accuracy among the quantization methods because the model explicitly learns to compensate for quantization errors.
- **Cons:**
  - **Requires Training/Fine-tuning:** Needs access to the training pipeline and data, and involves the computational cost of training.
  - **Most Complex:** More involved setup and understanding required compared to PTQ methods.
- **Use Case:** When post-training quantization results in an unacceptable accuracy drop, and retraining resources are available.

## Focus for Sprint 14

For this sprint, we are primarily focusing on **Post-Training Quantization (PTQ)**, exploring both **Dynamic** and **Static** methods as they don't require retraining the model from scratch.
