# Common Data Types in Quantization

When we talk about quantization, we're essentially converting from the standard 32-bit floating-point (`FP32`) format to formats that use fewer bits. Here are the most common ones encountered:

## 1. INT8 (8-bit Integer)

- **Bits:** 8
- **Representation:** Represents integers. Can be:
  - **Signed (`int8`):** Typically represents values from -128 to 127.
  - **Unsigned (`uint8`):** Typically represents values from 0 to 255.
- **Size Reduction:** Significant (~4x smaller than `FP32`).
- **Speed:** Integer arithmetic is often much faster than floating-point math on many CPUs and specialized hardware (NPUs, TPUs).
- **Accuracy:** Has the lowest precision and smallest range among these types. This means it has the highest potential for accuracy degradation if not applied carefully.
- **Mapping:** Requires **scaling factors (scale and zero-point)** to map the original floating-point range to the limited INT8 range. The formula is often like: `float_value ≈ (int8_value - zero_point) * scale`.
- **Usage:** Very common target for post-training quantization (both dynamic and static) due to maximum size/speed benefits, especially for deployment on edge devices or CPUs.

## 2. FP16 (16-bit Floating Point / Half Precision)

- **Bits:** 16 (Typically: 1 sign, 5 exponent, 10 mantissa/significand bits).
- **Representation:** Represents floating-point numbers with less precision and a smaller range than `FP32`.
- **Size Reduction:** Reduces model size by half (~2x smaller than `FP32`).
- **Speed:** Can provide significant speedups on hardware that supports FP16 computation natively (many modern GPUs, TPUs).
- **Accuracy:** Offers better precision than `INT8`. Less accuracy loss is generally expected compared to `INT8`.
- **Range Issue:** Has a much smaller maximum representable value compared to `FP32` (~65,504). This can sometimes lead to **overflow** (values becoming too large) or **underflow** (gradients becoming zero) issues, particularly during _training_ (less commonly an issue just for inference).
- **Usage:** Widely used in **mixed-precision training** (where some parts of the computation stay in FP32 for stability) and sometimes as an inference target, offering a balance between size/speed and accuracy.

## 3. BF16 (BFloat16 / Brain Floating Point)

- **Bits:** 16 (Typically: 1 sign, 8 exponent, 7 mantissa/significand bits).
- **Representation:** Also a 16-bit floating-point format, developed by Google.
- **Size Reduction:** Reduces model size by half (~2x smaller than `FP32`), same as `FP16`.
- **Speed:** Offers speedups similar to `FP16` on supported hardware (Google TPUs, recent NVIDIA GPUs like Ampere/Hopper onwards, some CPUs).
- **Key Difference vs FP16:** `BF16` keeps the **same exponent range as `FP32`** (8 exponent bits) but has fewer precision bits (7 mantissa bits) than `FP16` (10 mantissa bits).
- **Benefit:** The wider dynamic range (matching `FP32`) makes `BF16` much less prone to the overflow/underflow issues seen with `FP16`, making it more stable, especially for training large models.
- **Accuracy:** Precision is lower than `FP16`, but the stability often makes it preferable for training deep learning models where large activation or gradient values might occur.
- **Usage:** Increasingly popular for training and inference, especially on hardware that supports it, as it often provides a good combination of speed, size reduction, and numerical stability.

## Summary Table (Conceptual)

| Feature         | FP32 (Baseline) | INT8             | FP16          | BF16             |
| :-------------- | :-------------- | :--------------- | :------------ | :--------------- |
| Bits            | 32              | 8                | 16            | 16               |
| Type            | Float           | Integer          | Float         | Float            |
| Size vs FP32    | 1x              | ~0.25x           | ~0.5x         | ~0.5x            |
| Dynamic Range   | Very High       | Very Low         | Low           | Very High        |
| Precision       | High            | Very Low         | Medium        | Low              |
| Speed Potential | Baseline        | Highest          | High          | High             |
| Stability       | High            | Requires Mapping | Medium (Risk) | High             |
| Hardware        | Universal       | CPU/NPU/TPU Opt. | GPU/TPU Opt.  | TPU/New GPU Opt. |
