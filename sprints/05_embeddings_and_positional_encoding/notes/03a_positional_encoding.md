# Positional Encoding example output explanation

Let's break down the example output from the `sprints/05_embeddings_and_positional_encoding/results/03_positional_encoding.py` file.

```terminal
Input shape (token IDs): torch.Size([35, 4])
Shape after embedding: torch.Size([35, 4, 512])
Shape after positional encoding: torch.Size([35, 4, 512])

Positional Encoding values for position 0:
tensor([0., 1., 0., 1., 0., 1., 0., 1.])

Positional Encoding values for position 1:
tensor([0.8415, 0.5403, 0.8219, 0.5697, 0.8020, 0.5974, 0.7819, 0.6234])

Input embedding for first token, first batch element (first 8 dims):
tensor([ -6.3033,  -7.8703, -17.1175,  32.7640,  32.4037, -27.1068,  21.2027,
        -42.3517], grad_fn=<SliceBackward0>)

Output after PE for first token, first batch element (first 8 dims):
tensor([ -7.0037,  -7.6337, -19.0194,  37.5156,  36.0041, -29.0076,  23.5586,
        -45.9464], grad_fn=<SliceBackward0>)

Difference (should approx match PE[0] before dropout):
tensor([-0.7004,  0.2366, -1.9019,  4.7516,  3.6004, -1.9008,  2.3559, -3.5946],
       grad_fn=<SubBackward0>)

Difference without dropout (should exactly match PE[0]):
tensor([0., 1., 0., 1., 0., 1., 0., 1.], grad_fn=<SubBackward0>)
PE[0] for comparison:
tensor([0., 1., 0., 1., 0., 1., 0., 1.])

Caught expected error: d_model must be a positive integer.
Caught expected error: dropout_prob must be between 0 and 1.
Caught expected error: max_len must be a positive integer.
```

1.  **Shapes:**
    *   `Input shape (token IDs): torch.Size([35, 4])` - This is our starting point: a batch of 4 sequences, each 35 tokens long.
    *   `Shape after embedding: torch.Size([35, 4, 512])` - The embedding layer turned each token ID into a 512-dimensional vector (our gingerbread cookies).
    *   `Shape after positional encoding: torch.Size([35, 4, 512])` - Adding the positional encoding (the sprinkles!) doesn't change the shape, just the values within the vectors.

2.  **PE Values:**
    *   `Positional Encoding values for position 0:` `tensor([0., 1., 0., 1., 0., 1., 0., 1.])` - This shows the first 8 dimensions of the "label" for the very first position (`pos=0`). Remember the formulas? `sin(0 / ...) = 0` and `cos(0 / ...) = 1`. So, it alternates 0s and 1s.
    *   `Positional Encoding values for position 1:` `tensor([0.8415, 0.5403, ...])` - This is the label for the second position (`pos=1`), calculated using the `sin` and `cos` functions with `pos=1`. It's a unique vector!

3.  **The Addition:**
    *   It shows the first 8 values of an example embedding *before* PE.
    *   It shows the corresponding values *after* PE has been added (note they've changed!).
    *   `Difference (should approx match PE[0] before dropout):` Because the default `PositionalEncoding` includes dropout, the difference isn't *exactly* `PE[0]`, but it's close. Dropout randomly zeros out some elements.
    *   `Difference without dropout (should exactly match PE[0]):` When we run it again using an instance *without* dropout (`dropout_prob=0.0`), the difference between the output and the input embedding perfectly matches the `PE[0]` values (`[0., 1., 0., 1., ...]`). This proves the addition step is working exactly as planned!

4.  **Error Checks:** The last lines confirm that the little safety checks we put in the `__init__` method (like ensuring `d_model` is positive) are working correctly.

So, the example code successfully took token IDs, turned them into embeddings, and then added the unique positional "labels" from our factory. The shapes match up, and the math (when dropout is off) confirms the correct values are being added.
