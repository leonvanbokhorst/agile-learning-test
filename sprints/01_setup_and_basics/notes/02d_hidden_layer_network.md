# Understanding a Simple Neural Network with One Hidden Layer

## Network Architecture

Our network has the following structure:

- **Input Layer**: 3 features (e.g., height, weight, age)
- **Hidden Layer**: 4 neurons
- **Output Layer**: 1 neuron

```python
class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(3, 4)  # 3 inputs → 4 hidden neurons
        self.output = nn.Linear(4, 1)  # 4 hidden → 1 output

    def forward(self, x):
        x = self.hidden(x)
        x = torch.relu(x)  # Activation function
        x = self.output(x)
        return x
```

## Parameter Count

The network contains 21 total parameters:

- **Hidden Layer**:
  - Weights: 12 (4 neurons × 3 inputs)
  - Biases: 4 (1 per hidden neuron)
- **Output Layer**:
  - Weights: 4 (1 output × 4 hidden neurons)
  - Bias: 1 (1 for output)

## Signal Flow

1. **Input to Hidden Layer**:

   - Each input feature connects to all 4 hidden neurons
   - Each connection has its own weight
   - Each hidden neuron adds its own bias

2. **Hidden Layer Activation**:

   - ReLU activation function: max(0, x)
   - Turns off neurons with negative outputs (sets them to 0)
   - Allows for non-linear combinations of features

3. **Hidden to Output**:
   - Each hidden neuron connects to the output
   - Final weighted sum produces the prediction

## Complexity and Learning

- **Interconnectedness**: All parameters are connected through the network
- **Learning Process**:
  - Each weight is adjusted based on its contribution to the final error
  - Backpropagation calculates how much each weight should change
  - The network learns to combine features in complex ways

## Visualization

```
Input Layer        Hidden Layer        Output Layer
[Feature 1] ────┐
                ├─→ [Neuron 1] ───┐
[Feature 2] ────┼─→ [Neuron 2] ───┼─→ [Output]
                ├─→ [Neuron 3] ───┘
[Feature 3] ────┘
```

## Key Concepts

1. **Weight Sharing**: Each input feature affects all hidden neurons
2. **Non-linearity**: ReLU activation allows for complex decision boundaries
3. **Parameter Updates**: All 21 parameters are updated during training
4. **Feature Combination**: Hidden layer can learn to combine features in complex ways

## Practical Implications

- Even a "simple" network can learn complex patterns
- The number of parameters grows quickly with more hidden neurons
- Each neuron can specialize in different aspects of the input
- The network can learn hierarchical features (simple → complex)
