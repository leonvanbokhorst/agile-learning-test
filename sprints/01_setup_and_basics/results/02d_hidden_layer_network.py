import torch
import torch.nn as nn
import torch.nn.functional as F

print("--- Hidden Layer Network Example ---")


# Let's create a network with:
# - 3 input features (like before)
# - 4 hidden neurons (our "brain cells")
# - 1 output neuron
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


# Create our network
import itertools
net = SimpleNetwork()
print("\nNetwork structure:")
print(net)

# Let's see all the parameters (weights and biases)
print("\nAll parameters:")
for name, param in net.named_parameters():
    print(f"{name}: shape {param.shape}")

# Create some input data
x = torch.tensor([1.75, 70.0, 25.0], dtype=torch.float32)
print(f"\nInput features: {x}")

# Forward pass
output = net(x)
print(f"\nRaw output: {output}")

# Let's see what's happening in the hidden layer
with torch.no_grad():
    hidden_output = net.hidden(x)
    print(f"\nHidden layer output (before activation): {hidden_output}")
    hidden_activated = torch.relu(hidden_output)
    print(f"Hidden layer output (after ReLU): {hidden_activated}")

# Let's visualize the connections
print("\nConnection visualization:")
print("Input → Hidden Layer:")
for i, j in itertools.product(range(3), range(4)):
    weight = net.hidden.weight[j, i].item()
    print(f"Input {i} → Hidden {j}: weight = {weight:.4f}")

print("\nHidden Layer → Output:")
for j in range(4):  # hidden neurons
    weight = net.output.weight[0, j].item()
    print(f"Hidden {j} → Output: weight = {weight:.4f}")

# Let's do a forward pass and see how the signal flows
print("\nSignal flow example:")
with torch.no_grad():
    # Input to hidden
    hidden_input = torch.matmul(net.hidden.weight, x) + net.hidden.bias
    print(f"Hidden layer input: {hidden_input}")

    # Hidden activation
    hidden_act = torch.relu(hidden_input)
    print(f"After ReLU: {hidden_act}")

    # Hidden to output
    output = torch.matmul(net.output.weight, hidden_act) + net.output.bias
    print(f"Final output: {output}")

print("\nExample complete!")
