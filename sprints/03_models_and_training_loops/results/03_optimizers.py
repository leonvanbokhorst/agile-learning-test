import torch.nn as nn
import torch.optim as optim


# Dummy Model (replace with your actual model later)
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)  # Example: 10 inputs, 2 outputs

    def forward(self, x):
        return self.linear(x)


# Instantiate the model
model = SimpleModel()

# --- Optimizer Section ---

# Define hyperparameters
learning_rate = 0.001

# Instantiate an optimizer (e.g., Adam)
# It needs the model's parameters to know what to update
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Optimizer created:")
print(optimizer)

# Example of how it would be used in a training loop (conceptual)
# We don't have data or a loss function here yet

# मान लीजिए (maan lijiye - Hindi for 'suppose') we have inputs and calculate loss
# inputs = torch.randn(5, 10) # Example batch of 5 samples, 10 features
# outputs = model(inputs)
# labels = torch.randint(0, 2, (5,)) # Example labels
# loss_fn = nn.CrossEntropyLoss()
# loss = loss_fn(outputs, labels)

# --- These steps happen in the training loop ---
# 1. Zero gradients before backward pass
# optimizer.zero_grad()

# 2. Calculate gradients
# loss.backward()

# 3. Update weights
# optimizer.step()
# ---------------------------------------------

# You can also try SGD
sgd_optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
print("\nSGD optimizer also created:")
print(sgd_optimizer)

# Add more experiments or comparisons here!
