# First, install PyTorch Geometric following instructions at:
# https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

import torch
import torch.nn.functional as F
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv

# ─── 1. Load the dataset ───────────────────────────────────────────────────────
# This gives you a single graph with 34 nodes, each node feature is one‑hot (34‑dim),
# and labels are the two real‑world factions after the split.
dataset = KarateClub()
data = dataset[0]  # our graph

# ─── 2. Create train/test splits ───────────────────────────────────────────────
# As in the original GCN demo: we only give labels for the instructor (node 0)
# and the president (node 33), then test on the rest.
data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.train_mask[0] = True  # Instructor’s node
data.train_mask[33] = True  # President’s node
data.test_mask = ~data.train_mask  # Everything else for evaluation


# ─── 3. Define the GCN model ───────────────────────────────────────────────────
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # First graph conv: from input features → 16 hidden dims
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        # Second conv: from 16 dims → num_classes (2 factions)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # 1) Message‑passing: aggregate neighbor features
        x = self.conv1(x, edge_index)
        # 2) Nonlinearity
        x = F.relu(x)
        # 3) Dropout for regularization (50%)
        x = F.dropout(x, p=0.5, training=self.training)
        # 4) Second conv to get logits for each class
        x = self.conv2(x, edge_index)
        # 5) Log‑softmax for classification
        return F.log_softmax(x, dim=1)


# ─── 4. Setup device, model, optimizer ────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


# ─── 5. Training and testing functions ────────────────────────────────────────
def train():
    model.train()  # set dropout, etc.
    optimizer.zero_grad()  # clear old gradients
    out = model(data)  # forward pass
    # compute loss only on labeled nodes
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()  # backprop
    optimizer.step()  # gradient step
    return loss.item()


def test():
    model.eval()  # disable dropout
    out = model(data)
    pred = out.argmax(dim=1)  # highest log‑prob = predicted class
    # compare only on test nodes
    correct = pred[data.test_mask] == data.y[data.test_mask]
    return int(correct.sum()) / int(data.test_mask.sum())


# ─── 6. Run the training loop ─────────────────────────────────────────────────
for epoch in range(1, 201):
    loss = train()
    if epoch % 20 == 0:
        acc = test()
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Test Acc: {acc:.4f}")

# ─── 7. Extract node embeddings ────────────────────────────────────────────────
# After training, you can grab the hidden representations from conv1
model.eval()
with torch.no_grad():
    embeddings = model.conv1(data.x, data.edge_index)  # shape [34, 16]
# 'embeddings[i]' now holds a 16‑dim vector for node i
