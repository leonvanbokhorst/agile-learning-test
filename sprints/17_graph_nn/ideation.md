# Zachary’s Karate Club social graph

## 1. Introduction

It all started with Wayne W. Zachary’s 1977 study “An Information Flow Model for Conflict and Fission in Small Groups,” where he spent three years (1970–72) logging friendships and off‑mat interactions among the 34 members of a university karate club citeturn0search0. When a power struggle erupted between the instructor (node 1) and the club president (node 34), Zachary used a max‑flow/min‑cut algorithm to predict which side each person would end up on—and nailed it for everyone but one citeturn0search0.

The publicly‑shared network has 78 edges—though there’s a tiny ambiguity around one link—and captures exactly who socialized with whom outside practice. It’s small enough to toy with by hand, but rich enough to show clear community structure, which is why it became the go‑to benchmark citeturn0search0.

Then in 2002, Michelle Girvan and Mark Newman spotlighted it in their landmark PNAS paper on community detection, cementing “Zachary’s Karate Club” as the first example everyone pulls out when testing new graph algorithms citeturn0search0. For a first GNN experiment it’s perfect: known ground‑truth factions, a manageable size, and a real‑world backstory to boot. 

## 2. Graph Convolutional Network (GCN) in PyTorch

A simple GCN in PyTorch. You’ll see how messages flow across friendships, then you can riff on feeding those embeddings into an LLM later. 

First, grab PyTorch Geometric (it wraps all the graph ops you’d otherwise implement by hand). Then load the KarateClub dataset, define a two‑layer GCN, and train it to predict which “faction” each node belongs to. Under the hood you’re just doing sparse adjacency multiplications and ReLU activations—but seeing it in code is surprisingly rewarding.

```python
import torch
import torch.nn.functional as F
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv

dataset = KarateClub()
data = dataset[0]

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.conv2(x, edge_index), dim=1)

model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
```

Run that and you’ll start seeing loss drop as nodes learn to “agree” with their neighbors. Once you’ve got embeddings, you could translate each node’s vector into a natural‑language summary of its position (“Alice sits between the influencer clique and the newcomers”), then feed those summaries plus a prompt into your LLM for richer social reasoning.

Give this a spin and tell me how the loss curve looks or if the embeddings cluster nicely—then we’ll brainstorm how to hand off those vectors to GPT for, say, smart intro‑email drafts or conflict‑resolution advice.
