# First, install PyTorch Geometric following instructions at:
# https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

import torch
import torch.nn.functional as F
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import faiss
import numpy as np
import os
import litellm
from sentence_transformers import SentenceTransformer

# ─── 1. Load the dataset ───────────────────────────────────────────────────────
# This gives you a single graph with 34 nodes, each node feature is one‑hot (34‑dim),
# and labels are the two real‑world factions after the split.
dataset = KarateClub()
data = dataset[0]  # our graph

# ─── 2. Create train/test splits ───────────────────────────────────────────────
# As in the original GCN demo: we only give labels for the instructor (node 0)
# and the president (node 33), then test on the rest.
data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.train_mask[0] = True  # Instructor's node
data.train_mask[33] = True  # President's node
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
    print(embeddings.shape)
# 'embeddings[i]' now holds a 16‑dim vector for node i

# assume embeddings is your [34,16] tensor and data.y has the true 0/1 labels
emb = embeddings.cpu().numpy()
labels = data.y.cpu().numpy()

# squeeze down to 2D
proj = TSNE(n_components=2, random_state=42).fit_transform(emb)

plt.figure(figsize=(6,6))
# two scatter calls so matplotlib picks two default colors for you
plt.scatter(proj[labels==0,0], proj[labels==0,1], label='Faction 0', alpha=0.8)
plt.scatter(proj[labels==1,0], proj[labels==1,1], label='Faction 1', alpha=0.8)
plt.legend()
plt.title('t‑SNE on your Karate Club embeddings')
plt.xlabel('TSNE 1'); plt.ylabel('TSNE 2')
plt.tight_layout()
# plt.show() # Commented out show
plt.savefig('tsne_karate_club.png') # Save the plot to a file instead
print("Plot saved to tsne_karate_club.png")


# 1. Prepare embeddings and labels
emb_np = embeddings.cpu().numpy().astype('float32')
labels = data.y.cpu().numpy()

# 2. Build a flat L2 index
d = emb_np.shape[1]
index = faiss.IndexFlatL2(d)
index.add(emb_np)

# 3. Search 3 nearest (including itself at position 0)
k = 3
distances, neighbors = index.search(emb_np, k)

# 4. Identify bridge nodes
bridge_nodes = [
    i for i in range(len(neighbors))
    if any(labels[j] != labels[i] for j in neighbors[i, 1:])
]
print("Bridge nodes:", bridge_nodes)

# 6. Pick the best bridge node
scores = []
for i in range(len(emb_np)):
    dists = distances[i,1:]      # skip self
    nbrs  = neighbors[i,1:]
    same  = [d for d,j in zip(dists,nbrs) if labels[j]==labels[i]]
    opp   = [d for d,j in zip(dists,nbrs) if labels[j]!=labels[i]]
    score = 0
    if len(same) > 0 and len(opp) > 0:
        score = np.mean(opp) - np.mean(same)
    scores.append((i, score))

best_bridge = max(scores, key=lambda x: x[1])[0]
print("Best bridge node:", best_bridge)

# 7. Create RAG context
docs = [
    {"node_id": 8, "name": "Ava", "text": "Ava is an AI Researcher at Fontys, expert in LMs & KGs. Last led Project Phoenix connecting model dev and infra teams."},
    {"node_id": 9, "name": "Ben", "text": "Ben is our Data Scientist, built analytics pipelines for Orion and Phoenix."},
    {"node_id": 30, "name": "Cara", "text": "Cara, ML Engineer, focuses on deployment & MLOps. Co‑owned Phoenix with Ava."},
    # …and so on for all 34 nodes
]

# 3) Embed docs
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [d["text"] for d in docs]
embeddings = model.encode(texts, convert_to_numpy=True).astype('float32')

# 4) Build FAISS index
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)


# 5) Retrieve context using FAISS
def retrieve_context(node_id, k=3):
    # Find the index of the document corresponding to the node_id
    doc_index = next((idx for idx, d in enumerate(docs) if d['node_id'] == node_id), None)
    if doc_index is None:
        return [] # Node not found in docs

    query_embedding = embeddings[doc_index:doc_index+1] # Keep it 2D for FAISS search
    distances, neighbor_indices = index.search(query_embedding, k + 1) # Search for k+1 nearest

    # Get neighbor documents, skipping the first one (itself)
    hits = [docs[i] for i in neighbor_indices[0][1:]]
    return hits


# 6) Build your RAG prompt
# build a quick lookup
id2doc = {d['node_id']: d for d in docs}

def make_rag_prompt(node_id):
    entry   = id2doc[node_id]          # Ava, Ben, or Cara
    name    = entry['name']            # "Cara"
    own_txt = entry['text']            # Cara's bio
    hits    = retrieve_context(node_id)
    ctx_txt = "\n\n".join(h['text'] for h in hits)
    
    return (
        f"Here's some background on colleagues you'll work with:\n\n{ctx_txt}\n\n"
        f"Now draft a concise, friendly email introducing {name} "
        "into the AI ethics workshop, highlighting how they can bridge modeling and deployment teams."
    )


# 7) Call the LLM
litellm.api_key = os.getenv("OPENAI_API_KEY")
prompt = make_rag_prompt(8)
resp = litellm.completion(
    model="gpt-4o-mini",
    messages=[{"role":"system","content":"You're a helpful AI team coordinator."},
              {"role":"user","content":prompt}],
    temperature=0.7,
)
print(resp.choices[0].message.content)



# # 5. Generate introduction emails
# def make_intro_prompt(node_id):
#     nbrs = neighbors[node_id, 1:].tolist()
#     return (
#         f"Node {node_id} in this org‑graph bridges two camps. "
#         f"Their closest peers are nodes {nbrs}. "
#         f"Draft a concise email introducing them to help mediate between the two groups."
#     )

# # 7. Generate introduction email for the best bridge node
# print(make_intro_prompt(best_bridge))

