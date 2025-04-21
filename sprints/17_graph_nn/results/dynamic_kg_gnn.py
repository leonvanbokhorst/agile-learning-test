import os
import json
import networkx as nx
import torch
import numpy as np
import faiss
import litellm
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborSampler
from difflib import get_close_matches
import json
import litellm

"""
Complete dynamic KG + GNN pipeline with GPT-4.1-nano triple extraction via litellm.
- Events -> GPT-based triples -> networkx graph update -> GraphSAGE embeddings -> FAISS index
"""


# Updated extract and processing logic to handle cold-start seeding of entities

import json
import litellm
from difflib import get_close_matches

# Assume name2idx and register_node are already defined as before

def canonicalize(entity: str, known_entities, cutoff=0.6):
    lower_known = [ke.lower() for ke in known_entities]
    matches = get_close_matches(entity.lower(), lower_known, n=1, cutoff=cutoff)
    if matches:
        match_lower = matches[0]
        for orig in known_entities:
            if orig.lower() == match_lower:
                return orig
    return None

def extract_triples_with_seed(text: str):
    """
    1) Use GPT to extract raw triples.
    2) If no known entities yet, register all subjects/objects directly (seed phase).
    3) Otherwise, canonicalize subjects/objects against known_entities.
    """
    prompt = (
        "Extract all subject-verb-object triples from the text. "
        "Return a JSON array of triples, each an array of three strings [subject, relation, object].\n\n"
        f"Text: \"{text}\""
    )
    resp = litellm.completion(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "You are an assistant that extracts SVO triples."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.0
    )
    try:
        raw_triples = json.loads(resp.choices[0].message.content.strip())
    except json.JSONDecodeError:
        return []

    triples = []
    known = list(name2idx.keys())

    # If cold start: seed with raw entities
    if not known:
        for subj, rel, obj in raw_triples:
            register_node(subj)
            register_node(obj)
            triples.append((subj, rel, obj))
        return triples

    # Normal mode: canonicalize, but register if no match
    for subj, rel, obj in raw_triples:
        subj_can = canonicalize(subj, known)
        if subj_can is None:
            subj_can = subj # Use original name if new
            register_node(subj_can)
            known.append(subj_can) # Add to known list for subsequent checks in this batch

        obj_can = canonicalize(obj, known)
        if obj_can is None:
            obj_can = obj # Use original name if new
            register_node(obj_can)
            known.append(obj_can) # Add to known list

        # Now we always have valid subj_can and obj_can (either matched or newly registered)
        triples.append((subj_can, rel, obj_can))
    return triples

# Replace calls in process_event:
#   triples = extract_triples_with_seed(event_text)
# This ensures the first events seed your graph, and later events use canonicalization.


def extract_triples_with_entity_list(text: str):
    """
    1) If cold‑start (no entities yet), fall back to seed logic.
    2) Else, ask GPT-4.1-nano to only pick subjects/objects from your known list.
    """
    known = list(name2idx.keys())
    # Cold start: seed exactly once
    if not known:
        return extract_triples_with_seed(text)

    # Build a prompt that nails down subjects/objects
    prompt = (
        "You have these entity names: " + ", ".join(known) + ".\n"
        "Extract all facts as [subject, relation, object] triples where subject and object\n"
        "must be exactly one of those names. Relation should be only the verb phrase\n"
        "connecting them (no extra words). Return a JSON array of triples.\n\n"
        f"Text: \"{text}\""
    )
    resp = litellm.completion(
        model="gpt-4.1-nano",
        messages=[
            {"role":"system","content":"You’re a precise triple extractor."},
            {"role":"user",  "content":prompt}
        ],
        temperature=0.0,
    )
    try:
        raw = json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        return []

    # Now we know GPT only returned legal names, so just register any brand‑new ones
    triples = []
    for subj, rel, obj in raw:
        if subj not in name2idx:
            register_node(subj)
        if obj not in name2idx:
            register_node(obj)
        triples.append((subj, rel, obj))
    return triples

# Then in your pipeline, swap out:
#   triples = extract_triples_with_seed(event_text)
# for
#   triples = extract_triples_with_entity_list(event_text)


# ─── 0. LLM SETUP ────────────────────────────────────────────────────────────────
litellm.api_key = os.getenv("OPENAI_API_KEY")  # ensure your key is in the ENV

# ─── GLOBALS ────────────────────────────────────────────────────────────────────
G = nx.DiGraph()         # Directed graph for facts/triples
EMB_DIM = 16             # Embedding dimension
MAX_NODES = 100          # Maximum unique entities expected
name2idx = {}            # Maps node name -> integer index
next_idx = 0             # Next available index for a new node

# FAISS index for storing node embeddings
index = faiss.IndexFlatL2(EMB_DIM)

# ─── EMBEDDING TABLE & MODEL ───────────────────────────────────────────────────
embedding_table = torch.nn.Embedding(MAX_NODES, EMB_DIM)

class SimpleGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
    def forward(self, x, edge_index):
        return self.conv1(x, edge_index)

# Initialize GraphSAGE (in practice, load pre-trained weights)
model = SimpleGraphSAGE(in_channels=EMB_DIM, hidden_channels=EMB_DIM)

# ─── 1. REGISTER NEW NODES ─────────────────────────────────────────────────────
def register_node(name: str):
    """Assigns a unique index to each new node name."""
    global next_idx
    if name not in name2idx:
        name2idx[name] = next_idx
        next_idx += 1

# ─── 2. GPT-BASED TRIPLE EXTRACTION ─────────────────────────────────────────────
# def extract_triples(text: str):
#     """
#     Uses GPT-4.1-nano (via litellm) to extract (subject, relation, object) triples in JSON.
#     Falls back to empty list on parse errors.
#     """
#     prompt = (
#         "Extract all subject-verb-object triples from the text. "
#         "Return a JSON array of triples, each an array of three strings [subject, relation, object].\n\n"
#         f"Text: \"{text}\""
#     )
#     resp = litellm.completion(
#         model="gpt-4.1-nano",
#         messages=[
#             {"role": "system", "content": "You are an expert at extracting precise SVO triples (named entities) for a knowledge graph."},
#             {"role": "user",   "content": prompt}
#         ],
#         temperature=0.0
#     )
#     content = resp.choices[0].message.content.strip()
#     try:
#         return json.loads(content)
#     except json.JSONDecodeError:
#         # Fallback: no triples if GPT misformats
#         return []

# ─── 3. GRAPH UPDATE ────────────────────────────────────────────────────────────
def update_graph(triples):
    """
    Adds nodes and directed edges to the global graph from extracted triples.
    """
    for subj, rel, obj in triples:
        register_node(subj)
        register_node(obj)
        G.add_node(subj)
        G.add_node(obj)
        G.add_edge(subj, obj, relation=rel)

# ─── 4. EMBEDDING RECOMPUTATION ────────────────────────────────────────────────
def recompute_embedding(node: str):
    """
    Recomputes the embedding for `node` by running GraphSAGE on the full graph.
    Adds the result to the FAISS index.
    """
    # Build edge_index tensor from networkx
    edge_list = [(name2idx[u], name2idx[v]) for u, v in G.edges()]
    if not edge_list:
        return
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Feature matrix via embedding lookup
    x = embedding_table(torch.arange(next_idx))

    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        emb = out[name2idx[node]].cpu().numpy().astype('float32')
        index.add(np.expand_dims(emb, axis=0))

# ─── 5. END-TO-END EVENT PROCESSOR ─────────────────────────────────────────────
def process_event(event_text: str):
    """
    Full pipeline:
      1) Extract triples via GPT
      2) Update the graph
      3) Recompute and index embeddings for affected nodes
    """
    triples = extract_triples_with_entity_list(event_text)
    update_graph(triples)
    # Recompute embeddings for each subject/object in the event
    for subj, _, obj in triples:
        recompute_embedding(subj)
        recompute_embedding(obj)

# ─── 6. EXAMPLE USAGE ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_event = "Ava works with Cara on Phoenix and evaluates ethics protocols"
    process_event(sample_event)
    print("Registered nodes:", name2idx)
    print("Graph edges:", list(G.edges(data=True)))
    print("FAISS index size:", index.ntotal)

    sample_event = "Lonn is a good friend of Ava"
    process_event(sample_event)
    print("Registered nodes:", name2idx)
    print("Graph edges:", list(G.edges(data=True)))
    print("FAISS index size:", index.ntotal)

    sample_event = "Ben doesn't like to work with Lonn"
    process_event(sample_event)
    print("Registered nodes:", name2idx)
    print("Graph edges:", list(G.edges(data=True)))
    print("FAISS index size:", index.ntotal)

    sample_event = "Lonn travels to Barcelona"
    process_event(sample_event)
    print("Registered nodes:", name2idx)
    print("Graph edges:", list(G.edges(data=True)))
    print("FAISS index size:", index.ntotal)

    sample_event = "Ava doesn't like working with Lonn much"
    process_event(sample_event)
    print("Registered nodes:", name2idx)
    print("Graph edges:", list(G.edges(data=True)))
    print("FAISS index size:", index.ntotal)