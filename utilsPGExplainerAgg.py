print("🧠 I'm utils.py and I'm being imported!")

from dialogue_config import FAIL, SUCCESS
import csv
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_sparse import SparseTensor, matmul
from tqdm import tqdm
import math
import itertools
import random

# ---------- Utility Functions ----------
def convert_list_to_dict(lst):
    if len(lst) > len(set(lst)):
        raise ValueError('List must be unique!')
    return {k: v for v, k in enumerate(lst)}

def remove_empty_slots(dic):
    for id in list(dic.keys()):
        for key in list(dic[id].keys()):
            if dic[id][key] == '':
                dic[id].pop(key)

# ---------- Custom MinCutPool Layer ----------
class MinCutPoolLayerSparse(nn.Module):
    def __init__(self, in_dim, num_clusters):
        super(MinCutPoolLayerSparse, self).__init__()
        self.num_clusters = num_clusters
        self.assign_net = nn.Linear(in_dim, num_clusters)
        self.proj_net = nn.Linear(in_dim, in_dim)

    def forward(self, x, adj_sparse: SparseTensor):
        S = torch.softmax(self.assign_net(x), dim=-1)            # (N, num_clusters)
        X_proj = self.proj_net(x)                               # (N, F)
        Z = torch.matmul(S.T, X_proj)                           # (num_clusters, F)
        adj_S = matmul(adj_sparse, S)                           # (N, num_clusters)
        adj_new = torch.matmul(S.T, adj_S)                      # (num_clusters, num_clusters)

        deg = adj_sparse.sum(dim=1).to_dense()                  # (N,)
        D = deg.unsqueeze(-1) * torch.ones_like(S)              # (N, num_clusters)
        vol = torch.trace(torch.matmul(S.T, D))
        cut = torch.trace(adj_new)
        mincut_loss = -cut / (vol + 1e-9)

        SS = torch.matmul(S.T, S)
        I = torch.eye(self.num_clusters, device=S.device)
        ortho_loss = torch.norm(SS - I, p='fro')

        # Return adj_new as a dense matrix (we'll convert to edge_index later)
        return Z, adj_new.unsqueeze(0), mincut_loss, ortho_loss, S

# ---------- PGExplainer-style Aggregation Layer (PGExplainerAgg) ----------
class PGExplainerAggLayer(nn.Module):
    """
    A PGExplainer-inspired aggregation layer:
    - Learns an edge importance mask via a small MLP that scores (u,v) pairs.
    - Aggregates neighbor features weighted by learned mask values.
    This provides a trainable, explainable attention-like aggregation which can be
    used as a proxy to PGExplainer explanations for comparison against Shapley-based methods.
    """
    def __init__(self, in_channels, out_channels, hidden=64):
        super(PGExplainerAggLayer, self).__init__()
        # Mask generator / scorer: input is concat(h_u, h_v) -> score
        self.mask_net = nn.Sequential(
            nn.Linear(2 * in_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.out_act = nn.ReLU()

    def forward(self, x, edge_index):
        """
        x: (N, F)
        edge_index: (2, E) with format [src, dst] (i.e., messages from dst -> src if src is row)
        We'll compute mask for each edge (src, dst) using concat(x[src], x[dst]),
        then messages = x[dst] * mask, aggregated to src nodes.
        """
        if edge_index.numel() == 0:
            # No edges: just apply linear + activation on node features
            out = self.linear(x)
            return self.out_act(out), None

        row = edge_index[0]  # source (receiver) nodes
        col = edge_index[1]  # neighbor (sender) nodes

        x_row = x[row]       # (E, F)
        x_col = x[col]       # (E, F)
        concat = torch.cat([x_row, x_col], dim=1)  # (E, 2F)

        scores = self.mask_net(concat).squeeze(-1)   # (E,)
        masks = torch.sigmoid(scores)                # (E,)

        # Messages: neighbor features weighted by mask
        messages = x_col * masks.unsqueeze(-1)      # (E, F)

        # Aggregate: sum messages per target node (row)
        num_nodes = x.size(0)
        agg = torch.zeros_like(x)                   # (N, F)
        # index_add_ supports only 1d indices; use scatter-add via index_add_
        agg.index_add_(0, row, messages)

        # Also compute normalization: sum of masks per target node
        norm = torch.zeros((num_nodes, 1), device=x.device)
        norm.index_add_(0, row, masks.unsqueeze(-1))

        # Combine aggregated message with self feature (residual)
        # Normalize aggregated messages; avoid division by zero by adding 1.0 (self)
        normalized = agg / (norm + 1e-9)
        combined = normalized + x  # residual connection

        out = self.linear(combined)
        return self.out_act(out), masks  # return masks so they can be inspected if needed

# ---------- Full Model with Adaptive Loss Weights (using PGExplainerAgg) ----------
class MinCutPGExplainerGNN_Improved(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_clusters):
        super(MinCutPGExplainerGNN_Improved, self).__init__()
        self.mincut = MinCutPoolLayerSparse(in_channels, num_clusters)
        # PGExplainer-inspired aggregation layers operate on pooled nodes (num_clusters x F)
        self.pgagg1 = PGExplainerAggLayer(in_channels, hidden_channels)
        self.pgagg2 = PGExplainerAggLayer(hidden_channels, hidden_channels)
        self.linear = nn.Linear(hidden_channels, out_channels)
        # Learnable log-variance weights for adaptive loss balancing (4 terms)
        self.log_vars = nn.Parameter(torch.zeros(4))

    def forward(self, x, adj_sparse: SparseTensor):
        # 1) Pool to clusters
        Z, adj_pooled, mincut_loss, ortho_loss, S = self.mincut(x, adj_sparse)
        # adj_pooled is (1, num_clusters, num_clusters) dense-like tensor
        # create edge_index from it
        adj_mat = adj_pooled.squeeze(0)
        edge_index_pooled = (adj_mat > 0).nonzero(as_tuple=False).t().contiguous()
        # 2) Apply PGExplainer-style aggregation layers on pooled graph
        x_gnn, _masks1 = self.pgagg1(Z, edge_index_pooled)
        x_gnn, _masks2 = self.pgagg2(x_gnn, edge_index_pooled)
        out = self.linear(x_gnn)  # (num_clusters, out_channels)
        return out, mincut_loss, ortho_loss, Z, S

# ---------- Load Graph ----------
csv_file = "/content/drive/MyDrive/ArewardShap/ArewardShap/GO-Bot-DRL/dataset_state_after.csv"
G = nx.Graph()
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    previous_state = None
    for row in reader:
        if row[0].lower() == 'state_after':
            continue
        try:
            state = [int(float(x)) for x in row]
            if state.count(1) == 3 and state.count(0) == len(state) - 3:
                G.add_node(tuple(state))
                if previous_state is not None:
                    # original code removed this edge for some reason; keep same behavior
                    G.remove_edge(previous_state, tuple(state))
            else:
                G.add_node(tuple(state))
                if previous_state is not None:
                    G.add_edge(previous_state, tuple(state))
            previous_state = tuple(state)
        except ValueError:
            print(f"Skipping row with non-numeric value: {row}")

node_list = list(G.nodes())
node_index_map = {node: i for i, node in enumerate(node_list)}
if len(G.edges()) == 0:
    # Avoid constructing an empty edge_index with shape (0,0)
    edge_index = torch.empty((2,0), dtype=torch.long)
else:
    edge_index = torch.tensor([[node_index_map[u], node_index_map[v]] for u, v in G.edges()], dtype=torch.long).t().contiguous()
node_features = torch.tensor(node_list, dtype=torch.float32)
data = Data(x=node_features, edge_index=edge_index)

# ---------- Model Setup ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = data.x.to(device)
# Ensure edge_index is valid for SparseTensor; if empty, create empty SparseTensor
if data.edge_index.numel() == 0:
    adj_sparse = SparseTensor(row=torch.tensor([], dtype=torch.long),
                              col=torch.tensor([], dtype=torch.long),
                              sparse_sizes=(x.shape[0], x.shape[0])).to(device)
else:
    adj_sparse = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(x.shape[0], x.shape[0])).to(device)

num_clusters = 30
model = MinCutPGExplainerGNN_Improved(x.size(1), hidden_channels=64, out_channels=1, num_clusters=num_clusters).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ---------- Centrality Target ----------
centrality = nx.degree_centrality(G)
# Ensure ordering matches node_list; if a node isn't in centrality (isolated), default to 0
target_scalar = torch.tensor([centrality.get(node, 0.0) for node in node_list], dtype=torch.float32).view(-1, 1).to(device)

# ---------- Training Loop ----------
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out, mincut_loss, ortho_loss, Z, S = model(x, adj_sparse)

    # pooled_target: pool node-level centrality into cluster-level target via S^T * target_scalar
    pooled_target = torch.matmul(S.T, target_scalar)  # (num_clusters, 1)
    reward_loss = F.mse_loss(out, pooled_target)
    reg_loss = 0.5 * torch.norm(Z, p='fro') ** 2

    # Adaptive Loss Function with Learnable Weights (as in original)
    loss = (1 / (2 * torch.exp(model.log_vars[0]))) * reward_loss + \
           (1 / (2 * torch.exp(model.log_vars[1]))) * reg_loss + \
           (1 / (2 * torch.exp(model.log_vars[2]))) * mincut_loss + \
           (1 / (2 * torch.exp(model.log_vars[3]))) * ortho_loss + \
           torch.sum(model.log_vars)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Total Loss: {loss.item():.4f}")
        print(f"  ↳ reward: {reward_loss.item():.4f}, reg: {reg_loss.item():.4f}, "
              f"mincut: {mincut_loss.item():.4f}, ortho: {ortho_loss.item():.4f}")

# ---------- Final Embedding & Assignment Matrix ----------
model.eval()
with torch.no_grad():
    embeddings, _, _, _, S = model(x, adj_sparse)  # embeddings are cluster-level outputs (num_clusters, out_channels)
    embeddings = embeddings.detach().cpu().numpy()
    S = S.detach().cpu().numpy()

# ---------- Parameter Counter ----------
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_size_MB = total_params * 4 / (1024**2)
    print(f"\U0001F4CA Total Parameters: {total_params:,}")
    print(f"\U0001F3AF Trainable Parameters: {trainable_params:,}")
    print(f"\U0001F4BE Approximate Model Size: {total_size_MB:.2f} MB")

count_parameters(model)

# ---------- Reward Function ----------
def reward_function(success, max_round, state):
    state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
    # Standardize state length to 224
    if len(state_tensor) > 224:
        partial_tensor = state_tensor[:224]
    elif len(state_tensor) < 224:
        pad = torch.zeros(224 - len(state_tensor), device=device)
        partial_tensor = torch.cat([state_tensor, pad])
    else:
        partial_tensor = state_tensor

    # Use the mincut assignment net on the standardized partial tensor
    assign_logits = model.mincut.assign_net(partial_tensor)   # (num_clusters,)
    assign_probs = torch.softmax(assign_logits, dim=-1)
    cluster_label = torch.argmax(assign_probs).item()

    # embeddings computed above are cluster-level outputs (num_clusters, out_channels)
    # If embeddings not available or cluster_label out of range, fallback to 0
    try:
        rewardd = float(embeddings[cluster_label][0])
    except Exception:
        rewardd = 0.0

    reward = -1 + rewardd
    if success == FAIL:
        reward += -max_round
    elif success == SUCCESS:
        reward += 2 * max_round
    return reward
