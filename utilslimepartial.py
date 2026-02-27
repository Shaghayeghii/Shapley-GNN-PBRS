print("🧠 I'm utils.py and I'm being imported lime!")

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
        S = torch.softmax(self.assign_net(x), dim=-1)
        X_proj = self.proj_net(x)
        Z = torch.matmul(S.T, X_proj)
        adj_S = matmul(adj_sparse, S)
        adj_new = torch.matmul(S.T, adj_S)

        deg = adj_sparse.sum(dim=1).to_dense()
        D = deg.unsqueeze(-1) * torch.ones_like(S)
        vol = torch.trace(torch.matmul(S.T, D))
        cut = torch.trace(adj_new)
        mincut_loss = -cut / (vol + 1e-9)

        SS = torch.matmul(S.T, S)
        I = torch.eye(self.num_clusters, device=S.device)
        ortho_loss = torch.norm(SS - I, p='fro')

        return Z, adj_new.unsqueeze(0), mincut_loss, ortho_loss, S

# ---------- LIME-inspired Local Surrogate GNN Layer ----------
class LIMEGNNLayer(nn.Module):
    """
    LIME-inspired local surrogate aggregator.
    For each node i:
      - collect neighbors embeddings (k x F)
      - sample `num_samples` binary masks over neighbors
      - for each mask, compute aggregated embedding and scalar target (similarity with center)
      - solve linear regression y = M @ beta to get neighbor importance beta (length k)
      - apply softmax(beta) as aggregation weights over neighbors
      - produce blended output = 0.5*center + 0.5*aggregated_neighbors
    Returns: (out_embeddings, None) to match Shapley layer signature.
    """
    def __init__(self, in_channels, out_channels, num_samples=40, perturb_prob=0.5):
        super(LIMEGNNLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.num_samples = num_samples
        self.perturb_prob = perturb_prob

    def forward(self, x, edge_index):
        device = x.device
        num_nodes = x.size(0)
        row, col = edge_index  # edge_index as tuple (row, col)
        out_tensor = torch.zeros_like(x)

        for i in range(num_nodes):
            neighbors = col[row == i]
            if len(neighbors) == 0:
                out_tensor[i] = x[i]
                continue

            neigh = x[neighbors]  # shape: [k, F]
            k = neigh.size(0)
            center = x[i].unsqueeze(0)  # [1, F]

            # Decide number of samples adaptive to neighborhood size (for stability)
            samples = min(max(self.num_samples, k * 4), 200)

            M_list = []
            y_list = []

            # create perturbation samples
            for _ in range(samples):
                # mask: binary vector length k
                bern = torch.bernoulli(self.perturb_prob * torch.ones(k, device=device))
                if bern.sum() == 0:
                    # ensure at least one neighbor selected
                    idx = torch.randint(0, k, (1,), device=device)
                    bern[idx] = 1.0
                M_list.append(bern)

                # aggregated embedding for this mask (mean of selected neighbors)
                mask = bern.bool()
                selected = neigh[mask]
                agg = selected.mean(dim=0)  # [F]

                # scalar target: dot(center, agg) — measures alignment
                y_val = torch.dot(center.squeeze(0), agg)
                y_list.append(y_val)

            M = torch.stack(M_list)  # [samples, k]
            y = torch.stack(y_list)  # [samples]

            # Solve linear regression y = M @ beta  => beta shape [k]
            # Use pseudo-inverse for stability
            # Add tiny ridge for numerical stability
            ridge = 1e-6
            try:
                # pinv on (M^T M + ridge I) is more stable
                MtM = M.t().matmul(M)
                MtM = MtM + ridge * torch.eye(k, device=device)
                Mt_y = M.t().matmul(y)
                beta = torch.linalg.solve(MtM, Mt_y)  # [k]
            except RuntimeError:
                # fallback to pinverse
                beta = torch.matmul(torch.pinverse(M), y)  # [k]

            # Convert beta to positive aggregation weights
            weights = torch.softmax(beta, dim=0)  # [k]

            # Weighted aggregate of neighbor embeddings
            agg_neigh = (weights.unsqueeze(1) * neigh).sum(dim=0)  # [F]

            # Blend center and aggregated neighbor
            blended = 0.5 * center.squeeze(0) + 0.5 * agg_neigh
            out_tensor[i] = blended

        out = self.linear(out_tensor)
        return F.relu(out), None

# ---------- Full Model with Adaptive Loss Weights (LIME) ----------
class MinCutLIMEGNN_Improved(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_clusters):
        super(MinCutLIMEGNN_Improved, self).__init__()
        self.mincut = MinCutPoolLayerSparse(in_channels, num_clusters)
        self.lime1 = LIMEGNNLayer(in_channels, hidden_channels)
        self.lime2 = LIMEGNNLayer(hidden_channels, hidden_channels)
        self.linear = nn.Linear(hidden_channels, out_channels)
        self.log_vars = nn.Parameter(torch.zeros(4))  # Learnable loss weights

    def forward(self, x, adj_sparse: SparseTensor):
        Z, adj_pooled, mincut_loss, ortho_loss, S = self.mincut(x, adj_sparse)
        edge_index_pooled = (adj_pooled.squeeze(0) > 0).nonzero(as_tuple=False).t().contiguous()
        x_gnn, _ = self.lime1(Z, edge_index_pooled)
        x_gnn, _ = self.lime2(x_gnn, edge_index_pooled)
        out = self.linear(x_gnn)
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
edge_index = torch.tensor([[node_index_map[u], node_index_map[v]] for u, v in G.edges()], dtype=torch.long).t().contiguous()
node_features = torch.tensor(node_list, dtype=torch.float32)
data = Data(x=node_features, edge_index=edge_index)

# ---------- Model Setup ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = data.x.to(device)
adj_sparse = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(x.shape[0], x.shape[0])).to(device)
num_clusters = 30
model = MinCutLIMEGNN_Improved(x.size(1), hidden_channels=64, out_channels=1, num_clusters=num_clusters).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ---------- Centrality Target ----------
centrality = nx.degree_centrality(G)
target_scalar = torch.tensor([centrality[node] for node in node_list], dtype=torch.float32).view(-1, 1).to(device)

# ---------- Training Loop ----------
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out, mincut_loss, ortho_loss, Z, S = model(x, adj_sparse)

    # pooled_target: cluster-wise pooling of centrality targets
    pooled_target = torch.matmul(S.T, target_scalar)
    reward_loss = F.mse_loss(out, pooled_target)
    reg_loss = 0.5 * torch.norm(Z, p='fro') ** 2

    # Adaptive Loss Function with Learnable Weights
    loss = (1 / (2 * torch.exp(model.log_vars[0]))) * reward_loss + \
           (1 / (2 * torch.exp(model.log_vars[1]))) * reg_loss + \
           (1 / (2 * torch.exp(model.log_vars[2]))) * mincut_loss + \
           (1 / (2 * torch.exp(model.log_vars[3]))) * ortho_loss + \
           torch.sum(model.log_vars)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Total Loss: {loss.item():.6f}")
        print(f"  ↳ reward: {reward_loss.item():.6f}, reg: {reg_loss.item():.6f}, "
              f"mincut: {mincut_loss.item():.6f}, ortho: {ortho_loss.item():.6f}")

# ---------- Final Embedding & Assignment Matrix ----------
model.eval()
with torch.no_grad():
    embeddings, _, _, _, S = model(x, adj_sparse)
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

    assign_probs = torch.softmax(model.mincut.assign_net(partial_tensor), dim=-1)
    cluster_label = torch.argmax(assign_probs).item()
    rewardd = embeddings[cluster_label][0]
    reward = -1 + rewardd
    if success == FAIL:
        reward += -max_round
    elif success == SUCCESS:
        reward += 2 * max_round
    return reward
