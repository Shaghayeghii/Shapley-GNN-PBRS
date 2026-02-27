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

# ---------- Shapley-based GNN Layer ----------
class ShapleyGNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ShapleyGNNLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        row, col = edge_index
        shapley_matrix = torch.zeros((num_nodes, x.size(1)), device=x.device)

        for i in range(num_nodes):
            neighbors = col[row == i]
            if len(neighbors) == 0:
                shapley_matrix[i] = x[i]
                continue

            neighbor_indices = neighbors.tolist()
            num_neighbors = len(neighbor_indices)
            contrib = torch.zeros_like(x[i])

            if num_neighbors <= 5:
                for subset in itertools.chain.from_iterable(
                        itertools.combinations(neighbor_indices, r)
                        for r in range(num_neighbors + 1)):
                    subset = list(subset)
                    S = torch.mean(x[subset], dim=0) if subset else torch.zeros_like(x[i])
                    with_i = torch.mean(torch.cat([x[subset], x[i].unsqueeze(0)], dim=0), dim=0) if subset else x[i]
                    marginal_contrib = with_i - S
                    weight = (math.factorial(len(subset)) *
                              math.factorial(num_neighbors - len(subset))) / math.factorial(num_neighbors + 1)
                    contrib += weight * marginal_contrib
            else:
                samples = 10
                for _ in range(samples):
                    perm = neighbor_indices.copy()
                    random.shuffle(perm)
                    S_list = []
                    for j in perm:
                        S_val = torch.mean(x[S_list], dim=0) if S_list else torch.zeros_like(x[i])
                        S_list.append(j)
                        new_val = torch.mean(x[S_list], dim=0)
                        contrib += (new_val - S_val) / samples

            shapley_matrix[i] = x[i] + contrib

        out = self.linear(shapley_matrix)
        return F.relu(out), None

# ---------- Full Model with Adaptive Loss Weights ----------
class MinCutShapleyGNN_Improved(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_clusters):
        super(MinCutShapleyGNN_Improved, self).__init__()
        self.mincut = MinCutPoolLayerSparse(in_channels, num_clusters)
        self.shapley1 = ShapleyGNNLayer(in_channels, hidden_channels)
        self.shapley2 = ShapleyGNNLayer(hidden_channels, hidden_channels)
        self.linear = nn.Linear(hidden_channels, out_channels)
        self.log_vars = nn.Parameter(torch.zeros(4))  # Learnable loss weights

    def forward(self, x, adj_sparse: SparseTensor):
        Z, adj_pooled, mincut_loss, ortho_loss, S = self.mincut(x, adj_sparse)
        edge_index_pooled = (adj_pooled.squeeze(0) > 0).nonzero(as_tuple=False).t().contiguous()
        x_gnn, _ = self.shapley1(Z, edge_index_pooled)
        x_gnn, _ = self.shapley2(x_gnn, edge_index_pooled)
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
model = MinCutShapleyGNN_Improved(x.size(1), hidden_channels=64, out_channels=1, num_clusters=num_clusters).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ---------- Centrality Target ----------
centrality = nx.degree_centrality(G)
target_scalar = torch.tensor([centrality[node] for node in node_list], dtype=torch.float32).view(-1, 1).to(device)

# ---------- Training Loop ----------
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out, mincut_loss, ortho_loss, Z, S = model(x, adj_sparse)

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
        print(f"Epoch {epoch}, Total Loss: {loss.item():.4f}")
        print(f"  ↳ reward: {reward_loss.item():.4f}, reg: {reg_loss.item():.4f}, "
              f"mincut: {mincut_loss.item():.4f}, ortho: {ortho_loss.item():.4f}")

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
