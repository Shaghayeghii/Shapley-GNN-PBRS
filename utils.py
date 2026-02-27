print("🧠 I'm utils.py and I'm being imported!")

from dialogue_config import FAIL, SUCCESS
import csv
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_sparse import SparseTensor, matmul
from torch import nn
from tqdm import tqdm

# ---------- Load CSV and Build Graph ----------
csv_file = "/content/drive/MyDrive/ArewardShap/ArewardShap/ArewardShap/GO-Bot-DRL/dataset_state_after.csv"
G = nx.Graph()

with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    previous_state = None
    for row in reader:
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

# ---------- Define MinCut + GraphSAGE + LSTM ----------
class MinCutPoolLayerSparse(nn.Module):
    def __init__(self, in_dim, num_clusters):
        super(MinCutPoolLayerSparse, self).__init__()
        self.num_clusters = num_clusters
        self.assign_net = nn.Linear(in_dim, num_clusters)
        self.proj_net = nn.Linear(in_dim, in_dim)

    def forward(self, x, adj_sparse: SparseTensor):
        S = torch.softmax(self.assign_net(x), dim=-1)  # [N, K]
        X_proj = self.proj_net(x)                      # [N, F]
        Z = torch.matmul(S.T, X_proj)                  # [K, F]

        adj_S = matmul(adj_sparse, S)                  # [N, K]
        adj_new = torch.matmul(S.T, adj_S)             # [K, K]

        deg = adj_sparse.sum(dim=1).to_dense()
        D = deg.unsqueeze(-1) * torch.ones_like(S)
        vol = torch.trace(torch.matmul(S.T, D))
        cut = torch.trace(adj_new)
        mincut_loss = -cut / (vol + 1e-9)

        SS = torch.matmul(S.T, S)
        I = torch.eye(self.num_clusters, device=S.device)
        ortho_loss = torch.norm(SS - I, p='fro')

        return Z, adj_new, mincut_loss, ortho_loss, S

class MinCutGraphSAGE_LSTM_Sparse(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_clusters):
        super(MinCutGraphSAGE_LSTM_Sparse, self).__init__()
        self.mincut = MinCutPoolLayerSparse(in_channels, num_clusters)
        self.sage = SAGEConv(in_channels, hidden_channels)
        self.lstm = nn.LSTM(input_size=hidden_channels, hidden_size=out_channels, batch_first=True)

    def forward(self, x, adj_sparse: SparseTensor):
        Z, adj_pooled, mincut_loss, ortho_loss, S = self.mincut(x, adj_sparse)  # Z: [K, F]
        edge_index_pooled = (adj_pooled > 0).nonzero(as_tuple=False).t().contiguous()
        x_sage = F.relu(self.sage(Z, edge_index_pooled))  # [K, hidden_channels]

        x_seq = x_sage.unsqueeze(0)  # [1, K, hidden_channels]
        x_lstm_out, _ = self.lstm(x_seq)  # [1, K, out_channels]
        x_final = x_lstm_out.squeeze(0)  # [K, out_channels]
        return x_final, mincut_loss, ortho_loss, S

# ---------- Model Setup ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = data.x.to(device)
adj_sparse = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(x.shape[0], x.shape[0])).to(device)

num_clusters = 30
model = MinCutGraphSAGE_LSTM_Sparse(
    in_channels=x.size(1),
    hidden_channels=64,
    out_channels=1,
    num_clusters=num_clusters
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ---------- Centrality Target ----------
centrality = nx.degree_centrality(G)
target_scalar = torch.tensor([centrality[node] for node in node_list], dtype=torch.float32).view(-1, 1).to(device)

# ---------- Training ----------
mu = 10
alpha = 1
beta = 1

model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out, mincut_loss, ortho_loss, S = model(x, adj_sparse)  # out: [K, 1]

    pooled_target = torch.matmul(S.T, target_scalar)  # [K, 1]

    bce_loss = F.binary_cross_entropy_with_logits(out, pooled_target)
    reg_loss = 0.5 * torch.norm(out, p='fro') ** 2

    loss = reg_loss + mu * bce_loss + alpha * mincut_loss + beta * ortho_loss
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, BCE: {bce_loss.item():.4f}, MinCut: {mincut_loss.item():.4f}, Ortho: {ortho_loss.item():.4f}")

# ---------- Final Embedding & Assignment Matrix ----------
model.eval()
with torch.no_grad():
    embeddings, _, _, S = model(x, adj_sparse)
    embeddings = embeddings.detach().cpu().numpy()
    S = S.detach().cpu().numpy()

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

# ---------- Reward Function ----------
def reward_function(success, max_round, state):
    state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
    partial_tensor = state_tensor[:224]
    assign_probs = torch.softmax(model.mincut.assign_net(partial_tensor), dim=-1)
    cluster_label = torch.argmax(assign_probs).item()
    rewardd = embeddings[cluster_label][0]
    reward = -1 + rewardd
    if success == FAIL:
        reward += -max_round
    elif success == SUCCESS:
        reward += 2 * max_round
    return reward
