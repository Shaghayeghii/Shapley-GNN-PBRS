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

# ---------- Multi-head GAT Layer ----------
class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList([
            CustomGATHead(in_channels, out_channels) for _ in range(heads)
        ])

    def forward(self, h, edge_index):
        head_outs, attns = zip(*[head(h, edge_index) for head in self.heads])
        return F.relu(torch.cat(head_outs, dim=1)), torch.stack(attns).mean(dim=0)

class CustomGATHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomGATHead, self).__init__()
        self.W = nn.Linear(in_channels, out_channels, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * out_channels, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, h, edge_index):
        Wh = self.W(h)
        row, col = edge_index
        a_input = torch.cat([Wh[row], Wh[col]], dim=1)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze())

        alpha = torch.zeros(h.size(0), h.size(0), device=h.device)
        alpha[row, col] = e
        alpha = F.softmax(alpha, dim=1)

        h_prime = torch.matmul(alpha, Wh)
        return h_prime, alpha

# ---------- Model with MinCut + Multi-head GAT ----------
class MinCutGAT_Sparse(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_clusters):
        super(MinCutGAT_Sparse, self).__init__()
        self.mincut = MinCutPoolLayerSparse(in_channels, num_clusters)
        self.gat1 = MultiHeadGATLayer(in_channels, hidden_channels // 4, heads=4)
        self.gat2 = MultiHeadGATLayer(hidden_channels, hidden_channels // 4, heads=4)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, adj_sparse: SparseTensor):
        Z, adj_pooled, mincut_loss, ortho_loss, S = self.mincut(x, adj_sparse)
        edge_index_pooled = (adj_pooled.squeeze(0) > 0).nonzero(as_tuple=False).t().contiguous()
        x_gat, attn = self.gat1(Z, edge_index_pooled)
        x_gat, attn2 = self.gat2(x_gat, edge_index_pooled)
        out = self.linear(x_gat)
        return out, mincut_loss, ortho_loss, (attn + attn2) / 2, Z, S

# ---------- Load Graph ----------
csv_file = "/content/drive/MyDrive/ArewardShap/ArewardShap/ArewardShap/GO-Bot-DRL/dataset_state_after.csv"
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
model = MinCutGAT_Sparse(x.size(1), hidden_channels=64, out_channels=1, num_clusters=num_clusters).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ---------- Centrality Target ----------
centrality = nx.degree_centrality(G)
target_scalar = torch.tensor([centrality[node] for node in node_list], dtype=torch.float32).view(-1, 1).to(device)

# ---------- Training Loop ----------
#lambda_att = 0.01
#mu_reg = 0.1
alpha = 1.0    # reward_loss
beta = 0.1    # reg_loss
gamma = 0.5    # mincut_loss
delta = 0.1    # ortho_loss
epsilon = 0.01 # att_loss

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out, mincut_loss, ortho_loss, attn, Z, S = model(x, adj_sparse)

    pooled_target = torch.matmul(S.T, target_scalar)
    reward_loss = F.mse_loss(out, pooled_target)

    attn = attn + 1e-8
    entropy_loss = -torch.mean(attn * torch.log(attn))

    reg_loss = 0.5 * torch.norm(Z, p='fro') ** 2

    #total_loss = reward_loss + lambda_att * entropy_loss + mu_reg * reg_loss + mincut_loss + ortho_loss
    total_loss = alpha * reward_loss + beta * reg_loss + gamma * mincut_loss + delta * ortho_loss + epsilon * entropy_loss
    
    total_loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss.item():.4f}, Reward: {reward_loss.item():.4f}, "
              f"Entropy: {entropy_loss.item():.4f}, Reg: {reg_loss.item():.4f}, "
              f"MinCut: {mincut_loss.item():.4f}, Ortho: {ortho_loss.item():.4f}")

# ---------- Final Embedding & Assignment Matrix ----------
model.eval()
with torch.no_grad():
    embeddings, _, _, _, _, S = model(x, adj_sparse)
    embeddings = embeddings.detach().cpu().numpy()
    S = S.detach().cpu().numpy()

# ---------- Count Model Parameters ----------
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_size_MB = total_params * 4 / (1024**2)  # assuming float32 (4 bytes per param)
    print(f"📊 Total Parameters: {total_params:,}")
    print(f"🎯 Trainable Parameters: {trainable_params:,}")
    print(f"💾 Approximate Model Size: {total_size_MB:.2f} MB")

count_parameters(model)
# ---------- Reward Function ----------
def reward_function(success, max_round, state):
    state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
    assign_probs = torch.softmax(model.mincut.assign_net(state_tensor), dim=-1)
    cluster_label = torch.argmax(assign_probs).item()
    rewardd = embeddings[cluster_label][0]
    reward = -1 + rewardd
    if success == FAIL:
        reward += -max_round
    elif success == SUCCESS:
        reward += 2 * max_round
    return reward
