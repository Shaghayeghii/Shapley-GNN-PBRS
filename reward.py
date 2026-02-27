from dialogue_config import FAIL, SUCCESS
import pandas as pd
import node2vec
import csv
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from node2vec import Node2Vec
import joblib


# Load your CSV file with states (modify the file path accordingly)
csv_file = "/content/drive/MyDrive/dataset/GO-Bot-DRL/dataset_state_after.csv"


# Initialize a graph to represent the sequence of states
G = nx.Graph()

with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    previous_state = None  # To keep track of the previous state

    for row in reader:
        try:
            # Convert floating-point values to integers
            state = [int(float(x)) for x in row]

            # Check if it's the start of a new episode
            if state.count(1) == 3 and state.count(0) == len(state) - 3:
                # If it's the start of an episode, add the state to the graph
                G.add_node(tuple(state))

                # Ensure there are no new edges to the previous state at the start of the episode
                if previous_state is not None:
                    G.remove_edge(previous_state, tuple(state))

            else:
                # If it's not the start of an episode, simply add the state to the graph
                G.add_node(tuple(state))

                # Add an edge to the previous state if it's not the start of the episode
                if previous_state is not None:
                    G.add_edge(previous_state, tuple(state))

            # Update the previous state for the next iteration
            previous_state = tuple(state)
        except ValueError:
            print(f"Skipping row with non-numeric value: {row}")

# Perform K-Means clustering on the nodes
num_clusters = 40  # Adjust as needed
kmeans = KMeans(n_clusters=num_clusters)
cluster_labels = kmeans.fit_predict(list(G.nodes()))

# Create a new directed smaller graph where each cluster is one node
smaller_graph = nx.DiGraph()
for i in range(num_clusters):
    smaller_graph.add_node(i)

# Iterate through the edges in the original graph
for edge in G.edges():
    source_node, target_node = edge

    # Get the cluster labels of the source and target nodes
    source_label = cluster_labels[list(G.nodes).index(source_node)]
    target_label = cluster_labels[list(G.nodes).index(target_node)]

    # Add a directed edge from source cluster to target cluster
    if source_label != target_label:
        smaller_graph.add_edge(source_label, target_label)

# Initialize Node2Vec with reduced parameters
node2vec = Node2Vec(smaller_graph, dimensions=1, walk_length=20, num_walks=30, p=1, q=0.5, workers=1)

# Precompute the walks
model = node2vec.fit(window=10, min_count=1)

# Get Node2Vec embeddings
embeddings = {str(node): model.wv[str(node)] for node in smaller_graph.nodes()}


# Save KMeans model
kmeans_filename = "kmeans_model.joblib"
joblib.dump(kmeans, kmeans_filename)

# Save Node2Vec model
node2vec_filename = "node2vec_model"
model.save(node2vec_filename)