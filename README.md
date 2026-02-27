# 🧠 RewardShaping-GNN

This repository provides implementations of various reward shaping strategies using **Graph Neural Networks (GNNs)** in dialogue systems. Supported GNNs: **GAT**, **GraphSAGE**, **Node2Vec** — with and without **CSR (Compressed Sparse Row)**.
Base models include: **DQN** and **D3QN**.

---

## 🔧 Run on Google Colab (Notebook-Style Instructions)

Follow the steps below just like in a Colab notebook:

---

### 🛠️ Step 1: Clone the Repository

```python
# Clone the repo and move into the directory
!git clone https://github.com/shaghayeghsi/RewardShaping-GNN.git
%cd RewardShaping-GNN.git
```

---

### 📦 Step 2: Unzip the Dataset

```python
# Unzip the dataset
!unzip dataset_state_after.csv.zip
```

---
### 🧰 Step 3: Install Dependencies

```python
# Download the pre-trained Word2Vec model (GoogleNews vectors)
!gdown --id '1sG0osAy9VV26HzQBoBkRWS4vT9X60VaB' --output /content/drive/MyDrive/ArewardShap/ArewardShap/GO-Bot-DRL/GoogleNews-vectors-negative300.bin.gz

# Remove any incompatible or broken PyG packages
!pip uninstall -y torch-scatter torch-sparse torch-cluster torch-spline-conv

# Install compatible versions of PyG packages for torch==2.6.0 + CPU
!pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
!pip install torch-sparse -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
!pip install torch-cluster -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
!pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
!pip install torch-geometric -U

# Install additional required libraries
!pip install gdown
!pip install gensim
!pip install node2vec
!pip install spektral
!pip install networkx
!pip install numpy
!pip install pandas
!pip install scikit-learn
```

---

### 🚀 Step 4: Train the Model

```python
# Train the model using default settings
!python train.py
```

---

## 🎯 Reward Shaping (without CSR)

Choose **one** of the following and run:

```python
# GAT-based shaping
!cp utilssparsestaticgat.py utils.py
```

```python
# GraphSAGE-based shaping
!cp utilsfinalsparseminsage.py utils.py
```

```python
# Node2Vec-based shaping
!cp utilsfinalN2Vsparse.py utils.py
```

Then:

```python
# Common tracker for non-CSR models
!cp state_trackerorg.py state_tracker.py
```

---

## 🧹 Reward Shaping (with CSR)

Choose **one** of the following:

```python
# GAT + CSR
!cp utilssparsestaticpartialgat.py utils.py
```

```python
# GraphSAGE + CSR
!cp utilspartialsage.py utils.py
```

```python
# Node2Vec + CSR
!cp utilsN2Vsparsepartial.py utils.py
```

Then:

```python
# Common tracker for CSR-based models
!cp state_trackerw2vfinal.py state_tracker.py
```

---

## 🤖 Run Base Models

### D3QN:

```python
# Use D3QN agent
!cp d3qn.py dqn_agent.py
```

Edit `constants.json`:

```json
"vanilla": false
```

---

### DQN:

```python
# Use DQN agent
!cp dqn_agentdqn.py dqn_agent.py
```

Keep `constants.json` as:

```json
"vanilla": true
```

---

> ⚠️ **Make sure all file replacements are done correctly before training any model.**
