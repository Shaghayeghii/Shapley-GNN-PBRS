# 🧠 RewardShaping-GNN

This repository provides implementations of reward shaping approch using **Shapley values based reward attribution and Graph Neural Networks (GNNs)** in dialogue systems. Supported **GAT** and explainability methods: **PGExplainer**, **GNNExplainer**, **LIME** — with and without **CSR (Contextual state representation)**.
Base models include: **DQN** and **D3QN**.

---

## 🔧 Run on Google Colab (Notebook-Style Instructions)

Follow the steps below just like in a Colab notebook:

---

### 🛠️ Step 1: Clone the Repository

```python
# Clone the repo and move into the directory
!git clone [https://github.com/shaghayeghsi/RewardShaping-GNN.git](https://github.com/Shaghayeghii/Shapley-GNN-PBRS.git)
%cd Shapley-GNN-PBRS.git
```

---

### 📦 Step 2: Unzip the Dataset

```python
# Unzip the dataset
!unzip dataset_state_after.csv.zip
```

---
### 🧰 Step 3: Install Dependencies

##### Install additional required libraries


```python
!pip install gdown
```

```python
# Download the pre-trained Word2Vec model (GoogleNews vectors)
!gdown --id '1sG0osAy9VV26HzQBoBkRWS4vT9X60VaB' --output /content/drive/MyDrive/ArewardShap/ArewardShap/GO-Bot-DRL/GoogleNews-vectors-negative300.bin.gz
```

#### Remove any incompatible or broken PyG packages
```python
!pip uninstall -y torch torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv
```

```python
# Install compatible versions of PyG packages for torch==2.6.0 + CPU
!pip install torch==2.6.0 torchvision torchaudio
!pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
!pip install torch-geometric -U
```

```python
!pip install gensim
```

```python
!pip install spektral
!pip install networkx
!pip install numpy
!pip install pandas
!pip install scikit-learn
```

```python
!pip show torch
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
# GraphSAGE-based shaping
!cp utilsshapgat.py utils.py
```

```python
# GAT-based shaping
!cp utilssparsestaticgat.py utils.py
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
# Shapley values + CSR
!cp utilsshapgatlearnablepartial.py utils.py
```

```python
# GAT + CSR
!cp utilssparsestaticpartialgat.py utils.py
```

```python
# lime + CSR
!cp utilslimepartial.py utils.py
```

```python
# GNNxplainer + CSR
!cp utilsPGExplainerAgg.py utils.py
```

```python
# PGExplainer + CSR
!cp utilsPGExplainerAgg.py utils.py
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
