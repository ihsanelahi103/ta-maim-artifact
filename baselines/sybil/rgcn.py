# ============================================================
# PLACEHOLDER — NOT FINAL
# This baseline will be fully implemented and tuned
# after the main TA-MAIM system is complete.
# Do not use these results for paper comparison yet.
# ============================================================

"""
RGCN-based Sybil Detection Baseline.
Relational Graph Convolutional Network for node classification.
Static GNN — trained once on graph snapshot, cannot adapt dynamically.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config.config import AGENT, TRAINING


class RGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        return self.relu(self.W(adj @ x))


class RGCNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.layer1 = RGCNLayer(in_dim, hidden_dim)
        self.layer2 = RGCNLayer(hidden_dim, num_classes)

    def forward(self, x, adj):
        x = self.layer1(x, adj)
        return self.layer2(x, adj)


class RGCNBaseline:
    """
    RGCN trained to classify nodes as honest or Sybil.
    Key weakness: trained on static graph — degrades when Sybil
    agents change behaviour (dynamic attack). TA-MAIM adapts; RGCN cannot.
    """

    def __init__(self, dataset="yelp"):
        self.dataset    = dataset
        self.num_agents = AGENT["num_agents"]
        self.num_honest = AGENT["num_honest"]
        self.num_sybil  = AGENT["num_sybil"]
        self.device     = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model      = RGCNModel(
            in_dim=16, hidden_dim=32, num_classes=2
        ).to(self.device)
        self.optimizer  = optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion  = nn.CrossEntropyLoss()

    def run(self, epochs=100):
        """Train RGCN and evaluate Sybil detection."""
        print(f"[RGCN] Training on {self.dataset} "
              f"({self.num_agents} agents, {epochs} epochs)")

        # Build graph features and adjacency
        features = self._build_features()
        adj      = self._build_adjacency()
        labels   = torch.LongTensor(
            [0] * self.num_honest + [1] * self.num_sybil
        ).to(self.device)

        x   = torch.FloatTensor(features).to(self.device)
        adj = torch.FloatTensor(adj).to(self.device)

        # Train
        best_f1 = 0.0
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            logits = self.model(x, adj)
            loss   = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 20 == 0:
                f1 = self._evaluate(logits, labels)
                if f1 > best_f1:
                    best_f1 = f1

        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x, adj)
        f1, p, r = self._evaluate_full(logits, labels)
        print(f"[RGCN] Best F1={best_f1:.4f} | Final F1={f1:.4f}")

        return {
            "baseline":   "RGCN",
            "dataset":    self.dataset,
            "sybil_f1":   f1,
            "sybil_prec": p,
            "sybil_rec":  r,
            "ndcg_10":    0.0,
            "dp_epsilon": float("inf"),
            "trust_sep":  0.0,
            "gas_per_tx": 0,
            "note": "Static GNN. Cannot adapt to dynamic Sybil behaviour.",
        }

    def _build_features(self):
        np.random.seed(42)
        features = []
        for i in range(self.num_agents):
            is_sybil = i >= self.num_honest
            if is_sybil:
                feat = np.random.randn(16) * 0.5 - 0.3
            else:
                feat = np.random.randn(16) * 0.5 + 0.3
            features.append(feat)
        return np.array(features)

    def _build_adjacency(self):
        np.random.seed(42)
        adj = np.eye(self.num_agents)
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if np.random.random() < 0.3:
                    adj[i][j] = adj[j][i] = 1.0
        row_sums = adj.sum(axis=1, keepdims=True)
        return adj / np.where(row_sums == 0, 1, row_sums)

    def _evaluate(self, logits, labels):
        preds = logits.argmax(dim=1)
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        p  = tp / max(tp + fp, 1)
        r  = tp / max(tp + fn, 1)
        return 2 * p * r / max(p + r, 1e-8)

    def _evaluate_full(self, logits, labels):
        preds = logits.argmax(dim=1)
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        p  = tp / max(tp + fp, 1)
        r  = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        return f1, p, r
