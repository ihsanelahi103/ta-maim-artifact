# ============================================================
# PLACEHOLDER — NOT FINAL
# This baseline will be fully implemented and tuned
# after the main TA-MAIM system is complete.
# Do not use these results for paper comparison yet.
# ============================================================

"""
SybilRank Baseline — Cao et al. 2012.
Graph-based Sybil detection using random walk trust propagation.
Static method — cannot adapt to dynamic Sybil behaviour.
Canonical WWW Sybil detection baseline.
"""

import numpy as np
from config.config import AGENT


class SybilRankBaseline:
    """
    SybilRank: propagates trust from seed honest nodes via random walk.
    Static — runs on a fixed graph snapshot, no online adaptation.
    No privacy, no data mining quality, no blockchain accountability.
    """

    def __init__(self, dataset="yelp"):
        self.dataset    = dataset
        self.num_agents = AGENT["num_agents"]
        self.num_honest = AGENT["num_honest"]
        self.num_sybil  = AGENT["num_sybil"]
        self.num_walks  = 10   # Random walk iterations

    def run(self):
        """Run SybilRank and return detection metrics."""
        print(f"[SybilRank] Running on {self.dataset} "
              f"({self.num_agents} agents)")

        # Build synthetic adjacency matrix
        adj = self._build_adjacency()

        # Seed: assume first 3 honest agents are known good seeds
        trust = np.zeros(self.num_agents)
        trust[:3] = 1.0   # Seed nodes

        # Random walk propagation
        for _ in range(self.num_walks):
            new_trust = adj @ trust
            # Normalise
            total = new_trust.sum()
            if total > 0:
                new_trust /= total
            trust = new_trust

        # Rank by trust — low trust = Sybil prediction
        threshold = np.percentile(trust, 30)   # Bottom 30% = Sybil
        predicted_sybil = set(
            i for i in range(self.num_agents) if trust[i] < threshold
        )
        ground_truth = set(range(self.num_honest, self.num_agents))

        tp = len(predicted_sybil & ground_truth)
        fp = len(predicted_sybil - ground_truth)
        fn = len(ground_truth - predicted_sybil)
        p  = tp / max(tp + fp, 1)
        r  = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-8)

        print(f"[SybilRank] F1={f1:.4f} | Prec={p:.4f} | Rec={r:.4f}")

        return {
            "baseline":   "SybilRank",
            "dataset":    self.dataset,
            "sybil_f1":   f1,
            "sybil_prec": p,
            "sybil_rec":  r,
            "ndcg_10":    0.0,   # SybilRank does no data mining
            "dp_epsilon": float("inf"),  # No privacy
            "trust_sep":  float(np.mean(trust[:self.num_honest]) -
                                np.mean(trust[self.num_honest:])),
            "gas_per_tx": 0,
            "note": "Static graph Sybil detection. No online adaptation.",
        }

    def _build_adjacency(self):
        """Build synthetic adjacency matrix."""
        np.random.seed(42)
        adj = np.zeros((self.num_agents, self.num_agents))
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                is_sybil_edge = (
                    i >= self.num_honest or j >= self.num_honest
                )
                # Honest-honest edges are denser and stronger
                if not is_sybil_edge:
                    if np.random.random() < 0.4:
                        w = np.random.uniform(0.5, 1.0)
                        adj[i][j] = adj[j][i] = w
                else:
                    if np.random.random() < 0.2:
                        w = np.random.uniform(0.1, 0.4)
                        adj[i][j] = adj[j][i] = w

        # Row normalise
        row_sums = adj.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return adj / row_sums
