# ============================================================
# PLACEHOLDER — NOT FINAL
# This baseline will be fully implemented and tuned
# after the main TA-MAIM system is complete.
# Do not use these results for paper comparison yet.
# ============================================================

"""
FedAvg Baseline — McMahan et al. 2017.
Standard federated learning with central aggregation.
No Byzantine defense, no agent autonomy, static DP.
Used to prove autonomous agents outperform passive FL.
"""

import numpy as np
from config.config import AGENT, TRAINING


class FedAvgBaseline:
    """
    Simulates FedAvg on the same data mining task.
    Passive nodes upload local model updates to central aggregator.
    No Sybil detection — Byzantine nodes corrupt the global model.
    """

    def __init__(self, dataset="yelp"):
        self.dataset      = dataset
        self.num_clients  = AGENT["num_honest"]   # Only honest clients in FL
        self.num_sybil    = AGENT["num_sybil"]    # Sybil clients attack FL
        self.num_rounds   = TRAINING["num_episodes"]
        self.model_dim    = 128                    # Simulated model size
        self.lr           = 0.01
        self.local_epochs = 5
        self.fixed_dp_eps = 1.0                   # Static DP — FL weakness

        # Global model (simulated as weight vector)
        self.global_model = np.random.randn(self.model_dim) * 0.01
        self.results      = []

    def run(self):
        """Run FedAvg for all rounds and return metrics."""
        print(f"[FedAvg] Running {self.num_rounds} rounds on {self.dataset}")
        print(f"[FedAvg] {self.num_clients} honest + {self.num_sybil} Sybil clients")

        for round_num in range(1, self.num_rounds + 1):
            # 1. Server broadcasts global model
            # 2. Clients compute local updates
            client_updates = self._get_client_updates()

            # 3. Sybil clients inject poisoned updates
            sybil_updates  = self._get_sybil_updates()

            # 4. Simple average — no defense
            all_updates = client_updates + sybil_updates
            self.global_model = np.mean(all_updates, axis=0)

            # 5. Evaluate
            if round_num % 10 == 0:
                metrics = self._evaluate(round_num)
                self.results.append(metrics)
                print(f"  Round {round_num:3d} | "
                      f"NDCG@10: {metrics['ndcg_10']:.4f} | "
                      f"Sybil-F1: {metrics['sybil_f1']:.4f} | "
                      f"DP ε: {self.fixed_dp_eps:.3f}")

        return self._final_metrics()

    def _get_client_updates(self):
        """Simulate honest client local training updates."""
        updates = []
        for _ in range(self.num_clients):
            # Honest update = gradient step toward good model
            noise   = np.random.randn(self.model_dim) * 0.05
            quality = np.random.uniform(0.65, 1.0)
            update  = self.global_model + noise * quality
            updates.append(update)
        return updates

    def _get_sybil_updates(self):
        """Simulate Sybil client poisoning attacks."""
        updates = []
        for _ in range(self.num_sybil):
            # Byzantine update — opposite direction to corrupt model
            poison_scale = np.random.uniform(1.5, 3.0)
            update = -self.global_model * poison_scale + \
                     np.random.randn(self.model_dim) * 0.5
            updates.append(update)
        return updates

    def _evaluate(self, round_num):
        """
        Evaluate FedAvg performance.
        FedAvg has no Sybil detection — F1 always 0.
        Mining quality degrades under Byzantine attack.
        """
        # Model quality degrades under Sybil poisoning
        sybil_ratio = self.num_sybil / (self.num_clients + self.num_sybil)
        model_quality = max(0.0, 1.0 - sybil_ratio * 1.5 +
                           np.random.uniform(-0.05, 0.05))

        # NDCG proxy: proportional to model quality
        ndcg_10 = model_quality * 0.75 + np.random.uniform(-0.03, 0.03)
        ndcg_10 = np.clip(ndcg_10, 0.0, 1.0)
        hr_10   = ndcg_10 * 0.85

        return {
            "round":      round_num,
            "ndcg_5":     ndcg_10 * 0.95,
            "ndcg_10":    ndcg_10,
            "hr_10":      hr_10,
            "sybil_f1":   0.0,     # FedAvg has NO Sybil detection
            "sybil_prec": 0.0,
            "sybil_rec":  0.0,
            "dp_epsilon": self.fixed_dp_eps,   # Static — never adapts
            "trust_sep":  0.0,     # No trust mechanism
            "gas_per_tx": 0,       # No blockchain
        }

    def _final_metrics(self):
        if not self.results:
            return {}
        last = self.results[-1]
        return {
            "baseline":   "FedAvg",
            "dataset":    self.dataset,
            "ndcg_10":    last["ndcg_10"],
            "hr_10":      last["hr_10"],
            "sybil_f1":   0.0,
            "dp_epsilon": self.fixed_dp_eps,
            "trust_sep":  0.0,
            "note": "No Sybil defense. Static DP. No blockchain accountability.",
        }
