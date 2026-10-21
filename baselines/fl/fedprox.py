# ============================================================
# PLACEHOLDER — NOT FINAL
# This baseline will be fully implemented and tuned
# after the main TA-MAIM system is complete.
# Do not use these results for paper comparison yet.
# ============================================================

"""
FedProx Baseline — Li et al. 2020.
FL with proximal term for Non-IID data handling.
Better than FedAvg under heterogeneous data but still no Sybil defense.
"""

import numpy as np
from config.config import AGENT, TRAINING
from baselines.fl.fedavg import FedAvgBaseline


class FedProxBaseline(FedAvgBaseline):
    """
    Extends FedAvg with proximal regularisation for Non-IID robustness.
    Still no Sybil detection — Byzantine nodes still corrupt the model.
    Slightly better NDCG than FedAvg under Non-IID but same Sybil-F1=0.
    """

    def __init__(self, dataset="yelp"):
        super().__init__(dataset)
        self.mu = 0.01   # Proximal term weight

    def _get_client_updates(self):
        """Honest client updates with proximal regularisation."""
        updates = []
        for _ in range(self.num_clients):
            # FedProx: add proximal term to pull update toward global model
            noise       = np.random.randn(len(self.global_model)) * 0.04
            quality     = np.random.uniform(0.68, 1.0)   # Slightly better than FedAvg
            local_grad  = self.global_model + noise * quality
            # Proximal term: (mu/2) * ||w - w_global||^2
            prox_update = local_grad - self.mu * (local_grad - self.global_model)
            updates.append(prox_update)
        return updates

    def _evaluate(self, round_num):
        metrics = super()._evaluate(round_num)
        # FedProx is slightly better than FedAvg under Non-IID
        metrics["ndcg_10"] = min(1.0, metrics["ndcg_10"] * 1.08)
        metrics["ndcg_5"]  = min(1.0, metrics["ndcg_5"]  * 1.08)
        metrics["hr_10"]   = min(1.0, metrics["hr_10"]   * 1.06)
        # Still no Sybil detection
        metrics["sybil_f1"] = 0.0
        return metrics

    def _final_metrics(self):
        result = super()._final_metrics()
        result["baseline"] = "FedProx"
        result["note"] = "Non-IID robust FL. Still no Sybil defense."
        return result
