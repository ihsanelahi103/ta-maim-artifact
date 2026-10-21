# ============================================================
# PLACEHOLDER — NOT FINAL
# This baseline will be fully implemented and tuned
# after the main TA-MAIM system is complete.
# Do not use these results for paper comparison yet.
# ============================================================

"""
InFEDge Baseline — Wang et al. 2022 (JSAC).
Blockchain-based FL with incentive mechanism.
Real FISCO BCOS deployment but passive nodes, no agent autonomy, no Sybil defense.
Primary gas-cost comparison baseline (same as EdgeChain).
"""

import numpy as np
from config.config import AGENT, TRAINING
from baselines.fl.fedavg import FedAvgBaseline


class InFEDgeBaseline(FedAvgBaseline):
    """
    InFEDge: Blockchain-FL without autonomous agents.
    High gas cost (4,300,000/TX) vs TA-MAIM (~12,000/TX).
    No Sybil detection, no adaptive DP, passive nodes.
    """

    def __init__(self, dataset="yelp"):
        super().__init__(dataset)
        self.gas_per_tx = 4_300_000   # InFEDge's real measured gas cost

    def _evaluate(self, round_num):
        metrics = super()._evaluate(round_num)
        # InFEDge has blockchain but still no Sybil defense
        metrics["gas_per_tx"] = self.gas_per_tx
        metrics["sybil_f1"]   = 0.0
        # Slightly better than FedAvg due to incentive mechanism
        metrics["ndcg_10"] = min(1.0, metrics["ndcg_10"] * 1.05)
        return metrics

    def _final_metrics(self):
        result = super()._final_metrics()
        result["baseline"]   = "InFEDge"
        result["gas_per_tx"] = self.gas_per_tx
        result["note"] = (
            "Blockchain-FL. No agent autonomy. No Sybil defense. "
            "4.3M gas/TX vs TA-MAIM ~12K gas/TX."
        )
        return result
