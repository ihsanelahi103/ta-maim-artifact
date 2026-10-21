"""
Evaluator — computes all metrics for TA-MAIM.
Metrics: NDCG@K, HR@K, Sybil-F1, Trust accuracy, DP ε, Gas cost.
"""

import numpy as np
from config.config import EVALUATION, AGENT


class Evaluator:
    def __init__(self, agents, env, logger):
        self.agents  = agents
        self.env     = env
        self.logger  = logger
        self.num_honest = AGENT["num_honest"]
        self.num_sybil  = AGENT["num_sybil"]

    def evaluate(self, episode):
        """Run full evaluation and return metrics dict."""
        metrics = {}
        metrics.update(self._compute_sybil_detection_metrics())
        metrics.update(self._compute_mining_quality_metrics())
        metrics.update(self._compute_privacy_metrics())
        metrics.update(self._compute_trust_metrics())
        metrics["episode"] = episode
        return metrics

    def _compute_sybil_detection_metrics(self):
        """Precision, Recall, F1 for Sybil detection."""
        ground_truth = set(range(self.num_honest, self.num_honest + self.num_sybil))
        predicted    = set(
            a.agent_id for a in self.agents if a.is_quarantined
        )

        tp = len(ground_truth & predicted)
        fp = len(predicted - ground_truth)
        fn = len(ground_truth - predicted)

        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = (2 * precision * recall / max(precision + recall, 1e-8))

        return {
            "sybil_precision": precision,
            "sybil_recall":    recall,
            "sybil_f1":        f1,
            "sybil_tp": tp, "sybil_fp": fp, "sybil_fn": fn,
        }

    def _compute_mining_quality_metrics(self):
        """NDCG@K and HR@K proxies based on mining quality scores."""
        honest_agents = [a for a in self.agents if not a.is_sybil]
        qualities     = [a.avg_mining_quality for a in honest_agents]

        # NDCG@10 proxy: proportion of high-quality contributions in top-10
        sorted_q = sorted(qualities, reverse=True)
        ndcg_10  = np.mean(sorted_q[:10]) if len(sorted_q) >= 10 \
            else np.mean(sorted_q)
        ndcg_5   = np.mean(sorted_q[:5])  if len(sorted_q) >= 5 \
            else np.mean(sorted_q)
        hr_10    = sum(1 for q in sorted_q[:10] if q > 0.7) / \
                   min(len(sorted_q), 10)

        return {
            "ndcg_5": ndcg_5, "ndcg_10": ndcg_10,
            "hr_10":  hr_10,
            "avg_mining_quality": np.mean(qualities),
        }

    def _compute_privacy_metrics(self):
        """DP epsilon statistics."""
        eps_values = [a.dp_epsilon for a in self.agents if not a.is_sybil]
        return {
            "avg_epsilon": np.mean(eps_values),
            "min_epsilon": np.min(eps_values),
            "max_epsilon": np.max(eps_values),
            "agents_at_floor": sum(1 for e in eps_values
                                   if e <= 0.31),
        }

    def _compute_trust_metrics(self):
        """Trust score accuracy and separation."""
        tpp = self.env.tpp
        honest_trust = np.mean(tpp.trust_scores[:self.num_honest])
        sybil_trust  = np.mean(tpp.trust_scores[self.num_honest:])
        separation   = honest_trust - sybil_trust

        return {
            "trust_honest_mean": float(honest_trust),
            "trust_sybil_mean":  float(sybil_trust),
            "trust_separation":  float(separation),
            "global_threat":     float(tpp.global_threat_level),
        }
