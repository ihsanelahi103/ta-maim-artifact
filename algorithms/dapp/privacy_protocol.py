"""
Distribution-Aware Adaptive Privacy Protocol (DAPP) — Contribution C4.
Extends EdgeChain's AAP to handle both IID and Non-IID distributions.
"""

import numpy as np
from config.config import DAPP as DAPP_CONFIG, AGENT
import os as _os
if _os.environ.get("TAMAIM_EPS_MIN"):
    DAPP_CONFIG["epsilon_min"] = float(_os.environ["TAMAIM_EPS_MIN"])


class DistributionAwarePrivacyProtocol:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        # Initialise per-agent epsilon from Gaussian
        self.epsilon_values = np.clip(
            np.random.normal(
                DAPP_CONFIG["epsilon_init_mean"],
                DAPP_CONFIG["epsilon_init_std"],
                num_agents
            ),
            DAPP_CONFIG["epsilon_min"],
            DAPP_CONFIG["epsilon_max"]
        )
        # Track local distribution skew per agent (KL divergence proxy)
        self.distribution_skew = np.zeros(num_agents)
        self.is_noniid = np.zeros(num_agents, dtype=bool)

    def update(self, agents, global_threat):
        """Called every episode — adapt epsilon per agent."""
        if getattr(self, '_fixed_epsilon', False):
            return  # A4 ablation: epsilon fixed, no adaptation
        regime = self._get_regime(global_threat)
        alpha  = DAPP_CONFIG["alpha"][regime]
        for agent in agents:
            aid = agent.agent_id
            new_eps = self.epsilon_values[aid] * alpha
            # Non-IID extra tightening — lower floor for high-skew agents
            if self.is_noniid[aid]:
                new_eps -= DAPP_CONFIG["noniid_epsilon_penalty"]
                # Non-IID agents get tighter floor (better privacy)
                eps_min = DAPP_CONFIG["epsilon_min"] * 0.67  # 0.300 * 0.67 = 0.200
            else:
                eps_min = DAPP_CONFIG["epsilon_min"]  # Standard floor = 0.300
            self.epsilon_values[aid] = np.clip(
                new_eps, eps_min, DAPP_CONFIG["epsilon_max"]
            )
            agent.update_dp_epsilon(self.epsilon_values[aid])

    def adjust_epsilon(self, agent_id, global_threat):
        """Single-agent epsilon adjustment (called from action executor)."""
        if getattr(self, '_fixed_epsilon', False):
            return float(self.epsilon_values[agent_id])  # Fixed — no change
        regime  = self._get_regime(global_threat)
        alpha   = DAPP_CONFIG["alpha"][regime]
        new_eps = self.epsilon_values[agent_id] * alpha
        if self.is_noniid[agent_id]:
            new_eps -= DAPP_CONFIG["noniid_epsilon_penalty"]
            eps_min = DAPP_CONFIG["epsilon_min"] * 0.67
        else:
            eps_min = DAPP_CONFIG["epsilon_min"]
        self.epsilon_values[agent_id] = np.clip(
            new_eps, eps_min, DAPP_CONFIG["epsilon_max"]
        )
        return float(self.epsilon_values[agent_id])

    def update_distribution_skew(self, agent_id, skew_value):
        """Called by data loader when Non-IID skew is detected."""
        self.distribution_skew[agent_id] = skew_value
        self.is_noniid[agent_id] = (
            skew_value > DAPP_CONFIG["distribution_skew_threshold"]
        )

    def _get_regime(self, global_threat):
        if global_threat >= 0.6:
            return "high"
        elif global_threat >= 0.3:
            return "medium"
        return "low"

    @property
    def average_epsilon(self):
        return float(np.mean(self.epsilon_values))

    def get_stats(self):
        return {
            "mean_epsilon": self.average_epsilon,
            "min_epsilon":  float(np.min(self.epsilon_values)),
            "max_epsilon":  float(np.max(self.epsilon_values)),
            "noniid_agents": int(np.sum(self.is_noniid)),
            "mean_skew":    float(np.mean(self.distribution_skew)),
        }