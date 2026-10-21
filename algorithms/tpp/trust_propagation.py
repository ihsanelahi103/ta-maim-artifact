import os as _os
_SYB_PRIOR = 1.0 if _os.environ.get("TAMAIM_NO_SYBIL_PRIOR") else 0.9
"""
Trust Propagation Protocol (TPP) — Contribution C1/C3.

Computes, propagates, and updates peer trust scores on-chain.
NOT a simple report counter like EdgeChain's bi —
a learned, graph-propagated trust metric.
"""

import numpy as np
from config.config import TPP as TPP_CONFIG, AGENT


class TrustPropagationProtocol:
    """
    Maintains and propagates trust scores across the agent network.

    Key differences from EdgeChain's Byzantine score bi:
    1. Trust is continuous and bidirectional [0,1] not a report count
    2. Trust propagates through the graph (transitive trust)
    3. Trust can RECOVER — not just binary quarantine
    4. Sybil suspicion is separate from trust (graph-structural signal)
    """

    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.num_honest = AGENT["num_honest"]

        # Core trust scores — all agents start at neutral
        self.trust_scores = np.full(
            num_agents,
            TPP_CONFIG["initial_trust"]
        )

        # Sybil suspicion scores (graph-structural, separate from trust)
        self.sybil_suspicion = np.zeros(num_agents)

        # Interaction history for trust computation
        self.validation_records = []        # (validator, target, correct)
        self.sybil_flags = []               # (reporter, target)

        # Global threat level Θ
        self.global_threat_level = 0.0

        # Trust adjacency matrix (weighted edges)
        # All agents start at neutral trust — scores diverge through interactions
        self.trust_matrix = np.full(
            (num_agents, num_agents),
            TPP_CONFIG["initial_trust"]
        )
        np.fill_diagonal(self.trust_matrix, 1.0)  # Self-trust = 1

        # Sybil agents have slightly lower initial outgoing trust variance
        # (models the reality that new/unknown nodes start with less credibility)
        num_honest = AGENT["num_honest"]
        for i in range(num_honest, num_agents):
            for j in range(num_agents):
                if j != i:
                    # Small downward bias for unknown Sybil agents
                    self.trust_matrix[i][j] *= _SYB_PRIOR
                    self.trust_matrix[j][i] *= _SYB_PRIOR

    def record_validation(self, validator_id, target_id, correct):
        """Record a validation event and update trust accordingly."""
        self.validation_records.append((validator_id, target_id, correct))

        if correct:
            # Reward target's trust (their data was validated correctly)
            self._update_trust(
                target_id,
                TPP_CONFIG["positive_reinforcement"]
            )
            # Also reward validator's trustworthiness
            self._update_trust(
                validator_id,
                TPP_CONFIG["positive_reinforcement"] * 0.5
            )
        else:
            # Penalise — either validator is wrong or target submitted bad data
            self._update_trust(
                target_id,
                -TPP_CONFIG["negative_reinforcement"]
            )

        # Update pairwise trust
        delta = (TPP_CONFIG["positive_reinforcement"] if correct
                 else -TPP_CONFIG["negative_reinforcement"])
        self.trust_matrix[validator_id][target_id] = np.clip(
            self.trust_matrix[validator_id][target_id] + delta,
            TPP_CONFIG["min_trust"],
            TPP_CONFIG["max_trust"]
        )

    def record_sybil_flag(self, reporter_id, target_id):
        """Record a Sybil flag — increases target's suspicion score."""
        self.sybil_flags.append((reporter_id, target_id))

        # Weight the flag by reporter's own trust score
        reporter_trust = self.trust_scores[reporter_id]
        suspicion_increase = 0.1 * reporter_trust

        self.sybil_suspicion[target_id] = np.clip(
            self.sybil_suspicion[target_id] + suspicion_increase,
            0.0, 1.0
        )

        # High suspicion → trust penalty
        if self.sybil_suspicion[target_id] > 0.5:
            self._update_trust(
                target_id,
                -TPP_CONFIG["sybil_penalty"]
            )

    def propagate(self, depth=None):
        """
        Propagate trust through the agent graph (transitive trust).
        Trust from A to C via B = trust(A,B) * trust(B,C) * damping.

        Fix: weight incoming trust by the TRUSTOR's own trust score —
        a low-trust agent's endorsement carries little weight.
        This creates natural separation: Sybil agents endorse each other
        but their endorsements are down-weighted by their own low trust.
        """
        if depth is None:
            depth = TPP_CONFIG["propagation_depth"]

        damping = TPP_CONFIG["propagation_damping"]

        # Weight trust_matrix rows by trustor's current trust score
        # so low-trust agents' opinions matter less
        trustor_weights = self.trust_scores.reshape(-1, 1)  # column vector
        weighted_matrix = self.trust_matrix * trustor_weights
        weighted_matrix = np.clip(weighted_matrix, 0.0, 1.0)

        propagated = weighted_matrix.copy()
        for _ in range(depth):
            propagated = propagated + damping * (
                propagated @ weighted_matrix
            )
            propagated = np.clip(propagated, 0.0, 1.0)

        # Update trust scores as WEIGHTED mean of incoming trust
        # (exclude self-trust from the average)
        for i in range(self.num_agents):
            incoming_weights = []
            incoming_values  = []
            for j in range(self.num_agents):
                if j != i:
                    incoming_weights.append(self.trust_scores[j])
                    incoming_values.append(propagated[j][i])

            total_w = sum(incoming_weights)
            if total_w > 0:
                weighted_avg = sum(
                    w * v for w, v in zip(incoming_weights, incoming_values)
                ) / total_w
                # Blend: 70% propagated, 30% current (smoothing)
                self.trust_scores[i] = np.clip(
                    0.7 * weighted_avg + 0.3 * self.trust_scores[i],
                    TPP_CONFIG["min_trust"],
                    TPP_CONFIG["max_trust"]
                )

    def update(self, agents, blockchain=None):
        """
        Called every step. Updates global threat level Θ.
        """
        # Apply trust decay
        self.trust_scores *= TPP_CONFIG["trust_decay"]

        # Propagate trust through graph
        self.propagate()

        # Push updated trust scores to agents
        for agent in agents:
            agent.update_trust_score(self.trust_scores[agent.agent_id])

        # Compute global threat level Θ
        self._compute_global_threat()

    def _compute_global_threat(self):
        """
        Compute global threat metric Θ ∈ [0,1].
        Extension of EdgeChain's formula — adds Sybil suspicion dimension.
        Θ = min(1.0, 0.35*low_trust_fraction + 0.35*max_suspicion
                    + 0.20*sybil_flag_rate + 0.10*trust_variance)
        """
        # Fraction of agents with low trust (threat proxy)
        low_trust_count = np.sum(self.trust_scores < 0.4)
        low_trust_fraction = low_trust_count / self.num_agents

        # Max suspicion score across all agents
        max_suspicion = float(np.max(self.sybil_suspicion))

        # Recent Sybil flag rate (last 20 flags)
        recent_flags = len(self.sybil_flags[-20:])
        sybil_flag_rate = min(1.0, recent_flags / max(20.0, 1.0))

        # Trust score variance (high variance = contested trust = threat)
        trust_variance = float(np.var(self.trust_scores))
        trust_variance_norm = min(1.0, trust_variance * 4.0)  # scale to [0,1]

        self.global_threat_level = min(1.0,
            0.35 * low_trust_fraction  +
            0.35 * max_suspicion       +
            0.20 * sybil_flag_rate     +
            0.10 * trust_variance_norm
        )

    def update_from_mining_quality(self, agent_id, quality, authentic=None):
        """
        Directly update trust based on observed mining quality.
        This is the primary driver of trust separation between
        honest (high quality) and Sybil (low quality) agents.
        Called by environment after each MineData action.
        """
        # TPP observes ONLY quality signal — authentic parameter intentionally ignored
        # This is by design: TPP cannot know ground truth Sybil labels
        # authentic=None is acceptable — quality is the sole detection signal
        # This makes camouflage effective: high-quality fake looks honest
        _ = authentic  # Explicitly unused — no label leakage
        if quality > 0.5:
            # Good quality → trust increase (honest OR camouflaged Sybil)
            delta = TPP_CONFIG["positive_reinforcement"] * quality
            self._update_trust(agent_id, delta)
            self.trust_matrix[:, agent_id] = np.clip(
                self.trust_matrix[:, agent_id] + delta * 0.5, 0.0, 1.0
            )
        elif quality < 0.3:
            # Low quality → trust decrease (obvious Sybil OR honest with bad ep)
            delta = TPP_CONFIG["negative_reinforcement"] * (1.0 - quality)
            self._update_trust(agent_id, -delta)
            self.trust_matrix[:, agent_id] = np.clip(
                self.trust_matrix[:, agent_id] - delta * 0.5, 0.0, 1.0
            )
        # 0.3 ≤ quality ≤ 0.5: ambiguous — no trust update (transition zone)

    def step_update(self, flag_count):
        """Light update called every environment step."""
        if flag_count > 0:
            self.global_threat_level = min(
                1.0,
                self.global_threat_level + 0.02 * flag_count
            )

    def curate_local_graph(self, agent_id):
        """
        Remove low-trust edges from agent's local view.
        Returns count of edges removed.
        """
        threshold = TPP_CONFIG["quarantine_threshold"]
        removed = 0
        for j in range(self.num_agents):
            if (j != agent_id and
                    0.0 < self.trust_matrix[agent_id][j] < threshold):
                self.trust_matrix[agent_id][j] = 0.0
                removed += 1
        return removed

    def _update_trust(self, agent_id, delta):
        """Update a single agent's trust score."""
        self.trust_scores[agent_id] = np.clip(
            self.trust_scores[agent_id] + delta,
            TPP_CONFIG["min_trust"],
            TPP_CONFIG["max_trust"]
        )

    def get_sybil_predictions(self, threshold=0.5):
        """
        Predict which agents are Sybil based on trust + suspicion.
        Returns list of predicted Sybil agent IDs.
        """
        predictions = []
        for i in range(self.num_agents):
            combined_score = (
                0.6 * (1.0 - self.trust_scores[i]) +
                0.4 * self.sybil_suspicion[i]
            )
            if combined_score > threshold:
                predictions.append(i)
        return predictions

    def get_stats(self):
        """Return current TPP statistics for logging."""
        return {
            "global_threat": self.global_threat_level,
            "mean_trust_honest": float(np.mean(
                self.trust_scores[:self.num_honest])),
            "mean_trust_sybil": float(np.mean(
                self.trust_scores[self.num_honest:])),
            "mean_suspicion": float(np.mean(self.sybil_suspicion)),
            "max_suspicion": float(np.max(self.sybil_suspicion)),
            "total_flags": len(self.sybil_flags),
            "total_validations": len(self.validation_records),
        }