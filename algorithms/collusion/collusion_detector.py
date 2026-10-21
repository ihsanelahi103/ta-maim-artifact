"""
Collusion-Aware Trust Signal (CATS) — Contribution C5, WWW 2027.

Motivation
----------
TrustPropagationProtocol (TPP) and ReputationManager (RMP) evaluate
trust at the individual / pairwise level: an agent's own reported
mining quality, and pairwise validation outcomes between two agents.
This is sufficient against Static, Camouflage, and Adaptive-Stealth
Sybils, because each of those attackers eventually produces behavior
that differs from honest agents when viewed in isolation.

Coordinated collusion breaks this assumption by design: a colluding
Sybil's individual behavior can be indistinguishable from noise while
the GROUP's behavior is highly structured — dense mutual connectivity
and correlated activity over time. Neither signal is visible to a
purely pairwise/individual trust mechanism. CATS adds two such
higher-order, group-level signals, computed only from information the
system already legitimately observes (the interaction graph and each
agent's own quality history) — no ground-truth Sybil labels are used.

Two signals, combined:

1. Structural suspicion — clustering coefficient + ego-network
   density of an agent's local neighbourhood in the interaction
   graph. A tightly-interconnected clique (dense mutual edges among a
   small group, sparse links to the rest of the network) is the
   classic signature of a coordinated Sybil cluster, consistent with
   community-detection-based Sybil defenses (cf. SybilRank).

2. Behavioral-correlation suspicion — Pearson correlation between an
   agent's own mining-quality time series and its DIRECT graph
   neighbours' time series over a sliding window. Independent honest
   agents mining different, unrelated data shards should show close
   to zero correlation with their neighbours; a coordinated campaign
   sharing a common "intensity" driver will not.

Both signals are restricted to direct graph edges (not all O(N^2)
pairs) — this is both computationally cheap and more principled,
since correlation between agents with no plausible channel to
coordinate is not evidence of collusion.
"""

import numpy as np


class CollusionAwareTrustSignal:
    """
    Group-level collusion detector, complementing TPP/RMP's
    individual and pairwise trust evaluation.

    Usage (per environment step or episode):
        cats.record_quality(agent_id, quality)   # every MineData action
        cats.update(current_graph)               # periodically (e.g. every episode)
        cats.is_suspicious(agent_id)              # queried by RMP.should_quarantine
    """

    def __init__(self, num_agents, window=15,
                 structural_weight=0.5, correlation_weight=0.5,
                 suspicion_threshold=0.55, min_history=4,
                 min_neighbors=5):
        self.num_agents = num_agents
        self.window = window
        self.structural_weight  = structural_weight
        self.correlation_weight = correlation_weight
        self.suspicion_threshold = suspicion_threshold
        self.min_history = min_history
        # Minimum graph degree before clustering coefficient / ego
        # density are treated as meaningful (see gate in update()) --
        # same statistical-reliability principle as min_history above,
        # applied to the structural signal instead of the temporal one.
        self.min_neighbors = min_neighbors

        self.suspicion = np.zeros(num_agents)
        self.structural_score  = np.zeros(num_agents)
        self.correlation_score = np.zeros(num_agents)

        # _history holds ONE aggregated value per agent per EPISODE
        # (see _finalize_episode_buffer). This is essential: an
        # agent's MineData action fires probabilistically, so raw
        # per-call readings accumulate at different rates across
        # agents. Comparing the "last N raw readings" between two
        # agents compares values from different episodes for each —
        # effectively random pairings that destroy any real
        # correlation signal (confirmed empirically: near-zero,
        # noisy correlation for both honest and colluding Sybils
        # before this fix). Aggregating to one value per episode
        # guarantees history[i][-L:] and history[j][-L:] refer to the
        # SAME L most recent episodes for every agent.
        self._history = {i: [] for i in range(num_agents)}
        self._episode_buffer = {i: [] for i in range(num_agents)}
        self._episodes_seen = 0

    # ── Observation ──────────────────────────────────────────────

    def record_quality(self, agent_id, quality):
        """Call every time an agent's mining quality is observed
        within the current episode. Buffered per-episode; see
        _finalize_episode_buffer() for how this becomes _history."""
        self._episode_buffer.setdefault(agent_id, []).append(float(quality))

    # ── Structural signal ────────────────────────────────────────

    def _finalize_episode_buffer(self):
        """Collapse the current episode's raw per-call readings into
        ONE aggregated (mean) value per agent, appended to _history.
        Called once per episode, before this episode's readings are
        available — i.e. it finalizes the PREVIOUS episode's buffer.
        Agents with no readings this episode (never chose MineData)
        are skipped for that episode rather than padded with a
        placeholder, to avoid injecting artificial correlation."""
        for i in range(self.num_agents):
            buf = self._episode_buffer.get(i, [])
            if buf:
                self._history[i].append(float(np.mean(buf)))
                if len(self._history[i]) > self.window:
                    self._history[i].pop(0)
            self._episode_buffer[i] = []

    def _clustering_coefficient(self, agent_id, graph):
        neighbours = list(graph.get(agent_id, {}).keys())
        if len(neighbours) < 2:
            return 0.0
        connected = sum(
            1 for i in neighbours for j in neighbours
            if i != j and j in graph.get(i, {})
        )
        possible = len(neighbours) * (len(neighbours) - 1)
        return connected / possible if possible > 0 else 0.0

    def _ego_density(self, agent_id, graph):
        neighbours = list(graph.get(agent_id, {}).keys())
        if len(neighbours) < 2:
            return 0.0
        possible = len(neighbours) * (len(neighbours) - 1) / 2
        actual = sum(
            1 for idx, n1 in enumerate(neighbours)
            for n2 in neighbours[idx + 1:]
            if n2 in graph.get(n1, {})
        )
        return actual / possible if possible > 0 else 0.0

    # ── Behavioral-correlation signal ────────────────────────────

    def _neighbour_correlation(self, agent_id, graph):
        own_hist = self._history.get(agent_id, [])
        if len(own_hist) < self.min_history:
            return 0.0
        neighbours = list(graph.get(agent_id, {}).keys())
        if not neighbours:
            return 0.0
        corrs = []
        for n in neighbours:
            nh = self._history.get(n, [])
            L = min(len(own_hist), len(nh))
            if L < self.min_history:
                continue
            a = np.asarray(own_hist[-L:])
            b = np.asarray(nh[-L:])
            if np.std(a) < 1e-6 or np.std(b) < 1e-6:
                continue
            c = np.corrcoef(a, b)[0, 1]
            if not np.isnan(c):
                corrs.append(c)
        return float(np.mean(corrs)) if corrs else 0.0

    # ── Combination ───────────────────────────────────────────────

    def update(self, graph):
        """Recompute suspicion scores for all agents from the current
        interaction graph and accumulated quality histories. Call
        once per episode (cheap: O(agents * avg_degree^2))."""
        self._finalize_episode_buffer()
        self._episodes_seen += 1
        if graph is None:
            return
        for i in range(self.num_agents):
            if len(graph.get(i, {})) < self.min_neighbors:
                # Too few neighbours for clustering/density to be
                # statistically meaningful -- a 2-neighbour node whose
                # neighbours happen to know each other trivially hits
                # clustering=1.0 regardless of any actual coordination
                # (confirmed empirically: every false positive during
                # testing showed struct=1.0000 exactly, while genuine
                # 30-member Sybil cliques showed 0.73-0.88 -- the
                # ceiling is a small-sample artifact, not a collusion
                # signal). Same principle as min_history above, applied
                # to the structural rather than temporal signal.
                self.structural_score[i] = 0.0
            else:
                clustering = self._clustering_coefficient(i, graph)
                density    = self._ego_density(i, graph)
                self.structural_score[i] = np.clip(
                    0.5 * clustering + 0.5 * density, 0.0, 1.0
                )

            correlation = self._neighbour_correlation(i, graph)
            # Only positive correlation is a collusion signal; agents
            # moving in opposite directions from neighbours is not.
            self.correlation_score[i] = np.clip(correlation, 0.0, 1.0)

            self.suspicion[i] = np.clip(
                self.structural_weight  * self.structural_score[i] +
                self.correlation_weight * self.correlation_score[i],
                0.0, 1.0
            )

    # ── Query ─────────────────────────────────────────────────────

    def is_suspicious(self, agent_id):
        if agent_id >= self.num_agents:
            return False
        # Require a minimum warm-up so early-episode noise (empty/short
        # histories) can't trigger false positives before signals stabilise.
        if self._episodes_seen < 3:
            return False
        return bool(self.suspicion[agent_id] > self.suspicion_threshold)

    def get_stats(self):
        return {
            "mean_suspicion":       float(np.mean(self.suspicion)),
            "max_suspicion":        float(np.max(self.suspicion)),
            "mean_structural":      float(np.mean(self.structural_score)),
            "mean_correlation":     float(np.mean(self.correlation_score)),
            "episodes_seen":        self._episodes_seen,
        }
