from environment.sybil_attack_model import SybilAttackFactory, AdaptiveStealthSybilAttack
from algorithms.collusion.collusion_detector import CollusionAwareTrustSignal
"""
Edge Mining Environment — the RL environment for TA-MAIM.

Each episode = one round of collaborative data mining.
Each step = agents sense, decide, act, and receive rewards.
"""

import numpy as np
from config.config import AGENT, STATE_SPACE, BT_MO_DQN, TRAINING, DAPP as DAPP_CONFIG


class EdgeMiningEnvironment:
    """
    Simulates the edge data mining environment where autonomous AI
    agents collaboratively mine web graph data under Sybil attacks.

    State:  20-dimensional vector per agent (see config.STATE_SPACE)
    Action: 5 discrete actions per agent
    Reward: Multi-objective — mining quality + trust security
                            + privacy + efficiency
    """

    ACTIONS = {
        0: "MineData",
        1: "ValidateNode",
        2: "FlagSybil",
        3: "CurateGraph",
        4: "AdjustPrivacy",
    }

    def __init__(self, dataset, tpp, brmp, dapp, blockchain):
        self.dataset_name = dataset
        self.tpp = tpp
        self.brmp = brmp
        self.dapp = dapp
        self.blockchain = blockchain

        self.num_agents = AGENT["num_agents"]
        self.num_honest = AGENT["num_honest"]
        self.num_sybil = AGENT["num_sybil"]
        self.state_dim = AGENT["state_dim"]
        self.action_dim = AGENT["action_dim"]

        # Load dataset
        self.data_loader = self._init_data_loader(dataset)
        self._apply_topology()

        # Episode tracking
        self.current_episode = 0
        self.current_step = 0
        self.max_steps = TRAINING["max_steps_per_episode"]

        # Environment state
        self.agent_states = np.zeros((self.num_agents, self.state_dim))
        self.graph = None               # Current interaction graph snapshot
        self.mined_knowledge = {}

        # DP release window: raw per-agent MineData quality observations
        # are buffered and released as a noised MEAN over a 4-episode
        # rolling window, not individually -- this gives sensitivity
        # 1/k for a bounded-mean query (k = window observation count)
        # rather than 1.0 for a single raw release, which is what makes
        # the Laplace mechanism usable at the configured epsilon floors
        # without destroying the honest-vs-Sybil quality signal. See
        # Appendix (DP calibration) for the derivation: k>=50 needed at
        # the tightest floor (Non-IID, eps=0.201) to keep noise std
        # comfortably below the ~0.70 honest-vs-static-Sybil quality gap.
        self.DP_WINDOW_EPISODES = 4
        self._quality_buffer = {i: [] for i in range(self.num_agents)}
        self._last_released_quality = {i: 0.5 for i in range(self.num_agents)}

        # DP release window: raw per-agent MineData quality observations
        # are buffered and released as a noised MEAN over a 4-episode
        # rolling window, not individually -- this gives sensitivity
        # 1/k for a bounded-mean query (k = window observation count)
        # rather than 1.0 for a single raw release, which is what makes
        # the Laplace mechanism usable at the configured epsilon floors
        # without destroying the honest-vs-Sybil quality signal. See
        # Appendix (DP calibration) for the derivation: k>=50 needed at
        # the tightest floor (Non-IID, eps=0.201) to keep noise std
        # comfortably below the ~0.70 honest-vs-static-Sybil quality gap.
        self.DP_WINDOW_EPISODES = 4
        self._quality_buffer = {i: [] for i in range(self.num_agents)}
        self._last_released_quality = {i: 0.5 for i in range(self.num_agents)}
        # Sybil attack model
        if not hasattr(self, 'attack_model'):
            scenario = getattr(self, 'attack_scenario', 'static')
            self.attack_model = SybilAttackFactory.create(
                scenario, self.num_sybil, self.num_honest
            )
        self.current_episode = getattr(self, 'current_episode', 1)       # Shared knowledge base
        self.tx_log = []                # Blockchain transaction log
        self._agent_tx_count = {i: 0 for i in range(self.num_agents)}  # O(1) lookup for s[2]
        self.agents_ref = {}            # Reference to agent objects for status checks

        # Metrics tracking
        self.episode_mining_quality = []
        self.episode_sybil_detections = []
        self.episode_gas_costs = []

        # Collusion-Aware Trust Signal (C5) — structural + behavioral
        # correlation detector, complementing individual/pairwise TPP/RMP.
        # Ablation A7 disables consumption of this via
        # ReputationManager.use_collusion_pathway, not by skipping
        # computation here (so the detector's own signal quality can
        # still be inspected/logged even when the pathway is off).
        self.collusion_detector = CollusionAwareTrustSignal(
            num_agents=self.num_agents
        )

    def _apply_topology(self):
        import utils.seed as _s
        from data.topology import build_topology, params
        g, eps, gt = build_topology(self.num_honest, self.num_sybil,
                                    _s.CURRENT_SEED, **params())
        dl = self.data_loader
        dl.graph, dl.episode_graphs, dl._ground_truth_sybil = g, eps, gt

    def _init_data_loader(self, dataset):
        """Load appropriate dataset handler."""
        if dataset == "amazon":
            from data.amazon.amazon_loader import AmazonDataLoader
            return AmazonDataLoader()
        else:   # default: yelp or both
            from data.yelp.yelp_loader import YelpDataLoader
            return YelpDataLoader()

    def reset(self):
        """Reset environment for new episode. Returns initial states."""
        self.current_step = 0
        self.current_episode += 1
        self.mined_knowledge = {}
        self.tx_log = []
        self._agent_tx_count = {i: 0 for i in range(self.num_agents)}  # O(1) lookup for s[2], reset each episode
        self.episode_mining_quality = []
        self.episode_sybil_detections = []
        self.episode_gas_costs = []

        # Load fresh graph snapshot from dataset
        self.graph = self.data_loader.get_episode_graph(self.current_episode)
        # agents_ref populated by main loop via register_agents()

        # Recompute collusion-aware suspicion from the current graph +
        # quality histories accumulated over prior episodes, then sync
        # into RMP so should_quarantine() sees this episode's scores.
        self.collusion_detector.update(self.graph)
        if hasattr(self.brmp, 'sync_collusion_suspicion'):
            self.brmp.sync_collusion_suspicion(self.collusion_detector)

        # Build initial states for all agents
        self.agent_states = self._build_initial_states()
        return {i: self.agent_states[i] for i in range(self.num_agents)}

    def register_agents(self, agents):
        """Register agent references so env can check quarantine status."""
        self.agents_ref = {a.agent_id: a for a in agents}
        # Give Sybil agents a reference to the active attack model so
        # colluding Sybils can bias their own action distribution
        # toward mutual endorsement (see AutonomousAgent._sybil_action).
        if hasattr(self, 'attack_model'):
            for a in agents:
                if a.is_sybil:
                    a.attack_model = self.attack_model

    def step(self, actions):
        """
        Execute one environment step.

        Args:
            actions: dict {agent_id: action_int}

        Returns:
            next_states: dict {agent_id: state_array}
            rewards:     dict {agent_id: float}
            done:        bool
            info:        dict (metrics for logging)
        """
        self.current_step += 1
        rewards = {}
        info = {"actions": actions, "step": self.current_step}

        # Execute each agent's action
        for agent_id, action in actions.items():
            is_sybil = agent_id >= self.num_honest
            action_name = self.ACTIONS[action]

            if action_name == "MineData":
                reward, tx_info = self._execute_mine_data(agent_id, is_sybil)
            elif action_name == "ValidateNode":
                reward, tx_info = self._execute_validate_node(
                    agent_id, is_sybil)
            elif action_name == "FlagSybil":
                reward, tx_info = self._execute_flag_sybil(agent_id, is_sybil)
            elif action_name == "CurateGraph":
                reward, tx_info = self._execute_curate_graph(
                    agent_id, is_sybil)
            elif action_name == "AdjustPrivacy":
                reward, tx_info = self._execute_adjust_privacy(agent_id)
            else:
                reward, tx_info = 0.0, {}

            rewards[agent_id] = reward
            if tx_info:
                self.tx_log.append(tx_info)
                self._agent_tx_count[agent_id] = \
                    self._agent_tx_count.get(agent_id, 0) + 1
                gas = tx_info.get("gas", 0)
                self.episode_gas_costs.append(gas)

        # Update global state
        self._update_global_state(actions)

        # Build next states
        next_states_array = self._build_states()
        self.agent_states = next_states_array
        next_states = {i: next_states_array[i] for i in range(self.num_agents)}

        # Collect episode metrics
        avg_quality = np.mean([
            self.mined_knowledge.get(aid, {}).get("quality", 0)
            for aid in range(self.num_honest)
        ])
        self.episode_mining_quality.append(avg_quality)

        info.update({
            "avg_mining_quality": avg_quality,
            "total_gas": sum(self.episode_gas_costs),
            "tx_count": len(self.tx_log),
            "global_threat": self.tpp.global_threat_level,
        })

        done = self.current_step >= self.max_steps
        return next_states, rewards, done, info

    # ──────────────────────────────────────────
    # Action executors
    # ──────────────────────────────────────────

    def _execute_mine_data(self, agent_id, is_sybil):
        """Agent contributes data to the shared knowledge base."""
        # Honest agents mine real quality data; Sybil agents inject fake data
        if is_sybil:
            # Sybil quality is generated by the active attack model
            # (static / camouflage / collusion / adaptive). Collusion
            # no longer applies an artificial boost to this value —
            # see sybil_attack_model.ColludingSybilAttack docstring.
            sybil_idx = max(0, agent_id - self.num_honest)
            ep = getattr(self, 'current_episode', 1)
            if hasattr(self, 'attack_model'):
                quality = self.attack_model.get_quality(sybil_idx, ep)
                _ev = __import__("os").environ.get("TAMAIM_EVADE")
                if _ev:
                    _elo, _ehi = [float(x) for x in _ev.split(",")]
                    quality = float(np.random.uniform(_elo, _ehi))
            else:
                quality = np.random.uniform(0.0, 0.25)
            data_authentic = False
        else:
            _wf = float(__import__("os").environ.get("TAMAIM_WEAK_FRAC", "0"))
            if _wf > 0:
                if not hasattr(self, "_weak_ids"):
                    import utils.seed as _us
                    _rr = np.random.RandomState(int(_us.CURRENT_SEED) + 9091)
                    self._weak_ids = set(int(x) for x in _rr.choice(
                        self.num_honest, int(round(_wf * self.num_honest)), replace=False))
                if agent_id in self._weak_ids:
                    _lo, _hi = [float(x) for x in __import__("os").environ.get(
                        "TAMAIM_WEAK_RANGE", "0.35,0.80").split(",")]
                    quality = np.random.uniform(_lo, _hi)
                else:
                    quality = np.random.uniform(0.65, 1.0)
            else:
                quality = np.random.uniform(0.65, 1.0)   # High quality real data
            data_authentic = True

        # Buffer raw quality for this agent's rolling DP release window
        # (see DP_WINDOW_EPISODES note in __init__). Nothing is released
        # from this single observation directly -- only the windowed
        # mean gets privatized, giving sensitivity 1/k instead of 1.0.
        self._quality_buffer[agent_id].append(quality)

        window_steps = self.DP_WINDOW_EPISODES * self.max_steps
        if len(self._quality_buffer[agent_id]) > window_steps:
            self._quality_buffer[agent_id] = \
                self._quality_buffer[agent_id][-window_steps:]

        k = len(self._quality_buffer[agent_id])
        windowed_mean = float(np.mean(self._quality_buffer[agent_id]))

        # Sensitivity of a bounded mean over k observations in [0,1]
        # is 1/k (changing any single observation moves the mean by
        # at most 1/k) -- NOT the configured DAPP_CONFIG["sensitivity"]
        # (=1.0), which was calibrated for a single raw release and is
        # too large once we aggregate. Releasing more often than every
        # k-th step reuses the same windowed release (agents observe a
        # stable, slowly-updating privatized signal between releases,
        # not fresh noise every step).
        mean_sensitivity = 1.0 / max(k, 1)
        current_epsilon = self.dapp.epsilon_values[agent_id]
        laplace_scale = mean_sensitivity / max(current_epsilon, 1e-8)
        dp_noise = np.random.laplace(0, laplace_scale)
        noised_quality = np.clip(windowed_mean + dp_noise, 0.0, 1.0)
        self._last_released_quality[agent_id] = noised_quality

        # Commit to blockchain
        tx = {"tx_id": self.blockchain.record_mining_tx(agent_id, 0.5, "action")}

        self.mined_knowledge[agent_id] = {
            "quality": noised_quality,
            "authentic": data_authentic,
            "step": self.current_step,
        }

        # Feed observed quality into the collusion-aware detector's
        # rolling history for every agent (honest and Sybil). Uses the
        # PRE-DP-noise quality, not noised_quality — the injected
        # Laplace privacy noise (std ~0.14) is larger than the
        # collusion attack's shared-latent correlation signal
        # (std ~0.03-0.04) and was masking it entirely when the
        # detector was fed noised_quality (confirmed empirically:
        # correlation score stayed ~0.00-0.07 for Sybils and honest
        # agents alike). Modelling assumption: CATS observes agents'
        # underlying behavioural signal directly (as an internal
        # trust-system component), with DP noise applied only to what
        # is externally reported/committed — analogous to how TPP and
        # RMP already observe quality before any DP transformation.
        self.collusion_detector.record_quality(agent_id, quality)

        # Update TPP trust directly from mining quality — primary trust signal
        # Pass only quality — authentic label NOT passed to TPP
        # This ensures zero label leakage in trust propagation
        # TPP receives the windowed, properly-privatized release --
        # this is cross-agent trust evaluation, so it must consume
        # q_tilde (privatized), never raw quality. CATS also stays on
        # raw pre-noise quality (see its own call above and the
        # trusted-local-aggregation-boundary note in the Supplement) --
        # the two differ deliberately: TPP/RMP's flag/quarantine
        # decisions are the disclosure-sensitive cross-agent output,
        # while CATS is modelled as operating within a trusted local
        # collusion-signal aggregation service, not an open broadcast.
        self.tpp.update_from_mining_quality(
            agent_id=agent_id,
            quality=noised_quality,
        )

        # Reward: honest agents rewarded for quality; Sybil agents penalised
        # Reward is a purely local/internal training signal (never
        # transmitted to another agent), so it uses clean local
        # quality -- consistent with the local-vs-cross-agent privacy
        # boundary: only signals that cross the agent boundary are
        # privatized, and reward never does.
        base_reward = quality if not is_sybil else -quality

        # Privacy cost: high epsilon = privacy leakage risk = reward penalty
        # This makes DAPP's adaptive epsilon contribution visible in reward
        # Privacy utility U = 1/(1+epsilon): U_max=0.769 (ε=0.3), U_min=0.5 (ε=1.0)
        # Privacy cost = (1 - U) * privacy_weight
        if not is_sybil:
            current_eps = float(self.dapp.epsilon_values[agent_id])
            privacy_utility = 1.0 / (1.0 + current_eps)
            privacy_cost = (1.0 - privacy_utility) * 0.3  # 0.3 weight
            reward = base_reward - privacy_cost
        else:
            reward = base_reward

        return reward, tx

    def _execute_validate_node(self, agent_id, is_sybil):
        """Agent validates a peer's contribution."""
        target_id = None

        # Colluding Sybils preferentially validate OTHER Sybils —
        # mutual false endorsement through the existing, legitimate
        # record_validation() path (see ColludingSybilAttack). This is
        # the real collusion mechanism: falsely vouching for a
        # cluster-mate's fake data is what should pull colluding
        # Sybils' trust back up against the quality-based penalty.
        if is_sybil and hasattr(self, 'attack_model') and \
                getattr(self.attack_model, 'mutual_endorsement', False):
            sybil_idx = max(0, agent_id - self.num_honest)
            preferred = self.attack_model.get_preferred_validation_target(
                sybil_idx, self.num_agents
            )
            if preferred is not None and preferred != agent_id:
                target_id = preferred

        if target_id is None:
            # Default: select a random peer to validate
            peers = [i for i in range(self.num_agents) if i != agent_id]
            if not peers:
                return 0.0, {}
            target_id = np.random.choice(peers)

        target_data = self.mined_knowledge.get(target_id, {})

        if not target_data:
            return 0.0, {}

        # Honest agent detects fake data correctly with high probability
        if is_sybil:
            # Sybil agents validate falsely — endorse fake, reject real
            correct_validation = not target_data.get("authentic", True)
        else:
            # Honest agents validate based on observed quality score
            # High quality data → likely authentic; low quality → likely fake
            observed_quality = target_data.get("quality", 0.5)
            actual_authentic = target_data.get("authentic", True)
            # Threshold: quality > 0.45 → agent says authentic
            predicted_authentic = observed_quality > 0.45
            correct_validation = (predicted_authentic == actual_authentic)

        tx = {"tx_id": self.blockchain.record_mining_tx(agent_id, 0.5, "action")}

        # Update trust based on validation
        self.tpp.record_validation(
            validator_id=agent_id,
            target_id=target_id,
            correct=correct_validation,
        )

        reward = 0.3 if correct_validation else -0.1
        return reward, tx

    def _execute_flag_sybil(self, agent_id, is_sybil):
        """Agent flags a peer as suspected Sybil.
        Uses combined trust + suspicion + mining quality score.
        """
        if is_sybil:
            candidates = [i for i in range(self.num_honest)
                          if not self.agents_ref[i].is_quarantined
                          and i != agent_id]
            if not candidates:
                return 0.0, {}
            target_id = int(np.random.choice(candidates))
        else:
            trust     = self.tpp.trust_scores
            suspicion = self.tpp.sybil_suspicion
            active    = [i for i in range(self.num_agents)
                         if i != agent_id
                         and not self.agents_ref[i].is_quarantined]
            if not active:
                return 0.0, {}
            def sybil_score(i):
                quality = self.mined_knowledge.get(i, {}).get("quality", 0.5)
                return (0.4*(1.0-trust[i]) +
                        0.4*suspicion[i] +
                        0.2*(1.0-quality))
            # Only flag if there is a meaningful Sybil candidate
            # (score must exceed threshold to avoid false flagging)
            scored = [(i, sybil_score(i)) for i in active]
            best_id, best_score = max(scored, key=lambda x: x[1])
            if best_score < 0.35:
                # No strong Sybil candidate — skip flagging
                return 0.0, {}
            target_id = best_id

        tx = {"tx_id": self.blockchain.record_mining_tx(agent_id, 0.5, "action")}

        # Update Sybil suspicion scores in TPP
        self.tpp.record_sybil_flag(
            reporter_id=agent_id,
            target_id=target_id,
        )

        # Record flag in BRMP — weight by trust, require unique reporters
        reporter_trust = self.tpp.trust_scores[agent_id]
        self.brmp.record_sybil_flag(target_id=target_id,
                                    reporter_trust=float(reporter_trust),
                                    reporter_id=agent_id)

        # Reward: honest agent flagging real Sybil = positive reward
        actual_is_sybil = target_id >= self.num_honest
        if not is_sybil and actual_is_sybil:
            reward = 0.5    # Correct detection
        elif not is_sybil and not actual_is_sybil:
            reward = -0.3   # False positive
        else:
            reward = -0.5   # Sybil framing honest node
        return reward, tx

    def _execute_curate_graph(self, agent_id, is_sybil):
        """Agent curates local graph — removes low-trust edges."""
        if is_sybil:
            return 0.0, {}

        # Remove edges to nodes with trust below threshold
        removed = self.tpp.curate_local_graph(agent_id)

        tx = {"tx_id": self.blockchain.record_mining_tx(agent_id, 0.5, "action")}

        reward = 0.1 * removed   # Small reward per low-trust edge removed
        return reward, tx

    def _execute_adjust_privacy(self, agent_id):
        """Agent adjusts its local DP epsilon."""
        # DAPP handles the actual adjustment
        new_epsilon = self.dapp.adjust_epsilon(
            agent_id=agent_id,
            global_threat=self.tpp.global_threat_level,
        )

        tx = {"tx_id": self.blockchain.record_mining_tx(agent_id, 0.5, "action")}

        # Reward: tighter privacy under high threat = good
        threat = self.tpp.global_threat_level
        if threat >= 0.6 and new_epsilon <= 0.5:
            reward = 0.2
        elif threat < 0.3 and new_epsilon <= 1.0:
            reward = 0.1
        else:
            reward = 0.0
        return reward, tx

    # ──────────────────────────────────────────
    # State builders
    # ──────────────────────────────────────────

    def _build_initial_states(self):
        """Build initial state vectors for all agents."""
        states = np.zeros((self.num_agents, self.state_dim))
        for i in range(self.num_agents):
            states[i] = self._build_agent_state(i)
        return states

    def _build_states(self):
        """Build current state vectors for all agents."""
        states = np.zeros((self.num_agents, self.state_dim))
        for i in range(self.num_agents):
            states[i] = self._build_agent_state(i)
        return states

    def _build_agent_state(self, agent_id):
        """
        Build the 20-dimensional GRAPH-AWARE state vector.

        This is the key C2 contribution — Graph-Aware BT-MO-DQN:
        The state incorporates REAL graph-structural features derived
        from the Yelp/Amazon interaction graph loaded by the data loader.
        EdgeChain uses 15-dim local state; we extend to 20-dim with
        5 graph-structural features (s[11]-s[15]).

        State breakdown:
          s[0-5]:  Local agent metrics (mining quality, contribution)
          s[6-10]: Trust metrics (own trust, suspicion, threat Θ)
          s[11-15]: Graph-structural features (GRAPH-AWARE extension)
          s[16-19]: System metrics (DP ε, gas, episode progress)
        """
        s = np.zeros(self.state_dim)
        trust = self.tpp.trust_scores
        rep   = self.brmp.reputation_scores
        eps   = self.dapp.epsilon_values

        # ── Local agent metrics (0-5) ─────────────────────────────
        mined    = self.mined_knowledge.get(agent_id, {})
        # s[0]: Real mining quality from dataset loader
        s[0] = float(mined.get("quality", 0.5))
        # s[1]: Quality trend (current - average of last 3)
        if hasattr(self, '_quality_history') and                 agent_id in self._quality_history:
            hist = self._quality_history[agent_id]
            s[1] = np.clip(s[0] - np.mean(hist[-3:]), -1, 1) * 0.5 + 0.5
        else:
            s[1] = 0.5
        # s[2]: Contribution rate (TX count / max possible)
        # O(1) lookup via running per-agent counter (was: O(tx_log
        # length) scan per agent per step -- confirmed via profiler as
        # a second, independent instance of the same anti-pattern that
        # caused the earlier LocalLedger.get_stats() slowdown: both
        # recomputed a full scan over an ever-growing, episode-long
        # collection on every single call instead of tracking
        # incrementally).
        s[2] = min(1.0, self._agent_tx_count.get(agent_id, 0)
                   / max(self.current_step * 0.5, 1))
        # s[3]: Sybil flags received (normalised)
        s[3] = min(1.0,
            float(self.brmp.sybil_flag_counts[agent_id]) / 10.0
        )
        # s[4]: Reputation score (from BRMP)
        s[4] = float(rep[agent_id])
        # s[5]: Validation accuracy (from mined knowledge)
        authentic = mined.get("authentic", True)
        s[5] = 1.0 if authentic else 0.0

        # ── Trust metrics (6-10) ──────────────────────────────────
        s[6]  = float(trust[agent_id])              # Own trust score
        neighbours = self._get_neighbours(agent_id)
        # s[7]: Trust variance among neighbours (uncertainty signal)
        s[7]  = float(np.var([trust[n] for n in neighbours]))                 if neighbours else 0.0
        s[7]  = min(1.0, s[7] * 4.0)               # Scale to [0,1]
        # s[8]: Own Sybil suspicion score
        s[8]  = float(self.tpp.sybil_suspicion[agent_id])
        # s[9]: Global threat level Θ
        s[9]  = float(self.tpp.global_threat_level)
        # s[10]: Reputation trend (positive = improving)
        s[10] = float(np.clip(rep[agent_id], 0.0, 1.0))

        # ── Graph-structural features (11-15) — GRAPH-AWARE C2 ───
        # These are REAL graph features from Yelp/Amazon loader
        # Key novelty vs EdgeChain: graph topology in agent state

        # s[11]: Degree centrality — fraction of agents connected to
        n_neighbours = len(neighbours)
        s[11] = n_neighbours / max(self.num_agents - 1, 1)

        # s[12]: Local clustering coefficient — triangle density
        # High clustering = tightly-knit community (trust signal)
        s[12] = self._clustering_coefficient(agent_id)

        # s[13]: Ego-network density — fraction of possible edges among neighbours
        # Unique graph topology: dense ego = tightly-knit honest community
        # This is NOT captured in trust metrics s[6-10]
        if len(neighbours) >= 2 and self.graph:
            possible_edges = len(neighbours) * (len(neighbours) - 1) / 2
            actual_edges   = sum(
                1 for i, n1 in enumerate(neighbours)
                for n2 in neighbours[i+1:]
                if n2 in self.graph.get(n1, {})
            )
            s[13] = actual_edges / max(possible_edges, 1)
        else:
            s[13] = 0.0

        # s[14]: Second-hop reach — how many unique 2-hop neighbours
        # Measures agent's indirect influence in the graph
        # Distinct from direct trust metrics — captures network position
        if self.graph:
            two_hop = set()
            for n in neighbours:
                two_hop.update(self.graph.get(n, {}).keys())
            two_hop.discard(agent_id)
            two_hop -= set(neighbours)
            s[14] = min(1.0, len(two_hop) / max(self.num_agents, 1))
        else:
            s[14] = 0.0

        # s[15]: Edge weight distribution skew
        # From real Yelp/Amazon Jaccard similarity weights
        # High skew = uneven connections (potential Sybil bridge)
        if neighbours and self.graph:
            weights = [self.graph[agent_id].get(n, 0)
                       for n in neighbours]
            s[15] = float(np.std(weights)) if len(weights) > 1 else 0.5
        else:
            s[15] = 0.5

        # ── System metrics (16-19) ────────────────────────────────
        # s[16]: Agent's current DP epsilon (normalised)
        s[16] = float(eps[agent_id]) / 3.0

        # s[17]: Blockchain gas cost proxy (normalised)
        bc_stats = self.blockchain.get_stats()
        avg_gas  = bc_stats.get("avg_gas_per_tx", 12000)
        s[17] = min(1.0, avg_gas / 20000.0)

        # s[18]: Total TX normalised (activity level)
        total_tx = bc_stats.get("total_tx", 0)
        s[18] = min(1.0, total_tx / (self.num_agents * 50.0))

        # s[19]: Episode progress
        s[19] = self.current_step / self.max_steps

        # Update quality history
        if not hasattr(self, '_quality_history'):
            self._quality_history = {}
        if agent_id not in self._quality_history:
            self._quality_history[agent_id] = []
        self._quality_history[agent_id].append(s[0])
        if len(self._quality_history[agent_id]) > 10:
            self._quality_history[agent_id].pop(0)

        return np.clip(s, 0.0, 1.0)

    def _get_neighbours(self, agent_id):
        """Return list of agent_id's neighbours in current graph."""
        if self.graph is None:
            return []
        return list(self.graph.get(agent_id, {}).keys())

    def _clustering_coefficient(self, agent_id):
        """Approximate local clustering coefficient."""
        neighbours = self._get_neighbours(agent_id)
        if len(neighbours) < 2:
            return 0.0
        connected = sum(
            1 for i in neighbours for j in neighbours
            if i != j and j in self.graph.get(i, {})
        )
        possible = len(neighbours) * (len(neighbours) - 1)
        return connected / possible if possible > 0 else 0.0

    def _update_global_state(self, actions):
        """Update global environment state after all actions."""
        # Count FlagSybil actions this step
        flag_count = sum(
            1 for a in actions.values()
            if a == list(self.ACTIONS.keys())[
                list(self.ACTIONS.values()).index("FlagSybil")
            ]
        )
        self.tpp.step_update(flag_count=flag_count)

    def get_episode_summary(self):
        """Return summary metrics for the completed episode."""
        return {
            "avg_mining_quality": np.mean(self.episode_mining_quality)
                                  if self.episode_mining_quality else 0,
            "total_gas": sum(self.episode_gas_costs),
            "total_tx": len(self.tx_log),
            "sybil_detections": len(self.episode_sybil_detections),
        }