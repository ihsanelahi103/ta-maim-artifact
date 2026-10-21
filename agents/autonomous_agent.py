"""
Autonomous AI Agent — the core decision-making unit of TA-MAIM.
Each edge node is one AutonomousAgent instance.
Implements the sense → decide → act → learn loop.
"""

import numpy as np
import torch
from config.config import AGENT, BT_MO_DQN, DAPP


class AutonomousAgent:
    """
    A single autonomous AI agent operating at an edge node.

    Responsibilities:
    - Perceive a 20-dimensional state vector (local + trust + graph + system)
    - Select actions via graph-aware BT-MO-DQN policy
    - Store experiences in replay buffer
    - Learn from experience (honest agents only)
    - Maintain local DP epsilon via DAPP
    """

    def __init__(self, agent_id, is_sybil, state_dim, action_dim,
                 tpp, brmp, dapp):
        self.agent_id = agent_id
        self.is_sybil = is_sybil
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Protocol references
        self.tpp = tpp
        self.brmp = brmp
        self.dapp = dapp

        # Agent status
        self.is_quarantined = False
        self.is_registered = False

        # Local state tracking
        self.current_trust_score = 0.5          # Starts at neutral
        self.current_reputation = 0.5
        self.dp_epsilon = np.random.normal(
            DAPP["epsilon_init_mean"],
            DAPP["epsilon_init_std"]
        )
        self.dp_epsilon = np.clip(
            self.dp_epsilon,
            DAPP["epsilon_min"],
            DAPP["epsilon_max"]
        )

        # Performance tracking
        self.mining_quality_history = []
        self.validation_accuracy_history = []
        self.contribution_count = 0
        self.sybil_flags_received = 0
        self.sybil_flags_issued = 0

        # Optional reference to the active attack model — set by
        # EdgeMiningEnvironment.register_agents() for Sybil agents
        # only, after the attack model is instantiated. Lets a
        # colluding Sybil bias its own action distribution toward
        # mutual endorsement (see _sybil_action below). None for
        # honest agents and for non-collusion attack types.
        self.attack_model = None

        # DQN policy (lazy import to avoid circular)
        self._policy = None
        self._replay_buffer = None
        self._step_count = 0

        # Exploration
        self.epsilon = BT_MO_DQN["epsilon_start"]

    def _init_policy(self):
        """Lazy initialisation of DQN policy."""
        if self._policy is None:
            from algorithms.bt_mo_dqn.bt_mo_dqn import GraphAwareBTMODQN
            from utils.replay_buffer import ReplayBuffer
            self._policy = GraphAwareBTMODQN(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
            )
            self._replay_buffer = ReplayBuffer(
                capacity=BT_MO_DQN["replay_buffer_size"]
            )

    def select_action(self, state, global_threat):
        """
        Select action using epsilon-greedy BT-MO-DQN policy.
        Sybil agents use a random/adversarial policy instead.
        """
        if self.is_sybil:
            return self._sybil_action(state)

        self._init_policy()

        # Byzantine-aware exploration boost
        exploration_boost = self._get_exploration_boost(global_threat)
        effective_epsilon = min(1.0, self.epsilon + exploration_boost)

        if np.random.random() < effective_epsilon:
            return np.random.randint(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self._policy.device)
        with torch.no_grad():
            q_values = self._policy.online_net(state_tensor)
        return q_values.argmax().item()

    def _sybil_action(self, state):
        """
        Sybil agent behaviour:
        - Mostly mines data (to appear legitimate)
        - Occasionally validates falsely
        - Never flags Sybil (they protect each other)
        Mimics coordinated fake reviewer behaviour.

        Colluding Sybils (attack_model.mutual_endorsement=True) shift
        heavily toward ValidateNode: mutual false endorsement is the
        mechanism that resists trust_decay's pull toward quarantine
        while individual quality sits in the ambiguous zone (neither
        boosted nor penalised). Without this elevated rate, plain
        trust decay alone completes detection before the collusion
        pattern (dense mutual endorsement + correlated quality) can
        be observed by CollusionAwareTrustSignal — see
        ColludingSybilAttack docstring in sybil_attack_model.py.
        """
        if getattr(self.attack_model, 'mutual_endorsement', False):
            # MineData=0.25 (still enough to populate quality history
            # for behavioral-correlation detection), ValidateNode=0.70
            # (mutual endorsement), FlagSybil=0.00, CurateGraph=0.05,
            # AdjustPrivacy=0.00
            action_probs = [0.25, 0.70, 0.00, 0.05, 0.00]
        else:
            action_probs = [0.60, 0.35, 0.00, 0.05, 0.00]
            # MineData=0.60, ValidateNode=0.35, FlagSybil=0.00,
            # CurateGraph=0.05, AdjustPrivacy=0.00
        return np.random.choice(self.action_dim, p=action_probs)

    def _get_exploration_boost(self, global_threat):
        """Return exploration boost based on global threat level Θ."""
        if global_threat >= 0.6:
            return BT_MO_DQN["exploration_boost"]["high"]
        elif global_threat >= 0.3:
            return BT_MO_DQN["exploration_boost"]["medium"]
        return BT_MO_DQN["exploration_boost"]["low"]

    def store_experience(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self._init_policy()
        self._replay_buffer.push(state, action, reward, next_state, done)

    def learn(self):
        """Trigger one BT-MO-DQN learning step if buffer is ready."""
        self._init_policy()
        if len(self._replay_buffer) < BT_MO_DQN["batch_size"]:
            return

        self._step_count += 1
        if self._step_count % BT_MO_DQN["update_frequency"] == 0:
            batch = self._replay_buffer.sample(BT_MO_DQN["batch_size"])
            self._policy.update(batch)

        if self._step_count % BT_MO_DQN["target_update_frequency"] == 0:
            self._policy.update_target_network()

        # Decay exploration
        self.epsilon = max(
            BT_MO_DQN["epsilon_end"],
            self.epsilon * BT_MO_DQN["epsilon_decay"]
        )

    def update_trust_score(self, new_score):
        """Called by TPP after each trust propagation step."""
        self.current_trust_score = new_score

    def update_reputation(self, new_reputation):
        """Called by BRMP after each reputation update."""
        self.current_reputation = new_reputation

    def update_dp_epsilon(self, new_epsilon):
        """Called by DAPP after privacy adaptation."""
        self.dp_epsilon = np.clip(
            new_epsilon,
            DAPP["epsilon_min"],
            DAPP["epsilon_max"]
        )

    def add_dp_noise(self, value):
        """Apply Laplace differential privacy noise to a value."""
        sensitivity = DAPP["sensitivity"]
        scale = sensitivity / max(self.dp_epsilon, 1e-8)
        noise = np.random.laplace(0, scale)
        return value + noise

    def record_mining_quality(self, quality):
        """Track mining quality over time."""
        self.mining_quality_history.append(quality)
        if len(self.mining_quality_history) > 50:
            self.mining_quality_history.pop(0)

    def record_validation(self, correct):
        """Track validation accuracy."""
        self.validation_accuracy_history.append(1.0 if correct else 0.0)
        if len(self.validation_accuracy_history) > 50:
            self.validation_accuracy_history.pop(0)

    @property
    def avg_mining_quality(self):
        if not self.mining_quality_history:
            return 0.5
        return np.mean(self.mining_quality_history)

    @property
    def validation_accuracy(self):
        if not self.validation_accuracy_history:
            return 0.5
        return np.mean(self.validation_accuracy_history)

    def get_state_dict(self):
        """Return agent state as dictionary for logging."""
        return {
            "agent_id": self.agent_id,
            "is_sybil": self.is_sybil,
            "is_quarantined": self.is_quarantined,
            "trust_score": self.current_trust_score,
            "reputation": self.current_reputation,
            "dp_epsilon": self.dp_epsilon,
            "avg_mining_quality": self.avg_mining_quality,
            "validation_accuracy": self.validation_accuracy,
            "contribution_count": self.contribution_count,
            "sybil_flags_received": self.sybil_flags_received,
            "epsilon_explore": self.epsilon,
        }

    def save(self, path):
        """Save policy network weights."""
        if self._policy is not None:
            self._policy.save(path)

    def load(self, path):
        """Load policy network weights."""
        self._init_policy()
        self._policy.load(path)

    def __repr__(self):
        role = "Sybil" if self.is_sybil else "Honest"
        status = "QUARANTINED" if self.is_quarantined else "Active"
        return (f"Agent(id={self.agent_id}, role={role}, "
                f"status={status}, trust={self.current_trust_score:.3f})")
