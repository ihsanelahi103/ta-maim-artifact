"""
Graph-Aware Byzantine-Tolerant Multi-Objective DQN (Graph-Aware BT-MO-DQN)
Contribution C2 — WWW 2027

Extension of EdgeChain's BT-MO-DQN (JSAC 2025) with three genuine additions:
  1. Graph-aware state space (20-dim vs EdgeChain's 15-dim):
     - Adds: degree centrality, clustering coefficient,
             neighbour avg trust, Sybil neighbour ratio,
             graph density (5 new graph-structural features)
  2. Collaborative context: agents perceive neighbours' trust scores
     (EdgeChain agents only perceive their own local state)
  3. Web data mining action semantics:
     MineData / ValidateNode / FlagSybil / CurateGraph / AdjustPrivacy
     (vs EdgeChain's SendData / VerifyData / DetectByzantine)

What is the SAME as EdgeChain (honestly acknowledged):
  - Double DQN backbone (van Hasselt et al. 2016)
  - Huber loss function
  - Soft target network update (τ = 0.01)
  - Experience replay buffer
  - Byzantine-aware exploration boost concept

Cite as: "Graph-Aware BT-MO-DQN extends our prior BT-MO-DQN [EdgeChain]
by incorporating graph-structural features into the agent state space,
enabling trust-aware collaborative mining decisions."
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from config.config import BT_MO_DQN as CFG, AGENT


class QNetwork(nn.Module):
    """Online / target Q-network: 20 → 128 → 128 → 5."""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class GraphAwareBTMODQN:
    """
    Double DQN with:
    - Byzantine-aware exploration boost
    - Multi-objective adaptive reward weighting
    - Huber loss for stability
    """

    def __init__(self, state_dim, action_dim):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        # Forced to CPU: this network is small (20-dim state, single
        # agent, single-row forward passes) and called individually
        # per-agent thousands of times per episode (no cross-agent
        # batching, since each of up to 100 agents has its own
        # independently-weighted network, not shared parameters).
        # GPU kernel-launch/transfer overhead for this access pattern
        # dominates over the tiny actual compute, measured empirically
        # at ~4.2s per step for 100 agents on CUDA vs. expected
        # sub-100ms on CPU for a network this size.
        self.device = torch.device("cpu")

        self.online_net = QNetwork(state_dim, action_dim,
                                   CFG["hidden_dim"]).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim,
                                   CFG["hidden_dim"]).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.Adam(
            self.online_net.parameters(), lr=CFG["learning_rate"]
        )
        self.loss_fn    = nn.SmoothL1Loss()   # Huber loss
        self.gamma      = CFG["gamma"]
        self.tau        = CFG["tau"]
        self._loss_history = []

    def update(self, batch):
        """One gradient step on a sampled batch."""
        states, actions, rewards, next_states, dones = batch

        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        # Double DQN target
        with torch.no_grad():
            # Action selected by online net
            next_actions = self.online_net(next_states).argmax(dim=1)
            # Value from target net
            next_q = self.target_net(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            targets = rewards + self.gamma * next_q * (1 - dones)

        current_q = self.online_net(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        loss = self.loss_fn(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()
        self._loss_history.append(loss.item())

    def update_target_network(self):
        """Soft update: θ_target ← τ*θ_online + (1-τ)*θ_target"""
        for target_param, online_param in zip(
            self.target_net.parameters(), self.online_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data +
                (1 - self.tau) * target_param.data
            )

    @staticmethod
    def compute_reward(r_mine, r_trust, r_privacy, r_efficiency,
                       global_threat):
        """
        Multi-objective reward with adaptive weights based on threat level.
        R = w1*r_mine + w2*r_trust + w3*r_privacy + w4*r_efficiency
        """
        if global_threat >= 0.6:
            w = CFG["reward_weights"]["high"]
        elif global_threat >= 0.3:
            w = CFG["reward_weights"]["medium"]
        else:
            w = CFG["reward_weights"]["low"]
        return (w[0] * r_mine  + w[1] * r_trust +
                w[2] * r_privacy + w[3] * r_efficiency)

    @property
    def avg_loss(self):
        if not self._loss_history:
            return 0.0
        return float(np.mean(self._loss_history[-50:]))

    def save(self, path):
        torch.save({
            "online": self.online_net.state_dict(),
            "target": self.target_net.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online"])
        self.target_net.load_state_dict(ckpt["target"])
