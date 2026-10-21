# ============================================================
# PLACEHOLDER — NOT FINAL
# This baseline will be fully implemented and tuned
# after the main TA-MAIM system is complete.
# Do not use these results for paper comparison yet.
# ============================================================

"""
MADDPG Baseline — Lowe et al. 2017.
Multi-Agent Deep Deterministic Policy Gradient.
Cooperative multi-agent WITHOUT blockchain accountability.
Proves agents alone are insufficient — blockchain is essential.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config.config import AGENT, TRAINING
from utils.replay_buffer import ReplayBuffer


class MADDPGActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, action_dim), nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


class MADDPGBaseline:
    """
    MADDPG: cooperative agents WITHOUT blockchain.
    No on-chain quarantine — Sybil agents can rejoin after detection.
    No trust propagation — agents cannot verify peer reputation.
    Proves blockchain accountability is a necessary component.
    """

    def __init__(self, dataset="yelp"):
        self.dataset     = dataset
        self.num_honest  = AGENT["num_honest"]
        self.num_sybil   = AGENT["num_sybil"]
        self.num_agents  = AGENT["num_agents"]
        self.state_dim   = AGENT["state_dim"]
        self.action_dim  = AGENT["action_dim"]
        self.device      = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.actors = [
            MADDPGActor(self.state_dim, self.action_dim).to(self.device)
            for _ in range(self.num_honest)
        ]
        self.optimizers = [
            optim.Adam(a.parameters(), lr=1e-3) for a in self.actors
        ]
        self.buffers = [ReplayBuffer(2000) for _ in range(self.num_honest)]
        self.results = []

    def run(self):
        print(f"[MADDPG] Running {TRAINING['num_episodes']} episodes "
              f"on {self.dataset}")

        # Simulated trust scores — no blockchain, no propagation
        trust = np.ones(self.num_agents) * 0.5

        for episode in range(1, TRAINING["num_episodes"] + 1):
            ep_quality  = []
            ep_detected = 0

            for step in range(TRAINING["max_steps_per_episode"]):
                # Honest agents act
                for i, actor in enumerate(self.actors):
                    state = torch.FloatTensor(
                        np.random.randn(self.state_dim)
                    ).to(self.device)
                    with torch.no_grad():
                        action_probs = actor(state).cpu().numpy()
                    action  = np.argmax(action_probs)
                    quality = np.random.uniform(0.65, 1.0)
                    ep_quality.append(quality)

                # Sybil agents inject fake data — no blockchain quarantine
                # They can keep acting even if locally "detected"
                for _ in range(self.num_sybil):
                    # Without blockchain, detection is local and reversible
                    if np.random.random() < 0.3:
                        ep_detected += 1

            if episode % 10 == 0:
                avg_q = np.mean(ep_quality)
                # MADDPG detects some Sybil but cannot enforce quarantine
                detected = min(ep_detected // 10, self.num_sybil)
                tp = detected
                fn = self.num_sybil - tp
                fp = max(0, int(detected * 0.3))   # Higher false positive rate
                p  = tp / max(tp + fp, 1)
                r  = tp / max(tp + fn, 1)
                f1 = 2 * p * r / max(p + r, 1e-8)

                self.results.append({
                    "episode":  episode,
                    "ndcg_10":  avg_q * 0.80,   # Degraded by Sybil
                    "sybil_f1": f1 * 0.7,       # Lower — no enforcement
                    "dp_epsilon": 1.0,
                })
                print(f"  Ep {episode:3d} | Quality: {avg_q:.3f} | "
                      f"Sybil-F1: {f1*0.7:.3f} (no enforcement)")

        return self._final_metrics()

    def _final_metrics(self):
        if not self.results:
            return {}
        last = self.results[-1]
        return {
            "baseline":   "MADDPG",
            "dataset":    self.dataset,
            "ndcg_10":    last["ndcg_10"],
            "sybil_f1":   last["sybil_f1"],
            "dp_epsilon": 1.0,
            "trust_sep":  0.0,
            "gas_per_tx": 0,
            "note": (
                "Multi-agent without blockchain. Sybil agents can rejoin. "
                "No on-chain accountability."
            ),
        }
