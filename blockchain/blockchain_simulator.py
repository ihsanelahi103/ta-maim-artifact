"""
Blockchain Simulator — simulates FISCO BCOS PBFT behaviour.
Tracks transactions, agent registration, quarantine, and gas costs.
"""

import time
import hashlib
import numpy as np
from config.config import BLOCKCHAIN


class BlockchainSimulator:
    def __init__(self):
        self.agents         = {}          # agent_id → {registered, quarantined}
        self.transactions   = []
        self.total_gas      = 0
        self.block_height   = 0
        self.avg_latency_ms = BLOCKCHAIN["avg_latency_ms"]

    def register_agent(self, agent_id):
        self.agents[agent_id] = {"registered": True, "quarantined": False}
        self.submit_transaction(agent_id, "register_agent", {})

    def submit_transaction(self, agent_id, action, payload):
        """Simulate a PBFT-confirmed blockchain transaction."""
        if agent_id in self.agents and self.agents[agent_id]["quarantined"]:
            return None   # Quarantined agents cannot transact

        gas = BLOCKCHAIN["gas_per_tx"].get(action, 10000)
        tx_hash = hashlib.sha3_256(
            f"{agent_id}{action}{time.time()}".encode()
        ).hexdigest()

        tx = {
            "hash":     tx_hash,
            "agent_id": agent_id,
            "action":   action,
            "payload":  payload,
            "gas":      gas,
            "block":    self.block_height,
            "latency":  self.avg_latency_ms + np.random.normal(0, 20),
        }
        self.transactions.append(tx)
        self.total_gas   += gas
        self.block_height += 1
        return tx

    def quarantine_agent(self, agent_id):
        if agent_id in self.agents:
            self.agents[agent_id]["quarantined"] = True
            self.submit_transaction(
                agent_id=-1,
                action="flag_sybil",
                payload={"quarantined": agent_id}
            )

    def get_stats(self):
        return {
            "total_tx":      len(self.transactions),
            "total_gas":     self.total_gas,
            "block_height":  self.block_height,
            "avg_gas_per_tx": self.total_gas / max(len(self.transactions), 1),
            "quarantined":   sum(1 for a in self.agents.values()
                                 if a["quarantined"]),
        }
