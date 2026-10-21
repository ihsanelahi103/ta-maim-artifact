"""Logger for TA-MAIM experiments."""
import logging
import os
import json
from datetime import datetime


class Logger:
    def __init__(self, name, log_dir, exp_name):
        self.log_dir  = log_dir
        self.exp_name = exp_name
        self.metrics  = []
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(f"{log_dir}/{exp_name}.log"),
                logging.StreamHandler(),
            ]
        )
        self.logger = logging.getLogger(name)

    def info(self, msg):   self.logger.info(msg)
    def debug(self, msg):  self.logger.debug(msg)
    def warning(self, msg):self.logger.warning(msg)

    def log_episode(self, episode, reward, eval_metrics, blockchain_stats):
        record = {
            "episode": episode,
            "reward":  reward,
            **eval_metrics,
            **blockchain_stats,
            "timestamp": datetime.now().isoformat(),
        }
        self.metrics.append(record)
        self.info(
            f"Ep {episode:3d} | "
            f"NDCG@10: {eval_metrics.get('ndcg_10', 0):.4f} | "
            f"Sybil-F1: {eval_metrics.get('sybil_f1', 0):.4f} | "
            f"DP ε: {eval_metrics.get('avg_epsilon', 0):.3f} | "
            f"TX: {blockchain_stats.get('total_tx', 0)}"
        )
        # Save metrics to JSON
        with open(f"{self.log_dir}/metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)
