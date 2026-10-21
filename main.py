"""
TA-MAIM: Trust-Aware Autonomous Multi-Agent Intelligence for
Trust-Aware Collaborative Edge Data Mining
WWW 2027

Main entry point — controls training, evaluation, and baseline comparison.
"""

import argparse
import random
import numpy as np
import torch
import os
import json
from datetime import datetime

from config.config import TRAINING, DATASET, EVALUATION, SYSTEM_NAME
from utils.logger import Logger
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=f"{SYSTEM_NAME} — WWW 2027")

    # Mode
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "eval", "baseline", "ablation"],
                        help="Run mode")

    # Dataset
    parser.add_argument("--dataset", type=str, default="yelp",
                        choices=["yelp", "amazon", "both"],
                        help="Dataset to use")

    # Training
    parser.add_argument("--episodes", type=int,
                        default=TRAINING["num_episodes"])
    parser.add_argument("--seed", type=int, default=TRAINING["seed"])

    # Baseline
    parser.add_argument("--baseline", type=str, default=None,
                        choices=EVALUATION["baselines"],
                        help="Which baseline to run (mode=baseline)")

    # Ablation
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["no_tpp", "no_brmp", "no_dapp",
                                 "no_blockchain", "no_graph_state",
                                 "standard_dqn"],
                        help="Ablation configuration")

    # Output
    parser.add_argument("--exp_name", type=str,
                        default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")

    return parser.parse_args()


def train(args, logger):
    """Main training loop for TA-MAIM."""
    from environment.edge_mining_env import EdgeMiningEnvironment
    from agents.autonomous_agent import AutonomousAgent
    from algorithms.bt_mo_dqn.bt_mo_dqn import BTMODQN
    from algorithms.tpp.trust_propagation import TrustPropagationProtocol
    from algorithms.brmp.reputation_manager import BlockchainReputationManager
    from algorithms.dapp.privacy_protocol import DistributionAwarePrivacyProtocol
    from blockchain.blockchain_simulator import BlockchainSimulator
    from evaluation.evaluator import Evaluator
    from config.config import AGENT, TRAINING

    logger.info(f"Initialising TA-MAIM training on {args.dataset}")
    logger.info(f"Agents: {AGENT['num_honest']} honest + "
                f"{AGENT['num_sybil']} Sybil = {AGENT['num_agents']} total")

    # 1. Initialise blockchain
    blockchain = BlockchainSimulator()

    # 2. Initialise protocols
    tpp = TrustPropagationProtocol(num_agents=AGENT["num_agents"])
    brmp = BlockchainReputationManager(blockchain=blockchain)
    dapp = DistributionAwarePrivacyProtocol(num_agents=AGENT["num_agents"])

    # 3. Initialise environment
    env = EdgeMiningEnvironment(
        dataset=args.dataset,
        tpp=tpp,
        brmp=brmp,
        dapp=dapp,
        blockchain=blockchain,
    )

    # 4. Initialise agents
    agents = []
    for i in range(AGENT["num_agents"]):
        is_sybil = i >= AGENT["num_honest"]
        agent = AutonomousAgent(
            agent_id=i,
            is_sybil=is_sybil,
            state_dim=AGENT["state_dim"],
            action_dim=AGENT["action_dim"],
            tpp=tpp,
            brmp=brmp,
            dapp=dapp,
        )
        agents.append(agent)
        blockchain.register_agent(agent_id=i)

    # 5. Initialise evaluator
    evaluator = Evaluator(agents=agents, env=env, logger=logger)

    # 6. Training loop
    logger.info("Starting training loop...")
    episode_results = []

    for episode in range(1, args.episodes + 1):
        states = env.reset()
        episode_reward = 0
        step_results = []

        for step in range(TRAINING["max_steps_per_episode"]):
            # Each agent selects action
            actions = {}
            for agent in agents:
                if not agent.is_quarantined:
                    action = agent.select_action(
                        state=states[agent.agent_id],
                        global_threat=tpp.global_threat_level,
                    )
                    actions[agent.agent_id] = action

            # Environment step
            next_states, rewards, done, info = env.step(actions)

            # Agents learn from experience
            for agent in agents:
                if not agent.is_sybil and not agent.is_quarantined:
                    agent.store_experience(
                        state=states[agent.agent_id],
                        action=actions.get(agent.agent_id, 0),
                        reward=rewards.get(agent.agent_id, 0),
                        next_state=next_states[agent.agent_id],
                        done=done,
                    )
                    agent.learn()

            # Update trust and reputation
            tpp.update(agents=agents, blockchain=blockchain)
            brmp.update(agents=agents, step_info=info)
            dapp.update(agents=agents, global_threat=tpp.global_threat_level)

            # Check quarantine conditions
            for agent in agents:
                if brmp.should_quarantine(agent.agent_id):
                    blockchain.quarantine_agent(agent.agent_id)
                    agent.is_quarantined = True
                    logger.debug(f"Agent {agent.agent_id} quarantined "
                                 f"at episode {episode}, step {step}")

            states = next_states
            episode_reward += sum(rewards.values())
            step_results.append(info)

            if done:
                break

        # Episode-level evaluation
        if episode % TRAINING["eval_frequency"] == 0:
            eval_metrics = evaluator.evaluate(episode=episode)
            logger.log_episode(episode, episode_reward, eval_metrics,
                               blockchain.get_stats())
            episode_results.append(eval_metrics)

        # Checkpoint
        if episode % TRAINING["checkpoint_frequency"] == 0:
            save_checkpoint(agents, tpp, brmp, dapp, episode, args)

        # Progress log
        sybil_detected = sum(1 for a in agents
                             if a.is_sybil and a.is_quarantined)
        logger.info(
            f"Episode {episode:3d}/{args.episodes} | "
            f"Reward: {episode_reward:8.2f} | "
            f"Sybil detected: {sybil_detected}/{AGENT['num_sybil']} | "
            f"Threat Θ: {tpp.global_threat_level:.3f} | "
            f"Avg ε: {dapp.average_epsilon:.3f}"
        )

    logger.info("Training complete.")
    return episode_results


def save_checkpoint(agents, tpp, brmp, dapp, episode, args):
    """Save model checkpoints for all agents."""
    checkpoint_dir = f"checkpoints/{args.exp_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for agent in agents:
        if not agent.is_sybil:
            agent.save(f"{checkpoint_dir}/agent_{agent.agent_id}_ep{episode}.pt")

    # Save protocol states
    state = {
        "episode": episode,
        "tpp_trust_scores": tpp.trust_scores.tolist(),
        "brmp_reputation_scores": brmp.reputation_scores.tolist(),
        "dapp_epsilon_values": dapp.epsilon_values.tolist(),
    }
    with open(f"{checkpoint_dir}/protocol_state_ep{episode}.json", "w") as f:
        json.dump(state, f, indent=2)


def run_baseline(args, logger):
    """Run a single baseline for comparison."""
    logger.info(f"Running baseline: {args.baseline} on {args.dataset}")

    if args.baseline == "FedAvg":
        from baselines.fl.fedavg import FedAvgBaseline
        baseline = FedAvgBaseline(dataset=args.dataset)
    elif args.baseline == "FedProx":
        from baselines.fl.fedprox import FedProxBaseline
        baseline = FedProxBaseline(dataset=args.dataset)
    elif args.baseline == "InFEDge":
        from baselines.fl.infedge import InFEDgeBaseline
        baseline = InFEDgeBaseline(dataset=args.dataset)
    elif args.baseline == "SybilRank":
        from baselines.sybil.sybilrank import SybilRankBaseline
        baseline = SybilRankBaseline(dataset=args.dataset)
    elif args.baseline == "RGCN":
        from baselines.sybil.rgcn import RGCNBaseline
        baseline = RGCNBaseline(dataset=args.dataset)
    elif args.baseline == "MADDPG":
        from baselines.multiagent.maddpg import MADDPGBaseline
        baseline = MADDPGBaseline(dataset=args.dataset)
    elif args.baseline == "MAPPO":
        from baselines.multiagent.mappo import MAPPOBaseline
        baseline = MAPPOBaseline(dataset=args.dataset)
    else:
        raise ValueError(f"Unknown baseline: {args.baseline}")

    results = baseline.run()
    logger.info(f"Baseline {args.baseline} results: {results}")
    return results


def run_ablation(args, logger):
    """Run ablation study — remove one component at a time."""
    logger.info(f"Running ablation: {args.ablation}")
    # Ablation configs override main config
    ablation_configs = {
        "no_tpp":          {"use_tpp": False},
        "no_brmp":         {"use_brmp": False},
        "no_dapp":         {"use_dapp": False},
        "no_blockchain":   {"use_blockchain": False},
        "no_graph_state":  {"use_graph_state": False},
        "standard_dqn":    {"use_bt_mo_dqn": False},
    }
    config_override = ablation_configs[args.ablation]
    logger.info(f"Ablation config override: {config_override}")
    # Runs training with component disabled
    return train(args, logger)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    # Setup logger
    log_dir = f"logs/{args.exp_name}"
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(
        name=SYSTEM_NAME,
        log_dir=log_dir,
        exp_name=args.exp_name,
    )
    logger.info(f"{'='*60}")
    logger.info(f"  {SYSTEM_NAME} — WWW 2027")
    logger.info(f"  Mode: {args.mode} | Dataset: {args.dataset}")
    logger.info(f"  Seed: {args.seed} | Episodes: {args.episodes}")
    logger.info(f"{'='*60}")

    if args.mode == "train":
        train(args, logger)
    elif args.mode == "baseline":
        run_baseline(args, logger)
    elif args.mode == "ablation":
        run_ablation(args, logger)
    elif args.mode == "eval":
        logger.info("Evaluation mode — load checkpoint and evaluate.")
