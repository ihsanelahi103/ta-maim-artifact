"""
TA-MAIM Quick Run Script
========================
Run this first to verify your setup, then use main.py for full experiments.

Usage:
    python run.py              # Quick 10-episode test (all 3 scales)
    python run.py --full       # Full 200-episode training (medium scale)
    python run.py --baselines  # Run all 7 baselines
    python run.py --scale small|medium|large
"""

import argparse
import sys
import os
import numpy as np
import config.config as cfg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_quick_test(scale="medium", episodes=10, dataset="yelp", attack="static", seed=42, real_graph=None):
    """Quick test — verifies the pipeline works end to end."""
    from utils.seed import set_seed
    set_seed(seed)

    # Override scale
    import config.config as cfg
    cfg.SCALE = scale
    cfg.AGENT.update(cfg.SCALE_CONFIGS[scale])
    cfg.AGENT["sybil_fraction"] = 0.30
    cfg.AGENT["state_dim"]      = 20
    cfg.AGENT["action_dim"]     = 5

    from blockchain.local_ledger import LocalLedger
    from algorithms.tpp.trust_propagation import TrustPropagationProtocol
    from algorithms.brmp.reputation_manager import ReputationManager
    from algorithms.dapp.privacy_protocol import DistributionAwarePrivacyProtocol
    from agents.autonomous_agent import AutonomousAgent
    from environment.edge_mining_env import EdgeMiningEnvironment
    from evaluation.evaluator import Evaluator
    from config.config import AGENT, TRAINING

    print(f"\n{'='*60}")
    print(f"  TA-MAIM — Scale: {cfg.SCALE_CONFIGS[scale]['label']}")
    print(f"  Dataset: {dataset}")
    print(f"  Episodes: {episodes}")
    print(f"  Attack:   {attack.upper()}")
    print(f"{'='*60}\n")

    ledger = LocalLedger()
    tpp  = TrustPropagationProtocol(num_agents=AGENT["num_agents"])
    brmp = ReputationManager(ledger=ledger)
    dapp = DistributionAwarePrivacyProtocol(num_agents=AGENT["num_agents"])

    agents = []
    for i in range(AGENT["num_agents"]):
        is_sybil = i >= AGENT["num_honest"]
        agent = AutonomousAgent(i, is_sybil, AGENT["state_dim"],
                                AGENT["action_dim"], tpp, brmp, dapp)
        agents.append(agent)
        ledger.register_agent(i)

    env = EdgeMiningEnvironment(dataset, tpp, brmp, dapp, ledger)
    # Load real graph if provided
    if real_graph is not None and os.path.exists(real_graph):
        import pickle
        with open(real_graph, 'rb') as f:
            graph_data = pickle.load(f)
        env.graph = graph_data['adj']
        print(f"[RealGraph] Loaded: {real_graph}")
        print(f"[RealGraph] Edges: {len(graph_data['edges'])}, "
              f"Source: {graph_data.get('source','unknown')}")
    # Set attack scenario BEFORE first reset
    from environment.sybil_attack_model import SybilAttackFactory
    env.attack_scenario = attack
    env.attack_model = SybilAttackFactory.create(
        attack, env.num_sybil, env.num_honest
    )
    env.register_agents(agents)

    # Sync Non-IID distribution skew from data loader to DAPP
    # This activates DAPP's distribution-aware privacy adaptation
    if hasattr(env, 'data_loader') and env.data_loader is not None:
        skews = getattr(env.data_loader, 'distribution_skew', {})
        for agent_id, skew in skews.items():
            dapp.update_distribution_skew(int(agent_id), float(skew))
        noniid_count = sum(1 for s in skews.values() if s > 0.3)
        print(f"[DAPP] Non-IID sync: {noniid_count}/{AGENT['num_agents']} "
              f"agents flagged as Non-IID (skew>0.3)")   # Allow env to check quarantine status

    _nf = float(os.environ.get("TAMAIM_NONIID_FRAC", "0"))
    _inj = []
    if _nf > 0:
        _r = np.random.RandomState(int(seed) + 4242)
        _inj = sorted(int(x) for x in _r.choice(AGENT["num_honest"], int(round(_nf * AGENT["num_honest"])), replace=False))
        for _i in _inj:
            dapp.update_distribution_skew(_i, float(_r.uniform(0.45, 0.75)))
        print(f"[NonIID-INJECT] {len(_inj)} honest agents skewed: {_inj}")

    print(f"{'Ep':>4} | {'Reward':>9} | {'Sybil':>12} | "
          f"{'Θ':>6} | {'ε':>6} | {'H-trust':>8} | {'S-trust':>8}")
    print("-" * 70)

    for episode in range(1, episodes + 1):
        env.current_episode = episode
        states = env.reset()
        ep_reward = 0

        for step in range(TRAINING["max_steps_per_episode"]):
            actions = {
                a.agent_id: a.select_action(
                    states[a.agent_id], tpp.global_threat_level)
                for a in agents if not a.is_quarantined
            }
            next_states, rewards, done, info = env.step(actions)

            for agent in agents:
                if not agent.is_sybil and not agent.is_quarantined:
                    agent.store_experience(
                        states[agent.agent_id],
                        actions.get(agent.agent_id, 0),
                        rewards.get(agent.agent_id, 0.0),
                        next_states[agent.agent_id], done)
                    if step % 4 == 0:  # Learn every 4 steps — 4x faster
                        agent.learn()

            tpp.update(agents=agents, blockchain=None)
            if hasattr(brmp, 'sync_trust_scores'):
                brmp.sync_trust_scores(tpp.trust_scores)
            brmp.update(agents=agents, step_info=info)
            dapp.update(agents=agents, global_threat=tpp.global_threat_level)

            for agent in agents:
                if not agent.is_quarantined and \
                        brmp.should_quarantine(agent.agent_id):
                    ledger.quarantine_agent(agent.agent_id)
                    agent.is_quarantined = True

            states = next_states
            ep_reward += sum(rewards.values())
            if done:
                break

        sybil_det = sum(1 for a in agents if a.is_sybil and a.is_quarantined)
        h_trust   = np.mean(tpp.trust_scores[:AGENT["num_honest"]])
        s_trust   = np.mean(tpp.trust_scores[AGENT["num_honest"]:])

        # Track detection latency — first episode where ALL Sybil detected
        if not hasattr(run_quick_test, '_first_full_detection'):
            run_quick_test._first_full_detection = None
        if (run_quick_test._first_full_detection is None and
                sybil_det == AGENT['num_sybil']):
            run_quick_test._first_full_detection = episode

        print(f"{episode:>4} | {ep_reward:>9.2f} | "
              f"{sybil_det:>5}/{AGENT['num_sybil']} sybil | "
              f"{tpp.global_threat_level:>6.3f} | "
              f"{dapp.average_epsilon:>6.3f} | "
              f"{h_trust:>8.4f} | {s_trust:>8.4f}")

    # Final metrics
    sybil_q  = sum(1 for a in agents if a.is_sybil and a.is_quarantined)
    honest_q = sum(1 for a in agents if not a.is_sybil and a.is_quarantined)
    tp = sybil_q; fp = honest_q; fn = AGENT["num_sybil"] - sybil_q
    p  = tp / max(tp + fp, 1)
    r  = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-8)
    bc = ledger.get_stats()

    print(f"\n{'='*60}")
    print(f"  RESULTS — {episodes} episodes")
    print(f"{'='*60}")
    print(f"  Sybil detected:    {sybil_q}/{AGENT['num_sybil']}")
    print(f"  False positives:   {honest_q}/{AGENT['num_honest']}")
    print(f"  Sybil-F1:          {f1:.4f}")
    print(f"  Precision:         {p:.4f}")
    print(f"  Recall:            {r:.4f}")
    det_ep = getattr(run_quick_test, '_first_full_detection', None)
    run_quick_test._first_full_detection = None  # Reset for next run
    print(f"  Detection latency: Ep {det_ep if det_ep else 'never'} "
          f"(first full detection)")
    print(f"  Trust separation:  "
          f"{np.mean(tpp.trust_scores[:AGENT['num_honest']]) - np.mean(tpp.trust_scores[AGENT['num_honest']:]):.4f}")
    print(f"  DP epsilon:        {dapp.average_epsilon:.4f}")
    # TX logging removed — no blockchain in TA-MAIM
    print(f"  Global threat Θ:   {tpp.global_threat_level:.4f}")
    print(f"{'='*60}\n")

    if os.environ.get("TAMAIM_NONIID_DUMP"):
        import json
        json.dump({"seed": int(seed), "attack": attack, "scale": scale, "inject": _inj,
                   "eps": [float(x) for x in dapp.epsilon_values],
                   "is_noniid": [bool(x) for x in dapp.is_noniid],
                   "quarantined": [bool(x.is_quarantined) for x in agents],
                   "is_sybil": [bool(x.is_sybil) for x in agents],
                   "f1": float(f1)}, open(os.environ["TAMAIM_NONIID_DUMP"], "w"))
    return f1


def run_all_scales(episodes=10):
    """Run quick test on all 3 scales."""
    print("\nTA-MAIM — Scalability Test (all 3 scales)")
    results = []
    for scale in ["small", "medium", "large"]:
        f1 = run_quick_test(scale=scale, episodes=episodes)
        results.append((scale, f1))

    print("\nScalability Summary:")
    print(f"{'Scale':>8} | {'F1':>6}")
    print("-" * 20)
    for scale, f1 in results:
        import config.config as cfg
        label = cfg.SCALE_CONFIGS[scale]["label"]
        print(f"{label:>20} | {f1:.4f}")


def run_baselines():
    """Run all 7 baselines."""
    print("\nRunning all baselines...")
    from baselines.fl.fedavg   import FedAvgBaseline
    from baselines.fl.fedprox  import FedProxBaseline
    from baselines.fl.infedge  import InFEDgeBaseline
    from baselines.sybil.sybilrank import SybilRankBaseline
    from baselines.sybil.rgcn      import RGCNBaseline
    from baselines.multiagent.maddpg import MADDPGBaseline
    from baselines.multiagent.mappo  import MAPPOBaseline

    baselines = [
        FedAvgBaseline(),
        FedProxBaseline(),
        InFEDgeBaseline(),
        SybilRankBaseline(),
        RGCNBaseline(),
        MADDPGBaseline(),
        MAPPOBaseline(),
    ]

    all_results = []
    for b in baselines:
        result = b.run()
        all_results.append(result)

    print("\n" + "="*70)
    print("BASELINE COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Baseline':>12} | {'NDCG@10':>8} | {'Sybil-F1':>9} | {'DP ε':>8} | {'Notes':>10}")
    print("-"*70)
    for r in all_results:
        print(f"{r.get('baseline','?'):>12} | "
              f"{r.get('ndcg_10', 0):>8.4f} | "
              f"{r.get('sybil_f1', 0):>9.4f} | "
              f"{r.get('dp_epsilon', 0):>8.3f} | "
              "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TA-MAIM Quick Runner")
    parser.add_argument("--full",      action="store_true",
                        help="Run full 200-episode training")
    parser.add_argument("--seed",      type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--real_graph", type=str, default=None,
                        help="Path to real graph pkl file (overrides synthetic)")
    parser.add_argument("--attack",    type=str, default="static",
                        choices=["static","camouflage","collusion"],
                        help="Sybil attack scenario for security experiments")
    parser.add_argument("--baselines", action="store_true",
                        help="Run all 7 baselines")
    parser.add_argument("--dataset",    default="yelp",
                        choices=["yelp", "amazon"],
                        help="Dataset to use")
    parser.add_argument("--all-scales",action="store_true",
                        help="Test all 3 scales")
    parser.add_argument("--scale",     default=None,
                        choices=["small", "medium", "large"])
    parser.add_argument("--episodes",  type=int, default=10)
    args = parser.parse_args()

    if args.baselines:
        run_baselines()
    elif args.all_scales:
        run_all_scales(episodes=args.episodes)
    elif args.full:
        scale = args.scale if args.scale else cfg.SCALE
        run_quick_test(scale=scale, episodes=200, dataset=args.dataset, attack=args.attack, seed=args.seed, real_graph=args.real_graph)
    else:
        scale = args.scale if args.scale else cfg.SCALE
        run_quick_test(scale=scale, episodes=args.episodes, dataset=args.dataset, attack=args.attack, seed=args.seed, real_graph=args.real_graph)