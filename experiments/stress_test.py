"""
Stress Test — Extreme Sybil Ratios + Detection Latency
TA-MAIM WWW 2027

Tests:
1. Sybil ratios 10-70% (extends robustness to majority+ adversarial)
2. Reports detection latency as primary metric
3. Uses harder camouflage (90-95% quality similarity)

This addresses reviewer concern: "Is F1=1.000 too easy?"
At 60-70% Sybil, honest agents are minority → harder detection
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import config.config as cfg
from utils.seed import set_seed


def run_stress(sybil_fraction, attack='camouflage',
               dataset='yelp', episodes=200, seed=42):
    set_seed(seed)
    # Fixed agent count from SCALE config — never changes per ratio
    import config.config as _cfg
    base_agents = _cfg.SCALE_CONFIGS[_cfg.SCALE]['num_agents']
    num_sybil   = max(1, int(base_agents * sybil_fraction))
    num_honest  = base_agents - num_sybil

    # Update config IN-PLACE so loader uses correct values
    cfg.AGENT['num_agents'] = base_agents
    cfg.AGENT['num_honest'] = num_honest
    cfg.AGENT['num_sybil']  = num_sybil

    from config.config import AGENT, TRAINING
    from blockchain.local_ledger import LocalLedger
    from algorithms.tpp.trust_propagation import TrustPropagationProtocol
    from algorithms.brmp.reputation_manager import ReputationManager
    from algorithms.dapp.privacy_protocol import DistributionAwarePrivacyProtocol
    from agents.autonomous_agent import AutonomousAgent
    from environment.edge_mining_env import EdgeMiningEnvironment
    from environment.sybil_attack_model import SybilAttackFactory

    ledger = LocalLedger()
    tpp    = TrustPropagationProtocol(num_agents=base_agents)
    rmp    = ReputationManager(ledger=ledger)
    dapp   = DistributionAwarePrivacyProtocol(num_agents=base_agents)
    agents = []
    for i in range(base_agents):
        is_sybil = i >= num_honest
        agents.append(AutonomousAgent(i, is_sybil, AGENT['state_dim'],
                                      AGENT['action_dim'], tpp, rmp, dapp))
        ledger.register_agent(i)

    env = EdgeMiningEnvironment(dataset, tpp, rmp, dapp, ledger)
    env.attack_scenario = attack
    env.attack_model = SybilAttackFactory.create(attack, num_sybil, num_honest)
    env.register_agents(agents)

    first_full_detection = None
    partial_detection_ep = None
    ep_rewards = []

    for episode in range(1, episodes + 1):
        env.current_episode = episode
        states = env.reset()
        ep_reward = 0  # Reset per episode
        for step in range(TRAINING['max_steps_per_episode']):
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
                    if step % 4 == 0:
                        agent.learn()
            tpp.update(agents=agents, blockchain=None)
            rmp.sync_trust_scores(tpp.trust_scores)
            rmp.update(agents=agents, step_info=info)
            dapp.update(agents=agents, global_threat=tpp.global_threat_level)
            for agent in agents:
                if not agent.is_quarantined and \
                        rmp.should_quarantine(agent.agent_id):
                    ledger.quarantine_agent(agent.agent_id)
                    agent.is_quarantined = True
            states = next_states
            ep_reward += sum(rewards.values())
            if done: break

        ep_rewards.append(ep_reward)
        sybil_q = sum(1 for a in agents if a.is_sybil and a.is_quarantined)

        # Track partial detection (first 50% detected)
        if partial_detection_ep is None and sybil_q >= num_sybil * 0.5:
            partial_detection_ep = episode

        # Track full detection
        if first_full_detection is None and sybil_q == num_sybil:
            first_full_detection = episode

    # Final metrics
    sybil_q  = sum(1 for a in agents if a.is_sybil and a.is_quarantined)
    honest_q = sum(1 for a in agents if not a.is_sybil and a.is_quarantined)
    tp = sybil_q; fp = honest_q; fn = num_sybil - tp
    p  = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-8)

    return {
        'sybil_pct':           int(sybil_fraction * 100),
        'attack':              attack,
        'sybil_f1':            round(f1, 4),
        'false_positives':     honest_q,
        'detection_latency':   first_full_detection,
        'partial_detection':   partial_detection_ep,
        'trust_sep':           round(float(np.mean(tpp.trust_scores[:num_honest]) -
                                     np.mean(tpp.trust_scores[num_honest:])), 4),
        'dp_epsilon':          round(float(dapp.average_epsilon), 4),
        'avg_reward':          round(float(np.mean(ep_rewards[-20:])), 2),
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',  default='yelp')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--attack',   default='camouflage')
    args = parser.parse_args()

    fractions = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    print("=" * 75)
    print(f"  TA-MAIM — Stress Test: Extreme Sybil Ratios")
    print(f"  Dataset: {args.dataset} | Attack: {args.attack} | "
          f"Episodes: {args.episodes}")
    print(f"  Camouflage: 90-95% quality similarity (harder than before)")
    print("=" * 75)
    print(f"\n{'Sybil%':>7} | {'F1':>7} | {'FP':>4} | "
          f"{'Det.Ep':>7} | {'TrustSep':>9} | {'ε':>6} | {'Reward':>8}")
    print('-' * 65)

    results = []
    for frac in fractions:
        print(f"  {int(frac*100)}% Sybil...", end=' ', flush=True)
        r = run_stress(frac, args.attack, args.dataset, args.episodes)
        results.append(r)
        det = f"Ep {r['detection_latency']}" if r['detection_latency'] else "Never"
        print(f"F1={r['sybil_f1']:.4f}")
        print(f"{r['sybil_pct']:>6}% | {r['sybil_f1']:>7.4f} | "
              f"{r['false_positives']:>4} | {det:>7} | "
              f"{r['trust_sep']:>9.4f} | {r['dp_epsilon']:>6.3f} | "
              f"{r['avg_reward']:>8.2f}")

    print()
    print("=" * 75)
    print("  KEY FINDINGS:")
    perfect = [r for r in results if r['sybil_f1'] == 1.0]
    degraded = [r for r in results if r['sybil_f1'] < 1.0]
    if perfect:
        print(f"  F1=1.000 maintained up to {perfect[-1]['sybil_pct']}% Sybil")
    if degraded:
        print(f"  F1 degrades at {degraded[0]['sybil_pct']}%+ Sybil ratio")
        for r in degraded:
            print(f"    {r['sybil_pct']}%: F1={r['sybil_f1']:.4f} "
                  f"Det.Ep={r['detection_latency']}")

    os.makedirs('results', exist_ok=True)
    fname = f'results/stress_test_{args.dataset}_{args.attack}.json'
    with open(fname, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {fname}")