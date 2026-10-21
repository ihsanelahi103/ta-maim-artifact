"""
Robustness Analysis — Sybil Ratio Variation
TA-MAIM WWW 2027

Two experiments:
1. Ratio × Static attack (3 seeds) → mean ± std
2. Ratio × Camouflage attack (3 seeds) → mean ± std

Reports mean ± std across seeds for credibility.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import config.config as cfg
from utils.seed import set_seed


def run_single(sybil_fraction, attack, dataset, episodes, seed):
    """Run one configuration with one seed.

    IMPORTANT: base_agents stays FIXED at SCALE config value (e.g. 100).
    The sybil_fraction only changes how many of the 100 agents are Sybil.
    This keeps the graph structure constant while varying adversarial ratio.
    """
    set_seed(seed)

    # Fixed agent count from SCALE config — never changes per ratio
    import config.config as _cfg
    base_agents = _cfg.SCALE_CONFIGS[_cfg.SCALE]['num_agents']
    num_sybil   = max(1, int(base_agents * sybil_fraction))
    num_honest  = base_agents - num_sybil

    # Update config IN-PLACE so loader picks up the correct values
    cfg.AGENT['num_agents'] = base_agents
    cfg.AGENT['num_honest'] = num_honest
    cfg.AGENT['num_sybil']  = num_sybil
    # DO NOT reset scale — keep whatever was set externally

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
        agents.append(AutonomousAgent(
            i, is_sybil, AGENT['state_dim'],
            AGENT['action_dim'], tpp, rmp, dapp
        ))
        ledger.register_agent(i)

    env = EdgeMiningEnvironment(dataset, tpp, rmp, dapp, ledger)
    env.attack_scenario = attack
    env.attack_model = SybilAttackFactory.create(attack, num_sybil, num_honest)
    env.register_agents(agents)

    for episode in range(1, episodes + 1):
        env.current_episode = episode
        states = env.reset()
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
            if hasattr(rmp, 'sync_trust_scores'):
                rmp.sync_trust_scores(tpp.trust_scores)
            rmp.update(agents=agents, step_info=info)
            dapp.update(agents=agents, global_threat=tpp.global_threat_level)
            for agent in agents:
                if not agent.is_quarantined and \
                        rmp.should_quarantine(agent.agent_id):
                    ledger.quarantine_agent(agent.agent_id)
                    agent.is_quarantined = True
            states = next_states
            if done: break

    # Metrics
    sybil_q  = sum(1 for a in agents if a.is_sybil and a.is_quarantined)
    honest_q = sum(1 for a in agents if not a.is_sybil and a.is_quarantined)
    tp = sybil_q; fp = honest_q; fn = num_sybil - tp
    p  = tp / max(tp + fp, 1)
    r  = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-8)
    h_trust = float(np.mean(tpp.trust_scores[:num_honest]))
    s_trust = float(np.mean(tpp.trust_scores[num_honest:]))

    return {
        'f1':       f1,
        'precision':p,
        'recall':   r,
        'fp':       honest_q,
        'trust_sep':h_trust - s_trust,
        'dp_eps':   float(dapp.average_epsilon),
    }


def run_robustness(attack, dataset='yelp', episodes=200, seeds=[42, 123, 7]):
    """Run ratio experiment across multiple seeds."""
    fractions = [0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90]

    print(f"\n{'='*70}")
    print(f"  Attack: {attack.upper()} | Dataset: {dataset} | Seeds: {seeds}")
    print(f"{'='*70}")
    print(f"\n{'Ratio':>6} | {'F1 mean':>9} | {'F1 std':>8} | "
          f"{'TrustSep':>10} | {'ε':>6} | {'FP':>4}")
    print('-' * 56)

    all_results = []

    for frac in fractions:
        seed_results = []
        for seed in seeds:
            print(f"  {int(frac*100)}% Sybil seed={seed}...",
                  end=' ', flush=True)
            r = run_single(frac, attack, dataset, episodes, seed)
            seed_results.append(r)
            print(f"F1={r['f1']:.4f}")

        # Aggregate across seeds
        f1_vals  = [r['f1']       for r in seed_results]
        ts_vals  = [r['trust_sep'] for r in seed_results]
        eps_vals = [r['dp_eps']    for r in seed_results]
        fp_vals  = [r['fp']        for r in seed_results]

        mean_f1  = np.mean(f1_vals)
        std_f1   = np.std(f1_vals)
        mean_ts  = np.mean(ts_vals)
        mean_eps = np.mean(eps_vals)
        mean_fp  = np.mean(fp_vals)

        row = {
            'sybil_pct':   int(frac * 100),
            'attack':      attack,
            'f1_mean':     round(mean_f1,  4),
            'f1_std':      round(std_f1,   4),
            'trust_sep':   round(mean_ts,  4),
            'dp_epsilon':  round(mean_eps, 4),
            'avg_fp':      round(mean_fp,  2),
            'seed_results':seed_results,
        }
        all_results.append(row)

        print(f"{int(frac*100):>5}% | "
              f"{mean_f1:.4f}±{std_f1:.4f} | "
              f"{mean_ts:>10.4f} | "
              f"{mean_eps:>6.3f} | "
              f"{mean_fp:>4.1f}")

    return all_results


if __name__ == '__main__':
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',  default='yelp')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--seeds',    nargs='+', type=int,
                        default=[42, 123, 7])
    parser.add_argument('--attacks',  nargs='+',
                        default=['static', 'camouflage', 'collusion'])
    args = parser.parse_args()

    print("=" * 70)
    print("  TA-MAIM — Robustness Analysis")
    print(f"  Dataset: {args.dataset} | Episodes: {args.episodes}")
    print(f"  Seeds: {args.seeds}")
    print("=" * 70)

    all_results = {}

    for attack in args.attacks:
        results = run_robustness(
            attack=attack,
            dataset=args.dataset,
            episodes=args.episodes,
            seeds=args.seeds
        )
        all_results[attack] = results

    # Print combined summary
    print("\n" + "=" * 70)
    print("  COMBINED SUMMARY — Mean ± Std (for paper table)")
    print("=" * 70)
    print(f"\n{'Ratio':>6} | {'Static F1':>14} | {'Camouflage F1':>16}")
    print('-' * 42)

    fractions = [10, 20, 30, 40, 50]
    print(f"\n{'Ratio':>6} | {'Static F1':>14} | {'Camouflage F1':>16} | {'Collusion F1':>14}")
    print('-' * 58)
    for pct in fractions:
        row_s = next((r for r in all_results.get('static', [])
                      if r['sybil_pct'] == pct), None)
        row_c = next((r for r in all_results.get('camouflage', [])
                      if r['sybil_pct'] == pct), None)
        row_col = next((r for r in all_results.get('collusion', [])
                      if r['sybil_pct'] == pct), None)
        s_str   = (f"{row_s['f1_mean']:.3f}±{row_s['f1_std']:.3f}"
                   if row_s else "N/A")
        c_str   = (f"{row_c['f1_mean']:.3f}±{row_c['f1_std']:.3f}"
                   if row_c else "N/A")
        col_str = (f"{row_col['f1_mean']:.3f}±{row_col['f1_std']:.3f}"
                   if row_col else "N/A")
        print(f"{pct:>5}% | {s_str:>14} | {c_str:>16} | {col_str:>14}")

    os.makedirs('results', exist_ok=True)
    fname = f"results/robustness_{args.dataset}_{'-'.join(args.attacks)}_s{'-'.join(map(str, args.seeds))}.json"
    with open(fname, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {fname}")