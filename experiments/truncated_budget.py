"""
Truncated Episode-Budget Analysis — WWW 2027 rebuttal experiment
TA-MAIM

Addresses reviewer concern: "F1=1.000 across every condition is
statistically unusual and hard to accept at face value."

This experiment does NOT change the detection mechanism. It re-reads
the SAME quarantine trajectory already produced by a normal 200-episode
run, but reports F1/Precision/Recall/FP as they would have been had
the experiment been stopped at an earlier checkpoint (10, 20, 30, 50,
75, 100, 125, 150, 175, 200 episodes). Quarantine is monotonic (an
agent, once quarantined, never returns to active status), so this is
a legitimate, single-pass way to answer: "how much detection time does
TA-MAIM actually need before F1 looks perfect?" rather than treating
Episode 200's F1 as the only number that exists.

Expected result (based on already-confirmed detection latencies):
  Static      -> F1 already 1.000 by budget=10  (Det.Ep ~1-2)
  Collusion   -> F1 climbs from 0 to 1.000 somewhere in [10,20] (Det.Ep ~11-12)
  Camouflage  -> F1 = 0.000 at every budget <= 150 (Det.Ep ~165-170),
                 climbing sharply only near budget=175-200
  Adaptive    -> F1 = 0.000 at every budget <= 150 (Det.Ep ~179),
                 climbing only near budget=200

This is the falsifiable evidence a reviewer is asking for: F1 is NOT
uniformly 1.000 -- it is 1.000 only once the episode budget exceeds
each attack's real detection latency, and well below 1.000 (often
literally 0.000) for camouflage/adaptive at any budget a real-time
system would plausibly allow.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import config.config as cfg
from utils.seed import set_seed

CHECKPOINTS = [10, 20, 30, 50, 75, 100, 125, 150, 175, 200]


def run_one(attack, dataset='yelp', episodes=200, seed=42):
    set_seed(seed)
    cfg.SCALE = 'medium'
    cfg.AGENT.update(cfg.SCALE_CONFIGS['medium'])

    from config.config import AGENT, TRAINING
    from blockchain.local_ledger import LocalLedger
    from algorithms.tpp.trust_propagation import TrustPropagationProtocol
    from algorithms.brmp.reputation_manager import ReputationManager
    from algorithms.dapp.privacy_protocol import DistributionAwarePrivacyProtocol
    from agents.autonomous_agent import AutonomousAgent
    from environment.edge_mining_env import EdgeMiningEnvironment
    from environment.sybil_attack_model import SybilAttackFactory

    ledger = LocalLedger()
    tpp    = TrustPropagationProtocol(num_agents=AGENT['num_agents'])
    rmp    = ReputationManager(ledger=ledger)
    dapp   = DistributionAwarePrivacyProtocol(num_agents=AGENT['num_agents'])
    agents = []
    for i in range(AGENT['num_agents']):
        is_sybil = i >= AGENT['num_honest']
        agents.append(AutonomousAgent(i, is_sybil, AGENT['state_dim'],
                                      AGENT['action_dim'], tpp, rmp, dapp))
        ledger.register_agent(i)

    env = EdgeMiningEnvironment(dataset, tpp, rmp, dapp, ledger)
    env.attack_scenario = attack
    env.attack_model = SybilAttackFactory.create(
        attack, env.num_sybil, env.num_honest)
    env.register_agents(agents)

    checkpoint_metrics = {}

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
            for agent in agents:
                if not agent.is_quarantined and \
                        rmp.should_quarantine(agent.agent_id):
                    ledger.quarantine_agent(agent.agent_id)
                    agent.is_quarantined = True
            states = next_states
            if done:
                break

        # Snapshot F1 at this episode IF it's a checkpoint -- quarantine
        # is monotonic, so this is exactly what F1 would have been had
        # the experiment stopped here.
        if episode in CHECKPOINTS:
            sybil_q  = sum(1 for a in agents if a.is_sybil and a.is_quarantined)
            honest_q = sum(1 for a in agents if not a.is_sybil and a.is_quarantined)
            tp = sybil_q; fp = honest_q; fn = env.num_sybil - tp
            p  = tp / max(tp + fp, 1)
            r  = tp / max(tp + fn, 1)
            f1 = 2 * p * r / max(p + r, 1e-8)
            checkpoint_metrics[episode] = {
                'f1': round(f1, 4), 'precision': round(p, 4),
                'recall': round(r, 4), 'fp': fp,
                'sybil_detected': sybil_q, 'num_sybil': env.num_sybil,
            }

    return checkpoint_metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='yelp')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--attacks', nargs='+',
                        default=['static', 'camouflage', 'collusion', 'adaptive'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 7])
    args = parser.parse_args()

    print("=" * 90)
    print("  TA-MAIM -- Truncated Episode-Budget Analysis")
    print(f"  Dataset: {args.dataset} | Checkpoints: {CHECKPOINTS}")
    print("  Answers: 'How much detection time is actually needed before")
    print("  F1 looks perfect?' -- NOT a claim that F1=1.000 always/instantly.")
    print("=" * 90)

    all_results = {}
    for attack in args.attacks:
        print(f"\n--- Attack: {attack.upper()} ---")
        seed_runs = []
        for seed in args.seeds:
            print(f"  seed={seed}...", end=' ', flush=True)
            cp = run_one(attack, args.dataset, args.episodes, seed)
            seed_runs.append(cp)
            print("done")

        # Aggregate F1 mean+-std per checkpoint across seeds
        agg = {}
        for ep in CHECKPOINTS:
            f1_vals = [run[ep]['f1'] for run in seed_runs if ep in run]
            fp_vals = [run[ep]['fp'] for run in seed_runs if ep in run]
            if f1_vals:
                agg[ep] = {
                    'f1_mean': round(float(np.mean(f1_vals)), 4),
                    'f1_std':  round(float(np.std(f1_vals)), 4),
                    'fp_mean': round(float(np.mean(fp_vals)), 2),
                }
        all_results[attack] = agg

        print(f"\n  {'Budget':>7} | {'F1 mean':>9} | {'F1 std':>7} | {'FP mean':>8}")
        print('  ' + '-' * 42)
        for ep in CHECKPOINTS:
            if ep in agg:
                a = agg[ep]
                print(f"  {ep:>7} | {a['f1_mean']:>9.4f} | "
                      f"{a['f1_std']:>7.4f} | {a['fp_mean']:>8.2f}")

    os.makedirs('results', exist_ok=True)
    fname = f'results/truncated_budget_{args.dataset}.json'
    with open(fname, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {fname}")

    print("\n" + "=" * 90)
    print("  SUMMARY TABLE (paste into paper)")
    print("=" * 90)
    header = f"{'Budget':>7}"
    for attack in args.attacks:
        header += f" | {attack.capitalize():>12}"
    print(header)
    for ep in CHECKPOINTS:
        row = f"{ep:>7}"
        for attack in args.attacks:
            if ep in all_results[attack]:
                row += f" | {all_results[attack][ep]['f1_mean']:>12.3f}"
            else:
                row += f" | {'---':>12}"
        print(row)