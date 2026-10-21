"""
Ablation Study for TA-MAIM — WWW 2027
Tests component removal, one at a time.

A1: w/o Graph-Aware State  → 15-dim state (remove s[11-15])
A2: w/o TPP               → no trust propagation
A3: w/o RMP               → no reputation management / quarantine
A4: w/o DAPP              → fixed ε=1.0 (static DP)
A5: Full TA-MAIM          → all components (baseline, static attack)
A6: w/o BT-MO-DQN         → random policy instead of learned policy
A7: Full TA-MAIM under COLLUSION attack (Pathway 3 ON)
    — reference point for the A8 comparison below
A8: w/o Collusion-Aware Trust Signal (Pathway 3 OFF), collusion attack
    — isolates C5's contribution. A1-A6 all use the static attack
    (component removal is attack-agnostic there); A7/A8 specifically
    need the collusion attack, since Pathway 3 only matters for that
    threat model. F1 alone doesn't discriminate A7 vs A8 (both reach
    1.000 by episode 200) — detection_latency is the metric that
    shows Pathway 3's contribution, so it's now tracked for every
    variant.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import config.config as cfg
from utils.seed import set_seed


def run_variant(variant, dataset='yelp', episodes=200,
                 attack_scenario='static'):
    """Run one ablation variant."""
    set_seed(int(__import__('os').environ.get('TAMAIM_SEED', 42)))
    cfg.SCALE = 'medium'
    cfg.AGENT.update(cfg.SCALE_CONFIGS['medium'])

    from config.config import AGENT, TRAINING, DAPP
    from blockchain.local_ledger import LocalLedger
    from algorithms.tpp.trust_propagation import TrustPropagationProtocol
    from algorithms.brmp.reputation_manager import ReputationManager
    from algorithms.dapp.privacy_protocol import DistributionAwarePrivacyProtocol
    from agents.autonomous_agent import AutonomousAgent
    from environment.edge_mining_env import EdgeMiningEnvironment
    from environment.sybil_attack_model import SybilAttackFactory

    # Adjust state dim for A1
    state_dim = 15 if variant == 'A1' else AGENT['state_dim']
    cut = (lambda s: __import__('numpy').concatenate([s[:11], s[16:]])) if variant == 'A1' else (lambda s: s)

    ledger = LocalLedger()
    tpp    = TrustPropagationProtocol(num_agents=AGENT['num_agents'])
    rmp    = ReputationManager(ledger=ledger)
    dapp   = DistributionAwarePrivacyProtocol(num_agents=AGENT['num_agents'])

    # A4: Fix epsilon to 1.0 (disable adaptive DP completely)
    if variant == 'A4':
        dapp.epsilon_values[:] = 1.0
        dapp._fixed_epsilon = True  # Flag to disable all adaptation

    # A6: Override agent action selection with random policy
    use_random_policy = (variant == 'A6')
    use_const_policy = (variant == 'A9')

    # A8: Disable Pathway 3 (Collusion-Aware Trust Signal). A7 leaves
    # it enabled (default True) — same collusion attack, only this
    # flag differs, isolating C5's contribution.
    if variant == 'A8':
        rmp.use_collusion_pathway = False

    agents = []
    for i in range(AGENT['num_agents']):
        is_sybil = i >= AGENT['num_honest']
        agents.append(AutonomousAgent(
            i, is_sybil, state_dim,
            AGENT['action_dim'], tpp, rmp, dapp
        ))
        ledger.register_agent(i)

    env = EdgeMiningEnvironment(dataset, tpp, rmp, dapp, ledger)
    attack_scenario = __import__('os').environ.get('TAMAIM_ATTACK', attack_scenario)
    env.attack_scenario = attack_scenario
    env.attack_model = SybilAttackFactory.create(
        attack_scenario, env.num_sybil, env.num_honest
    )
    env.register_agents(agents)

    ep_rewards = []
    first_full_detection = None
    quar_ep = {}
    def _sybT(env):
        out = {}
        objs = list(vars(env).values())
        for v in list(objs):
            if hasattr(v, '__dict__'): objs += list(vars(v).values())
        for v in objs:
            for nm in ('transitions', 't1', 't2'):
                d = getattr(v, nm, None) if hasattr(v, '__dict__') else None
                if isinstance(d, dict) and d: out[nm] = {str(k): int(x) for k, x in d.items()}
        return out
    eps_traj = []
    thr_traj = []

    for episode in range(1, episodes + 1):
        env.current_episode = episode
        states = env.reset()
        ep_reward = 0

        for step in range(TRAINING['max_steps_per_episode']):
            if use_const_policy:
                actions = {a.agent_id: (int(os.environ.get('TAMAIM_CONST_ACTION',3)) if not getattr(a,'is_sybil',False) else a.select_action(cut(states[a.agent_id]), tpp.global_threat_level)) for a in agents if not a.is_quarantined}
            elif use_random_policy:
                # A6: Random policy instead of BT-MO-DQN
                import random
                actions = {
                    a.agent_id: random.randint(0, AGENT['action_dim']-1)
                    for a in agents if not a.is_quarantined
                }
            else:
                actions = {
                    a.agent_id: a.select_action(
                        cut(states[a.agent_id]),
                        tpp.global_threat_level
                    )
                    for a in agents if not a.is_quarantined
                }
            next_states, rewards, done, info = env.step(actions)

            for agent in agents:
                if not agent.is_sybil and not agent.is_quarantined:
                    s  = cut(states[agent.agent_id])
                    ns = cut(next_states[agent.agent_id])
                    agent.store_experience(
                        s, actions.get(agent.agent_id, 0),
                        rewards.get(agent.agent_id, 0.0), ns, done
                    )
                    if step % 4 == 0:
                        agent.learn()

            # A2: Skip TPP update
            if variant != 'A2':
                tpp.update(agents=agents, blockchain=None)

            if hasattr(rmp, 'sync_trust_scores'):
                rmp.sync_trust_scores(tpp.trust_scores)

            # A3: Skip RMP update and quarantine
            if variant != 'A3':
                rmp.update(agents=agents, step_info=info)
                for agent in agents:
                    if not agent.is_quarantined and \
                            rmp.should_quarantine(agent.agent_id):
                        ledger.quarantine_agent(agent.agent_id)
                        agent.is_quarantined = True
                        quar_ep.setdefault(agent.agent_id, episode)

            # A4: Skip DAPP adaptation
            if variant != 'A4':
                dapp.update(
                    agents=agents,
                    global_threat=tpp.global_threat_level
                )

            states = next_states
            ep_reward += sum(rewards.values())
            if done:
                break

        ep_rewards.append(ep_reward)
        eps_traj.append(float(dapp.average_epsilon))
        thr_traj.append(float(tpp.global_threat_level))
        for _a in agents:
            if _a.is_quarantined:
                quar_ep.setdefault(_a.agent_id, episode)

        # Track detection latency — first episode all Sybil quarantined
        sybil_q_now = sum(1 for a in agents if a.is_sybil and a.is_quarantined)
        if first_full_detection is None and sybil_q_now == AGENT['num_sybil']:
            first_full_detection = episode

    import json as _j, os as _o
    _o.makedirs('results', exist_ok=True)
    _j.dump({'variant': variant, 'attack': attack_scenario, 'episodes': episodes,
             'quar_ep': {str(k): v for k, v in quar_ep.items()},
             'sybil_T': _sybT(env), 'eps_traj': eps_traj, 'thr_traj': thr_traj, 'ep_rewards': [float(x) for x in ep_rewards], 'sybil_ids': list(range(AGENT['num_honest'], AGENT['num_agents']))},
            open('results/quar_%s_s%d%s.json' % (variant, int(_o.environ.get('TAMAIM_SEED', 42)), _o.environ.get('TAMAIM_TAG','')), 'w'))
    # Metrics
    sybil_q  = sum(1 for a in agents if a.is_sybil and a.is_quarantined)
    honest_q = sum(1 for a in agents if not a.is_sybil and a.is_quarantined)
    tp = sybil_q; fp = honest_q; fn = AGENT['num_sybil'] - tp
    p  = tp / max(tp + fp, 1)
    r  = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-8)
    h_trust = float(np.mean(tpp.trust_scores[:AGENT['num_honest']]))
    s_trust = float(np.mean(tpp.trust_scores[AGENT['num_honest']:]))
    avg_eps = float(dapp.average_epsilon)

    # Privacy utility: higher is better (lower epsilon = better privacy)
    # U = 1 / (1 + epsilon) — standard privacy utility formulation
    privacy_utility = round(1.0 / (1.0 + avg_eps), 4)

    # Graph feature contribution: measure ego-network trust quality
    # How well does the graph state help separate honest from Sybil?
    h_trust_var = float(np.var(tpp.trust_scores[:AGENT['num_honest']]))
    s_trust_var = float(np.var(tpp.trust_scores[AGENT['num_honest']:]))
    # Lower Sybil trust variance = cleaner separation
    graph_quality = round(1.0 - s_trust_var, 4)

    return {
        'variant':            variant,
        'attack_scenario':    attack_scenario,
        'sybil_f1':           round(f1, 4),
        'precision':          round(p, 4),
        'recall':             round(r, 4),
        'fp':                 honest_q,
        'trust_sep':          round(h_trust - s_trust, 4),
        'dp_epsilon':         round(avg_eps, 4),
        'privacy_utility':    privacy_utility,
        'graph_quality':      graph_quality,
        'avg_reward':         round(float(np.mean(ep_rewards[-20:])), 2),
        'detection_latency':  first_full_detection,
    }


VARIANT_DESC = {
    'A1': 'w/o Graph-Aware State (15-dim)',
    'A2': 'w/o TPP (no trust propagation)',
    'A3': 'w/o RMP (no quarantine)',
    'A4': 'w/o DAPP (fixed ε=1.0)',
    'A5': 'Full TA-MAIM (all components)',
    'A6': 'w/o BT-MO-DQN (random policy)',
    'A9': 'Constant policy (honest agents always action 3; Sybils unchanged)',
    'A7': 'Full TA-MAIM, collusion attack (Pathway 3 ON)',
    'A8': 'w/o Collusion-Aware Signal (Pathway 3 OFF), collusion attack',
}

VARIANT_ATTACK = {
    'A1': 'static', 'A2': 'static', 'A3': 'static',
    'A4': 'static', 'A5': 'static', 'A6': 'static', 'A9': 'static',
    'A7': 'collusion', 'A8': 'collusion',
}


if __name__ == '__main__':
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',  default='yelp')
    parser.add_argument('--episodes', type=int, default=200)
    args = parser.parse_args()

    print("=" * 70)
    print("  TA-MAIM — Ablation Study")
    print(f"  Dataset: {args.dataset} | Episodes: {args.episodes}")
    print("=" * 70)
    print(f"\n{'Variant':>4} | {'Description':>30} | {'F1':>7} | "
          f"{'TrustSep':>9} | {'ε':>6} | {'PrivUtil':>8} | "
          f"{'Reward':>8} | {'Det.Ep':>7}")
    print('-' * 96)

    results = []
    # Run A5 first (full model), then ablations, then collusion pair
    for variant in __import__('os').environ.get('TAMAIM_VARIANTS', 'A5,A1,A2,A3,A4,A6,A7,A8').split(','):
        desc = VARIANT_DESC[variant]
        attack = VARIANT_ATTACK[variant]
        print(f"  Running {variant} ({attack}): {desc}...",
              end=' ', flush=True)
        r = run_variant(variant, dataset=args.dataset,
                        episodes=args.episodes, attack_scenario=attack)
        results.append(r)
        print(f"F1={r['sybil_f1']:.4f}")
        marker = " ← FULL MODEL" if variant == 'A5' else ""
        det = r['detection_latency'] if r['detection_latency'] else 'Never'
        print(f"{variant:>4} | {desc:>30} | {r['sybil_f1']:>7.4f} | "
              f"{r['trust_sep']:>9.4f} | {r['dp_epsilon']:>6.3f} | "
              f"{r['privacy_utility']:>8.4f} | {r['avg_reward']:>8.2f} | "
              f"{str(det):>7}{marker}")

    print()
    print("=" * 70)
    print("  KEY FINDINGS:")
    full = next(r for r in results if r['variant'] == 'A5')
    for r in results:
        if r['variant'] not in ('A5', 'A7', 'A8'):
            f1_drop = full['sybil_f1'] - r['sybil_f1']
            rw_drop = full['avg_reward'] - r['avg_reward']
            print(f"  Removing {r['variant'][2:]}: "
                  f"F1 drop={f1_drop:+.4f} "
                  f"Reward drop={rw_drop:+.2f}")

    a7 = next((r for r in results if r['variant'] == 'A7'), None)
    a8 = next((r for r in results if r['variant'] == 'A8'), None)
    if a7 and a8:
        lat7 = a7['detection_latency'] or args.episodes
        lat8 = a8['detection_latency'] or args.episodes
        print(f"\n  Collusion-Aware Trust Signal (C5) ablation:")
        print(f"    A7 (Pathway 3 ON):  Det.Ep={lat7}  F1={a7['sybil_f1']:.4f}")
        print(f"    A8 (Pathway 3 OFF): Det.Ep={lat8}  F1={a8['sybil_f1']:.4f}")
        if lat8 > lat7:
            improvement = 100.0 * (lat8 - lat7) / lat8
            print(f"    Pathway 3 improves detection latency by "
                  f"{lat8-lat7} episodes ({improvement:.1f}% faster)")

    os.makedirs('results', exist_ok=True)
    with open(f'results/ablation_{args.dataset}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: results/ablation_{args.dataset}.json")