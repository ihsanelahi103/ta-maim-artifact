"""
Adaptive Stealth Attack Evaluation — Addresses Reviewer MC1
TA-MAIM WWW 2027

Three-phase adaptive stealth attacker:
Phase 1: 95-99% quality mimicry (harder than camouflage 90-95%)
Phase 2: Slow behavioral drift with trust feedback adaptation
Phase 3: Selective malicious actions based on trust buffer

Expected: F1=0.93-0.98, longer detection latency, reduced TrustSep
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from utils.seed import set_seed
import config.config as cfg


def run_adaptive_stealth(dataset='yelp', episodes=200, seed=42):
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
    from environment.sybil_attack_model import AdaptiveStealthSybilAttack

    ledger = LocalLedger()
    tpp    = TrustPropagationProtocol(num_agents=AGENT['num_agents'])
    rmp    = ReputationManager(ledger=ledger)
    dapp   = DistributionAwarePrivacyProtocol(num_agents=AGENT['num_agents'])
    agents = []
    for i in range(AGENT['num_agents']):
        agents.append(AutonomousAgent(
            i, i >= AGENT['num_honest'],
            AGENT['state_dim'], AGENT['action_dim'], tpp, rmp, dapp))
        ledger.register_agent(i)

    env    = EdgeMiningEnvironment(dataset, tpp, rmp, dapp, ledger)
    attack = AdaptiveStealthSybilAttack(env.num_sybil, env.num_honest)
    env.attack_scenario = 'adaptive'
    env.attack_model    = attack
    env.register_agents(agents)

    first_full = None
    first_partial = None
    ep_rewards = []
    det_history = []

    print(f"  Ep | Reward | Det | TrSep | eps")
    print(f"  ---+--------+-----+-------+------")

    for episode in range(1, episodes + 1):
        env.current_episode = episode
        states = env.reset()
        ep_reward = 0

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
            # Feed trust estimates back to adaptive attacker
            for si in range(env.num_sybil):
                idx = AGENT['num_honest'] + si
                attack.update_trust_estimate(si, float(tpp.trust_scores[idx]))
            for agent in agents:
                if not agent.is_quarantined and \
                        rmp.should_quarantine(agent.agent_id):
                    ledger.quarantine_agent(agent.agent_id)
                    agent.is_quarantined = True
            states = next_states
            ep_reward += sum(rewards.values())
            if done:
                break

        sq = sum(1 for a in agents if a.is_sybil and a.is_quarantined)
        hq = sum(1 for a in agents if not a.is_sybil and a.is_quarantined)
        h  = float(np.mean(tpp.trust_scores[:AGENT['num_honest']]))
        s  = float(np.mean(tpp.trust_scores[AGENT['num_honest']:]))
        ep_rewards.append(ep_reward)
        det_history.append(sq)
        if first_partial is None and sq >= env.num_sybil * 0.5:
            first_partial = episode
        if first_full is None and sq == env.num_sybil:
            first_full = episode
        if episode % 25 == 0 or episode <= 5:
            print(f"  {episode:>3} | {ep_reward:>6.0f} | "
                  f"{sq:>2}/{env.num_sybil} | {h-s:.4f} | "
                  f"{dapp.average_epsilon:.4f}")

    sq  = sum(1 for a in agents if a.is_sybil and a.is_quarantined)
    hq  = sum(1 for a in agents if not a.is_sybil and a.is_quarantined)
    tp  = sq; fp = hq; fn = env.num_sybil - tp
    p   = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    f1  = 2 * p * r / max(p + r, 1e-8)
    h   = float(np.mean(tpp.trust_scores[:AGENT['num_honest']]))
    s   = float(np.mean(tpp.trust_scores[AGENT['num_honest']:]))
    return {
        'attack': 'adaptive_stealth', 'dataset': dataset, 'seed': seed,
        'sybil_f1': round(f1, 4), 'precision': round(p, 4),
        'recall': round(r, 4), 'false_positives': hq,
        'partial_det_ep': first_partial,
        'full_det_ep': first_full or episodes,
        'trust_sep': round(h - s, 4),
        'dp_epsilon': round(float(dapp.average_epsilon), 4),
        'avg_reward': round(float(np.mean(ep_rewards[-20:])), 2),
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',  default='yelp')
    parser.add_argument('--episodes', type=int, default=200)
    args = parser.parse_args()

    print("=" * 65)
    print("  TA-MAIM — Adaptive Stealth Attack Evaluation")
    print(f"  Dataset: {args.dataset} | Episodes: {args.episodes}")
    print("  Phase 1: 95-99% mimicry  Phase 2: Slow drift  Phase 3: Selective")
    print("=" * 65)

    results = []
    for seed in [42, 123, 7]:
        print(f"\n  === Seed {seed} ===")
        r = run_adaptive_stealth(args.dataset, args.episodes, seed)
        results.append(r)
        print(f"\n  F1={r['sybil_f1']:.4f}  FP={r['false_positives']}  "
              f"PartDet=Ep{r['partial_det_ep']}  "
              f"FullDet=Ep{r['full_det_ep']}  "
              f"TrustSep={r['trust_sep']:.4f}")

    f1s  = [r['sybil_f1'] for r in results]
    lats = [r['full_det_ep'] for r in results]
    seps = [r['trust_sep'] for r in results]

    print("\n" + "=" * 65)
    print("  RESULTS — mean±std over 3 seeds")
    print("=" * 65)
    print(f"  F1:           {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  Detection Ep: {np.mean(lats):.0f} ± {np.std(lats):.0f}")
    print(f"  Trust Sep:    {np.mean(seps):.4f} ± {np.std(seps):.4f}")
    print()
    print("  ATTACK DIFFICULTY RANKING:")
    print(f"  Static     | F1=1.000 | Det.Ep=1   | Difficulty: Low")
    print(f"  Collusion  | F1=1.000 | Det.Ep=3   | Difficulty: Medium")
    print(f"  Camouflage | F1=1.000 | Det.Ep=165 | Difficulty: High")
    print(f"  Adaptive   | F1={np.mean(f1s):.4f} | "
          f"Det.Ep={int(np.mean(lats))} | Difficulty: Very High")

    os.makedirs('results', exist_ok=True)
    fname = f'results/adaptive_stealth_{args.dataset}.json'
    with open(fname, 'w') as f:
        json.dump({
            'per_seed': results, 'f1_mean': float(np.mean(f1s)),
            'f1_std': float(np.std(f1s)), 'lat_mean': float(np.mean(lats)),
            'lat_std': float(np.std(lats)), 'sep_mean': float(np.mean(seps)),
            'sep_std': float(np.std(seps)),
        }, f, indent=2)
    print(f"\n  Saved: {fname}")