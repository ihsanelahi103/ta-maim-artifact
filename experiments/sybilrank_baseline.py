"""
SybilRank Baseline — WWW 2027 submission
TA-MAIM

Addresses reviewer expectation for this venue's audience: TA-MAIM's
CATS module explicitly builds on SybilRank's graph-clique insight
(Cao et al., NSDI 2012) but never directly compares against it. WWW's
Trust/Security/Web-Mining reviewer pool is exactly the community that
knows SybilRank and will expect to see it run, not just cited.

Algorithm (faithful to Cao et al. 2012):
  1. A small trusted "seed set" (SybilRank REQUIRES this by design --
     it is a semi-supervised algorithm, unlike TA-MAIM's fully online,
     label-free operation. This asymmetry is itself worth stating
     explicitly in the paper -- it is a real advantage of TA-MAIM,
     not something to obscure).
  2. Normalized trust mass starts concentrated at the seeds and
     diffuses via weighted random walk for R = O(log N) steps
     (early termination is what prevents trust from just converging
     to the stationary distribution, which would carry no signal).
  3. Final trust is degree-normalized (SybilRank's correction for
     the tendency of high-degree nodes to accumulate more trust mass
     regardless of Sybil status).
  4. Nodes are ranked ascending by final trust; the lowest-trust
     nodes are the Sybil prediction.

This script reuses the EXACT SAME cached interaction graph the rest
of TA-MAIM's experiments use (same loader, same edges, same Sybil
clique structure) -- this is a fair, apples-to-apples structural
comparison, not a different dataset or a weaker setup.

Important expected finding to verify, not assume: because the Sybil
clique is injected at the GRAPH CONSTRUCTION level in *_loader.py
(independent of which attack model later generates quality behavior
on top of it), SybilRank's result should be IDENTICAL regardless of
attack type (static/camouflage/collusion/adaptive) -- it has no way
to observe behavior at all, only structure. This is itself a
meaningful point for the paper: a purely structural baseline cannot
distinguish attack sophistication the way TA-MAIM's combined
structural+behavioral approach does. Report what the numbers
actually show; do not assume this holds without checking.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import config.config as cfg
from utils.seed import set_seed


def get_connected_components(graph, num_agents):
    """BFS connected-component finder over the full graph."""
    visited = [False] * num_agents
    components = []
    for start in range(num_agents):
        if visited[start]:
            continue
        comp = []
        stack = [start]
        visited[start] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in graph.get(u, {}):
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        components.append(comp)
    return components


def select_seeds(graph, num_agents, num_honest, deg_honest, num_seeds):
    """
    Component-aware, highest-degree seed selection.

    Diagnostic history (kept for transparency -- see paper Appendix):
    (1) Sequential low-index seeds gave AUC ~0.41-0.11 regardless of
        seed count (7 to 35) -- ruled out "insufficient seed mass".
    (2) Highest-degree-only seeding improved AUC somewhat but not
        reliably above 0.5 -- because this graph's honest population
        fragments into multiple disconnected components; seeds
        concentrated in one component cannot reach others regardless
        of how well-connected they are locally.
    (3) Component-aware seeding (>=1 seed per honest-containing
        component, proportional to component size, highest-degree
        within each) is the correct, fair implementation -- it
        maximizes genuine reachability rather than concentrating
        seed mass in a single region.
    """
    components = get_connected_components(graph, num_agents)
    honest_components = []
    for comp in components:
        honest_in_comp = [n for n in comp if n < num_honest]
        if honest_in_comp:
            honest_components.append(honest_in_comp)
    total_honest_reachable = sum(len(c) for c in honest_components)
    if total_honest_reachable == 0:
        return list(range(min(num_seeds, num_honest)))
    seeds = []
    for comp in honest_components:
        share = max(1, round(num_seeds * len(comp) / total_honest_reachable))
        comp_sorted = sorted(comp, key=lambda i: deg_honest[i], reverse=True)
        seeds.extend(comp_sorted[:share])
    return seeds


def sybilrank(graph, num_agents, seed_ids, rounds=None):
    """
    Cao et al. (NSDI 2012) SybilRank.

    graph: dict-of-dicts adjacency (agent_id -> {neighbor_id: weight}),
           same structure used throughout the rest of the codebase.
    seed_ids: list of agent_ids treated as known-honest seeds.
    rounds: propagation steps; defaults to ceil(log2(num_agents)),
            the early-termination rule from the original paper.
    """
    if rounds is None:
        rounds = max(1, int(np.ceil(np.log2(max(num_agents, 2)))))

    # Weighted degree of every node (sum of outgoing edge weights)
    deg = np.zeros(num_agents)
    for u in range(num_agents):
        deg[u] = sum(graph.get(u, {}).values())
        if deg[u] <= 0:
            deg[u] = 1e-8  # isolated node guard

    # Initial trust mass concentrated at seeds
    trust = np.zeros(num_agents)
    if len(seed_ids) == 0:
        raise ValueError("SybilRank requires at least one seed node")
    for s in seed_ids:
        trust[s] = 1.0 / len(seed_ids)

    # Early-terminated normalized random-walk propagation
    for _ in range(rounds):
        new_trust = np.zeros(num_agents)
        for u in range(num_agents):
            if trust[u] <= 0:
                continue
            neighbors = graph.get(u, {})
            for v, w in neighbors.items():
                new_trust[v] += trust[u] * (w / deg[u])
        trust = new_trust

    # Degree normalization (SybilRank's correction step)
    final_score = trust / deg

    return final_score  # LOWER score = more suspicious (less trust reached it)


def evaluate_sybilrank(final_score, num_honest, num_sybil, num_agents):
    """Compute F1 (at true-k operating point), AUC, and Precision@k."""
    order = np.argsort(final_score)  # ascending: most suspicious first
    predicted_sybil_at_k = set(order[:num_sybil].tolist())
    ground_truth_sybil = set(range(num_honest, num_honest + num_sybil))

    tp = len(predicted_sybil_at_k & ground_truth_sybil)
    fp = len(predicted_sybil_at_k - ground_truth_sybil)
    fn = len(ground_truth_sybil - predicted_sybil_at_k)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # AUC via rank-sum (Mann-Whitney U), no external dependency needed.
    # Sybils should rank LOWER (more suspicious) than honest agents for
    # a good detector, so we compute AUC on (-final_score) as the "score
    # that should be high for Sybils" convention.
    scores_for_auc = -final_score
    labels = np.array([1 if i >= num_honest else 0 for i in range(num_agents)])
    pos = scores_for_auc[labels == 1]
    neg = scores_for_auc[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        auc = float('nan')
    else:
        # rank-based AUC
        all_scores = np.concatenate([pos, neg])
        ranks = np.argsort(np.argsort(all_scores)) + 1
        pos_ranks = ranks[:len(pos)]
        auc = (pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

    # Precision@k for a couple of standard k values from the literature
    prec_at_k = {}
    for k in [10, 20, num_sybil]:
        k = min(k, num_agents)
        topk = set(order[:k].tolist())
        prec_at_k[k] = len(topk & ground_truth_sybil) / max(k, 1)

    return {
        'f1_at_true_k': round(f1, 4),
        'precision_at_true_k': round(precision, 4),
        'recall_at_true_k': round(recall, 4),
        'auc': round(float(auc), 4) if not np.isnan(auc) else None,
        'precision_at_k': {str(k): round(v, 4) for k, v in prec_at_k.items()},
    }


def run_condition(dataset, attack, seed_fraction=0.05, seed=int(__import__('os').environ.get('TAMAIM_SEED', 42))):
    set_seed(seed)
    cfg.SCALE = 'medium'
    cfg.AGENT.update(cfg.SCALE_CONFIGS['medium'])

    from config.config import AGENT
    from blockchain.local_ledger import LocalLedger
    from algorithms.tpp.trust_propagation import TrustPropagationProtocol
    from algorithms.brmp.reputation_manager import ReputationManager
    from algorithms.dapp.privacy_protocol import DistributionAwarePrivacyProtocol
    from agents.autonomous_agent import AutonomousAgent
    from environment.edge_mining_env import EdgeMiningEnvironment
    from environment.sybil_attack_model import SybilAttackFactory

    num_agents = AGENT['num_agents']
    num_honest = AGENT['num_honest']
    num_sybil = AGENT['num_sybil']

    # Build env only to get the SAME cached graph the rest of the
    # experiments use -- no training, no episodes needed for a
    # purely structural baseline.
    ledger = LocalLedger()
    tpp    = TrustPropagationProtocol(num_agents=num_agents)
    rmp    = ReputationManager(ledger=ledger)
    dapp   = DistributionAwarePrivacyProtocol(num_agents=num_agents)
    agents = []
    for i in range(num_agents):
        is_sybil = i >= num_honest
        agents.append(AutonomousAgent(i, is_sybil, AGENT['state_dim'],
                                      AGENT['action_dim'], tpp, rmp, dapp))
        ledger.register_agent(i)

    env = EdgeMiningEnvironment(dataset, tpp, rmp, dapp, ledger)
    env.attack_scenario = attack
    env.attack_model = SybilAttackFactory.create(attack, env.num_sybil, env.num_honest)
    env.register_agents(agents)
    env.current_episode = 1
    env.reset()  # loads the episode-1 graph snapshot into env.graph

    graph = env.graph

    deg_all = np.array([sum(graph.get(i, {}).values()) for i in range(num_agents)])
    deg_honest = deg_all[:num_honest]
    deg_sybil = deg_all[num_honest:]
    num_components = len(get_connected_components(graph, num_agents))
    print(f"    [DIAG] connected components: {num_components} | "
          f"mean degree honest={deg_honest.mean():.4f} sybil={deg_sybil.mean():.4f}")

    # Seed set: component-aware, highest-degree-within-component HONEST
    # agents. See select_seeds() docstring for the full diagnostic
    # history explaining why this replaced two earlier, simpler
    # strategies (sequential low-index seeds, then highest-degree-only
    # seeds) that both failed on this graph's fragmented structure.
    num_seeds = max(1, int(num_honest * seed_fraction))
    seed_ids = select_seeds(graph, num_agents, num_honest, deg_honest, num_seeds)
    print(f"    [DIAG] component-aware seeds selected: {len(seed_ids)}")

    final_score = sybilrank(graph, num_agents, seed_ids)
    metrics = evaluate_sybilrank(final_score, num_honest, num_sybil, num_agents)
    metrics['dataset'] = dataset
    metrics['attack'] = attack
    metrics['num_seeds'] = len(seed_ids)
    metrics['num_components'] = num_components
    return metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['yelp', 'amazon'])
    parser.add_argument('--attacks', nargs='+',
                        default=['static', 'camouflage', 'collusion', 'adaptive'])
    parser.add_argument('--seed_fraction', type=float, default=0.05,
                        help='Fraction of honest agents used as SybilRank seeds')
    args = parser.parse_args()

    print("=" * 80)
    print("  SybilRank Baseline (Cao et al., NSDI 2012)")
    print("  Same cached graph as all other TA-MAIM experiments")
    print(f"  Datasets: {args.datasets} | Attacks: {args.attacks}")
    print("=" * 80)

    all_results = []
    for dataset in args.datasets:
        for attack in args.attacks:
            print(f"\n--- {dataset} / {attack} ---")
            m = run_condition(dataset, attack, args.seed_fraction)
            all_results.append(m)
            print(f"  F1@true-k:  {m['f1_at_true_k']:.4f}")
            print(f"  AUC:        {m['auc']}")
            print(f"  Precision@k: {m['precision_at_k']}")
            print(f"  Seeds used: {m['num_seeds']}")

    print("\n" + "=" * 80)
    print("  SUMMARY -- F1@true-k by dataset x attack")
    print("=" * 80)
    header = f"{'Dataset':>8} | " + " | ".join(f"{a:>12}" for a in args.attacks)
    print(header)
    for dataset in args.datasets:
        row = f"{dataset:>8} | "
        row += " | ".join(
            f"{next(m['f1_at_true_k'] for m in all_results if m['dataset']==dataset and m['attack']==a):>12.4f}"
            for a in args.attacks
        )
        print(row)

    print("\n  Note: if F1 is identical across attack types for a given")
    print("  dataset, this confirms SybilRank's purely structural score")
    print("  cannot distinguish attack behavioral sophistication -- the")
    print("  graph clique it detects does not change with attack type,")
    print("  only agent QUALITY behavior does, which SybilRank never")
    print("  observes. TA-MAIM's F1/latency DOES vary by attack type")
    print("  (Table: Security Evaluation) precisely because it also")
    print("  uses behavioral signal SybilRank structurally cannot see.")

    os.makedirs('results', exist_ok=True)
    with open('results/sybilrank_baseline.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\n  Saved: results/sybilrank_baseline.json")