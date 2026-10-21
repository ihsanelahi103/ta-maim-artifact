"""Go/no-go checks (pre-registration): (1) no-attack FedAvg NDCG vs popularity baseline,
(2) attack effect on FedAvg target HR@10, (3) AUC of peer-validation signal (-r) for Sybil vs honest, rounds 50-80.
usage: python experiments/fedrec_gonogo.py --dataset yelp --seed 7"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import fedrec_core as F
import fedrec_baselines as B


def auc(pos, neg):
    s = 0.0
    for p in pos:
        s += (neg < p).sum() + 0.5 * (neg == p).sum()
    return s / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="yelp"); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--eta", type=float, default=0.0)
    a = ap.parse_args()
    S, C, cands = B.load(a.dataset, a.seed)
    eta = a.eta if a.eta > 0 else B.select_eta(S, C, cands, a.seed)[0]
    print("eta", eta)
    from utility_eval import ndcg_at_k
    pop = C.sum(0); nd = []
    for ai, (c, g) in enumerate(cands):
        nd.append(ndcg_at_k(pop[c].astype(float), g, 10, np.random.RandomState(ai).uniform(0, 1e-9, len(c))))
    print("popularity-only NDCG@10 %.4f" % np.mean(nd))
    res = {}
    for att in (False, True):
        sim = F.FedSim(S, C, cands, a.seed, attack=att)
        aucs, aucs2 = [], []
        for r in range(1, a.rounds + 1):
            G, sy, act = sim.collect()
            if att and 50 <= r <= 80:
                rr = sim.peer_scores(G, list(range(sim.n_h)), eta)
                aucs.append(auc(-rr[sy], -rr[~sy]))        # Sybil = lower improvement -> higher -r
                r2 = sim.peer_scores_disturb(G, list(range(sim.n_h)), eta)
                aucs2.append(auc(-r2[sy], -r2[~sy]))
            sim.apply(B.F.agg_mean(G, np.nonzero(act)[0]), eta)
        res[att] = sim.evaluate()
        print("attack=%s  %s" % (att, {k: round(v, 4) for k, v in res[att].items()}))
        if aucs2:
            print("S2 (list-disturbance) AUC rounds 50-80: mean %.3f min %.3f max %.3f" % (np.mean(aucs2), np.min(aucs2), np.max(aucs2)))
        if aucs:
            print("peer-validation AUC rounds 50-80: mean %.3f min %.3f max %.3f" % (np.mean(aucs), np.min(aucs), np.max(aucs)))
    print("GO/NO-GO: (1) NDCG gain over popularity: %.4f; (2) HR@10 attack-minus-noattack: %.3f (needs >=0.10); (3) see AUC (needs >=0.7)" % (
        res[False]["ndcg"] - np.mean(nd), res[True]["hr10"] - res[False]["hr10"]))


if __name__ == "__main__":
    main()
