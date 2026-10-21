import os, json, argparse
import numpy as np
import utility_eval as U
import utility_eval2 as V
from utility_baselines2 import krum_select, normclip_w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dataset", default="yelp")
    ap.add_argument("--attack", default="camouflage")
    ap.add_argument("--checkpoints", default="75,100,150,200")
    ap.add_argument("--strengths", default="1,3")
    ap.add_argument("--poisons", default="identical,diverse")
    ap.add_argument("--n_targets", type=int, default=300)
    ap.add_argument("--n_honest", type=int, default=70)
    ap.add_argument("--n_sybil", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seed = a.seed
    cps = [int(x) for x in a.checkpoints.split(",")]
    n_h, n_s = a.n_honest, a.n_sybil; n = n_h + n_s
    qpath = ("results/quar_A5_s%d_AMZ_T_%s.json" if a.dataset == "amazon" else "results/quar_A5_s%d_T_%s.json") % (seed, a.attack)
    js = json.load(open(qpath))
    T = {n_h + int(k): int(v) for k, v in js["sybil_T"]["transitions"].items()}
    qe = {int(k): int(e) for k, e in js["quar_ep"].items()}
    shards = U.load_shards(a.dataset, n_h)
    n_items, S, C, test, tr_items = U.split_and_stats(shards, seed)
    cands = U.make_candidates(n_items, test, tr_items, seed)
    targets = np.random.RandomState(seed + 999).choice(n_items, min(a.n_targets, n_items), replace=False)
    rows = []
    for pk in a.poisons.split(","):
        for c in [float(x) for x in a.strengths.split(",")]:
            Sy, Cy = V.make_poison(pk, n_s, n_items, targets, c, seed)
            S_all = np.vstack([S, Sy]); C_all = np.vstack([C, Cy])
            for t in cps:
                act = [i for i in range(n_h, n) if T[i] <= t]
                ids = list(range(n_h)) + act
                k = np.array(ids); f = len(act)
                qfl = set(i for i, e in qe.items() if e <= t)
                def add(m, nd):
                    rows.append(dict(seed=seed, dataset=a.dataset, poison=pk, strength=c, ckpt=t, method=m, ndcg=float(nd[0]), poison_at10=float(nd[1]), f=f))
                def pooled(keep):
                    kk = np.array(keep)
                    return V.rank_eval(V.score_from(S_all[kk].sum(0), C_all[kk].sum(0)), cands, targets, seed)
                def wsum(kk, w):
                    return V.rank_eval(V.score_from((w[:, None] * S_all[kk]).sum(0), (w[:, None] * C_all[kk]).sum(0)), cands, targets, seed)
                Vv, obs, mu = V.agent_vectors(S_all[k], C_all[k])
                add("fedavg", pooled(ids))
                add("oracle", pooled(list(range(n_h))))
                add("TA-MAIM", pooled([i for i in ids if i not in qfl]))
                add("median", V.rank_eval(V.masked_agg(Vv, obs, mu, "median"), cands, targets, seed))
                w = V.foolsgold_weights(Vv, mu)
                add("foolsgold", wsum(k, w))
                add("krum", pooled(list(k[krum_select(Vv, f, 1)])))
                add("multikrum", pooled(list(k[krum_select(Vv, f, len(ids) - f)])))
                add("normclip", wsum(k, normclip_w(Vv, mu)))
                kh = np.arange(n_h)
                Vh, oh, muh = V.agent_vectors(S_all[kh], C_all[kh])
                add("median_nosybil", V.rank_eval(V.masked_agg(Vh, oh, muh, "median"), cands, targets, seed))
                add("normclip_nosybil", wsum(kh, normclip_w(Vh, muh)))
                add("foolsgold_nosybil", wsum(kh, V.foolsgold_weights(Vh, muh)))
                kq = np.array([i for i in ids if i not in qfl])
                Vq, oq, muq = V.agent_vectors(S_all[kq], C_all[kq])
                add("TA-MAIM+median", V.rank_eval(V.masked_agg(Vq, oq, muq, "median"), cands, targets, seed))
                add("TA-MAIM+normclip", wsum(kq, normclip_w(Vq, muq)))
    json.dump(rows, open(a.out, "w"))
    print("done seed", seed, a.dataset, len(rows), "rows")
if __name__ == "__main__":
    main()
