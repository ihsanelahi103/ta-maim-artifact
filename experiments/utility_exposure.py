import os, json, argparse, collections
import numpy as np
import utility_eval as U
import utility_eval2 as V

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,123,7,11,99")
    ap.add_argument("--attack", default="camouflage")
    ap.add_argument("--combined", action="store_true")
    ap.add_argument("--tkey", default="")   # adaptive: t2 (malicious phase, primary) or t1 (drift start, sensitivity)      # camouflage | adaptive
    ap.add_argument("--checkpoints", default="25,50,75,100,125,150,175,200")
    ap.add_argument("--strengths", default="3")
    ap.add_argument("--poisons", default="identical,diverse")
    ap.add_argument("--n_targets", type=int, default=300)
    ap.add_argument("--n_honest", type=int, default=70)
    ap.add_argument("--n_sybil", type=int, default=30)
    ap.add_argument("--out", default="results/utility_exposure.json")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    cps = [int(x) for x in a.checkpoints.split(",")]
    n_h, n_s = a.n_honest, a.n_sybil; n = n_h + n_s
    rows = []
    for seed in seeds:
        p = "results/quar_A5_s%d_T_%s.json" % (seed, a.attack)
        if not os.path.exists(p):
            print("MISSING", p); continue
        js = json.load(open(p))
        key = "transitions" if a.attack == "camouflage" else (a.tkey or "t2")
        T = {n_h + int(k): int(v) for k, v in js["sybil_T"][key].items()}
        qe = {int(k): int(e) for k, e in js["quar_ep"].items()}
        shards = U.load_shards("yelp", n_h)
        n_items, S, C, test, tr_items = U.split_and_stats(shards, seed)
        cands = U.make_candidates(n_items, test, tr_items, seed)
        targets = np.random.RandomState(seed + 999).choice(n_items, min(a.n_targets, n_items), replace=False)
        sr = U.sybilrank_flags(n, n_h, n_s, seed)
        for pk in a.poisons.split(","):
            for c in [float(x) for x in a.strengths.split(",")]:
                Sy, Cy = V.make_poison(pk, n_s, n_items, targets, c, seed)
                S_all = np.vstack([S, Sy]); C_all = np.vstack([C, Cy])
                for t in cps:
                    act = [i for i in range(n_h, n) if T[i] <= t]
                    ids = list(range(n_h)) + act
                    qfl = set(i for i, e in qe.items() if e <= t)
                    def add(m, nd, nfl=None, fp=None):
                        rows.append(dict(seed=seed, poison=pk, strength=c, ckpt=t, method=m, ndcg=nd[0],
                                         poison_at10=nd[1], n_active=len(act),
                                         n_active_unq=len([i for i in act if i not in qfl]), flagged=nfl, fp=fp))
                    def pooled(keep_ids):
                        k = np.array(keep_ids)
                        return V.rank_eval(V.score_from(S_all[k].sum(0), C_all[k].sum(0)), cands, targets, seed)
                    add("fedavg", pooled(ids))
                    add("oracle", pooled(list(range(n_h))))
                    add("sybilrank", pooled([i for i in ids if i not in sr]), len(sr), len([i for i in sr if i < n_h]))
                    add("TA-MAIM", pooled([i for i in ids if i not in qfl]), len(qfl), len([i for i in qfl if i < n_h]))
                    Vv, obs, mu = V.agent_vectors(S_all[ids], C_all[ids])
                    if a.combined:
                        kp = np.array([i for i in ids if i not in qfl])
                        V2, o2, m2 = V.agent_vectors(S_all[kp], C_all[kp])
                        add("TA-MAIM+median", V.rank_eval(V.masked_agg(V2, o2, m2, "median"), cands, targets, seed), len(qfl))
                        add("TA-MAIM+trimmed", V.rank_eval(V.masked_agg(V2, o2, m2, "trim"), cands, targets, seed), len(qfl))
                    add("median", V.rank_eval(V.masked_agg(Vv, obs, mu, "median"), cands, targets, seed))
                    add("trimmed-mean", V.rank_eval(V.masked_agg(Vv, obs, mu, "trim"), cands, targets, seed))
                    w = V.foolsgold_weights(Vv, mu)
                    k = np.array(ids)
                    add("foolsgold", V.rank_eval(V.score_from((w[:, None] * S_all[k]).sum(0),
                                                              (w[:, None] * C_all[k]).sum(0)), cands, targets, seed))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rows, open(a.out, "w"))
    agg = collections.defaultdict(list)
    for r in rows: agg[(r["poison"], r["strength"], r["ckpt"], r["method"])].append(r)
    print("attack=%s | NDCG@10 mean+-std | poison@10 | active | active-unquarantined | honest flagged" % a.attack)
    for k in sorted(agg):
        rs = agg[k]; nd = np.array([r["ndcg"] for r in rs])
        fps = [r["fp"] for r in rs if r["fp"] is not None]
        print("%-9s c=%-3s ep=%-4s %-13s | %.4f+-%.4f | %.3f | %.1f | %.1f | %s" % (
            k[0], k[1], k[2], k[3], nd.mean(), nd.std(), np.mean([r["poison_at10"] for r in rs]),
            np.mean([r["n_active"] for r in rs]), np.mean([r["n_active_unq"] for r in rs]),
            "%.1f" % np.mean(fps) if fps else "n/a"))
if __name__ == "__main__":
    main()
