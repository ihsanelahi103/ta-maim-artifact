"""
Baseline comparison for the downstream utility task (extends utility_eval.py).

Same data, split, candidates, poison targets and seeds as utility_eval.py.
Adds the standard Sybil/Byzantine-robust FL aggregators, operating ONLY on the
uploads (what a server without TA-MAIM would see):
  fedavg     count-weighted pooling of everyone (= 'none' in utility_eval.py)
  median     coordinate-wise median over the agents that observed the item
  trimmed    coordinate-wise trimmed mean (30% each side) over observers
  foolsgold  FoolsGold (Fung et al. 2020): down-weights agents whose uploads
             are mutually near-identical, then weighted pooling
and the structure/behaviour based rules:
  sybilrank  bottom-n_sybil SybilRank (n_sybil given to it), graph only
  oracle     true Sybils removed (upper bound)
  TA-MAIM:V  agents quarantined by episode t (results/quar_V_s<seed><TAG>.json)
Poison models:
  identical  every Sybil pushes the same 300 target items with 5 stars
  diverse    each Sybil pushes a random 40% of the targets and pads its upload
             with 200 honest-looking fake ratings on random items (evades
             upload-similarity defences; TA-MAIM's detector does not read uploads)
"""
import os, sys, json, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import utility_eval as U


def score_from(S, C, m=5.0):
    C = np.maximum(C, 0.0); S = np.clip(S, 0.0, 5.0 * C)
    mu = S.sum() / max(C.sum(), 1.0)
    return (S + m * mu) / (C + m)


def rank_eval(score, cands, targets, seed, k=10):
    tset = set(int(t) for t in targets)
    nd, pk = [], []
    for a, (c, g) in enumerate(cands):
        jit = np.random.RandomState(seed * 31 + a).uniform(0, 1e-9, size=len(c))
        sc = score[c]
        nd.append(U.ndcg_at_k(sc, g, k, jit))
        top = c[np.argsort(-(sc + jit))[:k]]
        pk.append(np.mean([int(t) in tset for t in top]))
    return float(np.mean(nd)), float(np.mean(pk))


def agent_vectors(S, C, m=5.0):
    mu = S.sum() / max(C.sum(), 1.0)
    return (S + m * mu) / (C + m), C > 0, mu


def masked_agg(V, obs, mu, mode, trim=0.3):
    out = np.full(V.shape[1], mu)
    for j in np.nonzero(obs.any(0))[0]:
        v = np.sort(V[obs[:, j], j])
        if mode == "median":
            out[j] = float(np.median(v))
        else:
            t = int(np.floor(trim * len(v)))
            out[j] = float(v[t:len(v) - t].mean()) if len(v) - 2 * t > 0 else float(np.median(v))
    return out


def foolsgold_weights(V, mu, kappa=1.0):
    X = V - mu
    nrm = np.linalg.norm(X, axis=1); nrm[nrm == 0] = 1.0
    cs = (X @ X.T) / np.outer(nrm, nrm)
    np.fill_diagonal(cs, -np.inf)
    mx = cs.max(1)
    n = len(mx)
    for i in range(n):
        for j in range(n):
            if i != j and mx[j] > mx[i]:
                cs[i, j] = cs[i, j] * mx[i] / mx[j]
    wv = 1.0 - cs.max(1)
    wv = np.clip(wv, 0.0, 1.0)
    if wv.max() > 0:
        wv = wv / wv.max()
    wv[wv == 1.0] = 0.99
    with np.errstate(divide="ignore", invalid="ignore"):
        wv = kappa * (np.log(wv / (1.0 - wv)) + 0.5)
    wv[np.isinf(wv) & (wv > 0)] = 1.0
    wv[np.isinf(wv) & (wv < 0)] = 0.0
    return np.clip(np.nan_to_num(wv, nan=0.0), 0.0, 1.0)


def make_poison(kind, n_s, n_items, targets, c, seed):
    Sy = np.zeros((n_s, n_items)); Cy = np.zeros((n_s, n_items))
    if kind == "identical":
        Sy[:, targets] = 5.0 * c; Cy[:, targets] = c
        return Sy, Cy
    rng = np.random.RandomState(seed + 777)
    for s in range(n_s):
        sub = rng.choice(targets, int(0.4 * len(targets)), replace=False)
        Sy[s, sub] = 5.0 * c; Cy[s, sub] = c
        pad = rng.choice(n_items, 200, replace=False)
        cnt = rng.randint(1, 4, size=200).astype(float)
        rate = np.clip(np.round(rng.normal(3.8, 1.0, size=200)), 1, 5)
        Sy[s, pad] += rate * cnt; Cy[s, pad] += cnt
    return Sy, Cy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="yelp")
    ap.add_argument("--seeds", default="42,123,7,11,99")
    ap.add_argument("--variants", default="A5,A7,A8")
    ap.add_argument("--checkpoints", default="5,25,200")
    ap.add_argument("--strengths", default="1,3")
    ap.add_argument("--poisons", default="identical,diverse")
    ap.add_argument("--n_targets", type=int, default=300)
    ap.add_argument("--n_honest", type=int, default=70)
    ap.add_argument("--n_sybil", type=int, default=30)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default="results/utility_eval2.json")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    cps = [int(x) for x in a.checkpoints.split(",")]
    n_h, n_s = a.n_honest, a.n_sybil; n = n_h + n_s
    tag = os.environ.get("TAMAIM_TAG", "")
    rows = []
    for seed in seeds:
        if a.selftest:
            r = np.random.RandomState(seed)
            shards = [[{"business_id": "b%d" % r.randint(800), "stars": float(r.randint(1, 6)),
                        "useful": int(r.randint(0, 3))} for _ in range(r.randint(300, 900))]
                      for _ in range(n_h)]
        else:
            shards = U.load_shards(a.dataset, n_h)
        n_items, S, C, test, tr_items = U.split_and_stats(shards, seed)
        cands = U.make_candidates(n_items, test, tr_items, seed)
        targets = np.random.RandomState(seed + 999).choice(n_items, min(a.n_targets, n_items), replace=False)
        sr = (set(int(i) for i in np.random.RandomState(seed).choice(n, n_s, replace=False))
              if a.selftest else U.sybilrank_flags(n, n_h, n_s, seed))
        rand = set(int(i) for i in np.random.RandomState(seed + 17).choice(n, n_s, replace=False))
        quar = {}
        for v in a.variants.split(","):
            p = "results/quar_%s_s%d%s.json" % (v, seed, tag)
            if a.selftest:
                quar[v] = {int(i): 2 for i in range(n_h, n)}
            elif os.path.exists(p):
                quar[v] = {int(k): int(e) for k, e in json.load(open(p))["quar_ep"].items()}
            else:
                print("MISSING", p)
        for pk_ in a.poisons.split(","):
            for c in [float(x) for x in a.strengths.split(",")]:
                Sy, Cy = make_poison(pk_, n_s, n_items, targets, c, seed)
                S_all = np.vstack([S, Sy]); C_all = np.vstack([C, Cy])
                def add(method, ckpt, nd_pk, nflag=None, fp=None):
                    rows.append(dict(seed=seed, poison=pk_, strength=c, method=method, ckpt=ckpt,
                                     ndcg=nd_pk[0], poison_at10=nd_pk[1], flagged=nflag, fp=fp))
                def pooled(flag):
                    keep = np.array([i not in flag for i in range(n)])
                    return rank_eval(score_from(S_all[keep].sum(0), C_all[keep].sum(0)), cands, targets, seed)
                add("fedavg", None, pooled(set()), 0, 0)
                add("oracle", None, pooled(set(range(n_h, n))), n_s, 0)
                add("random-removal", None, pooled(rand), n_s, len([i for i in rand if i < n_h]))
                add("sybilrank", None, pooled(sr), n_s, len([i for i in sr if i < n_h]))
                V, obs, mu = agent_vectors(S_all, C_all)
                add("median", None, rank_eval(masked_agg(V, obs, mu, "median"), cands, targets, seed))
                add("trimmed-mean", None, rank_eval(masked_agg(V, obs, mu, "trim"), cands, targets, seed))
                w = foolsgold_weights(V, mu)
                add("foolsgold", None, rank_eval(score_from((w[:, None] * S_all).sum(0), (w[:, None] * C_all).sum(0)),
                                                 cands, targets, seed), None, None)
                for v, qe in quar.items():
                    for t in cps:
                        fl = set(i for i, e in qe.items() if e <= t)
                        add("TA-MAIM:" + v, t, pooled(fl), len(fl), len([i for i in fl if i < n_h]))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rows, open(a.out, "w"))
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["poison"], r["strength"], r["method"], r["ckpt"])].append(r)
    print("\nNDCG@10 mean +- std over %d seeds | poison@10 | honest flagged" % len(seeds))
    for key in sorted(agg, key=lambda k: (k[0], k[1], str(k[2]), k[3] or 0)):
        rs = agg[key]
        nd = np.array([r["ndcg"] for r in rs]); pk = np.array([r["poison_at10"] for r in rs])
        fps = [r["fp"] for r in rs if r["fp"] is not None]
        print("%-9s c=%-3s %-15s ep=%-4s | %.4f +- %.4f | %.3f | %s" % (
            key[0], key[1], key[2], key[3] if key[3] else "-", nd.mean(), nd.std(), pk.mean(),
            "%.1f" % np.mean(fps) if fps else "n/a"))


if __name__ == "__main__":
    main()
