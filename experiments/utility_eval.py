"""
Downstream utility evaluation for TA-MAIM (item recommendation, NDCG@10).

Honest agents hold real review shards. Each uploads item sufficient statistics
(sum of stars, review count). Sybils upload poisoned statistics promoting a
fixed set of target items. The server aggregates the uploads of the agents a
filtering rule KEEPS and ranks items by a Bayesian-mean score. Utility is
NDCG@10 on held-out reviews of the honest agents (sampled evaluation: held-out
items + random negatives, 100 candidates, identical for every method).

Filtering rules compared (all on the same seed / data split / poison):
  none      keep everyone (plain averaging)
  random    drop n_sybil random agents (control: is it just "less data"?)
  sybilrank drop the n_sybil lowest SybilRank scores (n_sybil is given to it)
  oracle    drop exactly the true Sybils (upper bound)
  <variant> drop agents TA-MAIM had quarantined by episode t
            (read from results/quar_<variant>_s<seed>.json)
"""
import os, sys, json, argparse
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np

ATTACK_OF = {"A5": "static", "A7": "collusion", "A8": "collusion, CATS off"}


def load_shards(dataset, n_honest):
    if dataset == "amazon":
        from data.amazon.amazon_loader import AmazonDataLoader
        dl = AmazonDataLoader()
        return [[dict(r, business_id=r["product_id"], useful=r.get("helpful_votes", 0))
                 for r in dl.agent_data[i]] for i in range(n_honest)]
    if dataset != "yelp":
        raise SystemExit("only yelp and amazon are supported here")
    from data.yelp.yelp_loader import YelpDataLoader
    dl = YelpDataLoader()
    return [dl.agent_data[i] for i in range(n_honest)]


def split_and_stats(shards, seed, test_frac=0.2, m_prior=5.0):
    items = sorted({r["business_id"] for s in shards for r in s})
    idx = {b: k for k, b in enumerate(items)}
    n_items, n_h = len(items), len(shards)
    S = np.zeros((n_h, n_items)); C = np.zeros((n_h, n_items))
    test, train_items = [], []
    for a, shard in enumerate(shards):
        rng = np.random.RandomState(seed * 1000 + a)
        perm = rng.permutation(len(shard))
        n_te = max(5, int(test_frac * len(shard)))
        te = [shard[i] for i in perm[:n_te]]
        tr = [shard[i] for i in perm[n_te:]]
        for r in tr:
            k = idx[r["business_id"]]
            S[a, k] += float(r["stars"]); C[a, k] += 1
        rel = {}
        for r in te:
            k = idx[r["business_id"]]
            g = float(r["stars"]) * (1 + int(r.get("useful", 0)))
            rel[k] = max(rel.get(k, 0.0), g)
        test.append(rel)
        train_items.append(set(np.nonzero(C[a])[0].tolist()))
    return n_items, S, C, test, train_items


def make_candidates(n_items, test, train_items, seed, n_cand=100, max_pos=30):
    cands = []
    for a, rel in enumerate(test):
        rng = np.random.RandomState(seed * 7919 + a)
        pos = list(rel.keys())
        if len(pos) > max_pos:
            pos = list(rng.choice(pos, max_pos, replace=False))
        banned = set(train_items[a]) | set(rel.keys())
        negs = []
        while len(negs) < n_cand - len(pos):
            k = int(rng.randint(n_items))
            if k not in banned and k not in negs:
                negs.append(k)
        c = np.array(pos + negs)
        g = np.array([rel.get(int(k), 0.0) for k in c])
        cands.append((c, g))
    return cands


def ndcg_at_k(scores, gains, k, jitter):
    order = np.argsort(-(scores + jitter))[:k]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((gains[order] * disc[:len(order)]).sum())
    ideal = np.sort(gains)[::-1][:k]
    idcg = float((ideal * disc[:len(ideal)]).sum())
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(S_all, C_all, keep, cands, targets, seed, m_prior=5.0, k=10):
    S = S_all[keep].sum(0); C = C_all[keep].sum(0)
    C = np.maximum(C, 0.0); S = np.clip(S, 0.0, 5.0 * C)
    mu = S.sum() / max(C.sum(), 1.0)
    score = (S + m_prior * mu) / (C + m_prior)
    tset = set(int(t) for t in targets)
    nd, pk = [], []
    for a, (c, g) in enumerate(cands):
        rng = np.random.RandomState(seed * 31 + a)
        jit = rng.uniform(0, 1e-9, size=len(c))
        sc = score[c]
        nd.append(ndcg_at_k(sc, g, k, jit))
        top = c[np.argsort(-(sc + jit))[:k]]
        pk.append(np.mean([int(t) in tset for t in top]))
    return float(np.mean(nd)), float(np.mean(pk))


def sybilrank_flags(n, n_h, n_s, seed):
    from data.topology import build_topology, params
    g, _, _ = build_topology(n_h, n_s, seed, **params())
    W = np.zeros((n, n))
    if isinstance(g, dict):
        for i, nb in g.items():
            for j, w in nb.items():
                W[int(i), int(j)] = float(w)
    else:
        W = np.array(g, dtype=float)
    W = np.maximum(W, W.T)
    deg = W.sum(1); deg[deg == 0] = 1.0
    rng = np.random.RandomState(seed + 5)
    seeds = rng.choice(n_h, max(1, int(round(0.05 * n_h))), replace=False)
    t = np.zeros(n); t[seeds] = 1.0 / len(seeds)
    for _ in range(int(np.ceil(np.log2(n)))):
        t = (W / deg[:, None]).T @ t
    score = t / deg
    flagged = np.argsort(score)[:n_s]
    return set(int(i) for i in flagged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="yelp")
    ap.add_argument("--seeds", default="42,123,7")
    ap.add_argument("--variants", default="A5,A7,A8")
    ap.add_argument("--checkpoints", default="5,25,100,200")
    ap.add_argument("--strengths", default="1,3")
    ap.add_argument("--eps", default="inf,10")
    ap.add_argument("--n_targets", type=int, default=300)
    ap.add_argument("--n_honest", type=int, default=70)
    ap.add_argument("--n_sybil", type=int, default=30)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default="results/utility_eval.json")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    variants = a.variants.split(",")
    cps = [int(x) for x in a.checkpoints.split(",")]
    strengths = [float(x) for x in a.strengths.split(",")]
    epss = [float(x) for x in a.eps.split(",")]
    n_h, n_s = a.n_honest, a.n_sybil
    n = n_h + n_s
    rows = []
    for seed in seeds:
        if a.selftest:
            rng = np.random.RandomState(seed)
            shards = [[{"business_id": "b%d" % rng.randint(800), "stars": float(rng.randint(1, 6)),
                        "useful": int(rng.randint(0, 3))} for _ in range(rng.randint(300, 900))]
                      for _ in range(n_h)]
        else:
            shards = load_shards(a.dataset, n_h)
        n_items, S, C, test, tr_items = split_and_stats(shards, seed)
        cands = make_candidates(n_items, test, tr_items, seed)
        trng = np.random.RandomState(seed + 999)
        targets = trng.choice(n_items, min(a.n_targets, n_items), replace=False)
        if a.selftest:
            sr_flag = set(int(i) for i in np.random.RandomState(seed).choice(n, n_s, replace=False))
        else:
            sr_flag = sybilrank_flags(n, n_h, n_s, seed)
        rrng = np.random.RandomState(seed + 17)
        rand_flag = set(int(i) for i in rrng.choice(n, n_s, replace=False))
        quar = {}
        for v in variants:
            p = "results/quar_%s_s%d%s.json" % (v, seed, os.environ.get("TAMAIM_TAG", ""))
            if a.selftest:
                quar[v] = {int(i): 2 for i in range(n_h, n)}
            elif os.path.exists(p):
                d = json.load(open(p))
                quar[v] = {int(k): int(e) for k, e in d["quar_ep"].items()}
            else:
                print("MISSING", p)
        for eps in epss:
            nrng = np.random.RandomState(seed + 4242)
            if np.isfinite(eps):
                Sn = S + nrng.laplace(0, 6.0 / eps, S.shape)
                Cn = C + nrng.laplace(0, 6.0 / eps, C.shape)
            else:
                Sn, Cn = S, C
            for c in strengths:
                Sy = np.zeros((n_s, n_items)); Cy = np.zeros((n_s, n_items))
                Sy[:, targets] = 5.0 * c; Cy[:, targets] = c
                S_all = np.vstack([Sn, Sy]); C_all = np.vstack([Cn, Cy])
                def run(flag):
                    keep = np.array([i not in flag for i in range(n)])
                    return evaluate(S_all, C_all, keep, cands, targets, seed)
                base = {"none": set(), "random": rand_flag, "sybilrank": sr_flag,
                        "oracle": set(range(n_h, n))}
                for name, fl in base.items():
                    nd, pk = run(fl)
                    rows.append(dict(seed=seed, eps=eps, strength=c, method=name, ckpt=None,
                                     ndcg=nd, poison_at10=pk, flagged=len(fl),
                                     fp=len([i for i in fl if i < n_h])))
                for v, qe in quar.items():
                    for t in cps:
                        fl = set(i for i, e in qe.items() if e <= t)
                        nd, pk = run(fl)
                        rows.append(dict(seed=seed, eps=eps, strength=c, method="TA-MAIM:" + v, ckpt=t,
                                         ndcg=nd, poison_at10=pk, flagged=len(fl),
                                         fp=len([i for i in fl if i < n_h])))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rows, open(a.out, "w"))
    # summary
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["eps"], r["strength"], r["method"], r["ckpt"])].append(r)
    print("\nNDCG@10 (mean +- std over %d seeds) | poison@10 | flagged | honest flagged" % len(seeds))
    for key in sorted(agg, key=lambda k: (k[0], k[1], str(k[2]), k[3] or 0)):
        rs = agg[key]
        nd = np.array([r["ndcg"] for r in rs]); pk = np.array([r["poison_at10"] for r in rs])
        print("eps=%-5s c=%-3s %-14s ep=%-4s | %.4f +- %.4f | %.3f | %5.1f | %4.1f" % (
            key[0], key[1], key[2], key[3] if key[3] else "-", nd.mean(), nd.std(),
            pk.mean(), np.mean([r["flagged"] for r in rs]), np.mean([r["fp"] for r in rs])))


if __name__ == "__main__":
    main()
