"""Real-data-derived quality signal + end-to-end ranking (review-2 experiment E3). See preregistration_review2 (E3).
Chronological split inside every honest shard: oldest 60% = platform HISTORY (reference), next 20% = STREAM released over the
episodes, newest 20% = TEST (future; never used for quality or pooling).  Quality of a stream batch =
mean over its reviews whose item has >=M history ratings of 1-|stars-ref_item|/4 (no Sybil labels, no future data); other batches
carry the previous value (initial Q0)."""
import os, sys, pickle
import numpy as np
sys.path.insert(0, os.getcwd())

M_MIN = 3
SIGNAL = os.environ.get("RS_SIGNAL", "dev")
Q0 = 0.75
CACHE = {"yelp": ("data/processed/yelp/yelp_graph_cache_100.pkl", "business_id", "useful"),
         "amazon": ("data/processed/amazon/amazon_graph_cache.pkl", "product_id", "helpful_votes")}


def load(dataset, n_h=70, n_s=30, seed=7, episodes=200, n_targets=300, fab_per_ep=3, fab_stars=5.0, tmax=3.0, tmin_hist=5):
    path, ik, gk = CACHE[dataset]
    ad = pickle.load(open(path, "rb"))["agent_data"]
    shards = []
    for a in range(n_h):
        rs = sorted(ad[a], key=lambda r: str(r.get("date", "")))
        n = len(rs); i1, i2 = int(0.6 * n), int(0.8 * n)
        shards.append(dict(hist=rs[:i1], stream=rs[i1:i2], test=rs[i2:]))
    items = sorted({r[ik] for s in shards for part in ("hist", "stream", "test") for r in s[part]})
    idx = {b: k for k, b in enumerate(items)}; n_items = len(items)
    HS = np.zeros(n_items); HC = np.zeros(n_items)
    for s in shards:
        for r in s["hist"]:
            k = idx[r[ik]]; HS[k] += float(r["stars"]); HC[k] += 1
    ref = np.where(HC >= M_MIN, HS / np.maximum(HC, 1), np.nan)
    rng = np.random.RandomState(seed + 555)
    cand = np.nonzero((HC >= tmin_hist) & (ref <= tmax))[0]
    targets = rng.choice(cand, min(n_targets, len(cand)), replace=False) if len(cand) else np.array([], dtype=int)
    src = rng.choice(n_h, n_s, replace=False)
    # per-agent per-episode review lists: (item index, stars)
    def chunks(rs):
        out = [[] for _ in range(episodes + 1)]
        n = len(rs)
        for j, r in enumerate(rs):
            out[1 + (j * episodes) // max(n, 1)].append((idx[r[ik]], float(r["stars"])))
        return out
    real = [chunks(s["stream"]) for s in shards]
    batches = [[list(b) for b in real[a]] for a in range(n_h)]
    fab = [[[] for _ in range(episodes + 1)] for _ in range(n_s)]
    for t in range(1, episodes + 1):
        pool = np.random.RandomState(seed * 1000 + t).choice(targets, min(20, len(targets)), replace=False) if len(targets) else []
        for s in range(n_s):
            r2 = np.random.RandomState(seed * 100000 + s * 1000 + t)
            for k in (r2.choice(pool, min(fab_per_ep, len(pool)), replace=False) if len(pool) else []):
                fab[s][t].append((int(k), fab_stars))
    sy_batches = [[list(real[int(src[s])][t]) + fab[s][t] for t in range(episodes + 1)] for s in range(n_s)]
    return dict(shards=shards, items=items, idx=idx, n_items=n_items, ref=ref, HC=HC, targets=targets, src=src,
                batches=batches + sy_batches, fab=fab, real=real, n_h=n_h, n_s=n_s, episodes=episodes, ik=ik, gk=gk,
                hist_S=HS, hist_C=HC)


def quality_table(D):
    ref = D["ref"]; E = D["episodes"]; n = D["n_h"] + D["n_s"]
    Q = np.full((n, E + 1), Q0); cov = np.zeros((n, E + 1), dtype=bool)
    for a in range(n):
        last = Q0
        for t in range(1, E + 1):
            if SIGNAL == "ext":      # signal B (pre-registered 16 Oct): 1 - share of five-star reviews in the batch
                ag = [0.0 if st >= 5.0 else 1.0 for k, st in D["batches"][a][t]]
            else:                    # signal A (default, used for all earlier E3 results)
                ag = [1.0 - abs(st - ref[k]) / 4.0 for k, st in D["batches"][a][t] if not np.isnan(ref[k])]
            if ag: last = float(np.mean(ag)); cov[a, t] = True
            Q[a, t] = last
    return Q, cov


def agent_cum(D, a, upto, with_fab=True):
    """item-sum vectors (S, C) of agent a's stream reviews in episodes 1..upto (Sybil: includes fabricated)."""
    S = np.zeros(D["n_items"]); C = np.zeros(D["n_items"])
    for t in range(1, upto + 1):
        for k, st in D["batches"][a][t]:
            S[k] += st; C[k] += 1
    return S, C


def ranking_setup(D, seed):
    import utility_eval as U
    ik, gk = D["ik"], D["gk"]
    test, train_items = [], []
    for a in range(D["n_h"]):
        rel = {}
        for r in D["shards"][a]["test"]:
            k = D["idx"][r[ik]]; g = float(r["stars"]) * (1 + int(r.get(gk, 0) or 0))
            rel[k] = max(rel.get(k, 0.0), g)
        test.append(rel)
        ti = {D["idx"][r[ik]] for r in D["shards"][a]["hist"]} | {D["idx"][r[ik]] for r in D["shards"][a]["stream"]}
        train_items.append(ti)
    return U.make_candidates(D["n_items"], test, train_items, seed)
