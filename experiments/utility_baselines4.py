"""Recent robust-aggregation baselines on the utility task (pre-registered:
claude/preregistration_recent_baselines_2026-10-13.md).  Same data, split, poison,
exposure model and checkpoints as utility_baselines3.py; reference rows are recomputed
with the SAME code so results can be regression-checked against results/ubase3.

  fltrust   FLTrust-style cosine trust (Cao et al., NDSS 2021): trust = ReLU(cos(g_i, g_0)),
            g_i rescaled to ||g_0||, per-item trust-weighted mean over observers.
            Server root g_0 = pooled data of 3 clean honest agents (an advantage no other
            method has).  g_i = V_i - mu (shrunk item scores minus global mean).
  flame     FLAME-style (Nguyen et al., USENIX Sec 2022): HDBSCAN on pairwise cosine
            distance (min_cluster_size=n//2+1, min_samples=1), keep majority cluster,
            clip to median L2 norm, per-item mean over admitted observers, Gaussian
            noise sigma = 0.001 * clip bound.  No hyper-parameter is tuned.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

LAMBDA = 0.001
ROOT_K = 3


def cos_mat(X, y=None):
    nrm = np.linalg.norm(X, axis=1); nrm[nrm == 0] = 1.0
    if y is None:
        return (X @ X.T) / np.outer(nrm, nrm)
    ny = np.linalg.norm(y) or 1.0
    return (X @ y) / (nrm * ny)


def fltrust_scores(Vv, obs, mu, V0):
    """Per-item FLTrust-style aggregate; returns (score_vector, trust_scores)."""
    X = Vv - mu; g0 = V0 - mu
    ts = np.maximum(cos_mat(X, g0), 0.0)
    n0 = np.linalg.norm(g0)
    ni = np.linalg.norm(X, axis=1); ni[ni == 0] = 1.0
    Xn = X * (n0 / ni)[:, None]
    W = obs * ts[:, None]                       # trust only counts where the agent observed the item
    den = W.sum(0)
    num = (W * Xn).sum(0)
    agg = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    return mu + agg, ts


def flame_scores(Vv, obs, mu, noise_rng):
    """Per-item FLAME-style aggregate; returns (score_vector, admitted_idx)."""
    from sklearn.cluster import HDBSCAN
    n = len(Vv)
    X = Vv - mu
    D = np.clip(1.0 - cos_mat(X), 0.0, 2.0)
    D = (D + D.T) / 2.0; np.fill_diagonal(D, 0.0)
    lab = HDBSCAN(min_cluster_size=n // 2 + 1, min_samples=1, metric="precomputed",
                  allow_single_cluster=True).fit(D.astype(np.float64)).labels_
    good = lab[lab >= 0]
    if len(good) == 0:
        adm = np.arange(n)
    else:
        top = np.bincount(good).argmax()
        adm = np.nonzero(lab == top)[0]
    nrm = np.linalg.norm(X, axis=1)
    e = float(np.median(nrm))
    sc = np.minimum(1.0, e / np.maximum(nrm, 1e-12))
    Xc = X * sc[:, None]
    O = obs[adm].astype(float)
    den = O.sum(0)
    agg = np.where(den > 0, (O * Xc[adm]).sum(0) / np.where(den > 0, den, 1.0), 0.0)
    agg = agg + noise_rng.normal(0.0, LAMBDA * e, size=agg.shape)
    return mu + agg, adm


def selftest():
    rng = np.random.RandomState(0)
    n_items, n_h, n_s = 400, 70, 30
    base = rng.normal(0, 0.5, n_items)
    X = base + rng.normal(0, 0.3, (n_h + n_s, n_items))
    X[n_h:, :60] += 3.0                              # Sybils promote items 0..59
    mu = 3.5
    Vv = mu + X; obs = np.ones_like(Vv, dtype=bool)
    V0 = mu + base + rng.normal(0, 0.1, n_items)     # clean root
    sc, ts = fltrust_scores(Vv, obs, mu, V0)
    print("FLTrust trust honest mean %.3f, sybil mean %.3f" % (ts[:n_h].mean(), ts[n_h:].mean()))
    opp = mu - (V0 - mu)
    _, ts2 = fltrust_scores(np.vstack([Vv, opp]), np.vstack([obs, obs[:1]]), mu, V0)
    assert ts2[-1] == 0.0, "opposite-direction vector must get zero trust"
    assert ts[:n_h].mean() > 0.3
    sc2, adm = flame_scores(Vv, obs, mu, np.random.RandomState(1))
    print("FLAME admitted %d agents; sybils admitted %d" % (len(adm), int((adm >= n_h).sum())))
    assert len(adm) >= (n_h + n_s) // 2 + 1
    # no-attack control: nearly everyone admitted, scores close to honest mean
    Vh = Vv[:n_h]; sc3, adm3 = flame_scores(Vh, obs[:n_h], mu, np.random.RandomState(1))
    print("FLAME no-sybil admitted %d/%d" % (len(adm3), n_h))
    print("selftest ok")


def main():
    import utility_eval as U
    import utility_eval2 as V
    from utility_baselines2 import normclip_w
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dataset", default="yelp")
    ap.add_argument("--attack", default="camouflage")
    ap.add_argument("--checkpoints", default="75,100,150,200")
    ap.add_argument("--strengths", default="1,3,6,10")
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
    root_ids = np.random.RandomState(seed + 4141).choice(n_h, ROOT_K, replace=False)
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
                def add(m, nd, **extra):
                    rows.append(dict(seed=seed, dataset=a.dataset, poison=pk, strength=c, ckpt=t,
                                     method=m, ndcg=float(nd[0]), poison_at10=float(nd[1]), f=f, **extra))
                def pooled(keep):
                    kk = np.array(keep)
                    return V.rank_eval(V.score_from(S_all[kk].sum(0), C_all[kk].sum(0)), cands, targets, seed)
                def wsum(kk, w):
                    return V.rank_eval(V.score_from((w[:, None] * S_all[kk]).sum(0), (w[:, None] * C_all[kk]).sum(0)), cands, targets, seed)
                Vv, obs, mu = V.agent_vectors(S_all[k], C_all[k])
                # reference rows (identical code to utility_baselines3 -> regression check)
                add("fedavg", pooled(ids))
                add("oracle", pooled(list(range(n_h))))
                add("TA-MAIM", pooled([i for i in ids if i not in qfl]))
                add("median", V.rank_eval(V.masked_agg(Vv, obs, mu, "median"), cands, targets, seed))
                add("normclip", wsum(k, normclip_w(Vv, mu)))
                # root vector from clean honest agents (same mu as the cell)
                Sr = S_all[root_ids].sum(0); Cr = C_all[root_ids].sum(0)
                V0 = (Sr + 5.0 * mu) / (Cr + 5.0)
                sc_f, ts = fltrust_scores(Vv, obs, mu, V0)
                syb = k >= n_h
                add("fltrust", V.rank_eval(sc_f, cands, targets, seed),
                    ts_honest=float(ts[~syb].mean()), ts_sybil=float(ts[syb].mean()) if syb.any() else None)
                nrng = np.random.RandomState(seed * 101 + int(c * 10) + t)
                sc_m, adm = flame_scores(Vv, obs, mu, nrng)
                add("flame", V.rank_eval(sc_m, cands, targets, seed),
                    admitted=int(len(adm)), sybil_admitted=int((k[adm] >= n_h).sum()))
                kh = np.arange(n_h)
                Vh, oh, muh = V.agent_vectors(S_all[kh], C_all[kh])
                Sr2 = S_all[root_ids].sum(0); Cr2 = C_all[root_ids].sum(0)
                V0h = (Sr2 + 5.0 * muh) / (Cr2 + 5.0)
                add("fltrust_nosybil", V.rank_eval(fltrust_scores(Vh, oh, muh, V0h)[0], cands, targets, seed))
                add("flame_nosybil", V.rank_eval(flame_scores(Vh, oh, muh, np.random.RandomState(seed * 101 + t))[0], cands, targets, seed))
    json.dump(rows, open(a.out, "w"))
    print("done seed", seed, a.dataset, len(rows), "rows")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
