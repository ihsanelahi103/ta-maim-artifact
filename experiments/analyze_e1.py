"""Offline baselines D1-D6 on the privatized quality stream of each run vs. TA-MAIM (from quar json).
Rules fixed in preregistration_reviewer_experiments_2026-10-13 (D7 dropped: CATS code API not replayed; TA-MAIM full run IS the CATS result)."""
import json, glob, os, sys
import numpy as np
E = 200; SEEDS = [7, 11, 42, 99, 123]
SCN = ["static","camou","collude","qmatch","band1","band2","band3","band4","band5"]
THR = [0.3, 0.4, 0.5, 0.6]

def load(scn, s):
    q = json.load(open("results/quar_A5_s%d_%s.json" % (s, scn)))
    d = json.load(open("results/qdump_s%d_%s.json" % (s, scn)))
    recs = d["records"]
    eps = sorted(set(r[0] for r in recs))
    assert max(eps) >= 150, "episode counter looks wrong: %s" % eps[:5]
    n = 1 + max(max(int(k) for k in q["quar_ep"]) if q["quar_ep"] else 0, max(q["sybil_ids"]), max(r[2] for r in recs))
    P = np.full((n, E + 1), np.nan)
    for ep, _, a, raw, pr in recs:
        if ep <= E: P[a, ep] = pr          # last release in episode
    neigh = {int(k): v for k, v in d["neigh"].items()}
    return q, P, neigh

def d1(P, th, W=5):
    f = {}
    for a in range(P.shape[0]):
        for ep in range(3, E + 1):
            w = P[a, max(1, ep - W + 1):ep + 1]; w = w[~np.isnan(w)]
            if len(w) >= 3 and w.mean() < th: f[a] = ep; break
    return f

def d2(P, th, lam=0.1):
    f = {}
    for a in range(P.shape[0]):
        m = None; c = 0
        for ep in range(1, E + 1):
            v = P[a, ep]
            if np.isnan(v): continue
            m = v if m is None else (1 - lam) * m + lam * v; c += 1
            if ep >= 3 and c >= 3 and m < th: f[a] = ep; break
    return f

def d3(P, th=0.5, g=0.9):
    f = {}
    for a in range(P.shape[0]):
        al = be = 1.0; c = 0
        for ep in range(1, E + 1):
            v = P[a, ep]
            if np.isnan(v): continue
            al = g * al + v; be = g * be + (1 - v); c += 1
            if c >= 10 and al / (al + be) < th: f[a] = ep; break
    return f

def corr_ep(P, neigh, a, ep, win=15):
    lo = max(1, ep - win + 1); x = P[a, lo:ep + 1]; cs = []
    for j in neigh.get(a, []):
        y = P[j, lo:ep + 1]; m = ~np.isnan(x) & ~np.isnan(y)
        if m.sum() < 4: continue
        xs, ys = x[m], y[m]
        if xs.std() < 1e-9 or ys.std() < 1e-9: continue
        cs.append(np.corrcoef(xs, ys)[0, 1])
    return np.mean(cs) if len(cs) >= 3 else np.nan

def d4(P, neigh, th=0.3):
    f = {}
    for a in range(P.shape[0]):
        if len(neigh.get(a, [])) < 5: continue
        for ep in range(4, E + 1):
            c = corr_ep(P, neigh, a, ep)
            if not np.isnan(c) and c > th: f[a] = ep; break
    return f

def d5(P, neigh, th=0.5):
    f = {}; nb = {a: set(v) for a, v in neigh.items()}
    for a, s in nb.items():
        d = len(s)
        if d < 5: continue
        e = sum(1 for i in s for j in s if i < j and j in nb.get(i, ()))
        if e / (d * (d - 1) / 2) > th: f[a] = 3
    return f

def score(flags, q, nS_total, nH):
    syb = set(q["sybil_ids"]); qe = {int(k): v for k, v in q["quar_ep"].items()}
    tp = [a for a in flags if a in syb]; fp = [a for a in flags if a not in syb]
    trunc = [a for a in syb if a not in flags and a in qe]    # TA-MAIM had already removed the stream
    lat = [flags[a] for a in tp]
    return dict(tp=len(tp), fp=len(fp), trunc=len(trunc), lat=lat)

def main():
    rows = []
    for scn in SCN:
        res = {}
        ok = True
        for s in SEEDS:
            try: q, P, nb = load(scn, s)
            except Exception as e:
                print("skip", scn, s, e); ok = False; continue
            nS = len(q["sybil_ids"]); nH = P.shape[0] - nS
            qe = {int(k): v for k, v in q["quar_ep"].items()}
            res.setdefault("TA-MAIM", []).append(score(qe, q, nS, nH))
            for th in THR:
                res.setdefault("D1 q<%.1f" % th, []).append(score(d1(P, th), q, nS, nH))
                res.setdefault("D2 ewma<%.1f" % th, []).append(score(d2(P, th), q, nS, nH))
            res.setdefault("D3 beta<0.5", []).append(score(d3(P), q, nS, nH))
            f4 = d4(P, nb); f5 = d5(P, nb); f1 = d1(P, 0.5)
            res.setdefault("D4 corr>0.3", []).append(score(f4, q, nS, nH))
            res.setdefault("D5 clust>0.5", []).append(score(f5, q, nS, nH))
            f6 = dict(f1)
            for a, e in f4.items(): f6[a] = min(e, f6.get(a, 10**9))
            res.setdefault("D6 q0.5|corr", []).append(score(f6, q, nS, nH))
        if not res: continue
        print("\n=== %s (n=%d seeds; Sybils 30, honest 70) ===" % (scn, len(res["TA-MAIM"])))
        print("%-14s %8s %8s %8s %10s" % ("detector", "TP/30", "FP/70", "trunc*", "med.lat"))
        for k, v in res.items():
            tp = np.mean([r["tp"] for r in v]); fp = np.mean([r["fp"] for r in v]); tr = np.mean([r["trunc"] for r in v])
            lat = [x for r in v for x in r["lat"]]
            print("%-14s %8.1f %8.1f %8.1f %10s" % (k, tp, fp, tr, ("%.0f" % np.median(lat)) if lat else "-"))
    print("\n*trunc = Sybils NOT flagged by the baseline whose stream TA-MAIM cut off by quarantining them first (baseline had no chance to flag later); lower bound counts them as misses.")
if __name__ == "__main__":
    main()
