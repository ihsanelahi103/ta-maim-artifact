"""Audit-period sweep (reviewer request: is the in-loop benefit more than checking more often?).
Replays the SAME CATS rule (S>0.5 & C>0.3 & suspicion>0.55, raw quality) on the same uncensored shadow-mode streams
results/fs/<regime>_s<seed>.json, but evaluated only every P episodes (P = 1 is the in-loop schedule).
Reports Sybils flagged, honest flagged, median first-flag episode, exposure (fraction of Sybil MineData calls before the flag),
and the cost: number of CATS evaluations per 200 episodes and measured seconds per evaluation.
usage: python experiments/fs_audit_sweep.py [regime ...]   (default regimes below)  -> results/fs_audit_sweep.json
"""
import json, glob, os, sys, time
import numpy as np
sys.path.insert(0, os.getcwd())
from experiments import fs_analyze as FA

PERIODS = [1, 2, 5, 10, 25, 50]
REGS = ["qmatch", "collude", "qm_ae10", "qm_l050", "camou", "bc10", "bc20"]
N = FA.N


def one(path):
    d, E, rec, g1, diff = FA.load(path)
    FA.COMM.clear(); FA.COMM.update(d.get("comm") or [])
    syb = set(d["sybil_ids"] or range(70, 100))
    T = None
    if d.get("sybil_T") and d["sybil_T"].get("transitions"):
        T = {int(k): int(v) for k, v in d["sybil_T"]["transitions"].items()}
    calls = np.zeros((N, E + 2))
    for r in rec:
        if int(r[0]) <= E: calls[int(r[2]), int(r[0])] += 1
    t0 = time.time()
    S, C, SUS = FA.cats_replay(rec, E, g1, diff, 3)          # raw quality column
    sec_per_eval = (time.time() - t0) / E
    cond = (S > 0.5) & (C > 0.3) & (SUS > 0.55)
    out = {}
    for P in PERIODS:
        per = np.zeros_like(cond)
        for ep in range(P, E + 1, P): per[ep] = cond[ep]
        sc = FA.score(FA.first(per, E), syb, calls, E, T)
        sc["evals"] = len(range(P, E + 1, P)); sc["sec_per_eval"] = sec_per_eval
        out[P] = sc
    return out


def main():
    regs = sys.argv[1:] or REGS
    allres = {}
    for rg in regs:
        files = sorted(glob.glob("results/fs/%s_s[0-9]*.json" % rg))
        if not files: continue
        runs = [one(p) for p in files]
        allres[rg] = runs
        print("\n=== %s: n=%d runs (30 Sybils / 70 honest), CATS rule on raw quality ===" % (rg, len(runs)))
        print("%-7s %6s %6s %6s %7s %7s %6s %9s" % ("period", "TP/30", "FP/70", "30/30", "medLat", "expo", "evals", "sec/eval"))
        for P in PERIODS:
            v = [r[P] for r in runs]
            lat = [x for r in v for x in r["lat"]]
            print("%-7d %6.1f %6.1f %6d %7.1f %7.3f %6d %9.3f" % (
                P, np.mean([r["tp"] for r in v]), np.mean([r["fp"] for r in v]), sum(1 for r in v if r["tp"] == 30),
                np.median(lat) if lat else float("nan"), np.mean([r["expo"] for r in v]), v[0]["evals"], v[0]["sec_per_eval"]))
    os.makedirs("results", exist_ok=True)
    json.dump({k: [{str(p): v for p, v in r.items()} for r in runs] for k, runs in allres.items()}, open("results/fs_audit_sweep.json", "w"))


if __name__ == "__main__":
    main()
