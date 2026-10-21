"""Transient-evidence analysis of the audit-period sweep (18 Oct 2026; descriptive, no tuning, uses only the existing shadow streams of seeds 7,11,42,99,123).
Question: when a periodic audit (period P, evaluations at P,2P,...) flags fewer Sybils than the in-loop schedule (P=1), is it because the CATS rule fires
for a Sybil only during short windows (transient evidence) that a coarse grid can miss?
Per Sybil a, F_a = episodes >=3 in which the CATS rule (S>.5 & C>.3 & suspicion>.55, raw quality) holds for a.
  ever      = |F_a| >= 1 (= flagged at P=1)
  exact     = F_a meets the grid {P,2P,..}  (what the audit sweep reports)
  phase_avg = expected flagged if the audit phase were uniform over 0..P-1
  guaranteed= longest contiguous run in F_a >= P (flagged for every phase)
  phase_dep = Sybils with 0 < hit probability < 1 (detection depends on audit phase)
usage: python experiments/transient_audit.py [regime ...]   -> results/transient_audit.json + printed table"""
import json, glob, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
from experiments import fs_analyze as FA
SEEDS = [7, 11, 42, 99, 123]; PERIODS = [1, 2, 5, 10, 25, 50]
REGS = ["qmatch", "collude", "qm_ae10", "qm_l050", "camou", "bc10", "bc20"]
N = FA.N


def longest_run(ep_set, E):
    best = cur = 0
    for e in range(3, E + 1):
        cur = cur + 1 if e in ep_set else 0; best = max(best, cur)
    return best


def one(path):
    d, E, rec, g1, diff = FA.load(path)
    FA.COMM.clear(); FA.COMM.update(d.get("comm") or [])
    syb = sorted(d["sybil_ids"] or range(70, 100))
    S, C, SUS = FA.cats_replay(rec, E, g1, diff, 3)
    cond = (S > 0.5) & (C > 0.3) & (SUS > 0.55)
    res = {"E": E, "ever": 0, "n_ep": [], "L": [], "P": {}}
    F = {a: {e for e in range(3, E + 1) if cond[e][a]} for a in syb}
    res["ever"] = sum(1 for a in syb if F[a]); res["n_ep"] = [len(F[a]) for a in syb if F[a]]; res["L"] = [longest_run(F[a], E) for a in syb if F[a]]
    for P in PERIODS:
        grid = set(range(P, E + 1, P)); ex = pa = gu = pd = 0
        for a in syb:
            if not F[a]: continue
            ex += bool(F[a] & grid)
            hits = sum(bool(F[a] & set(range(o if o else P, E + 1, P))) for o in range(P)); p = hits / P
            pa += p; pd += (0 < p < 1); gu += (longest_run(F[a], E) >= P)
        res["P"][P] = dict(exact=ex, phase_avg=pa, guaranteed=gu, phase_dep=pd)
    return res


def main():
    regs = sys.argv[1:] or REGS; out = {}
    for rg in regs:
        files = [p for s in SEEDS for p in glob.glob("results/fs/%s_s%d.json" % (rg, s))]
        if not files: print(rg, "no files"); continue
        runs = [one(p) for p in sorted(files)]; out[rg] = runs
        print("\n=== %s: n=%d seed runs; Sybils (of 30) whose rule ever fires: %.1f; median firing episodes %.0f, median longest run %.0f ===" % (
            rg, len(runs), np.mean([r["ever"] for r in runs]), np.median([x for r in runs for x in r["n_ep"]] or [np.nan]), np.median([x for r in runs for x in r["L"]] or [np.nan])))
        print("%-4s %7s %7s %9s %11s %9s" % ("P", "ever", "exact", "phase_avg", "guaranteed", "phase_dep"))
        for P in PERIODS:
            v = [r["P"][P] for r in runs]
            print("%-4d %7.1f %7.1f %9.1f %11.1f %9.1f" % (P, np.mean([r["ever"] for r in runs]), np.mean([x["exact"] for x in v]), np.mean([x["phase_avg"] for x in v]),
                  np.mean([x["guaranteed"] for x in v]), np.mean([x["phase_dep"] for x in v])))
    json.dump({k: [{**r, "P": {str(p): v for p, v in r["P"].items()}} for r in runs] for k, runs in out.items()}, open("results/transient_audit.json", "w"))
main()
