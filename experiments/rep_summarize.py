"""Independent replication (pre-registered 17 Oct 2026, preregistration_replication_2026-10-17): fresh seeds 1-5 only, 10 regimes of the detector-comparison table.
usage: python experiments/rep_summarize.py  -> prints table + criteria R1-R6, writes results/rep_results.json (does NOT touch fs_analysis.json)"""
import json, glob, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
from experiments import fs_analyze as FA
SEEDS = [1, 2, 3, 4, 5]
REG = "static collude qmatch band1 band2 band3 band4 camou bc10 bc20".split()
COL = [("TA-MAIM", "TA-MAIM(shadow)"), ("Q", "Q<0.5 [priv]"), ("Beta", "Beta<.5 [priv]"), ("S&C", "S>0.5&C>0.3 [raw]"),
       ("Struct", "Struct>0.5 [priv]"), ("Audit", "Periodic-audit25 CATS [raw]")]
res = {}
for rg in REG:
    files = [p for s in SEEDS for p in glob.glob("results/fs/%s_s%d.json" % (rg, s))]
    runs = [FA.run_file(p) for p in sorted(files)]
    res[rg] = {"n": len(runs), "cells": {nm: [float(np.mean([r[k]["tp"] for r in runs])), float(np.mean([r[k]["fp"] for r in runs]))] for nm, k in COL} if runs else {}}
json.dump(res, open("results/rep_results.json", "w"))
print("%-8s %2s " % ("regime", "n") + " ".join("%-12s" % c for c, _ in COL))
for rg in REG:
    c = res[rg]["cells"]
    print("%-8s %2d " % (rg, res[rg]["n"]) + " ".join("%-12s" % ("%.1f/%.1f" % tuple(c[n])) if c else "-" for n, _ in COL))
g = lambda rg, m: res[rg]["cells"][m]
ok = {}
ok["R1"] = all(g(r, "TA-MAIM")[0] >= 29 and g(r, "TA-MAIM")[1] <= 1.0 for r in ["static", "collude", "qmatch", "camou", "bc10"])
ok["R2"] = all(g(r, m)[0] <= 1 for r in ["band3", "band4"] for m in ["TA-MAIM", "Q", "Beta"])
ok["R3"] = all(g(r, "Q")[0] - g(r, "TA-MAIM")[0] >= 25 for r in ["band1", "band2"])
ok["R4"] = g("bc20", "TA-MAIM")[1] >= 10
ok["R5"] = g("camou", "S&C")[0] <= 28 or g("camou", "S&C")[1] >= g("camou", "TA-MAIM")[1] + 0.5
ok["R6"] = g("qmatch", "Q")[0] <= 1 and g("qmatch", "Beta")[0] <= 1
for k, v in ok.items(): print(k, "PASS" if v else "FAIL")
print("complete (50 runs):", all(res[r]["n"] == 5 for r in REG))
