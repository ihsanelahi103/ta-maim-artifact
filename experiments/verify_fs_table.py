"""Recompute the detector-comparison table (paper Table 3) from raw shadow streams on the five main seeds 7,11,42,99,123 only
(fresh-seed files 1-5 are excluded) and diff against the values printed in the paper. usage: python experiments/verify_fs_table.py"""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
from experiments import fs_analyze as FA
SEEDS = [7, 11, 42, 99, 123]
COL = [("TA-MAIM", "TA-MAIM(shadow)"), ("Q", "Q<0.5 [priv]"), ("Beta", "Beta<.5 [priv]"), ("S&C", "S>0.5&C>0.3 [raw]"),
       ("Struct", "Struct>0.5 [priv]"), ("Audit", "Periodic-audit25 CATS [raw]")]
# paper values (Sybils/30, honest FPs/70) in COL order
PAPER = {
 "static":  [(30,0),(30,.2),(30,0),(0,.6),(30,4.0),(0,0)],
 "collude": [(30,.2),(30,0),(30,0),(30,1.0),(30,4.0),(30,0)],
 "qmatch":  [(30,.2),(0,0),(0,0),(30,.8),(30,4.0),(30,0)],
 "band1":   [(4.4,0),(30,.2),(30,0),(0,1.0),(30,4.0),(0,0)],
 "band2":   [(0,0),(30,.2),(30,0),(0,1.2),(30,4.0),(0,0)],
 "band3":   [(0,0),(0,.2),(0,0),(0,1.2),(30,4.0),(0,0)],
 "band4":   [(0,0),(0,.2),(0,0),(0,1.2),(30,4.0),(0,0)],
 "camou":   [(30,0),(30,0),(30,0),(25,.8),(30,4.0),(7,0)],
 "bc10":    [(30,.4),(0,0),(0,0),(30,1.2),(30,3.8),(30,.2)],
 "bc20":    [(30,16.8),(0,0),(0,0),(30,18.4),(30,23.0),(30,15.2)],
}
bad = 0
for rg, want in PAPER.items():
    files = [p for s in SEEDS for p in glob.glob("results/fs/%s_s%d.json" % (rg, s))]
    if len(files) < 5: print("%-8s only %d/5 seed files -> skipped" % (rg, len(files))); continue
    runs = [FA.run_file(p) for p in sorted(files)]
    row = []
    for (nm, k), (wt, wf) in zip(COL, want):
        tp = np.mean([r[k]["tp"] for r in runs]); fp = np.mean([r[k]["fp"] for r in runs])
        ok = abs(tp - wt) <= 0.05 + 1e-9 and abs(fp - wf) <= 0.05 + 1e-9; bad += (not ok)
        row.append("%s %.1f/%.2f%s" % (nm, tp, fp, "" if ok else " <-- paper %.1f/%.2f" % (wt, wf)))
    print("%-8s " % rg + " | ".join(row))
print("MISMATCHES:", bad)
