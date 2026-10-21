import json, glob, collections
import numpy as np
rows = []
for f in sorted(glob.glob("results/ubase2_amz/s*.json")): rows += json.load(open(f))
by = collections.defaultdict(dict)
for r in rows: by[(r["poison"], r["strength"], r["ckpt"], r["method"])][r["seed"]] = r["ndcg"]
T = 2.776  # t(0.975, df=4)
base = ["fedavg", "krum", "multikrum", "normclip", "median", "foolsgold", "oracle"]
print("TA-MAIM minus baseline, paired over 5 seeds: mean [95% CI] (seeds with TA-MAIM >= baseline)")
for ck in (75, 200):
    for poison in ("identical", "diverse"):
        for c in (1.0, 3.0):
            ta = by[(poison, c, ck, "TA-MAIM")]
            print("\n%s c=%d ep=%d" % (poison, c, ck))
            for b in base:
                bb = by[(poison, c, ck, b)]
                d = np.array([ta[s] - bb[s] for s in sorted(ta)])
                m = d.mean(); h = T * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
                print("  vs %-10s %+.4f [%+.4f, %+.4f]  (%d/5)" % (b, m, m - h, m + h, int((d >= -1e-12).sum())))
