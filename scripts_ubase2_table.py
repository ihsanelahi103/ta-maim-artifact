import json, glob, collections
import numpy as np
rows = []
for f in sorted(glob.glob("results/ubase2/s*.json")): rows += json.load(open(f))
agg = collections.defaultdict(list)
for r in rows: agg[(r["poison"], r["strength"], r["ckpt"], r["method"])].append(r)
order = ["fedavg", "krum", "multikrum", "normclip", "median", "foolsgold", "TA-MAIM", "oracle"]
print("seeds:", sorted(set(r["seed"] for r in rows)))
for poison in ("identical", "diverse"):
    for c in (1.0, 3.0):
        for ck in (75, 100, 150, 200):
            line = []
            for m in order:
                v = agg.get((poison, c, ck, m))
                if v: line.append("%s %.4f+-%.4f" % (m, np.mean([x["ndcg"] for x in v]), np.std([x["ndcg"] for x in v])))
            print("%-9s c=%d ep=%-3d | %s" % (poison, c, ck, " | ".join(line)))
