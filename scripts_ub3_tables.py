import json, glob, collections, sys
import numpy as np
T = 2.776  # t(0.975, df=4)
CORE = ["fedavg","krum","multikrum","normclip","median","foolsgold","TA-MAIM","oracle"]
BASE = ["fedavg","krum","multikrum","normclip","median","foolsgold","oracle"]
def load(tag):
    rows = []
    for f in sorted(glob.glob("results/ubase3/%s_s*.json" % tag)): rows += json.load(open(f))
    by = collections.defaultdict(dict)
    for r in rows: by[(r["poison"], r["strength"], r["ckpt"], r["method"])][r["seed"]] = r["ndcg"]
    return rows, by
def ci(a, b):
    d = np.array([a[s] - b[s] for s in sorted(a)]); m = d.mean()
    h = T * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    return m, m - h, m + h
def lab(lo, hi):
    if lo > 1e-12: return "WIN "
    if hi < -1e-12: return "LOSS"
    return "tie "
for tag, name in (("yelp","YELP"),("amz","AMAZON")):
    rows, by = load(tag)
    seeds = sorted(set(r["seed"] for r in rows))
    print("\n##########", name, "seeds", seeds, "rows", len(rows))
    for sect, strengths in (("HEADLINE c=1,3", (1.0,3.0)), ("SENSITIVITY c=6,10", (6.0,10.0))):
        print("\n=====", name, sect, "=====")
        for poison in ("identical","diverse"):
            for c in strengths:
                for ck in (75,200):
                    line = []
                    for m in CORE:
                        v = by.get((poison,c,ck,m))
                        if v: line.append("%s %.4f" % (m, np.mean(list(v.values()))))
                    print("%-9s c=%-2d ep=%-3d | %s" % (poison, c, ck, " | ".join(line)))
                    ta = by[(poison,c,ck,"TA-MAIM")]
                    out = []
                    for b in BASE:
                        m_, lo, hi = ci(ta, by[(poison,c,ck,b)])
                        out.append("%s %+.4f[%+.4f,%+.4f] %s" % (b, m_, lo, hi, lab(lo, hi)))
                    print("    TA-MAIM minus: " + " ; ".join(out))
    print("\n===== %s DIAGNOSTICS (ep=200; mean NDCG; paired diff vs oracle) =====" % name)
    for poison in ("identical","diverse"):
        for c in (1.0,3.0,6.0,10.0):
            ck = 200
            orc = by[(poison,c,ck,"oracle")]
            parts = ["oracle %.4f" % np.mean(list(orc.values()))]
            for m in ("median_nosybil","normclip_nosybil","foolsgold_nosybil"):
                m_, lo, hi = ci(by[(poison,c,ck,m)], orc)
                parts.append("%s %+.4f[%+.4f,%+.4f]" % (m, m_, lo, hi))
            print("%-9s c=%-2d | %s" % (poison, c, " ; ".join(parts)))
    print("\n===== %s COMBINATIONS (ep=200; paired diff) =====" % name)
    for poison in ("identical","diverse"):
        for c in (1.0,3.0,6.0,10.0):
            ck = 200
            ta = by[(poison,c,ck,"TA-MAIM")]
            parts = []
            for m in ("TA-MAIM+median","TA-MAIM+normclip"):
                x = by[(poison,c,ck,m)]
                a1 = ci(x, ta); a2 = ci(x, by[(poison,c,ck,m.split("+")[1])])
                parts.append("%s mean %.4f | minus TA-MAIM %+.4f[%+.4f,%+.4f] | minus %s %+.4f[%+.4f,%+.4f]" % (m, np.mean(list(x.values())), a1[0], a1[1], a1[2], m.split("+")[1], a2[0], a2[1], a2[2]))
            print("%-9s c=%-2d" % (poison, c)); [print("    " + p) for p in parts]
