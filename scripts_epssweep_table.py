import re, glob, os, collections, statistics as st
cells = collections.defaultdict(list)
for f in sorted(glob.glob("logs/epssweep/e*.log")):
    m = re.match(r"e([\d.]+)_(\w+)_s(\d+)\.log", os.path.basename(f))
    eps, atk, seed = float(m.group(1)), m.group(2), int(m.group(3))
    t = open(f).read()
    def g(p):
        r = re.search(p, t); return r
    det = g(r"Sybil detected:\s*(\d+)/(\d+)")
    fp  = g(r"False positives:\s*(\d+)/(\d+)")
    f1  = g(r"Sybil-F1:\s*([-\d.]+)")
    lat = g(r"Detection latency:\s*Ep\s*(\d+)")
    ts  = g(r"Trust separation:\s*([-\d.]+)")
    ep  = g(r"DP epsilon:\s*([-\d.]+)")
    cells[(eps, atk)].append(dict(seed=seed, det=int(det.group(1)), nsyb=int(det.group(2)),
        fp=int(fp.group(1)), nh=int(fp.group(2)), f1=float(f1.group(1)),
        lat=int(lat.group(1)) if lat else None, ts=float(ts.group(1)), eps=float(ep.group(1))))
print("eps_floor attack | runs | Sybils det | honest FP (runs w/ FP) | F1 mean+-sd [min] | latency mean (n) | trustsep mean | final eps mean")
for (eps, atk) in sorted(cells, key=lambda k: (k[1], -k[0])):
    r = cells[(eps, atk)]; n = len(r)
    f1s = [x["f1"] for x in r]; lats = [x["lat"] for x in r if x["lat"] is not None]
    print(f"{eps:<5} {atk:<10} | {n} | {sum(x['det'] for x in r)}/{sum(x['nsyb'] for x in r)} | "
          f"{sum(x['fp'] for x in r)}/{sum(x['nh'] for x in r)} ({sum(x['fp']>0 for x in r)}) | "
          f"{st.mean(f1s):.4f}+-{(st.pstdev(f1s)):.4f} [{min(f1s):.4f}] | "
          f"{(st.mean(lats) if lats else float('nan')):.1f} ({len(lats)}) | {st.mean(x['ts'] for x in r):.3f} | {st.mean(x['eps'] for x in r):.4f}")
print("\nper-run FP>0 or missed Sybils:")
for (eps, atk) in sorted(cells, key=lambda k: (k[1], -k[0])):
    for x in cells[(eps, atk)]:
        if x["fp"] > 0 or x["det"] < x["nsyb"]:
            print(f"  eps {eps} {atk} s{x['seed']}: det {x['det']}/{x['nsyb']} FP {x['fp']}/{x['nh']} F1 {x['f1']:.4f} lat {x['lat']} ts {x['ts']:.3f}")
