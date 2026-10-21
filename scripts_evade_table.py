import re, glob, os, collections, statistics as st
cells = collections.defaultdict(list)
for f in sorted(glob.glob("logs/evade/E*_s*.log")):
    m = re.match(r"(E\d)_s(\d+)\.log", os.path.basename(f)); name, seed = m.group(1), int(m.group(2))
    t = open(f).read()
    g = lambda p: re.search(p, t)
    det = g(r"Sybil detected:\s*(\d+)/(\d+)"); fp = g(r"False positives:\s*(\d+)/(\d+)")
    f1 = g(r"Sybil-F1:\s*([-\d.]+)"); lat = g(r"Detection latency:\s*([^\n]*)")
    ts = g(r"Trust separation:\s*([-\d.]+)"); ep = g(r"DP epsilon:\s*([-\d.]+)")
    if not (det and fp and f1 and ts and ep):
        print("PARSE PROBLEM", f); continue
    cells[name].append(dict(seed=seed, det=int(det.group(1)), n=int(det.group(2)), fp=int(fp.group(1)), nh=int(fp.group(2)),
        f1=float(f1.group(1)), lat=(lat.group(1).strip() if lat else "n/a"), ts=float(ts.group(1)), eps=float(ep.group(1))))
band = {"E1": "U(0.35,0.48)", "E2": "U(0.50,0.55)", "E3": "U(0.65,1.00)"}
print("\ncell band | runs | Sybils detected | honest FP | F1 mean [min] | trustsep mean | final eps mean")
for k in sorted(cells):
    r = cells[k]; f1s = [x["f1"] for x in r]
    print("%s %s | %d | %d/%d | %d/%d | %.4f [%.4f] | %.3f | %.4f" % (k, band[k], len(r), sum(x["det"] for x in r), sum(x["n"] for x in r),
        sum(x["fp"] for x in r), sum(x["nh"] for x in r), st.mean(f1s), min(f1s), st.mean(x["ts"] for x in r), st.mean(x["eps"] for x in r)))
print("\nper run:")
for k in sorted(cells):
    for x in sorted(cells[k], key=lambda d: d["seed"]):
        print("  %s s%-3d det %d/%d FP %d/%d F1 %.4f ts %.3f latency: %s" % (k, x["seed"], x["det"], x["n"], x["fp"], x["nh"], x["f1"], x["ts"], x["lat"]))
