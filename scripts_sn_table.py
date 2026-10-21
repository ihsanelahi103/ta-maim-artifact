import json, glob, re, collections, statistics as st
rows = collections.defaultdict(list)
for f in sorted(glob.glob("results/quar_A[57]_s*_sn_*.json")):
    m = re.match(r"results/quar_(A\d)_s(\d+)_sn_d([\d.]+)_r([\d.]+)_p([\d.]+)_a(\d+)\.json", f)
    if not m: continue
    v, s, d, r, p, a = m.groups()
    j = json.load(open(f))
    q = {int(k): x for k, x in j["quar_ep"].items() if x is not None and x >= 0}
    syb = set(j["sybil_ids"]); n = 100; hon = set(range(n)) - syb
    missed = len(syb - set(q)); fp = len(set(q) & hon)
    lat = max(q[i] for i in syb if i in q) if missed == 0 else None
    rows[(j["attack"], d, r, p, a)].append((int(s), missed, fp, lat))
print("attack | honest_deg rewire density attack_edges | runs | Sybils missed | honest FP | first-full-detection ep (per seed)")
for k in sorted(rows):
    v = rows[k]; nr = len(v)
    lats = [x[3] for x in v]
    ls = ",".join("-" if x is None else str(x) for x in lats)
    print(k[0], "|", " ".join(k[1:]), "|", nr, "|", f"{sum(x[1] for x in v)}/{30*nr}", "|", f"{sum(x[2] for x in v)}/{70*nr}", "|", ls)
