import json, glob, os, sys, datetime, collections
FIX = datetime.datetime(2026, 10, 9, 0, 31).timestamp()
pat = sys.argv[1]
N_AGENTS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
files = sorted(glob.glob(pat))
post = [f for f in files if os.path.getmtime(f) > FIX]
print(f"{len(files)} files match, {len(post)} are post-fix, {len(files)-len(post)} excluded as stale")
d0 = json.load(open(post[0]))
print("keys:", list(d0.keys()), "| quar_ep type:", type(d0["quar_ep"]).__name__,
      "| sample:", str(d0["quar_ep"])[:120], "| sybil_ids sample:", str(d0["sybil_ids"])[:60])
def quarantined(q):
    if isinstance(q, dict):
        return {int(k) for k, v in q.items() if v is not None and v >= 0}
    return {i for i, v in enumerate(q) if v is not None and v >= 0}
rows = collections.defaultdict(list)
for f in post:
    d = json.load(open(f)); syb = set(int(x) for x in d["sybil_ids"]); q = quarantined(d["quar_ep"])
    key = os.path.basename(f).split("_s")[0] + "|" + d.get("attack", "?")
    rows[key].append((len(syb), len(syb - q), len(q - syb), N_AGENTS - len(syb), f))
for k, v in sorted(rows.items()):
    S = sum(r[0] for r in v); M = sum(r[1] for r in v); FP = sum(r[2] for r in v); H = sum(r[3] for r in v)
    print(f"{k}: runs {len(v)} | Sybils {S} missed {M} | honest {H} FP {FP} ({sum(r[2]>0 for r in v)} runs with FP)")
    for r in v:
        if r[1] or r[2]: print("    ", os.path.basename(r[4]), "missed", r[1], "FP", r[2])
