import json
for new, old in (("results/ubase3_reg/yelp_s42.json","results/ubase2/s42.json"),("results/ubase3_reg/amz_s42.json","results/ubase2_amz/s42.json")):
    N = {(r["poison"],r["strength"],r["ckpt"],r["method"]): r["ndcg"] for r in json.load(open(new))}
    O = json.load(open(old))
    bad = sum(abs(N[(r["poison"],r["strength"],r["ckpt"],r["method"])]-r["ndcg"]) > 1e-12 for r in O)
    print(new, "old rows:", len(O), "mismatches:", bad)
