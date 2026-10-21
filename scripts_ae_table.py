import re, glob, statistics as st
G=[1,2,5,10,20]; S=[7,11,42,99,123]
print("g | SybilRank F1 per seed (7,11,42,99,123) | mean")
for g in G:
    v=[]
    for s in S:
        t=open(f"logs/ae/sr_g{g}_s{s}.log").read()
        m=re.search(r"F1@true-k:\s+([0-9.]+)", t)
        v.append(float(m.group(1)) if m else float("nan"))
    print(g, " ".join(f"{x:.3f}" for x in v), f"| mean {st.mean(v):.3f}")
