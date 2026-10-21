"""E4: quarantine delay after camouflage onset T_k (existing results, no new runs)."""
import json, numpy as np
rows=[]; cens=0; pre=0
for s in [7,11,42,99,123]:
    d=json.load(open("results/quar_A5_s%d_T_camouflage.json"%s))
    ids=sorted(d["sybil_ids"]); T=d["sybil_T"]["transitions"]; qe={int(k):v for k,v in d["quar_ep"].items()}
    dl=[]
    for k,a in enumerate(ids):
        t=T[str(k)]
        if a in qe:
            x=qe[a]-t; dl.append(x)
            if x<0: pre+=1
        else: cens+=1
    rows+=dl
    print("seed %3d: n=%d median %.0f IQR %.0f-%.0f max %d" % (s,len(dl),np.median(dl),*np.percentile(dl,[25,75]),max(dl)))
r=np.array(rows)
print("pooled: n=%d median %.0f IQR %.0f-%.0f p90 %.0f max %d; never quarantined %d; quarantined BEFORE onset %d"%(len(r),np.median(r),*np.percentile(r,[25,75]),np.percentile(r,90),r.max(),cens,pre))
