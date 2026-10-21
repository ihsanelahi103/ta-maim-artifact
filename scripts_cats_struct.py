import numpy as np
from data.topology import build_topology, params
from algorithms.collusion.collusion_detector import CollusionAwareTrustSignal as C

c = C(100)
def score(g, i):
    if len(g.get(i, {})) < c.min_neighbors:
        return 0.0
    return float(np.clip(0.5*c._clustering_coefficient(i, g) + 0.5*c._ego_density(i, g), 0, 1))

print("floor structural = 0.5; suspicion needs s+corr > 1.1 with corr > 0.3")
for ns in (10, 20, 30, 50):
    sy_all, ho_all = [], []
    for seed in range(1, 11):
        g, _, gt = build_topology(100 - ns, ns, seed, **params())
        sy = [score(g, i) for i in gt]
        ho = [score(g, i) for i in range(100) if i not in gt]
        sy_all += sy; ho_all += ho
    sy_all, ho_all = np.array(sy_all), np.array(ho_all)
    print(f"Sybils={ns:2d}: Sybil struct mean {sy_all.mean():.3f} min {sy_all.min():.3f} max {sy_all.max():.3f} "
          f"frac>0.5 {np.mean(sy_all>0.5):.3f} | honest struct mean {ho_all.mean():.3f} max {ho_all.max():.3f} "
          f"frac>0.5 {np.mean(ho_all>0.5):.4f}  (honest with deg>=5: {np.mean([x>0 for x in ho_all]):.3f})")
