"""EXPLORATORY diagnosis of the NV-C failure (18 Oct 2026). No loop is run, nothing is tuned. For each (dataset, signal, seed) it rebuilds the same
quality table the loop receives (raw and NV-C calibrated) and reports what the group pathway can see:
 sep   = median over episodes 20-200 of (honest median q - Sybil median q), in units of the honest cross-sectional MAD (calibrated table)
 syb_r = mean pairwise Pearson correlation among Sybils over sliding 15-episode windows (episodes 20-200), honest_r the same for 70 random honest agents
 act   = share of episodes in which >=70% of Sybils have an observed (non-NaN, non-constant-fill) value
 med_shift = std over episodes of the per-episode median used by the calibration (a moving median distorts z-scores)
usage: python experiments/nvc_diag.py [seeds...]  -> results/nvc_diag.json + printed table"""
import os, sys, json
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "experiments"))
import numpy as np
SEEDS = [int(a) for a in sys.argv[1:]] or [7, 11, 42, 99, 123]
VARS = {"F5x3": dict(fab_per_ep=3, fab_stars=5.0)}
EPIS, W = 200, 15
out = {}
def corr_mean(X):                      # X: agents x time (NaN allowed); mean pairwise Pearson, windows of W episodes
    rs = []
    for lo in range(20, EPIS - W + 1, 5):
        Y = X[:, lo:lo + W]; ok = (~np.isnan(Y)).sum(1) >= 8
        Y = Y[ok]
        if len(Y) < 5: continue
        mu = np.nanmean(Y, 1, keepdims=True); Y = np.where(np.isnan(Y), mu, Y) - mu
        sd = np.sqrt((Y ** 2).sum(1)); good = sd > 1e-9
        Y, sd = Y[good], sd[good]
        if len(Y) < 5: continue
        C = (Y @ Y.T) / np.outer(sd, sd); iu = np.triu_indices(len(Y), 1); rs.append(C[iu].mean())
    return float(np.mean(rs)) if rs else float("nan")
print("%-7s %-4s %4s | %8s %8s %8s | %7s %7s | %6s" % ("data", "sig", "seed", "sep_raw", "sep_cal", "medSD", "syb_r", "hon_r", "act"))
for ds in ("yelp", "amazon"):
    for sig in ("A", "B"):
        os.environ["RS_SIGNAL"] = "dev" if sig == "A" else "ext"
        import importlib, rs_data as RD; importlib.reload(RD)
        for seed in SEEDS:
            D = RD.load(ds, seed=seed, episodes=EPIS, **VARS["F5x3"]); Q, cov = RD.quality_table(D)
            NH, NS = D["n_h"], D["n_s"]; Qr = Q.copy().astype(float); Qc = Qr.copy(); meds = []
            for t in range(1, Qr.shape[1]):
                col = Qr[:, t]; m = np.median(col); sc = max(1.4826 * np.median(np.abs(col - m)), 0.02); meds.append(m)
                Qc[:, t] = np.clip(0.825 + 0.101 * (col - m) / sc, 0, 1)
            def sep(M):
                s = []
                for t in range(20, EPIS + 1):
                    h, y = M[:NH, t], M[NH:, t]; mad = max(1.4826 * np.median(np.abs(h - np.median(h))), 1e-3)
                    s.append((np.median(h) - np.median(y)) / mad)
                return float(np.median(s))
            Hs = np.random.RandomState(seed).choice(NH, NS, replace=False)
            sr, hr = corr_mean(Qc[NH:, :]), corr_mean(Qc[Hs, :])
            act = float(np.mean([(np.abs(Qr[NH:, t] - np.median(Qr[NH:, t])) > 0).mean() >= 0.7 for t in range(20, EPIS + 1)]))
            row = dict(sep_raw=sep(Qr), sep_cal=sep(Qc), med_sd=float(np.std(meds)), syb_r=sr, hon_r=hr, act=act)
            out["%s|%s|%d" % (ds, sig, seed)] = row
            print("%-7s %-4s %4d | %8.2f %8.2f %8.3f | %7.2f %7.2f | %6.2f" % (ds, sig, seed, row["sep_raw"], row["sep_cal"], row["med_sd"], sr, hr, act))
json.dump(out, open("results/nvc_diag.json", "w"))
