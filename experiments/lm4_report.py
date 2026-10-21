"""Complete per-stream / per-seed report for the frozen lm4_run.py (companion; same deterministic evaluation, nothing tuned).
Recomputes every per-stream outcome with the same calls as lm4_run.run_budget, asserts that its family means equal the values saved by
lm4_run.py (results/lm4_results_B<b>.json), prints the binding diagnostic, and writes results/lm4_per_stream_B<b>.csv.
usage: python experiments/lm4_report.py --budgets 1 2      (--smoke for an execution check)"""
import os, sys, json, csv, argparse
import numpy as np
sys.path.insert(0, os.getcwd())
from experiments import lm3_run as M
from experiments import lm4_run as L4
from experiments.lm2_run import CFG, HELD_OUT, RULES, Rule, boot_ci

ap = argparse.ArgumentParser(); ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2]); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
if a.smoke:
    M.FIXED.update(cem_iters=1, cem_pop=4, cem_streams=2)
    train = M.load_set([7], True); test = M.load_set([7], True); held_list = [h for h in HELD_OUT if h in test] or ["quality-matching"]
else:
    train = M.load_set(L4.TRAIN_SEEDS); test = M.load_set(L4.TEST_SEEDS); held_list = HELD_OUT
for d in (train, test):
    for v in d.values():
        for s in v: L4.add_rate(s)
F = ("U", "poison", "cost", "hd", "sm", "checks")
for B in a.budgets:
    CFG.update(B=B)
    cov = [M.coverage(s, B) for v in test.values() for s in v]
    print("\n##### B=%d binding diagnostic on the NEW streams (RANDOM): %s  -> binding=%s (rule: unresolved at ep 100 >= 0.25)" % (
        B, {k: round(float(np.mean([c[k] for c in cov])), 3) for k in cov[0]}, float(np.mean([c["unresolved_100"] for c in cov])) >= 0.25))
    saved = {}
    p = "results/lm4_results_B%d%s.json" % (B, "_smoke" if a.smoke else "")
    if os.path.exists(p): saved = json.load(open(p))
    rows_csv = []; seed_acc = {}
    for held in held_list:
        if held not in test: continue
        tr = [s for f, v in train.items() if f != held for s in v]; te = test[held]
        w = L4.fit_prior(tr); wl = M.fit_tuned_lin(tr, B)
        per = []
        for si, st in enumerate(te):
            r = {k: M.simulate3(st, "verify", policy=Rule(k, 11 + si), B=B) for k in RULES}
            r["UNC"] = M.simulate3(st, "verify", policy=M.UncRule("UNC", 11 + si), B=B)
            r["TUNED-LIN"] = M.simulate3(st, "verify", policy=M.LinRule(wl, 13 + si), B=B)
            r["VOI"] = M.simulate3(st, "verify", policy=L4.VoiRule(w, 17 + si), B=B)
            for k in ("HARD", "ALL", "ORACLE"): r[k] = M.simulate3(st, k, B=B)
            per.append((st["name"], r))
        rules = RULES + ["UNC", "TUNED-LIN"]
        mean = lambda k, f: float(np.mean([r[k][f] for _, r in per]))
        best = max(rules, key=lambda k: mean(k, "U"))
        if saved.get(held):
            assert abs(saved[held]["means"]["VOI"]["U"] - mean("VOI", "U")) < 1e-6 and saved[held]["best_rule"] == best, "report differs from lm4_run output"
        reg = lambda k: np.array([r["ORACLE"]["U"] - r[k]["U"] for _, r in per])
        U = lambda k: np.array([r[k]["U"] for _, r in per])
        headroom = float(reg(best).mean() / abs(U(best).mean()))
        rr = reg(best); rv = reg("VOI"); ok = rr > 1e-9; rel_reg = (rr[ok] - rv[ok]) / rr[ok]; lo, hi = boot_ci(rel_reg)
        rel_u = (U("VOI") - U(best)) / np.abs(U(best)); ulo, uhi = boot_ci(rel_u)
        ok_hd = mean("VOI", "hd") - mean(best, "hd") <= 0.02; ok_sm = mean("VOI", "sm") - mean(best, "sm") <= 0.02
        print("\n=== B=%d  %s (n=%d)  strongest rule: %s  oracle headroom (regret/U of the rule): %.3f ===" % (B, held, len(per), best, headroom))
        print("%-10s %9s %9s %9s %8s %8s %8s" % ("policy", "net U", "regret", "poison", "cost", "honestDrop", "sybKept"))
        for k in ["ALL", "HARD"] + rules + ["VOI", "ORACLE"]:
            print("%-10s %9.1f %9.1f %9.1f %8.1f %8.3f %8.3f" % (k, mean(k, "U"), float(reg(k).mean()), mean(k, "poison"), mean(k, "cost"), mean(k, "hd"), mean(k, "sm")))
        print("PRIMARY regret reduction vs %s: %.3f  CI [%.3f, %.3f]  honest-ok %s sybil-ok %s -> %s" % (best, rel_reg.mean(), lo, hi, ok_hd, ok_sm, L4.verdict(rel_reg, lo, hi, ok_hd, ok_sm, headroom)))
        print("relative net-utility improvement vs %s: %.4f  CI [%.4f, %.4f]   (old +5%% criterion: %s)" % (best, rel_u.mean(), ulo, uhi, "met" if rel_u.mean() >= .05 and ulo > 0 else "not met"))
        print("per-stream (regret_%s -> regret_VOI, reduction):" % best)
        for (name, r), a_, b_ in zip(per, rr, rv):
            seed = name.split("_s")[-1].split(".")[0]; red = (a_ - b_) / a_ if a_ > 1e-9 else float("nan")
            print("   %-18s %8.1f -> %8.1f  %+.3f   poison %7.1f -> %7.1f   honestDrop %.3f -> %.3f" % (name, a_, b_, red, r[best]["poison"], r["VOI"]["poison"], r[best]["hd"], r["VOI"]["hd"]))
            rows_csv.append([B, held, name, seed, best, a_, b_, red] + [r[k][f] for k in ("VOI", best) for f in F])
            seed_acc.setdefault(seed, []).append(red)
    print("\nPER-SEED mean regret reduction over all streams (B=%d):" % B, {s: round(float(np.nanmean(v)), 3) for s, v in sorted(seed_acc.items())})
    out = "results/lm4_per_stream_B%d%s.csv" % (B, "_smoke" if a.smoke else "")
    with open(out, "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["B", "family", "stream", "seed", "best_rule", "regret_rule", "regret_voi", "reg_reduction"] + ["voi_" + f for f in F] + ["rule_" + f for f in F]); wr.writerows(rows_csv)
    print("wrote", out)
