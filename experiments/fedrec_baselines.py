"""Federated-recommendation comparison of aggregation defences under a PoisonFRS-style Sybil
attack (see fedrec_core.py; protocol: claude/preregistration_fedrec_2026-10-14.md).
usage: python fedrec_baselines.py --dataset yelp --seed 7 --out results/fedrec/yelp_s7.json
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import fedrec_core as F

METHODS = ["fedavg", "median", "trimmed", "normclip", "krum", "multikrum", "foolsgold",
           "fltrust", "flame", "fednga", "fednga_decay", "recess", "conda"]
ETA_GRID = [0.03, 0.1, 0.3, 1.0]   # <=1: repeated full step on one row stays a contraction
FNGA_DELTA = 0.1


def val_loss(sim):
    ls = []
    for a in range(sim.n_h):
        pv = sim.val[a]
        if len(pv) == 0:
            continue
        rng = np.random.RandomState(a + 11)
        neg = rng.randint(0, sim.n_items, len(pv) * F.NEG)
        ip = np.repeat(pv, F.NEG)
        x = (sim.V[ip] - sim.V[neg]) @ sim.U[a]
        ls.append(np.mean(np.log1p(np.exp(-np.clip(x, -30, 30)))))
    return float(np.mean(ls))


def aggregate(name, G, idx, sim, state, rnd):
    f = sim.n_s
    if name == "fedavg":
        return F.agg_mean(G, idx)
    if name == "median":
        return F.agg_median(G, idx)
    if name == "trimmed":
        return F.agg_trimmed(G, idx)
    if name == "normclip":
        return F.agg_normclip(G, idx)
    if name == "krum":
        return F.agg_krum(G, idx, f)
    if name == "multikrum":
        return F.agg_multikrum(G, idx, f)
    if name == "foolsgold":
        return state["fg"](G, idx)
    if name == "fltrust":
        root = G[state["root"]].mean(0)
        return F.agg_fltrust(G, idx, root)
    if name == "flame":
        out, adm = F.agg_flame(G, idx, state["rng"])
        state["adm"] = idx[adm]
        return out
    if name == "recess":
        return state["rc"].step(sim, G, idx, rnd, state["eta_now"])
    if name == "conda":
        out = state["cd"].step(sim, G, idx, state["eta_now"])
        state["wlast"] = (idx, state["cd"].last_w)
        return out
    if name in ("fednga", "fednga_decay"):
        alpha = sim.sizes.copy()
        return F.agg_fednga(G, idx, alpha)
    raise ValueError(name)


def run(name, attack, S, C, cands, seed, rounds, eta, log=None):
    sim = F.FedSim(S, C, cands, seed, attack=attack)
    dim = sim.n_items * sim.d
    state = {"rng": np.random.RandomState(seed + 77),
             "root": np.random.RandomState(seed + 4141).choice(sim.n_h, 3, replace=False)}
    if name == "foolsgold":
        state["fg"] = F.FoolsGold(sim.n, dim)
    if name == "recess":
        state["rc"] = F.Recess(sim.n, dim, np.random.RandomState(seed + 99))
        state["eta_now"] = eta
    if name == "conda":
        state["cd"] = F.Conda(np.random.RandomState(seed + 123))
        state["eta_now"] = eta
    scale, flame_sy = 1.0, []
    cd_sy = []
    for r in range(1, rounds + 1):
        G, sy, active = sim.collect()
        idx = np.nonzero(active)[0]
        agg = aggregate(name, G, idx, sim, state, r)
        if agg is None:                      # RECESS detection round: no aggregation
            continue
        if r == 1:
            ref = F.agg_mean(G, idx)
            nr = np.linalg.norm(agg)
            scale = float(np.linalg.norm(ref) / nr) if nr > 0 else 1.0
        e = eta * scale
        if name == "fednga_decay":
            e = e / (r ** (0.5 + FNGA_DELTA))
        sim.apply(agg, e)
        if name == "flame" and sy.any():
            flame_sy.append(int(np.isin(state["adm"], np.nonzero(sy)[0]).sum()))
        if name == "conda" and sy.any() and state["wlast"][1] is not None:
            ii, ww = state["wlast"]
            cd_sy.append(float(ww[sy[ii]].sum()))              # total aggregation weight on Sybils
        if log and r % 50 == 0:
            log("  %s attack=%s round %d" % (name, attack, r))
    out = sim.evaluate()
    out.update(method=name, attack=bool(attack), eta=eta, scale=scale, target=sim.target)
    if name == "recess":
        ex = state["rc"].excluded
        out["recess_sybils_excluded"] = int(ex[sim.n_h:].sum()); out["recess_honest_excluded"] = int(ex[:sim.n_h].sum())
    if cd_sy:
        out["conda_sybil_weight_last"] = cd_sy[-1]; out["conda_sybil_weight_mean"] = float(np.mean(cd_sy))
    if flame_sy:
        out["flame_sybils_admitted_last"] = flame_sy[-1]
    return out


def select_eta(S, C, cands, seed, rounds=100):
    best, bl = None, 1e9
    for eta in ETA_GRID:
        sim = F.FedSim(S, C, cands, seed, attack=False)
        for r in range(rounds):
            G, sy, active = sim.collect()
            sim.apply(F.agg_mean(G, np.nonzero(active)[0]), eta)
        l = val_loss(sim)
        if np.isfinite(l) and l < bl:
            best, bl = eta, l
    return best, bl


def load(dataset, seed, n_h=70):
    import utility_eval as U
    shards = U.load_shards(dataset, n_h)
    n_items, S, C, test, tr_items = U.split_and_stats(shards, seed)
    cands = U.make_candidates(n_items, test, tr_items, seed)
    return S, C, cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="yelp")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--eta", type=float, default=0.0, help="0 = select from grid (seed 7 rule)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    S, C, cands = load(a.dataset, a.seed)
    lg = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
    eta = a.eta
    if eta <= 0:
        S7, C7, c7 = (S, C, cands) if a.seed == 7 else load(a.dataset, 7)
        eta, bl = select_eta(S7, C7, c7, 7)
        lg("selected eta %.2f (val loss %.4f)" % (eta, bl))
    rows = []
    if os.path.exists(a.out):                                  # resume: keep finished runs (runs are independent and seeded)
        rows = json.load(open(a.out))
        lg("resuming: %d runs already in %s" % (len(rows), a.out))
    done = {(r["method"], bool(r["attack"])) for r in rows}
    for m in a.methods.split(","):
        for attack in (False, True):
            if (m, attack) in done:
                continue
            lg("run %s attack=%s" % (m, attack))
            rows.append(run(m, attack, S, C, cands, a.seed, a.rounds, eta, lg))
            rows[-1].update(seed=a.seed, dataset=a.dataset)
            json.dump(rows, open(a.out, "w"))
    lg("done")


if __name__ == "__main__":
    main()
