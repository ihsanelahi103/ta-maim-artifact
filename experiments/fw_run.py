"""Framework-level experiments F1 (coupling cuts). Pre-registered in preregistration_framework_2026-10-15.md.
FW_VARIANT in {L,B,T,G,C,R,P1,P2,X7,X8}. In-loop quarantine (no shadow). usage: TAMAIM_SEED=7 FW_VARIANT=B python experiments/fw_run.py [--attack static]
"""
import os, sys, json, time, argparse
os.environ["TAMAIM_NO_SYBIL_PRIOR"] = "1"
sys.path.insert(0, os.getcwd())
import numpy as np
V = os.environ.get("FW_VARIANT", "L")
SEED = int(os.environ.get("TAMAIM_SEED", "42"))
ap = argparse.ArgumentParser(); ap.add_argument("--attack", default="static"); ap.add_argument("--episodes", type=int, default=200)
a = ap.parse_args()
os.environ["TAMAIM_ATTACK"] = a.attack
from environment.edge_mining_env import EdgeMiningEnvironment as Env
from agents.autonomous_agent import AutonomousAgent as AA
from algorithms.dapp.privacy_protocol import DistributionAwarePrivacyProtocol as DP

EP, HR, H0, EPN = [0], [], [], []
cur = {"hr": 0.0, "h0": 0, "hn": 0, "ac": {}, "ar": {}}
ACT = []
_step, _reset, _bs = Env.step, Env.reset, Env._build_agent_state

def reset(self, *x, **k):
    if EP[0] > 0 or cur["hn"] > 0:
        pass
    EP[0] += 1
    return _reset(self, *x, **k)

def step(self, actions):
    out = _step(self, actions)
    rew = out[1]; nh = self.num_honest
    for i, r in rew.items():
        if i < nh:
            cur["hr"] += float(r); cur["hn"] += 1
            if actions.get(i) == 0: cur["h0"] += 1
            ac = actions.get(i)
            cur["ac"][ac] = cur["ac"].get(ac, 0) + 1; cur["ar"][ac] = cur["ar"].get(ac, 0.0) + float(r)
    return out
Env.step, Env.reset = step, reset

ZERO = {"T": range(6, 11), "G": range(11, 16)}.get(V)
if ZERO is not None:
    def bs(self, aid):
        s = _bs(self, aid)
        for d in ZERO: s[d] = 0.0
        return s
    Env._build_agent_state = bs
if V == "B":
    AA._get_exploration_boost = lambda self, th: 0.0
if V in ("P1", "P2"):
    val = 0.30 if V == "P1" else 1.0
    _init = DP.__init__
    def init(self, *x, **k):
        _init(self, *x, **k); self.epsilon_values[:] = val
    DP.__init__ = init
    DP.update = lambda self, *x, **k: None
    DP.adapt_epsilon = lambda self, aid, *x, **k: float(self.epsilon_values[aid]) if hasattr(self, "epsilon_values") else val


if V == "M":
    import torch, torch.nn as nn
    torch.manual_seed(SEED)
    SD, AD = 20, 5
    def mlp(i, o): return nn.Sequential(nn.Linear(i, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, o))
    actor, critic = mlp(SD, AD), mlp(2 * SD, 1)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
    BUF, STASH, MEAN = {}, {}, [np.zeros(SD)]
    def upd_mean(states, nh):
        try: MEAN[0] = np.mean([np.asarray(states[i]) for i in range(nh)], axis=0)
        except Exception: pass
    _r3, _st3 = Env.reset, Env.step
    def r3(self, *x, **k):
        st = _r3(self, *x, **k); upd_mean(st, self.num_honest); return st
    def st3(self, actions):
        out = _st3(self, actions); upd_mean(out[0], self.num_honest); return out
    Env.reset, Env.step = r3, st3
    _sel = AA.select_action
    def sel(self, state, th):
        if self.is_sybil: return self._sybil_action(state)
        s = torch.FloatTensor(np.asarray(state)); g = torch.FloatTensor(MEAN[0])
        with torch.no_grad():
            dist = torch.distributions.Categorical(logits=actor(s)); a = dist.sample()
            v = critic(torch.cat([s, g]))
        STASH[self.agent_id] = (g.numpy(), float(dist.log_prob(a)), float(v))
        return int(a)
    def store(self, s, a, r, ns, done):
        g, lp, v = STASH.get(self.agent_id, (MEAN[0], 0.0, 0.0))
        BUF.setdefault(self.agent_id, []).append((np.asarray(s), g, a, lp, v, r, bool(done), np.asarray(ns)))
    def ppo_update():
        S, G, A, LP, RET, ADV = [], [], [], [], [], []
        for aid, tr in BUF.items():
            n = len(tr); adv = np.zeros(n); last = 0.0
            lastv = 0.0
            if not tr[-1][6]:
                with torch.no_grad(): lastv = float(critic(torch.cat([torch.FloatTensor(tr[-1][7]), torch.FloatTensor(MEAN[0])])))
            for t in reversed(range(n)):
                nv = lastv if t == n - 1 else tr[t + 1][4]
                nd = 0.0 if tr[t][6] else 1.0
                delta = tr[t][5] + 0.95 * nv * nd - tr[t][4]
                last = delta + 0.95 * 0.95 * nd * last
                adv[t] = last
            for t in range(n):
                S.append(tr[t][0]); G.append(tr[t][1]); A.append(tr[t][2]); LP.append(tr[t][3])
                ADV.append(adv[t]); RET.append(adv[t] + tr[t][4])
        S, G = torch.FloatTensor(np.array(S)), torch.FloatTensor(np.array(G)); A = torch.LongTensor(A)
        LP, RET, ADV = torch.FloatTensor(LP), torch.FloatTensor(RET), torch.FloatTensor(ADV)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        for _ in range(4):
            idx = torch.randperm(len(S))
            for b in range(0, len(S), 256):
                j = idx[b:b + 256]
                dist = torch.distributions.Categorical(logits=actor(S[j]))
                ratio = torch.exp(dist.log_prob(A[j]) - LP[j])
                pl = -torch.min(ratio * ADV[j], torch.clamp(ratio, 0.8, 1.2) * ADV[j]).mean()
                vl = ((critic(torch.cat([S[j], G[j]], 1)).squeeze(1) - RET[j]) ** 2).mean()
                loss = pl + 0.5 * vl - 0.01 * dist.entropy().mean()
                opt.zero_grad(); loss.backward(); opt.step()
        BUF.clear()
    def learn(self):
        if sum(len(v) for v in BUF.values()) >= 3000 and self.agent_id == min(BUF): ppo_update()
    AA.select_action, AA.store_experience, AA.learn = sel, store, learn

variant = {"C": "A9", "R": "A6", "X7": "A7", "X8": "A8"}.get(V, "A5")
if V == "C": os.environ["TAMAIM_CONST_ACTION"] = "0"
tag = "_fw_%s_%s" % (V, a.attack); os.environ["TAMAIM_TAG"] = tag
from experiments import ablation_study as ab
# per-episode accumulators: wrap run_variant loop via reset hook
_r2 = Env.reset
def reset2(self, *x, **k):
    if EP[0] > 0:
        HR.append(cur["hr"]); H0.append(cur["h0"]); ACT.append([dict(cur["ac"]), dict(cur["ar"])])
    cur["hr"] = 0.0; cur["h0"] = 0; cur["hn"] = 0; cur["ac"] = {}; cur["ar"] = {}
    return _r2(self, *x, **k)
Env.reset = reset2
t0 = time.time()
res = ab.run_variant(variant, dataset="yelp", episodes=a.episodes, attack_scenario=a.attack)
HR.append(cur["hr"]); H0.append(cur["h0"]); ACT.append([dict(cur["ac"]), dict(cur["ar"])])
q = json.load(open("results/quar_%s_s%d%s.json" % (variant, SEED, tag)))
nq = q["quar_ep"]; sy = set(map(str, q["sybil_ids"]))
out = {"variant": V, "attack": a.attack, "seed": SEED, "res": res,
       "honest_reward_last20": float(np.mean(HR[-20:])), "honest_mine_last20": float(np.mean(H0[-20:])),
       "honest_reward": HR, "act_counts_last20": [{str(k): v for k, v in x[0].items()} for x in ACT[-20:]], "act_reward_last20": [{str(k): v for k, v in x[1].items()} for x in ACT[-20:]], "sybil_q": sum(1 for k in nq if k in sy), "honest_fp": sum(1 for k in nq if k not in sy),
       "eps_last": q["eps_traj"][-1], "secs": time.time() - t0}
os.makedirs("results/fw", exist_ok=True)
json.dump(out, open("results/fw/%s_%s_s%d.json" % (V, a.attack, SEED), "w"))
print("fw done", V, a.attack, SEED, "%.0fs" % out["secs"], out["sybil_q"], out["honest_fp"], round(out["honest_reward_last20"], 1))
