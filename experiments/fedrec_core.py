"""Federated recommendation with Sybil clients: core simulator, PoisonFRS-style attack and
defences (pre-registered: claude/preregistration_fedrec_2026-10-14.md).

Client = one agent/shard (the same 70 honest shards + 30 Sybils as the rest of the paper).
Model: matrix factorisation; user embedding u_a stays local, item embeddings V (n_items x d)
are global.  A client uploads the gradient of the mean BPR loss w.r.t. V (dense n_items x d).
The server applies V <- V - eta_m * agg_m(G).   Update convention: uploads are DESCENT
gradients (to be subtracted).
"""
import numpy as np

D = 16
BATCH = 256
NEG = 4
LR_U = 0.05
LAMBDA, K_POP, N_FILL, ATTACK_START = 10.0, 5, 59, 50     # PoisonFRS defaults (WWW'24)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FedSim:
    def __init__(self, S, C, cands, seed, n_sybil=30, d=D, target=None, attack=True):
        self.n_h, self.n_items = C.shape
        self.n_s, self.d, self.cands = n_sybil, d, cands
        self.n = self.n_h + n_sybil
        self.rng = np.random.RandomState(seed * 17 + 5)
        self.C = C
        self.pos = [np.nonzero(C[a])[0] for a in range(self.n_h)]
        # validation: 10% of each agent's training positives (used only for peer validation)
        self.val, self.trn = [], []
        for a in range(self.n_h):
            p = self.rng.permutation(self.pos[a])
            nv = max(2, int(0.1 * len(p))) if len(p) >= 12 else 0
            self.val.append(p[:nv]); self.trn.append(p[nv:] if nv else p)
        self.V = (self.rng.normal(0, 0.1, (self.n_items, d))).astype(np.float32)
        self.U = (self.rng.normal(0, 0.1, (self.n, d))).astype(np.float32)
        # Sybil j is a *camouflage clone* of honest shard j % n_h until the attack starts
        for j in range(n_sybil):
            self.trn.append(self.trn[j % self.n_h]); self.val.append(self.val[j % self.n_h])
        tot = C.sum(0)
        cand_t = np.nonzero(tot > 0)[0]
        self.target = int(target if target is not None else cand_t[np.argmin(tot[cand_t])])
        self.attack = attack
        self.round = 0
        self.Vs = None; self.vt_amp = None; self.pop = None
        self.sizes = np.array([max(1, len(self.trn[a])) for a in range(self.n)], float)

    # ---------------- client side ----------------
    def honest_grad(self, a, update_u=True):
        d = self.d
        G = np.zeros((self.n_items, d), np.float32)
        p = self.trn[a]
        if len(p) == 0:
            return G
        if len(p) > BATCH:
            p = self.rng.choice(p, BATCH, replace=False)
        i = np.repeat(p, NEG)
        j = self.rng.randint(0, self.n_items, len(i))
        u = self.U[a]
        vi, vj = self.V[i], self.V[j]
        x = (vi - vj) @ u
        w = sigmoid(-x)[:, None].astype(np.float32)          # = -dL/dx per pair
        m = float(len(i))
        np.add.at(G, i, -(w * u))          # client gradient = SUM of pair gradients (not mean)
        np.add.at(G, j, (w * u))
        if update_u:
            gu = -(w * (vi - vj)).sum(0) / m
            self.U[a] = u - LR_U * gu * 10.0
        return G

    def start_attack(self):
        V = self.V
        vavg = V.mean(0)
        sc = V @ vavg
        self.pop = np.argsort(sc)[:K_POP]                     # smallest <v_i, v_avg> = "popular" (PoisonFRS)
        self.vt_amp = LAMBDA * V[self.pop].mean(0)
        self.Vs = V.copy()

    def fake_grad(self):
        G = np.zeros((self.n_items, self.d), np.float32)
        t = self.target
        G[t] = -(self.vt_amp - self.V[t])                     # descent step moves v_t toward amplified popular embedding
        drift = np.linalg.norm(self.Vs - self.V, axis=1); drift[t] = -1
        fill = np.argsort(-drift)[:N_FILL]
        G[fill] = self.V[fill] - self.Vs[fill]                # pull filler items back toward round-s values (benign-looking)
        return G

    def collect(self):
        """All client uploads for this round: (n, n_items*d) float32 + is_sybil mask."""
        self.round += 1
        if self.attack and self.round == ATTACK_START and self.Vs is None:
            self.start_attack()
        Gs = [self.honest_grad(a).ravel() for a in range(self.n_h)]
        sy = np.zeros(self.n, bool)
        if self.attack and self.Vs is not None:
            fg = self.fake_grad().ravel()
            Gs += [fg.copy() for _ in range(self.n_s)]
            sy[self.n_h:] = True
        else:                                                   # camouflage phase / no-attack control
            Gs += [self.honest_grad(self.n_h + j).ravel() for j in range(self.n_s)]
        active = np.ones(self.n, bool)
        return np.stack(Gs), sy, active

    def apply(self, agg, eta):
        self.V -= (eta * agg).reshape(self.n_items, self.d).astype(np.float32)

    def respond(self, i, probe, eta):
        """Client i's response to a server probe (RECESS): it treats `probe` as the aggregated gradient,
        applies it to its local copy of V and uploads the gradient it computes at that model."""
        Vsave = self.V
        self.V = (Vsave - (eta * probe).reshape(self.n_items, self.d)).astype(np.float32)
        try:
            if self.attack and self.Vs is not None and i >= self.n_h:
                g = self.fake_grad()
            else:
                g = self.honest_grad(i, update_u=False)
        finally:
            self.V = Vsave
        return g.ravel()

    # ---------------- peer validation (TA-MAIM quality signal) ----------------
    def peer_scores(self, G, validators, eta, n_val=3, ids=None):
        """r_j = mean over K validators of relative BPR-validation-loss improvement when
        update j alone is applied.  Validators are honest agents (never j itself)."""
        r = np.zeros(self.n)
        Vm = self.V
        for j in (range(self.n) if ids is None else ids):
            vals = [v for v in validators if v != j]
            ch = self.rng.choice(vals, n_val, replace=False)
            gj = G[j].reshape(self.n_items, self.d)
            acc = []
            for v in ch:
                pv = self.val[v]
                if len(pv) == 0:
                    continue
                negs = self.rng.randint(0, self.n_items, len(pv) * NEG)
                ip = np.repeat(pv, NEG)
                rows = np.unique(np.concatenate([ip, negs]))
                def loss(dV):
                    vi, vj_ = Vm[ip] - dV(ip), Vm[negs] - dV(negs)
                    x = (vi - vj_) @ self.U[v]
                    return float(np.mean(np.log1p(np.exp(-np.clip(x, -30, 30)))))
                l0 = loss(lambda idx: 0.0)
                l1 = loss(lambda idx: eta * gj[idx])
                acc.append((l0 - l1) / max(l0, 1e-6))
            r[j] = np.mean(acc) if acc else 0.0
        return r


    def peer_scores_disturb(self, G, validators, eta, n_val=3, k=10):
        """Signal S2: r_j = - mean over K validators of the share of the validator's own top-k
        recommendation list (items it has not interacted with) replaced when update j alone is applied."""
        r = np.zeros(self.n)
        for j in range(self.n):
            vals = [v for v in validators if v != j]
            ch = self.rng.choice(vals, n_val, replace=False)
            gj = G[j].reshape(self.n_items, self.d)
            rows = np.nonzero(np.abs(gj).sum(1) > 0)[0]
            dis = []
            for v in ch:
                s = self.V @ self.U[v]
                s[self.pos[v]] = -1e9
                top0 = set(np.argpartition(-s, k)[:k].tolist())
                s2 = s.copy()
                s2[rows] -= eta * (gj[rows] @ self.U[v])
                s2[self.pos[v]] = -1e9
                top1 = set(np.argpartition(-s2, k)[:k].tolist())
                dis.append(1.0 - len(top0 & top1) / k)
            r[j] = -np.mean(dis)
        return r

    # ---------------- evaluation ----------------
    def evaluate(self, gains_fn=None, k=10):
        from utility_eval import ndcg_at_k
        nd = []
        for a, (c, g) in enumerate(self.cands):
            rng = np.random.RandomState(a * 31 + 1)
            sc = self.V[c] @ self.U[a]
            nd.append(ndcg_at_k(sc, g, k, rng.uniform(0, 1e-9, len(c))))
        hit5, hit10, cnt = 0, 0, 0
        t = self.target
        S_all = self.V @ self.U.T                              # (items, n_h)
        for a in range(self.n_h):
            if self.C[a, t] > 0:
                continue
            s = S_all[:, a].copy()
            s[self.pos[a]] = -1e9                              # exclude items the agent already has
            top = np.argpartition(-s, 10)[:10]
            top = top[np.argsort(-s[top])]
            hit10 += int(t in top); hit5 += int(t in top[:5]); cnt += 1
        return dict(ndcg=float(np.mean(nd)), hr5=hit5 / max(cnt, 1), hr10=hit10 / max(cnt, 1))


# ---------------------------------------------------------------- aggregators
def _norms(G):
    n = np.linalg.norm(G, axis=1)
    n[n == 0] = 1.0
    return n


def cosmat(G):
    X = G / _norms(G)[:, None]
    return X @ X.T


D_GLOBAL = [D]


def rowmean(X, w=None):
    """Observer mean: per item row, sum_i w_i g_i / sum_i w_i [g_i row != 0].  Sparse uploads are
    averaged over the clients that touched the row (as in the rating-level experiments)."""
    m = len(X); d = D_GLOBAL[0]
    X3 = X.reshape(m, -1, d)
    w = np.ones(m, np.float32) if w is None else np.asarray(w, np.float32)
    nz = (np.abs(X3).sum(-1) > 0).astype(np.float32)
    num = np.tensordot(w, X3, axes=(0, 0))
    cnt = np.tensordot(w, nz, axes=(0, 0))
    return (num / np.maximum(cnt, 1e-12)[:, None]).reshape(-1)


def agg_mean(G, idx):
    return rowmean(G[idx])


def agg_median(G, idx):
    return np.median(G[idx], axis=0)


def agg_trimmed(G, idx, beta=0.3):
    X = np.sort(G[idx], axis=0)
    b = int(beta * len(idx))
    return X[b:len(idx) - b].mean(0)


def agg_normclip(G, idx):
    X = G[idx]; nr = np.linalg.norm(X, axis=1)
    e = np.median(nr[nr > 0]) if (nr > 0).any() else 1.0
    sc = np.minimum(1.0, e / np.maximum(nr, 1e-12))
    return rowmean(X * sc[:, None])


def _krum_scores(X, f):
    m = len(X)
    sq = (X ** 2).sum(1)
    D2 = np.maximum(sq[:, None] + sq[None] - 2 * X @ X.T, 0)
    np.fill_diagonal(D2, np.inf)
    kk = max(1, m - f - 2)
    return np.sort(D2, axis=1)[:, :kk].sum(1)


def agg_krum(G, idx, f):
    X = G[idx]
    return X[int(np.argmin(_krum_scores(X, f)))]


def agg_multikrum(G, idx, f):
    X = G[idx]
    s = _krum_scores(X, f)
    m = max(1, len(X) - f)
    return rowmean(X[np.argsort(s)[:m]])


class FoolsGold:
    def __init__(self, n, dim):
        self.H = np.zeros((n, dim), np.float32)

    def __call__(self, G, idx):
        self.H[idx] += G[idx]
        cs = cosmat(self.H[idx]); np.fill_diagonal(cs, -np.inf)
        mx = cs.max(1)
        cs2 = cs.copy()
        for i in range(len(idx)):
            for j in range(len(idx)):
                if i != j and mx[j] > mx[i]:
                    cs2[i, j] = cs[i, j] * mx[i] / mx[j]
        np.fill_diagonal(cs2, -np.inf)
        wv = 1 - cs2.max(1)
        wv = np.clip(wv, 0, 1); wv = wv / max(wv.max(), 1e-12)
        wv[wv == 1] = 0.99
        with np.errstate(divide='ignore'):
            wv = np.clip(np.log(wv / (1 - wv)) + 0.5, 0, 1)
        return rowmean(G[idx], wv)


def agg_fltrust(G, idx, root):
    """FLTrust (Cao et al., NDSS'21) = scoring rule of RFLPA (Mai et al., NeurIPS'24):
    ts_i = ReLU(cos(g_i, g0)), g_i rescaled to ||g0||, trust-weighted mean."""
    g0 = root
    n0 = np.linalg.norm(g0)
    X = G[idx]
    nr = _norms(X)
    ts = np.maximum((X @ g0) / (nr * max(n0, 1e-12)), 0.0)
    Xn = X * (n0 / nr)[:, None]
    if ts.sum() <= 0:
        return np.zeros_like(g0)
    return rowmean(Xn, ts)


def agg_flame(G, idx, rng):
    from sklearn.cluster import HDBSCAN
    X = G[idx]
    m = len(X)
    D = np.clip(1.0 - cosmat(X), 0.0, 2.0); D = (D + D.T) / 2; np.fill_diagonal(D, 0.0)
    lab = HDBSCAN(min_cluster_size=m // 2 + 1, min_samples=1, metric="precomputed",
                  allow_single_cluster=True).fit(D.astype(np.float64)).labels_
    good = lab[lab >= 0]
    adm = np.arange(m) if len(good) == 0 else np.nonzero(lab == np.bincount(good).argmax())[0]
    nr = np.linalg.norm(X, axis=1)
    e = float(np.median(nr[nr > 0])) if (nr > 0).any() else 1.0
    sc = np.minimum(1.0, e / np.maximum(nr, 1e-12))
    out = rowmean(X[adm] * sc[adm, None])
    return out + rng.normal(0.0, 0.001 * e, size=out.shape).astype(np.float32), adm


def agg_fednga(G, idx, alpha):
    """Fed-NGA (Zuo et al., NeurIPS'25): sum_m alpha_m g_m/||g_m||."""
    X = G[idx]
    a = alpha[idx] / alpha[idx].sum()
    return ((X / _norms(X)[:, None]) * a[:, None]).sum(0)


class Recess:
    """RECESS (Yan et al., NeurIPS'23) as described in the paper; ambiguities are fixed in the
    pre-registration addendum: probe = unit-norm previous upload with noise added to the first 10% of
    dimensions until cos(probe, original) <= A, rescaled to the previous upload's norm; detection round every
    PERIOD rounds (aggregation skipped); alpha = -cos(probe, response)/||response||; alpha>0 -> deduct
    alpha*BS (marker reset), alpha<=0 -> good streak, +BS after 10 good detections; TS<=0 -> excluded;
    weights = softmax(TS) (observer-weighted mean as for every other rule)."""
    def __init__(self, n, dim, rng, ts0=1.0, bs=0.1, A=0.95, period=4, limit=10):
        self.n, self.dim, self.rng = n, dim, rng
        self.TS = np.full(n, ts0); self.TS0, self.BS, self.A, self.period, self.limit = ts0, bs, A, period, limit
        self.prev = None; self.streak = np.zeros(n, int); self.excluded = np.zeros(n, bool)

    def _probe(self, g):
        nrm = np.linalg.norm(g)
        if nrm == 0:
            return g.copy()
        t = g / nrm
        k = max(1, int(0.1 * len(t)))
        sigma = 0.01
        for _ in range(30):
            p = t.copy(); p[:k] += (sigma * self.rng.normal(0, 1, k)).astype(np.float32)
            if float(p @ t) / max(float(np.linalg.norm(p)), 1e-12) <= self.A:
                break
            sigma *= 2.0
        return p / max(float(np.linalg.norm(p)), 1e-12) * nrm

    def step(self, sim, G, idx, r, eta):
        if self.prev is None:
            self.prev = G.copy()
        act = np.array([i for i in idx if not self.excluded[i]])
        if r % self.period == 0:
            for i in act:
                probe = self._probe(self.prev[i])
                resp = sim.respond(i, probe, eta)
                nr = float(np.linalg.norm(resp))
                cos = float(probe @ resp) / max(float(np.linalg.norm(probe)) * nr, 1e-12)
                alpha = -cos / max(nr, 1e-12)
                if alpha > 0:
                    self.TS[i] -= alpha * self.BS; self.streak[i] = 0
                else:
                    self.streak[i] += 1
                    if self.streak[i] >= self.limit:
                        self.TS[i] = min(self.TS0, self.TS[i] + self.BS); self.streak[i] = 0
                if self.TS[i] <= 0:
                    self.excluded[i] = True
            return None                                       # aggregation skipped in a detection round
        self.prev[act] = G[act]
        w = np.exp(self.TS[act]); w = w / w.sum()
        return rowmean(G[act], w)


class Conda:
    """ConDA (Liang et al., ACM TKDD 19(1):27, 2025) adapted to the federated-MF simulator; every choice the paper leaves
    open is fixed in the pre-registration (Addendum 3).
    1. E-SHAP (Sec. 4.1): each validation client v (alpha*|training clients| honest clients drawn at random each round)
       evaluates the K submitted updates on its own held-out items.  Utility of a subset S = - BPR validation loss of the
       ENSEMBLE whose scores are the mean of the single-update models' scores (Eq. 6).  For this bilinear model the
       ensemble margin is x0 - mean_{k in S} d_k with d_k = eta*(G_k[i]-G_k[j]) @ u_v.  L subsets are sampled (sizes
       from the Shapley-kernel mass (K-1)/(s(K-s)), uniform members), the utility is regressed on the subset indicator
       (Eq. 7-9, linear surrogate with intercept); the coefficients are the contributions C_v.
    2. Majority voting (Eq. 10-12): the bottom beta of each C_v vote 'low'; Cbar_k = 0 if votes > V/2 else V - votes.
    3. Bayesian aggregation (Eq. 13-14): W ~ Dirichlet(Cbar) (Gamma construction; zero shape -> weight 0), mean over T
       draws.  KD (Eq. 15) is NOT run: with a student of the same bilinear form, the optimum of Eq. 15 is the teacher
       itself, so the MC-averaged weights are used directly (documented deviation).
    4. The weighted aggregate is the observer-weighted mean (as for every other rule)."""
    def __init__(self, rng, alpha=0.3, beta=0.2, L=1000, T=20, ridge=1e-3):
        self.rng, self.alpha, self.beta, self.L, self.T, self.ridge = rng, alpha, beta, L, T, ridge
        self.last_w = None; self.last_C = None

    def _eshap(self, sim, G3, idx, v, eta):
        K = len(idx)
        pv = sim.val[v]
        if len(pv) == 0:
            return None
        rng = self.rng
        negs = rng.randint(0, sim.n_items, len(pv) * NEG)
        ip = np.repeat(pv, NEG)
        u = sim.U[v]
        x0 = (sim.V[ip] - sim.V[negs]) @ u                                   # (P,)
        dk = eta * (G3[:, ip, :] - G3[:, negs, :]) @ u                       # (K,P) single-update score shifts
        sizes = np.arange(1, K)
        pr = (K - 1.0) / (sizes * (K - sizes)); pr /= pr.sum()
        ss = rng.choice(sizes, self.L, p=pr)
        X = np.zeros((self.L, K), np.float32)
        for l, s in enumerate(ss):
            X[l, rng.choice(K, s, replace=False)] = 1.0
        marg = x0[None, :] - (X @ dk) / ss[:, None]                           # ensemble margin (mean of scores)
        y = -np.mean(np.log1p(np.exp(-np.clip(marg, -30, 30))), axis=1)       # utility = -BPR loss
        A = np.hstack([X, np.ones((self.L, 1), np.float32)])
        coef = np.linalg.solve(A.T @ A + self.ridge * np.eye(K + 1), A.T @ y)
        return coef[:K]

    def step(self, sim, G, idx, eta):
        K = len(idx)
        G3 = G[idx].reshape(K, sim.n_items, sim.d)
        nv = max(1, int(round(self.alpha * K)))
        pool = np.array([a for a in range(sim.n_h) if len(sim.val[a]) > 0])
        vals = self.rng.choice(pool, min(nv, len(pool)), replace=False)
        votes = np.zeros(K); nvote = 0
        nlow = int(self.beta * K)
        Cs = []
        for v in vals:
            c = self._eshap(sim, G3, idx, v, eta)
            if c is None:
                continue
            Cs.append(c); nvote += 1
            votes[np.argsort(c)[:nlow]] += 1                                   # bottom beta% -> low-contribution vote
        if nvote == 0:
            return rowmean(G[idx])
        Cbar = np.where(votes > 0.5 * nvote, 0.0, nvote - votes)               # Eq. 12
        if Cbar.sum() <= 0:
            Cbar = np.ones(K)
        gam = self.rng.gamma(np.maximum(Cbar, 1e-12)[None, :].repeat(self.T, 0))
        gam[:, Cbar == 0] = 0.0
        W = gam / np.maximum(gam.sum(1, keepdims=True), 1e-12)
        w = W.mean(0)
        self.last_w, self.last_C = w, np.mean(Cs, 0)
        return rowmean(G[idx], w)
