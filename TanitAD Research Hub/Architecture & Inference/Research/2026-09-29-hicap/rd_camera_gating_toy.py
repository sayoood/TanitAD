#!/usr/bin/env python3
"""rd_camera_gating_toy.py  --  Stream R-D (HiCAP), 2026-09-29.  numpy only, < 2 min.

QUESTION.  HiCAP wants a compute-elastic state: always front-wide, and *other* cameras only when the
situation makes them informative.  Is a gate that is trained from CHEAP cues (already-paid front-wide
features + ego kinematics + the previous tick's tactical token) under a per-camera cost penalty able to
match "encode everything" at a fraction of the cost -- and what do gate errors, warm-up (cold start) and
hysteresis do to that?

WHAT THIS IS NOT.  A synthetic generative model whose structure I chose.  It shows mechanisms and their
sign/size *under those assumptions*; it says nothing about how much a real rear camera helps a real
planner.  Every constant below is an ASSUMPTION (listed in ASSUMPTIONS and written to the JSON).

WORLD.  7 cameras [FW FT CL CR RL RR RT] (front-wide, front-tele, cross-L/R, rear-L/R, rear-tele).
  7 situations, each with exactly ONE camera that carries the decision-relevant bit z (proceed/hold):
    cruise->FW  follow_far->FT  junc_L->CL  junc_R->CR  lc_L->RL  lc_R->RR  pullover->RT
  FW carries a weak leak of z in every situation (0.7 in follow_far: wide sees the lead but not its range,
  0.15 elsewhere) and a weak situation code.  Situations and z are sticky Markov chains.
  Decision y = [z > 0]  (binary proceed/hold);  "miss" = true hold predicted proceed (the safety error).

POLICIES (all evaluated on the SAME held-out episodes; paired, episode-cluster bootstrap):
  fixed1   : FW only                                  cost 1
  fixedall : all 7 cameras (head H=32, and a 4x bigger head H=128 / 2x epochs as an honesty control)
  oracle   : perfect situation knowledge -> exactly the right camera (cost 2 unless cruise)
  gated    : learned gate over 8 options {FW, FW+FT, FW+CL, FW+CR, FW+RL, FW+RR, FW+RT, ALL},
             trained by FULL-INFORMATION cost-sensitive learning: because every camera is cached offline,
             the per-option error E_j(x) is computable for every training tick, so the gate minimises
             sum_j p_j(x) [E_j(x) + lam * cost_j]  -- no RL / Gumbel needed.  (This is the design insight.)
  Gate cues: FW features (already paid) + ego cues + previous-tick option (the 'tactical token' proxy).
  VISION-ONLY arms (no ego, no nav cue at inference; FW features [+ own previous option]) and a sensitivity to how much
  situation information the always-on FW embedding carries (fw_code_radius 1.5 / 3.0 / 4.5).

ANALYSES written to the JSON (per seed unless noted): policy table incl. published fixed-camera profiles (Alpamayo-R1 default 4
  cams, Alpamayo-2 trajectory profile 6 cams) and a privileged ceiling; lambda frontier; regret matrix (situation x camera set
  served); gate confusion; error injection; hysteresis variants x cold-start on/off; cue ablation (open/closed-loop previous
  token, supervised situation gate, vision-only); escalate-to-ALL and hedge-with-runner-up under uncertainty; sensitivity.
  CAVEAT: lam* is chosen by a knee rule on the TEST frontier (optimistic by <~0.2 pt); the per-lambda rows are unselected.

Run:  python rd_camera_gating_toy.py [--quick]   (writes rd_camera_gating_toy_result.json next to itself;
      full run = 3 seeds + sensitivity, ~90 s single-thread; --quick = 1 seed, 3 lambdas, no sensitivity, ~30 s)
"""
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # single thread: faster + reproducible for tiny matmuls
    os.environ.setdefault(_v, "1")
import numpy as np

T0 = time.time()
CAMS = ["FW", "FT", "CL", "CR", "RL", "RR", "RT"]
NC = 7
SIT = ["cruise", "follow_far", "junc_L", "junc_R", "lc_L", "lc_R", "pullover"]
NS = 7
OPT = ["FW", "FW+FT", "FW+CL", "FW+CR", "FW+RL", "FW+RR", "FW+RT", "ALL"]
NO = 8
COST = np.array([1, 2, 2, 2, 2, 2, 2, 7], float)          # camera-encode units
MASK = np.zeros((NO, NC), np.float32)
MASK[:, 0] = 1
for o in range(1, 7):
    MASK[o, o] = 1
MASK[7, :] = 1
ORACLE_OPT = np.arange(NS)                                  # situation s -> option s (bijection)

ASSUMPTIONS = dict(
    p_stay_s=0.97, p_stay_z=0.90, base_s=[0.35, 0.15, 0.10, 0.10, 0.10, 0.10, 0.10],
    amp_required=2.0, amp_fw_leak=0.15, amp_fw_leak_follow=0.70, fw_code_radius=1.5,
    ego_v_mean=[0.5, 1.0, -0.6, -0.6, 0.4, 0.4, -0.8], ego_v_sd=0.4,
    ego_yaw_mean=[0, 0, 1, -1, 0.35, -0.35, 0.2], ego_yaw_sd=0.5,
    nav_p_true=0.85, nav_p_false=0.08, prev_token_err_train=0.15,
    cold_factor_first_ticks=[0.3, 0.6], T=200, n_train_ep=300, n_test_ep=300,
    note="every constant is a chosen assumption of the toy; none is measured on driving data")

rng_global = np.random.default_rng(0)


# ----------------------------------------------------------------------------------------------
def gen_episodes(n_ep, T, rng):
    A = ASSUMPTIONS
    base = np.array(A["base_s"]); base = base / base.sum()
    s = np.zeros((n_ep, T), int)
    z = np.zeros((n_ep, T), np.float32)
    s[:, 0] = rng.choice(NS, n_ep, p=base)
    z[:, 0] = rng.choice([-1, 1], n_ep)
    for t in range(1, T):
        ch = rng.random(n_ep) > A["p_stay_s"]
        s[:, t] = np.where(ch, rng.choice(NS, n_ep, p=base), s[:, t - 1])
        cz = rng.random(n_ep) > A["p_stay_z"]
        z[:, t] = np.where(cz, rng.choice([-1, 1], n_ep), z[:, t - 1])
    amp = np.zeros((NS, NC), np.float32)                      # amp[s, cam]
    for si in range(NS):
        amp[si, si] = A["amp_required"]
        if si != 0:
            amp[si, 0] = A["amp_fw_leak_follow"] if si == 1 else A["amp_fw_leak"]
    sig = amp[s] * z[..., None]                               # [E,T,NC] clean decision signal
    feat = rng.standard_normal((n_ep, T, NC, 3)).astype(np.float32)
    feat[..., 0] += sig
    ang = 2 * np.pi * np.arange(NS) / NS
    code = A["fw_code_radius"] * np.stack([np.cos(ang), np.sin(ang)], 1).astype(np.float32)
    feat[:, :, 0, 1:3] += code[s]                             # FW carries a weak situation code
    v = np.array(A["ego_v_mean"], np.float32)[s] + A["ego_v_sd"] * rng.standard_normal(s.shape)
    yaw = np.array(A["ego_yaw_mean"], np.float32)[s] + A["ego_yaw_sd"] * rng.standard_normal(s.shape)
    navL = (rng.random(s.shape) < np.where(np.isin(s, [2, 4]), A["nav_p_true"], A["nav_p_false"]))
    navR = (rng.random(s.shape) < np.where(np.isin(s, [3, 5, 6]), A["nav_p_true"], A["nav_p_false"]))
    cues = np.stack([v, yaw, navL, navR], -1).astype(np.float32)
    return dict(s=s, z=z, sig=sig.astype(np.float32), feat=feat, cues=cues, y=(z > 0).astype(np.float32))


# ----------------------------------------------------------------------------------------------
class MLP:
    """1-hidden-layer relu MLP with hand-written backprop + Adam (numpy)."""

    def __init__(self, din, dh, dout, rng, lr=5e-3):
        self.W1 = (rng.standard_normal((din, dh)) * np.sqrt(2.0 / din)).astype(np.float32)
        self.b1 = np.zeros(dh, np.float32)
        self.W2 = (rng.standard_normal((dh, dout)) * np.sqrt(1.0 / dh)).astype(np.float32)
        self.b2 = np.zeros(dout, np.float32)
        self.lr = lr
        self.m = [np.zeros_like(p) for p in self.params()]
        self.v = [np.zeros_like(p) for p in self.params()]
        self.t = 0

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def fwd(self, x):
        h = np.maximum(x @ self.W1 + self.b1, 0)
        return h, h @ self.W2 + self.b2

    def step(self, x, h, dout):
        n = x.shape[0]
        gW2 = h.T @ dout / n
        gb2 = dout.mean(0)
        dh = (dout @ self.W2.T) * (h > 0)
        gW1 = x.T @ dh / n
        gb1 = dh.mean(0)
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params(), [gW1, gb1, gW2, gb2])):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            mh = self.m[i] / (1 - 0.9 ** self.t)
            vh = self.v[i] / (1 - 0.999 ** self.t)
            p -= (self.lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ----------------------------------------------------------------------------------------------
def head_inputs(feat, mask, cues):
    """feat [N,NC,3], mask [N,NC], cues [N,4] -> [N, 21+7+4]"""
    return np.concatenate([(feat * mask[..., None]).reshape(len(feat), -1), mask, cues], 1)


def flat(ep):
    E, T = ep["s"].shape
    return dict(s=ep["s"].reshape(-1), y=ep["y"].reshape(-1), z=ep["z"].reshape(-1),
                feat=ep["feat"].reshape(E * T, NC, 3), cues=ep["cues"].reshape(E * T, 4),
                sig=ep["sig"].reshape(E * T, NC))


def train_head(tr, mask_sampler, hidden, epochs, rng, bs=2048):
    """Decision head. mask_sampler(s, rng) -> per-sample mask [N,NC]."""
    N = len(tr["s"])
    mask_all = mask_sampler(tr["s"], rng)
    X = head_inputs(tr["feat"], mask_all, tr["cues"])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    net = MLP(X.shape[1], hidden, 1, rng)
    y = tr["y"][:, None]
    Xn = ((X - mu) / sd).astype(np.float32)
    for ep in range(epochs):
        perm = rng.permutation(N)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            h, o = net.fwd(Xn[idx])
            net.step(Xn[idx], h, (sigmoid(o) - y[idx]))
    net.mu, net.sd = mu.astype(np.float32), sd.astype(np.float32)
    return net


def head_prob(net, feat, mask, cues):
    X = (head_inputs(feat, mask, cues) - net.mu) / net.sd
    return sigmoid(net.fwd(X.astype(np.float32))[1])[:, 0]


def train_gate(tr, Err, lam, use_prev, rng, hidden=32, epochs=18, bs=2048, prev_in=None, use_ego=True):
    """Gate minimises sum_j p_j (Err_j + lam*cost_j)   (full-information cost-sensitive learning)."""
    X = gate_inputs(tr["feat"][:, 0, :], tr["cues"] if use_ego else None, prev_in if use_prev else None)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)
    net = MLP(X.shape[1], hidden, NO, rng)
    C = (Err + lam * COST[None, :]).astype(np.float32)
    N = len(Xn)
    for ep in range(epochs):
        perm = rng.permutation(N)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            h, z = net.fwd(Xn[idx])
            p = softmax(z)
            c = C[idx]
            net.step(Xn[idx], h, p * (c - (p * c).sum(1, keepdims=True)))
    net.mu, net.sd, net.use_prev, net.kind, net.use_ego = mu.astype(np.float32), sd.astype(np.float32), use_prev, "cost", use_ego
    return net


def train_sitclf(tr, use_prev, rng, prev_in=None, hidden=32, epochs=18, bs=2048):
    """Calibrated situation posterior q(s|x): cross-entropy on the (label-derived, offline) situation. Same inputs as the gate."""
    X = gate_inputs(tr["feat"][:, 0, :], tr["cues"], prev_in if use_prev else None)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)
    net = MLP(X.shape[1], hidden, NS, rng)
    Y = np.eye(NS, dtype=np.float32)[tr["s"]]
    N = len(Xn)
    for ep in range(epochs):
        perm = rng.permutation(N)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            h, z = net.fwd(Xn[idx])
            net.step(Xn[idx], h, softmax(z) - Y[idx])
    net.mu, net.sd, net.use_prev, net.kind = mu.astype(np.float32), sd.astype(np.float32), use_prev, "sit"
    return net


def gate_inputs(fw_feat, cues, prev):
    parts = [fw_feat]
    if cues is not None:
        parts.append(cues)
    if prev is not None:
        parts.append(prev)
    return np.concatenate(parts, 1)


def gate_probs(net, fw_feat, cues, prev):
    X = (gate_inputs(fw_feat, cues if getattr(net, "use_ego", True) else None, prev if net.use_prev else None) - net.mu) / net.sd
    p = softmax(net.fwd(X.astype(np.float32))[1])
    if getattr(net, "kind", "cost") == "sit":                 # situation posterior -> option posterior (bijection), ALL=0
        p = np.concatenate([p, np.zeros((len(p), 1), np.float32)], 1)
    return p


def onehot_opt(o, drop_mask=None):
    oh = np.zeros((len(o), NO), np.float32)
    oh[np.arange(len(o)), o] = 1
    if drop_mask is not None:
        oh[drop_mask] = 0
    return oh


# ----------------------------------------------------------------------------------------------
def mask_fixed(option):
    return lambda s, rng: np.tile(MASK[option], (len(s), 1))


def mask_curriculum(s, rng):
    """camera-dropout curriculum (budget-conditioned head), per sample:
       40% the situation's oracle single camera | 20% oracle + one random extra camera (hedged pair)
       15% ALL | 25% FW + a random subset of 0-2 cameras (includes wrong-camera cases)."""
    N = len(s)
    u = rng.random(N)
    m = np.zeros((N, NC), np.float32)
    m[:, 0] = 1
    orc = ORACLE_OPT[s]
    idx = np.arange(N)
    a = u < 0.40
    m[idx[a], orc[a]] = 1
    b = (u >= 0.40) & (u < 0.60)
    m[idx[b], orc[b]] = 1
    m[idx[b], rng.integers(0, NC, b.sum())] = 1
    c = (u >= 0.60) & (u < 0.75)
    m[c] = 1
    d = u >= 0.75
    k = rng.integers(0, 3, d.sum())
    for j in range(2):
        sel = idx[d][k > j]
        m[sel, rng.integers(0, NC, len(sel))] = 1
    return m


# ----------------------------------------------------------------------------------------------
def episode_metrics(p_correct_prob, y, z, E, T):
    """per-episode accuracy and miss rate from P(y=1)."""
    pred = (p_correct_prob > 0.5).astype(np.float32)
    corr = (pred == y).reshape(E, T)
    acc = corr.mean(1)
    hold = (z.reshape(E, T) < 0)
    miss = ((pred.reshape(E, T) == 1) & hold).sum(1) / np.maximum(hold.sum(1), 1)
    return acc, miss


def run_sequential(ep, head, gate, mode, hyst, cold_on, rng, prev_mode="closed", tau=None, lead=0, hedge_tau=None):
    """Tick-by-tick policy. mode in {'gate','oracle'}; hyst dict; returns per-episode acc/miss/cost/switch."""
    E, T = ep["s"].shape
    A = ASSUMPTIONS
    cf = np.array(A["cold_factor_first_ticks"] + [1.0], np.float32)
    cur = np.zeros(E, int)
    last_cand = np.zeros(E, int)
    cnt = np.zeros(E, int)
    age = np.full((E, NC), 99)
    prev_opt = np.full(E, -1)
    ema = None
    accs, miss_n, hold_n, costs, sw = np.zeros(E), np.zeros(E), np.zeros(E), np.zeros(E), np.zeros(E)
    chosen = np.zeros((E, T), int)
    for t in range(T):
        fw = ep["feat"][:, t, 0, :]
        cu = ep["cues"][:, t]
        if mode == "oracle":
            prop = ORACLE_OPT[ep["s"][:, t]]
            prob = onehot_opt(prop)
        else:
            if prev_mode == "closed":
                prv = onehot_opt(np.maximum(prev_opt, 0), drop_mask=(prev_opt < 0) | (prev_opt == 7))
            else:                                             # 'open': noisy oracle of previous situation
                so = ep["s"][:, max(t - 1, 0)]
                wrong = rng.random(E) < A["prev_token_err_train"]
                so = np.where(wrong, rng.integers(0, NO, E), ORACLE_OPT[so])
                prv = onehot_opt(so)
            prob = gate_probs(gate, fw, cu, prv)
            if hyst.get("ema", 0) > 0:
                ema = prob if ema is None else hyst["ema"] * ema + (1 - hyst["ema"]) * prob
                prob = ema
            prop = prob.argmax(1)
            if tau is not None:                               # escalate to ALL under uncertainty
                prop = np.where(prob.max(1) < tau, 7, prop)
        # ---- hysteresis (margin + dwell, asymmetric escalate-fast / de-escalate-slow) ----
        if mode == "gate" and hyst.get("dwell", False):
            diff = prop != cur
            same = prop == last_cand
            cnt = np.where(diff & same, cnt + 1, np.where(diff, 1, 0))
            last_cand = prop
            need = np.where(COST[prop] < COST[cur], hyst["h_down"], hyst["h_up"])   # only de-escalation is delayed
            margin_ok = (prob[np.arange(E), prop] - prob[np.arange(E), cur]) > hyst["margin"]
            if tau is not None:                               # forced escalation to ALL bypasses the margin test
                margin_ok = margin_ok | (prob.max(1) < tau)
            switch = diff & (cnt >= need) & margin_ok
            new = np.where(switch, prop, cur)
            cnt = np.where(switch, 0, cnt)
        else:
            new = prop
        sw += (new != cur) if t > 0 else 0
        cur = new
        chosen[:, t] = cur
        mask = MASK[cur]
        if mode == "gate" and hedge_tau is not None:          # hedge: add the runner-up camera when the gate is unsure (cost 3, not 7)
            pp = prob[:, :7].copy()
            pp[np.arange(E), np.minimum(cur, 6)] = -1.0
            alt = pp.argmax(1)
            unc = prob.max(1) < hedge_tau
            mask = np.where(unc[:, None], np.maximum(mask, MASK[alt]), mask)
        if mode == "oracle" and lead > 0:                     # prefetch: also switch on the NEXT situation's camera
            mask = np.maximum(mask, MASK[ORACLE_OPT[ep["s"][:, min(T - 1, t + lead)]]])
        age = np.where(mask > 0, np.where(age >= 99, 0, age + 1), 99)   # 0 on the tick a camera turns on
        feat = ep["feat"][:, t].copy()
        if cold_on:
            fac = cf[np.minimum(age, 2)]
            fac[:, 0] = 1.0
            fac = np.where(age >= 99, 1.0, fac)
            feat[..., 0] = feat[..., 0] - ep["sig"][:, t] * (1.0 - fac)
        p = head_prob(head, feat, mask, cu)
        pred = p > 0.5
        accs += (pred == (ep["y"][:, t] > 0.5))
        hold = ep["z"][:, t] < 0
        miss_n += (pred & hold)
        hold_n += hold
        costs += mask.sum(1)
        prev_opt = cur
    return dict(acc=accs / T, miss=miss_n / np.maximum(hold_n, 1), cost=costs / T, sw=sw / T * 100, chosen=chosen)


def wrong_run_len(chosen, s_true):
    wrongm = chosen != ORACLE_OPT[s_true]
    runs = []
    for e in range(wrongm.shape[0]):
        c = 0
        for t in range(wrongm.shape[1]):
            if wrongm[e, t]:
                c += 1
            elif c:
                runs.append(c); c = 0
        if c:
            runs.append(c)
    return float(np.mean(runs)) if runs else 0.0


def boot_ci(x, B=2000, seed=0, paired_y=None):
    """episode-cluster bootstrap of the mean (or of the paired mean difference)."""
    rng = np.random.default_rng(seed)
    d = x if paired_y is None else x - paired_y
    n = len(d)
    idx = rng.integers(0, n, (B, n))
    bm = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))


def summarise(name, r, ref=None):
    out = dict(acc=boot_ci(r["acc"]), miss=boot_ci(r["miss"]), cost=float(r["cost"].mean()), sw_per_100=float(r["sw"].mean()))
    if ref is not None:
        out["d_acc_vs_ref"] = boot_ci(r["acc"], paired_y=ref["acc"])
        out["d_miss_vs_ref"] = boot_ci(r["miss"], paired_y=ref["miss"])
    return out


# ----------------------------------------------------------------------------------------------
def one_seed(seed, quick=False):
    rng = np.random.default_rng(1000 + seed)
    A = ASSUMPTIONS
    T = A["T"]
    tr_ep = gen_episodes(A["n_train_ep"], T, rng)
    te_ep = gen_episodes(A["n_test_ep"], T, rng)
    tr, te = flat(tr_ep), flat(te_ep)
    E, N = A["n_test_ep"], A["n_test_ep"] * T
    R = {}
    # ---- heads ----
    ep_h = 16
    h_fixed1 = train_head(tr, mask_fixed(0), 32, ep_h, rng)
    h_all = train_head(tr, mask_fixed(7), 32, ep_h, rng)
    h_all_big = train_head(tr, mask_fixed(7), 128, ep_h * 2, rng)
    h_curr = train_head(tr, mask_curriculum, 32, ep_h, rng)
    # published fixed-camera profiles as baselines (cameras = indices into CAMS)
    M4 = np.zeros(NC, np.float32); M4[[0, 1, 2, 3]] = 1          # Alpamayo-R1 released default: CL, FW, CR, FT
    M6 = np.zeros(NC, np.float32); M6[[0, 1, 2, 3, 4, 5]] = 1    # Alpamayo-2-Super trajectory profile: all but rear-tele
    h_f4 = train_head(tr, lambda s_, r_: np.tile(M4, (len(s_), 1)), 32, ep_h, rng)
    h_f6 = train_head(tr, lambda s_, r_: np.tile(M6, (len(s_), 1)), 32, ep_h, rng)
    # ceiling reference: all cameras + true situation one-hot appended as extra cue (privileged; reference only)
    trc = dict(tr); tec = dict(te)
    trc["cues"] = np.concatenate([tr["cues"], np.eye(NS, dtype=np.float32)[tr["s"]]], 1)
    tec["cues"] = np.concatenate([te["cues"], np.eye(NS, dtype=np.float32)[te["s"]]], 1)
    h_ceiling = train_head(trc, mask_fixed(7), 32, ep_h, rng)

    # ---- static evaluation of every (situation, option) cell with the curriculum head: 'regret matrix' ----
    P_opt = np.stack([head_prob(h_curr, te["feat"], np.tile(MASK[o], (N, 1)), te["cues"]) for o in range(NO)], 1)  # [N,8]
    corr_opt = ((P_opt > 0.5) == (te["y"][:, None] > 0.5)).astype(np.float32)
    regret = np.zeros((NS, NO))
    for si in range(NS):
        regret[si] = corr_opt[te["s"] == si].mean(0)
    R["regret_acc_by_situation_option"] = regret.round(3).tolist()

    # ---- gate training data: per-option error of the curriculum head on the TRAIN set ----
    Ntr = len(tr["s"])
    def err_opt(o):
        pp = head_prob(h_curr, tr["feat"], np.tile(MASK[o], (Ntr, 1)), tr["cues"])
        return 1.0 - np.where(tr["y"] > 0.5, pp, 1.0 - pp)          # 1 - P(correct)
    Err = np.stack([err_opt(o) for o in range(NO)], 1)
    # previous-tick token for training the gate (teacher-forced, noisy)
    s_prev = np.concatenate([tr_ep["s"][:, :1], tr_ep["s"][:, :-1]], 1).reshape(-1)
    wrong = rng.random(Ntr) < A["prev_token_err_train"]
    prev_tr = onehot_opt(np.where(wrong, rng.integers(0, NO, Ntr), ORACLE_OPT[s_prev]))

    lams = [0.0, 0.02, 0.04, 0.08, 0.16, 0.32] if not quick else [0.0, 0.04, 0.16]
    gates = {}
    for lam in lams:
        gates[lam] = train_gate(tr, Err, lam, True, rng, prev_in=prev_tr)
    # ---- reference policies ----
    none = dict(dwell=False)
    cache = {}
    # static baselines through the same machinery (option fixed): emulate with 'oracle' style loop not needed -> direct
    def static_res(net, option):
        p = head_prob(net, te["feat"], np.tile(MASK[option], (N, 1)), te["cues"])
        acc, miss = episode_metrics(p, te["y"], te["z"], E, T)
        return dict(acc=acc, miss=miss, cost=np.full(E, COST[option]), sw=np.zeros(E))
    def static_res_mask(net, mv):
        p = head_prob(net, te["feat"], np.tile(mv, (N, 1)), te["cues"])
        acc, miss = episode_metrics(p, te["y"], te["z"], E, T)
        return dict(acc=acc, miss=miss, cost=np.full(E, float(mv.sum())), sw=np.zeros(E))
    r_f4 = static_res_mask(h_f4, M4)
    r_f6 = static_res_mask(h_f6, M6)
    r_f1 = static_res(h_fixed1, 0)
    r_all = static_res(h_all, 7)
    r_allb = static_res(h_all_big, 7)
    pc = head_prob(h_ceiling, te["feat"], np.tile(MASK[7], (N, 1)), tec["cues"])
    a, m = episode_metrics(pc, te["y"], te["z"], E, T)
    r_ceil = dict(acc=a, miss=m, cost=np.full(E, 7.0), sw=np.zeros(E))
    r_or_warm = run_sequential(te_ep, h_curr, None, "oracle", none, False, rng)
    r_or_cold = run_sequential(te_ep, h_curr, None, "oracle", none, True, rng)
    r_or_lead = run_sequential(te_ep, h_curr, None, "oracle", none, True, rng, lead=2)
    R["policies"] = {
        "fixed1_FW_only": summarise("fixed1", r_f1),
        "fixed4_AR1_default(CL,FW,CR,FT)": summarise("f4", r_f4),
        "fixed6_AR2_trajectory_profile(all but RT)": summarise("f6", r_f6),
        "fixedall_H32": summarise("fixedall", r_all),
        "fixedall_H128_2x_epochs": summarise("fixedall_big", r_allb),
        "ceiling_all_cams_plus_true_situation(privileged,reference)": summarise("ceil", r_ceil),
        "oracle_gate_warm(no cold start)": summarise("or_warm", r_or_warm, r_all),
        "oracle_gate_with_cold_start": summarise("or_cold", r_or_cold, r_all),
        "oracle_gate_with_cold_start_prefetch_2_ticks": summarise("or_lead", r_or_lead, r_all),
    }
    # ---- gated frontier (lambda sweep), asymmetric hysteresis vs none, cold start on ----
    hyst_asym = dict(dwell=True, h_up=1, h_down=3, margin=0.10)
    frontier = []
    for lam in lams:
        r = run_sequential(te_ep, h_curr, gates[lam], "gate", hyst_asym, True, rng)
        s = summarise("gate", r, r_all)
        s["lam"] = lam
        # gate accuracy vs oracle option
        s["gate_option_match_oracle"] = float((r["chosen"] == ORACLE_OPT[te_ep["s"]]).mean())
        s["frac_ticks_ALL"] = float((r["chosen"] == 7).mean())
        frontier.append(s)
        cache[("gate", lam)] = r
    R["frontier_gated_asym_hyst_cold_on"] = frontier
    # knee: smallest cost whose accuracy is within 1 pt of best gated accuracy
    best_acc = max(f["acc"][0] for f in frontier)
    knee = min([f for f in frontier if f["acc"][0] >= best_acc - 0.01], key=lambda f: f["cost"])
    lam_star = knee["lam"]
    R["lam_star"] = lam_star
    g = gates[lam_star]

    # ---- hysteresis study at lam_star ----
    H = {}
    for name, h in {"none": none, "ema0.6": dict(dwell=False, ema=0.6),
                    "dwell_sym(h=2,m=.10)": dict(dwell=True, h_up=2, h_down=2, margin=0.10),
                    "asym(up=1,down=3,m=.10)": hyst_asym,
                    "asym_long(up=1,down=8,m=.10)": dict(dwell=True, h_up=1, h_down=8, margin=0.10)}.items():
        for cold in (False, True):
            r = run_sequential(te_ep, h_curr, g, "gate", h, cold, rng)
            H[f"{name}|cold={'on' if cold else 'off'}"] = summarise(name, r, r_all)
            cache[("H", name, cold)] = r
    R["hysteresis_at_lam_star"] = H

    # ---- cue ablation: gate without previous-tick token; open-loop prev token; closed-loop lock-in ----
    g_noprev = train_gate(tr, Err, lam_star, False, rng)
    C = {}
    r = run_sequential(te_ep, h_curr, g_noprev, "gate", hyst_asym, True, rng)
    C["gate_ego+FW_only(no prev token)"] = summarise("g0", r, r_all)
    C["gate_ego+FW_only(no prev token)"]["match"] = float((r["chosen"] == ORACLE_OPT[te_ep["s"]]).mean())
    C["gate_ego+FW_only(no prev token)"]["mean_wrong_run_len_ticks"] = wrong_run_len(r["chosen"], te_ep["s"])
    r = run_sequential(te_ep, h_curr, g, "gate", hyst_asym, True, rng, prev_mode="closed")
    C["gate_+prev_token_CLOSED_loop"] = summarise("g1c", r, r_all)
    C["gate_+prev_token_CLOSED_loop"]["match"] = float((r["chosen"] == ORACLE_OPT[te_ep["s"]]).mean())
    C["gate_+prev_token_CLOSED_loop"]["mean_wrong_run_len_ticks"] = wrong_run_len(r["chosen"], te_ep["s"])
    r = run_sequential(te_ep, h_curr, g, "gate", hyst_asym, True, rng, prev_mode="open")
    C["gate_+prev_token_OPEN_loop(noisy oracle)"] = summarise("g1o", r, r_all)
    C["gate_+prev_token_OPEN_loop(noisy oracle)"]["match"] = float((r["chosen"] == ORACLE_OPT[te_ep["s"]]).mean())
    C["gate_+prev_token_OPEN_loop(noisy oracle)"]["mean_wrong_run_len_ticks"] = wrong_run_len(r["chosen"], te_ep["s"])
    q1 = train_sitclf(tr, True, rng, prev_in=prev_tr)
    r = run_sequential(te_ep, h_curr, q1, "gate", hyst_asym, True, rng)
    kq = "supervised_situation_gate(CE on label-derived s, cost-blind)+prev"
    C[kq] = summarise("sit", r, r_all)
    C[kq]["match"] = float((r["chosen"] == ORACLE_OPT[te_ep["s"]]).mean())
    R["cue_ablation_at_lam_star"] = C

    # ---- confusion of gate vs oracle (closed loop, asym hysteresis, cold on) ----
    r = cache[("gate", lam_star)]
    conf = np.zeros((NS, NO))
    for si in range(NS):
        sel = te_ep["s"] == si
        conf[si] = np.bincount(r["chosen"][sel], minlength=NO) / max(sel.sum(), 1)
    R["gate_confusion_rows=true_situation,cols=option"] = conf.round(3).tolist()

    # ---- cost of a wrong camera: oracle with injected error rate eps (warm & cold), with/without escalation ----
    Eps = []
    for eps in [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]:
        accs = []
        for rep in range(1):
            s_flat = te["s"]
            o = ORACLE_OPT[s_flat]
            bad = rng.random(N) < eps
            o2 = np.where(bad, (o + rng.integers(1, 7, N)) % 7, o)   # a wrong single-camera option (never ALL)
            p = P_opt[np.arange(N), o2]
            a_, m_ = episode_metrics(p, te["y"], te["z"], E, T)
            accs.append((a_.mean(), m_.mean()))
        Eps.append(dict(eps=eps, acc=float(np.mean([a for a, _ in accs])), miss=float(np.mean([m for _, m in accs])), cost=2.0))
    R["error_injection_oracle_with_wrong_single_camera"] = Eps

    # ---- escalation-on-uncertainty sweep: (i) the cost-trained gate's softmax, (ii) the calibrated situation head ----
    Esc = {}
    s_prev_te = np.concatenate([te_ep["s"][:, :1], te_ep["s"][:, :-1]], 1).reshape(-1)
    pg = gate_probs(g, te["feat"][:, 0, :], te["cues"], onehot_opt(ORACLE_OPT[s_prev_te]))
    Esc["cost_gate_frac_ticks_maxprob>0.99"] = float((pg.max(1) > 0.99).mean())
    for name, net in (("cost_gate", g), ("situation_head", q1)):
        rows = []
        for tau in [None, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
            r = run_sequential(te_ep, h_curr, net, "gate", hyst_asym, True, rng, tau=tau)
            sm = summarise("esc", r, r_all); sm["tau"] = tau; sm["frac_ALL"] = float((r["chosen"] == 7).mean())
            rows.append(sm)
        Esc[name] = rows
    R["escalate_to_ALL_if_max_prob_below_tau"] = Esc
    Hg = {}
    for name, net in (("cost_gate", g), ("situation_head", q1)):
        rows = []
        for ht in [None, 0.6, 0.8, 0.9, 0.95, 0.99]:
            r = run_sequential(te_ep, h_curr, net, "gate", hyst_asym, True, rng, hedge_tau=ht)
            sm = summarise("hedge", r, r_all); sm["hedge_tau"] = ht
            rows.append(sm)
        Hg[name] = rows
    R["hedge_add_runner_up_camera_if_max_prob_below_tau"] = Hg

    # ---- VISION-ONLY gate arms (no ego cues, no route/nav cue at inference; FW features [+ own previous option]) ----
    rng_v = np.random.default_rng(5000 + seed)
    V = {}
    gv0 = train_gate(tr, Err, lam_star, False, rng_v, use_ego=False)
    gv1 = train_gate(tr, Err, lam_star, True, rng_v, prev_in=prev_tr, use_ego=False)
    for name, net in (("gate_VISION_ONLY(FW feat)", gv0), ("gate_VISION_ONLY(FW feat)+prev_token_closed_loop", gv1)):
        r = run_sequential(te_ep, h_curr, net, "gate", hyst_asym, True, rng_v)
        V[name] = summarise(name, r, r_all)
        V[name]["match"] = float((r["chosen"] == ORACLE_OPT[te_ep["s"]]).mean())
        V[name]["mean_wrong_run_len_ticks"] = wrong_run_len(r["chosen"], te_ep["s"])
    R["vision_only_gate_arms_at_lam_star"] = V

    R["headline"] = dict(
        fixed1=(float(r_f1["acc"].mean()), float(r_f1["miss"].mean()), 1.0),
        fixedall=(float(r_all["acc"].mean()), float(r_all["miss"].mean()), 7.0),
        fixedall_big=(float(r_allb["acc"].mean()), float(r_allb["miss"].mean()), 7.0),
        oracle_cold=(float(r_or_cold["acc"].mean()), float(r_or_cold["miss"].mean()), float(r_or_cold["cost"].mean())),
        gated_star=(knee["acc"][0], knee["miss"][0], knee["cost"]))
    R["seed"] = seed
    return R


def vision_sensitivity(radius, seed=0):
    """How much SITUATION information must the always-on front-wide embedding carry for a VISION-ONLY gate to work?
    Re-generates the world with a different FW situation-code radius (SNR of the FW situation code) and re-runs only the
    gate arms (ego-cue gate vs vision-only gates), single seed. lam fixed at 0.04."""
    A = ASSUMPTIONS
    old = A["fw_code_radius"]
    A["fw_code_radius"] = radius
    try:
        rng = np.random.default_rng(7000 + seed)
        T = A["T"]
        tr_ep = gen_episodes(A["n_train_ep"], T, rng)
        te_ep = gen_episodes(A["n_test_ep"], T, rng)
        tr, te = flat(tr_ep), flat(te_ep)
        E, N = A["n_test_ep"], A["n_test_ep"] * T
        Ntr = len(tr["s"])
        h_curr = train_head(tr, mask_curriculum, 32, 16, rng)
        h_all = train_head(tr, mask_fixed(7), 32, 16, rng)
        m = np.tile(MASK[7], (N, 1))
        a, mi = episode_metrics(head_prob(h_all, te["feat"], m, te["cues"]), te["y"], te["z"], E, T)
        r_all = dict(acc=a, miss=mi, cost=np.full(E, 7.0), sw=np.zeros(E))
        def err_opt(o):
            pp = head_prob(h_curr, tr["feat"], np.tile(MASK[o], (Ntr, 1)), tr["cues"])
            return 1.0 - np.where(tr["y"] > 0.5, pp, 1.0 - pp)
        Err = np.stack([err_opt(o) for o in range(NO)], 1)
        s_prev = np.concatenate([tr_ep["s"][:, :1], tr_ep["s"][:, :-1]], 1).reshape(-1)
        wrong = rng.random(Ntr) < A["prev_token_err_train"]
        prev_tr = onehot_opt(np.where(wrong, rng.integers(0, NO, Ntr), ORACLE_OPT[s_prev]))
        hyst = dict(dwell=True, h_up=1, h_down=3, margin=0.10)
        out = dict(fw_code_radius=radius, fixedall_acc=float(a.mean()))
        arms = {"ego_cue_gate+prev": (True, True), "vision_only+prev": (False, True), "vision_only": (False, False)}
        for name, (ego, prv) in arms.items():
            g = train_gate(tr, Err, 0.04, prv, rng, prev_in=prev_tr, use_ego=ego)
            r = run_sequential(te_ep, h_curr, g, "gate", hyst, True, rng)
            out[name] = dict(acc=float(r["acc"].mean()), miss=float(r["miss"].mean()), cost=float(r["cost"].mean()),
                             match=float((r["chosen"] == ORACLE_OPT[te_ep["s"]]).mean()),
                             mean_wrong_run_len_ticks=wrong_run_len(r["chosen"], te_ep["s"]))
        return out
    finally:
        A["fw_code_radius"] = old


def main():
    quick = "--quick" in sys.argv
    nseed = 1 if quick else 3
    out = dict(assumptions=ASSUMPTIONS, cams=CAMS, situations=SIT, options=OPT, cost_units=COST.tolist(), seeds=[])
    for sd in range(nseed):
        out["seeds"].append(one_seed(sd, quick))
        print(f"seed {sd} done at {time.time() - T0:.1f}s", flush=True)
    hs = np.array([[v for v in s["headline"][k]] for s in out["seeds"] for k in ["fixed1", "fixedall", "fixedall_big", "oracle_cold", "gated_star"]])
    hs = hs.reshape(nseed, 5, 3)
    names = ["fixed1", "fixedall_H32", "fixedall_H128", "oracle_cold_start", "gated_lam_star"]
    out["headline_mean_sd_over_seeds"] = {n: dict(acc=[float(hs[:, i, 0].mean()), float(hs[:, i, 0].std())],
                                                  miss=[float(hs[:, i, 1].mean()), float(hs[:, i, 1].std())],
                                                  cost=[float(hs[:, i, 2].mean()), float(hs[:, i, 2].std())]) for i, n in enumerate(names)}
    if not quick:
        out["sensitivity_fw_code_radius_seed0_lam0.04"] = [vision_sensitivity(r_) for r_ in (3.0, 4.5)]
        print(f"sensitivity done at {time.time() - T0:.1f}s", flush=True)
    out["runtime_s"] = time.time() - T0
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rd_camera_gating_toy_result.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps(out["headline_mean_sd_over_seeds"], indent=1))
    print("runtime %.1fs -> %s" % (out["runtime_s"], path))


if __name__ == "__main__":
    main()
