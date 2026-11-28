"""leadacc.py -- card PREREG_V8F_PROOF_ADD153 (ACCHIST10K): a vision head that reads the lead's acceleration from a longer history of image tokens. Tier T0 (a head on
cached tokens of logged frames); never driving performance. Inference input: camera tokens only (no ego state, no label). Imports leadspeed.py (card ADD148) unchanged.

  ARMS              the four stage-1 arms of the card: history H, temporal stack (None = row 217's two plain Conv1d k 5; a tuple = residual dilated Conv1d k 5), motion
  closing_labels    the label closing with a run of up to run_max token frames (run_max 5 = leadspeed.token_labels exactly; 10 = the card's 2 s training label)
  d_labels          d(t) = closing(t) - closing(t - 5), NaN where either is missing
  window            the uint8 tokens of frames t-H+1 .. t, zeros before the clip start, and the valid mask
  make_acc_head     the head: per-frame encoder (optionally with token differences), temporal stack, 4 outputs (presence, gap / 50 m, closing / 10 m/s, d / 10 m/s)
  clip_vectors      the read pairs of one clip on row 217's frames (S, A record, A derived, A floor, A rel, A lab2)
  reads / paired    Pearson reads from per-clip sufficient sums; clip-cluster bootstrap of differences on the same frames
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, "..", "leadspeed"), HERE):          # dev box: the sibling folder; box 2: copies in this folder (first on the path)
    if os.path.isdir(_p):
        sys.path.insert(0, os.path.abspath(_p))
import leadspeed as L  # noqa: E402

D_SCALE = 10.0
LAG = 5                                       # 1 s at 0.2 s
T_MIN = 9                                     # row 217's first window end: every arm emits from frame 9 on
ARMS = {
    "K25D": {"H": 25, "dil": (1, 2, 4), "motion": False},
    "K25DM": {"H": 25, "dil": (1, 2, 4), "motion": True},
    "K10D": {"H": 10, "dil": None, "motion": False},
    "K40D": {"H": 40, "dil": (1, 2, 4, 8), "motion": False},
}


# ---------------------------------------------------------------- labels (numpy)
def closing_labels(pres, g, run_max=L.RUN_MAX):
    """presence [T] and gap [T] from leadspeed.token_labels -> closing [T]: least-squares slope of the gap over the trailing run of up to run_max frames
    (steps <= 3.0 m, at least 3 frames); NaN where the run is shorter. run_max 5 reproduces leadspeed.token_labels."""
    T = len(pres)
    clos = np.full(T, np.nan)
    for j in range(T):
        if not pres[j]:
            continue
        idx = [j]
        f = j - 1
        while f >= 0 and len(idx) < run_max and pres[f] and abs(g[f + 1] - g[f]) <= L.STEP_MAX_M:
            idx.append(f)
            f -= 1
        if len(idx) < L.RUN_MIN:
            continue
        idx = idx[::-1]
        G = g[idx]
        t = np.arange(len(idx), dtype=np.float64) * L.DT
        tb = t.mean()
        clos[j] = float(((t - tb) * (G - G.mean())).sum() / ((t - tb) ** 2).sum())
    return clos


def d_labels(clos, lag=LAG):
    d = np.full(len(clos), np.nan)
    if len(clos) > lag:
        d[lag:] = clos[lag:] - clos[:-lag]
    return d


def window(tok, t, H):
    """tok [T, N, D] (any array, memmap included) -> (uint8 [H, N, D], valid bool [H]) for frames t-H+1 .. t; frames before 0 are zeros, valid False."""
    s = t - H + 1
    if s >= 0:
        return np.asarray(tok[s:t + 1]), np.ones(H, bool)
    w = np.zeros((H,) + tuple(tok.shape[1:]), np.uint8)
    w[-s:] = np.asarray(tok[0:t + 1])
    v = np.zeros(H, bool)
    v[-s:] = True
    return w, v


def window_ends(T):
    return np.arange(T_MIN, T)


# ---------------------------------------------------------------- the head (torch)
def make_acc_head(H, dil=None, motion=False, d_in=1024, width=64):
    import torch
    import torch.nn as nn
    F = torch.nn.functional

    class AccHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.H, self.dil, self.motion = int(H), (tuple(dil) if dil else None), bool(motion)
            self.ln = nn.LayerNorm(d_in)
            self.proj = nn.Linear(d_in, width)
            if self.motion:
                self.lnd = nn.LayerNorm(d_in)
                self.projd = nn.Linear(d_in, width)
            self.att = nn.Linear(width, 1)
            if self.dil is None:
                self.c1 = nn.Conv1d(width, width, 5)
                self.c2 = nn.Conv1d(width, width, 5)
            else:
                self.convs = nn.ModuleList([nn.Conv1d(width, width, 5, dilation=d) for d in self.dil])
            self.out = nn.Linear(width, 4)

        def frame(self, x, valid):                            # x [B, H, N, d_in] float, valid [B, H] bool -> z [B, H, width] (0 on invalid frames)
            pre = self.proj(self.ln(x))
            if self.motion:
                dx = torch.zeros_like(x)
                dx[:, 1:] = x[:, 1:] - x[:, :-1]
                dv = torch.zeros_like(valid)
                dv[:, 1:] = valid[:, 1:] & valid[:, :-1]
                dx = dx * dv[:, :, None, None].to(x.dtype)
                pre = pre + self.projd(self.lnd(dx))
            h = F.gelu(pre)
            w = torch.softmax(self.att(h).squeeze(-1), dim=-1)
            z = (w.unsqueeze(-1) * h).sum(-2)
            return z * valid[:, :, None].to(z.dtype)

        def forward(self, x, valid):                          # -> [B, 4] at the last frame
            y = self.frame(x, valid).transpose(1, 2)          # [B, width, H]
            if self.dil is None:
                y = F.gelu(self.c1(F.pad(y, (4, 0))))
                y = F.gelu(self.c2(F.pad(y, (4, 0))))
            else:
                for c, d in zip(self.convs, self.dil):
                    y = y + F.gelu(c(F.pad(y, (4 * d, 0))))
            return self.out(y[:, :, -1])

    return AccHead()


def receptive_field(dil):
    return 9 if dil is None else 1 + 4 * int(sum(dil))


def loss_terms(out, pres, gap, clos, d):
    import torch
    t = L.loss_terms(out[:, :3], pres, gap, clos)
    md = torch.isfinite(d)
    t["d"] = (out[md, 3] - d[md] / D_SCALE).abs().mean() if md.any() else out.sum() * 0.0
    return t


# ---------------------------------------------------------------- reads (numpy)
PAIRS = ("S", "A", "A_derived", "A_floor", "A_rel", "A_lab2")


def clip_vectors(d):
    """d: v, clos_lab, clos_lab2, clos_head, d_head [T] (NaN where absent) -> {pair name: (x label, y read)} on row 217's frames.
    S frames: label closing and head closing finite. A frames: f - 5 >= 0, label closing at f and f - 5, v at f and f - 5, head closing at f and f - 5 finite."""
    v, cl, ch, dh = d["v"], d["clos_lab"], d["clos_head"], d["d_head"]
    cl2 = d.get("clos_lab2", np.full(len(v), np.nan))
    out = {k: ([], []) for k in PAIRS}
    vl = lambda vv, c: max(0.0, vv + c)  # noqa: E731
    for f in range(len(v)):
        if np.isfinite(cl[f]) and np.isfinite(ch[f]):
            out["S"][0].append(cl[f]); out["S"][1].append(ch[f])
        g = f - LAG
        if g < 0 or not all(np.isfinite(x) for x in (cl[f], cl[g], v[f], v[g], ch[f], ch[g], dh[f])):
            continue
        al = vl(v[f], cl[f]) - vl(v[g], cl[g])
        out["A"][0].append(al); out["A"][1].append(vl(v[f], ch[f]) - vl(v[g], ch[f] - dh[f]))
        out["A_derived"][0].append(al); out["A_derived"][1].append(vl(v[f], ch[f]) - vl(v[g], ch[g]))
        out["A_floor"][0].append(al); out["A_floor"][1].append(vl(v[f], 0.0) - vl(v[g], 0.0))
        out["A_rel"][0].append(cl[f] - cl[g]); out["A_rel"][1].append(dh[f])
        if np.isfinite(cl2[f]) and np.isfinite(cl2[g]):
            out["A_lab2"][0].append(al); out["A_lab2"][1].append(vl(v[f], cl2[f]) - vl(v[g], cl2[g]))
    return {k: (np.asarray(a, np.float64), np.asarray(b, np.float64)) for k, (a, b) in out.items()}


def sums(x, y):
    return np.array([len(x), x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum()], np.float64)


def r_of(s):
    n, sx, sy, sxx, syy, sxy = s
    if n < 3:
        return float("nan")
    vx, vy = sxx - sx * sx / n, syy - sy * sy / n
    return float((sxy - sx * sy / n) / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else float("nan")


def clip_sums(cv):
    return {k: sums(*cv[k]) for k in PAIRS}


def reads(per_clip_sums):
    tot = {k: np.sum([c[k] for c in per_clip_sums] + [np.zeros(6)], axis=0) for k in PAIRS}      # no clip: n = 0, r NaN
    o = {k: r_of(tot[k]) for k in PAIRS}
    o.update({"n_S": int(tot["S"][0]), "n_A": int(tot["A"][0]), "n_A_lab2": int(tot["A_lab2"][0])})
    return o


def paired(sa, ka, sb, kb, n_boot=2000, seed=150):
    """sa, sb: per-clip sums (same clips, same frames); difference r(sa[ka]) - r(sb[kb]), clip-cluster bootstrap."""
    A = np.stack([c[ka] for c in sa])
    B = np.stack([c[kb] for c in sb])
    if not np.array_equal(A[:, 0], B[:, 0]):
        raise ValueError("paired read on different frames")
    d0 = r_of(A.sum(0)) - r_of(B.sum(0))
    rng = np.random.default_rng(seed)
    n = len(A)
    ds = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        ds.append(r_of(A[ix].sum(0)) - r_of(B[ix].sum(0)))
    return {"diff": d0, "ci95": [float(np.nanpercentile(ds, 2.5)), float(np.nanpercentile(ds, 97.5))], "n_clips": n,
            "estimator": "clip-cluster bootstrap, paired, %d resamples, seed %d; another draw of clips, one training run per arm" % (n_boot, seed)}


def name_of(A, lo_floor, S, prefix="ACCHIST10K"):
    if A >= 0.30 and lo_floor > 0:
        return prefix + "-CARRIED"
    return prefix + ("-SPEED-ONLY" if S >= 0.30 else "-NOT-CARRIED")
