"""leadspeed.py -- card PREREG_V8F_PROOF_ADD148 (LEADSPEED10K): a vision read of the lead's closing speed. Tier T0 (a head on cached tokens of logged frames); never
driving performance. Inference input: camera tokens only (no ego state, no label).

Pieces (pure where possible, tested in test_leadspeed.py):
  LeadSpeedHead     per frame: LayerNorm -> Linear(d_in, w) -> GELU -> attention pool over the tokens -> z_t; over the last H frames: two causal Conv1d (kernel 5) -> at the
                    last frame: presence logit, gap (x 50 m), closing (x 10 m/s). temporal=False replaces the convolutions by the mean over frames (the 2026-11-15 form).
  token_labels      the lead file rows of one clip on the token clock (k = 2 j): presence, gap (valid lead with 0 <= gap <= 60 m) and the closing label = least-squares slope of
                    the label gap over the trailing run of up to 5 token frames (steps <= 3.0 m, at least 3 frames), the formula of LabelSource.read; NaN where the run is shorter.
  clock_check       the lead file's own speed at k = 2 (j + s) against the token cache's speed at j, s = -2 .. +2: mean absolute error per shift.
  windows           the H-frame windows ending at frames t = H-1 .. T-1 of a clip (past frames only).
"""
from __future__ import annotations

import numpy as np

H = 10
DT = 0.2
GAP_SCALE, CLOS_SCALE = 50.0, 10.0
GAP_MAX_M, STEP_MAX_M, RUN_MAX, RUN_MIN = 60.0, 3.0, 5, 3


# ---------------------------------------------------------------- labels (numpy)
def token_labels(k, has, lead, gap, T):
    """Rows of one clip (k on the 0.1 s clock) -> presence [T] (bool), gap [T] (NaN without a valid lead), closing [T] (NaN where the run is shorter than 3)."""
    pres = np.zeros(T, bool)
    g = np.full(T, np.nan)
    for kk, hh, ll, gg in zip(k, has, lead, gap):
        if kk % 2 or kk < 0 or kk // 2 >= T:
            continue
        j = int(kk // 2)
        if hh == 1 and ll == 1 and np.isfinite(gg) and 0.0 <= gg <= GAP_MAX_M:
            pres[j] = True
            g[j] = float(gg)
    clos = np.full(T, np.nan)
    for j in range(T):
        if not pres[j]:
            continue
        idx = [j]
        f = j - 1
        while f >= 0 and len(idx) < RUN_MAX and pres[f] and abs(g[f + 1] - g[f]) <= STEP_MAX_M:
            idx.append(f)
            f -= 1
        if len(idx) < RUN_MIN:
            continue
        idx = idx[::-1]
        G = g[idx]
        t = np.arange(len(idx), dtype=np.float64) * DT
        tb = t.mean()
        clos[j] = float(((t - tb) * (G - G.mean())).sum() / ((t - tb) ** 2).sum())
    return pres, g, clos


def clock_check(k, v_lead, v_tok, shifts=(-2, -1, 0, 1, 2)):
    """-> {shift: (mae, n)}: |v_lead(k = 2 (j + s)) - v_tok(j)| over the frames where both exist."""
    by_k = {int(a): float(b) for a, b in zip(k, v_lead) if np.isfinite(b)}
    out = {}
    for s in shifts:
        e = [abs(by_k[2 * (j + s)] - float(v_tok[j])) for j in range(len(v_tok)) if 2 * (j + s) in by_k and np.isfinite(v_tok[j])]
        out[s] = (float(np.mean(e)) if e else float("inf"), len(e))
    return out


def windows(T, h=H):
    """end frames t of the windows t-h+1 .. t (past frames only)."""
    return np.arange(h - 1, T)


# ---------------------------------------------------------------- the head (torch)
def make_head(d_in=1024, width=64, temporal=True):
    import torch
    import torch.nn as nn

    class LeadSpeedHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.temporal = bool(temporal)
            self.ln = nn.LayerNorm(d_in)
            self.proj = nn.Linear(d_in, width)
            self.att = nn.Linear(width, 1)
            self.c1 = nn.Conv1d(width, width, 5)
            self.c2 = nn.Conv1d(width, width, 5)
            self.out = nn.Linear(width, 3)

        def frame(self, x):                                   # x [B, Hf, N, d_in] -> z [B, Hf, width]
            h = torch.nn.functional.gelu(self.proj(self.ln(x.float())))
            w = torch.softmax(self.att(h).squeeze(-1), dim=-1)
            return (w.unsqueeze(-1) * h).sum(-2)

        def forward(self, x):                                 # x [B, H, N, d_in] (frames t-H+1 .. t) -> [B, 3] at frame t
            z = self.frame(x)
            if self.temporal:
                y = z.transpose(1, 2)                         # [B, width, H]
                y = torch.nn.functional.gelu(self.c1(torch.nn.functional.pad(y, (4, 0))))
                y = torch.nn.functional.gelu(self.c2(torch.nn.functional.pad(y, (4, 0))))
                last = y[:, :, -1]
            else:
                last = z.mean(1)
            return self.out(last)

    return LeadSpeedHead()


def loss_terms(out, pres, gap, clos):
    """out [B,3]; pres [B] float 0/1; gap, clos [B] (NaN where absent) -> dict of terms (presence BCE; gap and closing L1 on scaled values where finite)."""
    import torch
    t = {"pres": torch.nn.functional.binary_cross_entropy_with_logits(out[:, 0], pres)}
    mg = torch.isfinite(gap)
    mc = torch.isfinite(clos)
    t["gap"] = (out[mg, 1] - gap[mg] / GAP_SCALE).abs().mean() if mg.any() else out.sum() * 0.0
    t["clos"] = (out[mc, 2] - clos[mc] / CLOS_SCALE).abs().mean() if mc.any() else out.sum() * 0.0
    return t


def r2(y, p):
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
    ss = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ((y - p) ** 2).sum() / ss) if ss > 0 else float("nan")


def pearson(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() > 0 and y.std() > 0 else float("nan")
