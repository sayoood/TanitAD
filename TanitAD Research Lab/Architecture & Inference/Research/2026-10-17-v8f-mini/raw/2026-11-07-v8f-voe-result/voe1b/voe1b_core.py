"""PREREG_V8F_VOE1B core = VOE1's core (voe1/voe1_core.py, copied, not edited there) plus the deltas the VOE1B card names: the ACTION-SWAP-M donor rule,
the untrained-reference rule (c) and its labels (NOTICES / INIT-LEVEL / NULL-ONLY / BLIND), the calibrated control 1 (per-draw separation flags of the
reference draws), the leave-one-out control 2. Every other definition is VOE1's, unchanged (IMPL_CHOICES_VOE1.md 848a2c8c).
(pure numpy / torch; imports no tanitad code, so the synthetic test runs without the code export).

Tier T0: a world-model diagnostic on cached per-frame latents, never a driving claim.
Contents: the manipulation index maps, the per-level surprise traces (one-step, teacher-forced, sliding context 3), window scores,
the pairwise accuracy with the two-part rule (episode-cluster bootstrap + whole-episode label-swap permutation reference),
the label rules, and the event-alignment AP with the same two-part machinery.
Every definition here is the one fixed in IMPL_CHOICES_VOE1.md; this file computes, it decides nothing.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

# ------------------------------------------------------------------------------------------------ constants (IMPL_CHOICES section 1)
F_WIN = 16                       # window = 16 cache frames (0.2 s each), window frame k = clip frame g0 + k
T0 = 4                           # onset frame (window index): 4 frames of context, frames 5..12 are the 1.6 s after it
N_POST = 8                       # 1.6 s = 8 cache steps
NF = T0 + N_POST + 1             # 13 frames used per manipulated copy (0..12)
FREEZE_N = 4                     # 0.8 s
TELE_SHIFT = 8                   # 1.6 s
DT = 0.2
STRIDE = {1: 1, 2: 2, 3: 4}      # cache frames per step of level l (WMAux: strides (2, 2))
CTX = 3                          # the predictors' trained context
MANIPS = ("TELEPORT", "FREEZE", "REVERSE", "SPLICE", "ACTION-SWAP", "ACTION-SWAP-M")
NULL_COUNTS = ("TELEPORT", "REVERSE", "SPLICE")      # the Copy-Last null enters NOTICES only for these
B_BOOT = 2000
R_PERM = 199
CAL_SEEDS = tuple(range(1001, 1021))
EVENTS = ("brake", "turn", "lead")
EV_BAND = 2                      # +-0.4 s = +-2 cache steps
FREEZE_MIN_SPEED = 3.0
SWAP_LON, SWAP_YAW = 1.5, 0.15
SWM_REL, SWM_LON_MIN, SWM_YAW_MIN = 0.10, 0.5, 0.05      # ACTION-SWAP-M (VOE1B card): magnitude match +-10 %; sign test only where the mean is at least 0.5 m/s^2 / 0.05 rad/s
SWM_MIN_N = 300                                          # fewer qualifying windows: the ACTION-SWAP-M row is UNDERPOWERED
N_UNTRAINED, UT_SEED0 = 16, 100                          # the untrained reference: torch.manual_seed 100..115
LABELS_B = ("NOTICES", "INIT-LEVEL", "NULL-ONLY", "BLIND")


# ------------------------------------------------------------------------------------------------ index maps
def idx_unmod(base: np.ndarray, shift: int = 0):
    """(frame index [N, NF], action index [N, NF-1]) into the bank. Action j drives the transition j -> j+1."""
    base = np.asarray(base, np.int64)
    fr = base[:, None] + shift + np.arange(NF)[None]
    ac = base[:, None] + shift + np.arange(NF - 1)[None]
    return fr, ac


def idx_manip(name: str, base: np.ndarray, donor_base: np.ndarray | None = None):
    """Index maps of one manipulated copy. The onset frame T0 is kept; frames T0+1.. are replaced (IMPL_CHOICES section 2)."""
    base = np.asarray(base, np.int64)
    fr, ac = idx_unmod(base)
    j = np.arange(NF)
    post = j[T0 + 1:]
    if name == "TELEPORT":                          # the frames 1.6 s later in the same clip; actions stay logged
        fr[:, T0 + 1:] = base[:, None] + (post + TELE_SHIFT)[None]
    elif name == "FREEZE":                          # frame T0 held for 0.8 s, then the picture resumes from the held frame (a pause)
        fr[:, T0 + 1:T0 + 1 + FREEZE_N] = (base + T0)[:, None]
        rest = j[T0 + 1 + FREEZE_N:]
        fr[:, T0 + 1 + FREEZE_N:] = base[:, None] + (rest - FREEZE_N)[None]
    elif name == "REVERSE":                         # the 1.6 s after T0 play backwards: new[j] = old[2*T0 + N_POST + 1 - j]
        fr[:, T0 + 1:] = base[:, None] + (2 * T0 + N_POST + 1 - post)[None]
    elif name == "SPLICE":                          # frames from another clip (donor window, same window positions); actions stay logged
        fr[:, T0 + 1:] = np.asarray(donor_base, np.int64)[:, None] + post[None]
    elif name in ("ACTION-SWAP", "ACTION-SWAP-M"):  # real frames, the actions of transitions T0.. come from the donor window (M: the magnitude-matched donor)
        ac[:, T0:] = np.asarray(donor_base, np.int64)[:, None] + np.arange(T0, NF - 1)[None]
    else:
        raise ValueError(name)
    return fr, ac


def donor_derangement(episode_codes: np.ndarray, seed: int = 0) -> np.ndarray:
    """``tanitad.models.hjepa.donor_indices(n, episode_ids=..., seed)`` re-stated (same torch.Generator draws): a seeded derangement over
    EPISODES; window i of episode e takes the window of episode pi(e) with the same within-episode rank (modulo that episode's count)."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.as_tensor(np.asarray(episode_codes)).reshape(-1)
    uniq = torch.unique(ids)
    ne = len(uniq)
    pe = torch.randperm(ne, generator=g)
    for i in range(ne):
        if pe[i] == i:
            jj = (i + 1) % ne
            pe[i], pe[jj] = pe[jj].clone(), pe[i].clone()
    members = [torch.nonzero(ids == u).reshape(-1) for u in uniq]
    out = torch.empty(len(ids), dtype=torch.long)
    for ei, mem in enumerate(members):
        donor_mem = members[int(pe[ei])]
        for rank, wi in enumerate(mem.tolist()):
            out[wi] = donor_mem[rank % len(donor_mem)]
    return out.numpy()


def swap_donors(lon: np.ndarray, yaw: np.ndarray, ep: np.ndarray, seed: int = 0):
    """ACTION-SWAP donors. Window i qualifies if some window of ANOTHER episode has |mean a differs| >= 1.5 m/s^2 or |mean yaw rate differs| >= 0.15 rad/s;
    the donor is drawn uniformly among them, windows in ascending order, one ``default_rng(seed)`` stream. Returns (donor index or -1, n candidates)."""
    n = len(lon)
    rng = np.random.default_rng(seed)
    donor = np.full(n, -1, np.int64)
    ncand = np.zeros(n, np.int64)
    for i in range(n):
        ok = (ep != ep[i]) & ((np.abs(lon - lon[i]) >= SWAP_LON) | (np.abs(yaw - yaw[i]) >= SWAP_YAW))
        c = np.flatnonzero(ok)
        ncand[i] = len(c)
        if len(c):
            donor[i] = c[rng.integers(0, len(c))]
    return donor, ncand


def swap_donors_m(mabs_lon: np.ndarray, mabs_yaw: np.ndarray, lon: np.ndarray, yaw: np.ndarray, ep: np.ndarray, seed: int = 0):
    """ACTION-SWAP-M donors (VOE1B card). Over the swapped transitions T0..NF-2: ``mabs_*`` = mean |action| of the channel, ``lon`` / ``yaw`` = the SIGNED means.
    Window i's candidate donors are windows j of ANOTHER episode with (1) mean |lon| and mean |yaw| each within +-10 % of window i's own, AND (2) an opposite sign
    on at least one channel, where the sign is tested only on a channel on which BOTH signed means are at least 0.5 m/s^2 (lon) / 0.05 rad/s (yaw) in size (the
    stricter of the two readings of 'where that mean is at least ... in size': a donor that passes it passes the window-only reading too). The donor is one
    uniform draw among the candidates, windows in ascending order, one ``default_rng(seed)`` stream. Returns (donor or -1, n candidates)."""
    n = len(lon)
    rng = np.random.default_rng(seed)
    donor = np.full(n, -1, np.int64)
    ncand = np.zeros(n, np.int64)
    for i in range(n):
        ok = ep != ep[i]
        ok &= np.abs(mabs_lon - mabs_lon[i]) <= SWM_REL * mabs_lon[i]
        ok &= np.abs(mabs_yaw - mabs_yaw[i]) <= SWM_REL * mabs_yaw[i]
        opp_lon = (np.abs(lon[i]) >= SWM_LON_MIN) & (np.abs(lon) >= SWM_LON_MIN) & (np.sign(lon) == -np.sign(lon[i]))
        opp_yaw = (np.abs(yaw[i]) >= SWM_YAW_MIN) & (np.abs(yaw) >= SWM_YAW_MIN) & (np.sign(yaw) == -np.sign(yaw[i]))
        c = np.flatnonzero(ok & (opp_lon | opp_yaw))
        ncand[i] = len(c)
        if len(c):
            donor[i] = c[rng.integers(0, len(c))]
    return donor, ncand


# ------------------------------------------------------------------------------------------------ level traces
def score_cols(level: int) -> list:
    """Columns (step j - 1) of the level's trace whose state lies within the 1.6 s after T0: STRIDE*j in (T0, T0 + N_POST]."""
    s = STRIDE[level]
    return [j - 1 for j in range(1, 64) if T0 < s * j <= T0 + N_POST]


def step_frames(level: int, n_steps: int) -> np.ndarray:
    """Window frame of every step j = 1..n_steps of the level."""
    return STRIDE[level] * np.arange(1, n_steps + 1)


class HJEPATraces:
    """Per-level one-step surprise of a ``tanitad.models.hjepa.HJEPAStack`` on a (manipulated) latent sequence.

    ``traces(z1 [N, T, D1], a_sc [N, T-1, A])`` -> ``{level: {"sur": [N, T_l-1], "null": [N, T_l-1]}}``. Step j (>= 1) of level l is the state
    ``z^l_j`` (window frame STRIDE*j). sur = ||F(z_{j-c..j-1}, a_{j-c..j-1})[-1] - z_j||^2 with c = min(3, j), teacher-forced on the given
    (manipulated) states and the macro-actions ``encode`` builds from the given actions; null = ||z_{j-1} - z_j||^2 (Copy-Last)."""

    def __init__(self, stack, ctx: int = CTX):
        self.stack, self.ctx = stack, ctx
        self.levels = tuple(range(1, stack.n_levels + 1))

    @torch.no_grad()
    def traces(self, z1: torch.Tensor, a_sc: torch.Tensor, chunk: int = 512) -> dict:
        out = {l: {"sur": [], "null": []} for l in self.levels}
        for s0 in range(0, z1.shape[0], chunk):
            enc = self.stack.encode(z1[s0:s0 + chunk], a_sc[s0:s0 + chunk])
            for l in self.levels:
                z, e = enc.z[l - 1], enc.act[l - 1]
                pred = self.stack.predictors[l - 1]
                ctx = min(self.ctx, getattr(pred, "context", self.ctx))
                P = []
                for j in range(1, z.shape[1]):
                    c = min(ctx, j)
                    P.append(pred(z[:, j - c:j], e[:, j - c:j])[:, -1])
                P = torch.stack(P, 1)
                out[l]["sur"].append(((P - z[:, 1:]).double() ** 2).sum(-1).numpy())
                out[l]["null"].append(((z[:, :-1] - z[:, 1:]).double() ** 2).sum(-1).numpy())
        return {l: {k: np.concatenate(v) for k, v in d.items()} for l, d in out.items()}


def medians(tr_unmod: dict) -> dict:
    """Per level: the median surprise (model, Copy-Last) over the unmodified windows' scored steps. The accuracy is invariant to it; it scales traces and figures."""
    return {l: {k: float(np.median(tr_unmod[l][k][:, score_cols(l)])) for k in ("sur", "null")} for l in tr_unmod}


def window_scores(tr: dict, med: dict) -> dict:
    """{level: {"sur": [N], "null": [N]}}: the maximum over the level's steps within the 1.6 s after T0 of the median-normalised surprise."""
    return {l: {k: (tr[l][k][:, score_cols(l)] / med[l][k]).max(1) for k in ("sur", "null")} for l in tr}


def pair_x(s_manip: np.ndarray, s_unmod: np.ndarray) -> np.ndarray:
    """1 where the manipulated copy scores above its unmodified copy, 0.5 on an exact tie, else 0."""
    return (s_manip > s_unmod).astype(np.float64) + 0.5 * (s_manip == s_unmod)


# ------------------------------------------------------------------------------------------------ statistics: bootstrap, permutation, two-part rule
_WCACHE = {}


def boot_W(n_ep: int, B: int = B_BOOT, seed: int = 0) -> np.ndarray:
    """Episode-resampling counts [B, n_ep]. Same rng calls as ``wmact_lib.boot_weights`` (the draws depend on n_ep only); the run asserts equality."""
    k = (n_ep, B, seed)
    if k not in _WCACHE:
        rng = np.random.default_rng(seed)
        W = np.zeros((B, n_ep))
        for b in range(B):
            W[b] = np.bincount(rng.choice(n_ep, size=n_ep, replace=True), minlength=n_ep)
        _WCACHE[k] = W
    return _WCACHE[k]


def episode_codes(eids) -> tuple:
    u, inv = np.unique(np.asarray([str(e) for e in eids]), return_inverse=True)
    return inv.astype(np.int64), len(u)


def cell_seed(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def two_part(vals: np.ndarray, ep: np.ndarray, n_ep: int, claim: float, key: str, R: int = R_PERM, cal_seeds=CAL_SEEDS, B: int = B_BOOT,
             ref_flags: bool = False) -> dict:
    """One cell. ``vals`` are per-window values with the claim value ``claim`` (accuracy: 0.5 with vals in {0, .5, 1}; a difference of two accuracies: 0.0).
    Statistic = mean of vals (ratio of per-episode sums). Interval: episode-cluster bootstrap (B, seed 0). Permutation reference: R draws (seeds 1..R)
    that swap the two copies' labels within WHOLE episodes, i.e. each episode independently, with probability 1/2, maps v -> 2*claim - v (accuracy x -> 1 - x,
    difference d -> -d); the stream of draw s is ``default_rng([s, cell_seed(key)])``. Separated ABOVE the claim iff the interval's lower end > claim AND
    the point > the 97.5th percentile of the reference; BELOW symmetrically (2.5th percentile). The calibration draws (seeds 1001..1020, disjoint from the
    reference) are scored as if real by the same rule against the same reference. VOE1B addition ``ref_flags=True``: the R reference draws THEMSELVES are scored as
    if real by the same rule (bootstrap interval of the draw's per-episode sums AND the draw's point against the cell's own 97.5th / 2.5th percentiles):
    ``ref_sep_above`` / ``ref_sep_below`` [R] (control 1's reference count)."""
    vals = np.asarray(vals, np.float64)
    N = len(vals)
    cnt = np.bincount(ep, minlength=n_ep).astype(np.float64)
    num = np.bincount(ep, weights=vals, minlength=n_ep)
    W = boot_W(n_ep, B)
    den = W @ cnt
    cs = cell_seed(key)

    def perm_num(s):
        fl = np.random.default_rng([s, cs]).integers(0, 2, n_ep).astype(bool)
        return np.where(fl, 2.0 * claim * cnt - num, num)

    point = float(num.sum() / N)
    bs = (W @ num) / den
    lo, hi = (float(x) for x in np.percentile(bs, [2.5, 97.5]))
    ref = np.array([perm_num(s).sum() / N for s in range(1, R + 1)])
    t_hi, t_lo = (float(x) for x in np.percentile(ref, [97.5, 2.5]))
    out = {"point": point, "lo": lo, "hi": hi, "claim": claim, "ref_hi": t_hi, "ref_lo": t_lo, "ref_median": float(np.median(ref)),
           "rank_above": float((1 + (ref >= point).sum()) / (R + 1)), "rank_below": float((1 + (ref <= point).sum()) / (R + 1)),
           "boot_above": bool(lo > claim), "boot_below": bool(hi < claim),
           "sep_above": bool(lo > claim and point > t_hi), "sep_below": bool(hi < claim and point < t_lo), "n": int(N), "n_episodes": int(n_ep)}
    out["separated_any"] = bool(out["sep_above"] or out["sep_below"])
    if ref_flags:
        RN = np.stack([perm_num(s) for s in range(1, R + 1)], 1)             # [n_ep, R]
        rpt = RN.sum(0) / N
        rb = (W @ RN) / den[:, None]
        rl, rh = np.percentile(rb, 2.5, axis=0), np.percentile(rb, 97.5, axis=0)
        out["ref_sep_above"] = [bool(a and b) for a, b in zip(rl > claim, rpt > t_hi)]
        out["ref_sep_below"] = [bool(a and b) for a, b in zip(rh < claim, rpt < t_lo)]
    if cal_seeds:
        PN = np.stack([perm_num(s) for s in cal_seeds], 1)                  # [n_ep, n_cal]
        pt = PN.sum(0) / N
        cb = (W @ PN) / den[:, None]
        cl, ch = np.percentile(cb, 2.5, axis=0), np.percentile(cb, 97.5, axis=0)
        out["cal"] = {"sep_above": [bool(a and b) for a, b in zip(cl > claim, pt > t_hi)], "sep_below": [bool(a and b) for a, b in zip(ch < claim, pt < t_lo)],
                      "boot_above": [bool(a) for a in cl > claim], "points": [float(x) for x in pt]}
    return out


def label_cell(acc: dict, diff: dict | None, counts_null: bool) -> str:
    """VOE1's NOTICES / NULL-ONLY / BLIND of one (manipulation, level, seed) from rules (a) and (b) only (no untrained reference). Kept as ``label_ab``."""
    if not acc["sep_above"]:
        return "BLIND"
    if not counts_null:
        return "NOTICES"
    return "NOTICES" if (diff is not None and diff["sep_above"]) else "NULL-ONLY"


def label_cell_b(acc: dict, diff: dict | None, counts_null: bool, ut_points) -> dict:
    """VOE1B labels of one (manipulation, level, model seed). (a) accuracy two-part separated above 0.5; (b) [TELEPORT, REVERSE, SPLICE only] separated above the
    Copy-Last null (paired); (c) the accuracy is strictly above the MAXIMUM of the untrained accuracies ``ut_points`` of that manipulation and level.
    BLIND = (a) fails. NULL-ONLY = (a) holds, (b) fails. INIT-LEVEL = (a) and (b) hold, (c) fails. NOTICES = (a), (b), (c) all hold."""
    a = bool(acc["sep_above"])
    b = None if not counts_null else bool(diff is not None and diff["sep_above"])
    c = bool(acc["point"] > max(ut_points))
    if not a:
        lab = "BLIND"
    elif b is False:
        lab = "NULL-ONLY"
    elif c:
        lab = "NOTICES"
    else:
        lab = "INIT-LEVEL"
    return {"label": lab, "a": a, "b": b, "c": c, "ut_max": float(max(ut_points)), "ut_min": float(min(ut_points)), "ut_median": float(np.median(ut_points)), "n_ut": len(ut_points)}


def level_label(l0: str, l1: str) -> str:
    """VOE1's label of a (manipulation, level) over BOTH seeds: NOTICES iff both seeds NOTICES; BLIND iff both BLIND; NULL-ONLY iff both are above 0.5 and at least
    one is NULL-ONLY; anything else is DISCORDANT. (VOE1 tags only; VOE1B uses ``level_label_b``.)"""
    if l0 == l1 and l0 in ("NOTICES", "BLIND", "NULL-ONLY"):
        return l0
    if {l0, l1} <= {"NOTICES", "NULL-ONLY"}:
        return "NULL-ONLY"
    return "DISCORDANT"


def level_label_b(l0: str, l1: str) -> str:
    """VOE1B label of a (manipulation, level) over both model seeds. NOTICES needs NOTICES at both seeds (the card). Equal labels stay. Otherwise (VOE1's reading
    extended): both above 0.5 and at least one NULL-ONLY -> NULL-ONLY; both in {BLIND, INIT-LEVEL} (different) -> BLIND/INIT-LEVEL (neither seed notices: the
    triggers 'BLIND or INIT-LEVEL' count it); anything else DISCORDANT."""
    if l0 == l1 and l0 in LABELS_B:
        return l0
    if {l0, l1} <= {"NOTICES", "INIT-LEVEL", "NULL-ONLY"} and "NULL-ONLY" in (l0, l1):
        return "NULL-ONLY"
    if {l0, l1} == {"BLIND", "INIT-LEVEL"}:
        return "BLIND/INIT-LEVEL"
    return "DISCORDANT"


# ------------------------------------------------------------------------------------------------ VOE1B controls 1 and 2 (pure functions of the rows)
def control1_count(cells: list, which: str = "real") -> int | np.ndarray:
    """Control 1 (calibrated). ``cells`` = two_part dicts of the 6 M2b real-vs-real cells computed with ``ref_flags=True``. ``real``: the number of cells separated
    (either side). ``ref``: the same count for each of the R reference draws scored as if real ([R] ints; the cells' streams are independent)."""
    if which == "real":
        return int(sum(bool(c["separated_any"]) for c in cells))
    return np.sum([np.asarray(c["ref_sep_above"], int) | np.asarray(c["ref_sep_below"], int) for c in cells], 0)


def control1_verdict(cells: list) -> dict:
    """The real count must not exceed the 95th percentile (numpy linear, as every percentile in this instrument) of the reference-draw counts."""
    real, ref = control1_count(cells, "real"), control1_count(cells, "ref")
    p95 = float(np.percentile(ref, 95))
    return {"real_count": real, "n_cells": len(cells), "p95_linear": p95, "p95_inverted_cdf": float(np.percentile(ref, 95, method="inverted_cdf")),
            "ref_count_hist": {int(k): int(v) for k, v in zip(*np.unique(ref, return_counts=True))}, "ref_mean": float(ref.mean()), "R": int(len(ref)), "pass": bool(real <= p95)}


def loo_notices(cells: list) -> list:
    """Control 2 for one family of untrained seeds and one (manipulation, level): ``cells`` = [{"sep_above": bool, "point": float}] of the K untrained members.
    Each member in turn is 'the model' and the other K-1 its reference: NOTICES iff rule (a) holds at that member AND its accuracy is strictly above the maximum of
    the other K-1 (rule (c)). Returns the K booleans. Under exchangeability at most one member can satisfy (c)."""
    pts = np.array([c["point"] for c in cells])
    out = []
    for i, c in enumerate(cells):
        others = np.delete(pts, i)
        out.append(bool(c["sep_above"] and pts[i] > others.max()))
    return out


# ------------------------------------------------------------------------------------------------ events and average precision
def onset(cond: np.ndarray, refractory: int = 5) -> np.ndarray:
    """First True after at least ``refractory`` False steps (1 s); frame 0 is never an onset (its past is unobserved)."""
    cond = np.asarray(cond, bool)
    out = np.zeros(len(cond), bool)
    for g in range(1, len(cond)):
        if cond[g] and not cond[max(0, g - refractory):g].any():
            out[g] = True
    return out


def lead_event_cond(near: np.ndarray, gap: np.ndarray, closing_thr: float = 0.5, jump_m: float = 5.0) -> np.ndarray:
    """A lead change (a near lead appears after 2 steps without one, or the gap jumps by more than ``jump_m`` between consecutive frames: a different lead)
    with the gap closing over the next 0.4 s (closing speed > ``closing_thr`` m/s; needs a near lead 2 steps later)."""
    T = len(near)
    near = np.asarray(near, bool)
    gap = np.nan_to_num(np.asarray(gap, np.float64), nan=0.0)
    cond = np.zeros(T, bool)
    for g in range(2, T - 2):
        if not near[g]:
            continue
        appear = (not near[g - 1]) and (not near[g - 2])
        switch = bool(near[g - 1]) and abs(gap[g] - gap[g - 1]) > jump_m
        closing = bool(near[g + 2]) and (gap[g] - gap[g + 2]) / (2 * DT) > closing_thr
        cond[g] = (appear or switch) and closing
    return cond


def event_step_labels(on: np.ndarray, base: np.ndarray, clip_lo: np.ndarray, clip_hi: np.ndarray, level: int, n_steps: int) -> np.ndarray:
    """[N, n_steps] bool. ``on`` = the event onsets over the whole bank (global frame index); step j of the level sits at global frame ``base + STRIDE*j``;
    positive iff an onset lies within +-EV_BAND cache steps (0.4 s), inside the window's own clip [clip_lo, clip_hi)."""
    on = np.asarray(on, bool)
    cs = np.concatenate([[0], np.cumsum(on)])
    fr = np.asarray(base)[:, None] + step_frames(level, n_steps)[None]
    lo = np.maximum(fr - EV_BAND, np.asarray(clip_lo)[:, None])
    hi = np.minimum(fr + EV_BAND + 1, np.asarray(clip_hi)[:, None])
    return (cs[hi] - cs[lo]) > 0


def ap_point(score: np.ndarray, y: np.ndarray) -> float:
    ys = np.asarray(y, np.float64)[np.argsort(-score, kind="stable")]
    npos = ys.sum()
    if npos == 0:
        return float("nan")
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / npos)


def ap_boot(ys_sorted: np.ndarray, ep_sorted: np.ndarray, W: np.ndarray, chunk: int = 250):
    """Episode-bootstrap AP and prevalence. ``ys_sorted`` [P] labels in descending-score order, ``ep_sorted`` [P] episode code of each point, ``W`` [B, n_ep]
    resampling counts: a point of an episode drawn m times carries weight m (AP = sum w y precision@rank / sum w y, precision from weighted cumulative counts)."""
    B = W.shape[0]
    ap, prev = np.empty(B), np.empty(B)
    ys = np.asarray(ys_sorted, np.float32)
    for c0 in range(0, B, chunk):
        w = W[c0:c0 + chunk][:, ep_sorted].astype(np.float32)
        wy = w * ys
        tp = np.cumsum(wy, 1, dtype=np.float32)
        n = np.cumsum(w, 1, dtype=np.float32)
        prec = tp / np.maximum(n, 1e-9)
        sy = wy.sum(1, dtype=np.float64)
        ap[c0:c0 + chunk] = (wy * prec).sum(1, dtype=np.float64) / np.maximum(sy, 1e-12)
        prev[c0:c0 + chunk] = sy / w.sum(1, dtype=np.float64)
    return ap, prev


def window_perm(ep_w: np.ndarray, n_ep: int, rng) -> np.ndarray:
    """src[i]: the window whose label trace window i receives; always from ANOTHER episode. Windows are laid out in a random episode order (episodes contiguous),
    and each takes the label trace of the window ``k`` positions further on (cyclic), k in [maxsize + 1, N - maxsize - 1]."""
    N = len(ep_w)
    order = rng.permutation(n_ep)
    rank = np.empty(n_ep, np.int64)
    rank[order] = np.arange(n_ep)
    seq = np.lexsort((np.arange(N), rank[ep_w]))
    mx = int(np.bincount(ep_w, minlength=n_ep).max())
    k = int(rng.integers(mx + 1, N - mx))
    src = np.empty(N, np.int64)
    src[seq] = seq[(np.arange(N) + k) % N]
    return src


def ap_cell(sm: np.ndarray, sn: np.ndarray, Y: np.ndarray, ep_w: np.ndarray, n_ep: int, key: str, R: int = R_PERM, cal_seeds=CAL_SEEDS,
            B: int = B_BOOT, with_boot: bool = True) -> dict:
    """One event-alignment cell. ``sm`` / ``sn`` [N, S] the model / Copy-Last traces (steps of one level), ``Y`` [N, S] the event labels of the unmodified windows.
    Points = (window, step) pooled. Statistics: AP - chance (chance = the event rate) and AP - AP_null. Two-part rule as in ``two_part``: bootstrap interval above 0
    AND point above the 97.5th percentile of R label-permutation draws (the label traces of windows move to windows of other episodes, scores fixed; stream
    ``default_rng([s, cell_seed(key)])``); calibration draws are scored as if real."""
    N, S = sm.shape
    pm, pn = sm.reshape(-1), sn.reshape(-1)
    y0 = np.asarray(Y, bool)
    ep_pt = np.repeat(ep_w, S)
    om, on_ = np.argsort(-pm, kind="stable"), np.argsort(-pn, kind="stable")
    prev = float(y0.mean())

    def aps(src):
        yf = y0[src].reshape(-1).astype(np.float64)
        return ap_point(pm, yf), ap_point(pn, yf)

    ident = np.arange(N)
    apm, apn = aps(ident)
    out = {"ap": apm, "ap_null": apn, "chance": prev, "n_points": int(N * S), "n_events_pts": int(y0.sum()), "n_windows": int(N)}
    cs = cell_seed(key)
    if not with_boot:
        return out
    W = boot_W(n_ep, B)

    def boots(src):
        yf = y0[src].reshape(-1)
        bm, pv = ap_boot(yf[om].astype(np.float32), ep_pt[om], W)
        bn, _ = ap_boot(yf[on_].astype(np.float32), ep_pt[on_], W)
        return bm, bn, pv

    def perm_src(s):
        return window_perm(ep_w, n_ep, np.random.default_rng([s, cs]))

    bm, bn, pv = boots(ident)
    d1, d2 = bm - pv, bm - bn
    out["boot"] = {"ap": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))], "ap_null": [float(np.percentile(bn, 2.5)), float(np.percentile(bn, 97.5))]}
    out["boot_apc"] = [float(x) for x in d1]          # the bootstrap draws of AP - chance (shared W): lets the report put an interval on level differences
    refs = np.array([[a - prev, a - b] for a, b in (aps(perm_src(s)) for s in range(1, R + 1))])       # [R, 2]
    rows = {}
    for nm, d, rf in (("ap_minus_chance", d1, refs[:, 0]), ("ap_minus_null", d2, refs[:, 1])):
        pt = (apm - prev) if nm == "ap_minus_chance" else (apm - apn)
        lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
        t_hi = float(np.percentile(rf, 97.5))
        rows[nm] = {"point": float(pt), "lo": lo, "hi": hi, "ref_hi": t_hi, "boot_above": bool(lo > 0), "sep_above": bool(lo > 0 and pt > t_hi),
                    "rank_above": float((1 + (rf >= pt).sum()) / (R + 1))}
    if cal_seeds:
        cal = {"ap_minus_chance": [], "ap_minus_null": []}
        for s in cal_seeds:
            src = perm_src(s)
            a_m, a_n = aps(src)
            cbm, cbn, cpv = boots(src)
            for nm, cd, pt in (("ap_minus_chance", cbm - cpv, a_m - prev), ("ap_minus_null", cbm - cbn, a_m - a_n)):
                cal[nm].append(bool(np.percentile(cd, 2.5) > 0 and pt > rows[nm]["ref_hi"]))
        for nm in cal:
            rows[nm]["cal_sep_above"] = cal[nm]
    out["rows"] = rows
    return out


# ------------------------------------------------------------------------------------------------ the window table and one tag's manipulation rows
def window_table(act: np.ndarray, speed: np.ndarray, base: np.ndarray, g0: np.ndarray, T_clip: np.ndarray, ep: np.ndarray) -> dict:
    """Which windows qualify for which manipulation, and the donors (IMPL_CHOICES_VOE1 section 3 + IMPL_CHOICES_VOE1B). ``act`` [Ntot, 2] raw (a, yaw rate), ``speed`` [Ntot]
    logged speed, ``base`` [N] bank index of window frame 0, ``g0`` [N] clip frame of window frame 0, ``T_clip`` [N] length of the window's clip, ``ep`` [N] episode codes."""
    seg = act[base[:, None] + np.arange(T0, NF - 1)[None]]                  # actions of transitions T0..T0+7
    lon, yaw = seg[..., 0].mean(1), seg[..., 1].mean(1)
    mabs_lon, mabs_yaw = np.abs(seg[..., 0]).mean(1), np.abs(seg[..., 1]).mean(1)
    spd_min = speed[base[:, None] + np.arange(T0, T0 + FREEZE_N + 1)[None]].min(1)      # frames T0..T0+FREEZE_N; NaN propagates -> not > 3
    sw_donor, sw_ncand = swap_donors(lon, yaw, ep, seed=0)
    swm_donor, swm_ncand = swap_donors_m(mabs_lon, mabs_yaw, lon, yaw, ep, seed=0)
    sp_donor = donor_derangement(ep, seed=0)
    q = {"TELEPORT": g0 + (NF - 1) + TELE_SHIFT <= T_clip - 1, "FREEZE": spd_min > FREEZE_MIN_SPEED, "REVERSE": np.ones(len(base), bool),
         "SPLICE": np.ones(len(base), bool), "ACTION-SWAP": sw_donor >= 0, "ACTION-SWAP-M": swm_donor >= 0, "RVR": g0 + 1 + (NF - 1) <= T_clip - 1}
    return {"q": q, "donor": {"SPLICE": sp_donor, "ACTION-SWAP": sw_donor, "ACTION-SWAP-M": swm_donor}, "lon": lon, "yaw": yaw, "mabs_lon": mabs_lon, "mabs_yaw": mabs_yaw,
            "spd_min": spd_min, "swap_ncand": sw_ncand, "swapm_ncand": swm_ncand}


def eval_tag(model, z1: torch.Tensor, a_sc: torch.Tensor, base: np.ndarray, eid: np.ndarray, wt: dict, tag: str, cal: bool = True, keep_traces: bool = False,
             lean: bool = False, rvr_ref_flags: bool = False, median_traces: bool = False) -> dict:
    """One model / tag: the unmodified copy, the real-vs-real shifted copy, the six manipulated copies; per manipulation x level the pairwise accuracy of the model
    and of the Copy-Last null, their paired difference, each through the two-part rule, and the VOE1 label (a, b only: ``label_ab``). ``z1`` [Ntot, D] and ``a_sc``
    [Ntot, A] are the bank. VOE1B options: ``lean`` = the model's accuracy cell only (untrained reference members: rules (a) and (c) need nothing else);
    ``rvr_ref_flags`` = control 1's per-reference-draw flags on the real-vs-real cells; ``median_traces`` = keep per-step medians of the normalised traces (figure band)."""
    take = lambda fr, ac: model.traces(z1[torch.from_numpy(fr)], a_sc[torch.from_numpy(ac)])
    levels = tuple(model.levels)
    fr_u, ac_u = idx_unmod(base)
    tr_u = take(fr_u, ac_u)
    med = medians(tr_u)
    Su = window_scores(tr_u, med)
    out = {"median": {str(l): med[l] for l in levels}, "manips": {}, "rvr": {}}
    traces = {"unmod": {l: tr_u[l]["sur"] / med[l]["sur"] for l in levels}} if keep_traces else None
    mtr = {} if median_traces else None
    cal_seeds = CAL_SEEDS if cal else ()
    # ---- control 1: the same window, 0.2 s later
    q = wt["q"]["RVR"]
    fr_b, ac_b = idx_unmod(base[q], shift=1)
    Sb = window_scores(take(fr_b, ac_b), med)
    ep, n_ep = episode_codes(eid[q])
    out["rvr"] = {"n": int(q.sum()), "n_episodes": int(n_ep), "levels": {}}
    for l in levels:
        x = pair_x(Sb[l]["sur"], Su[l]["sur"][q])
        out["rvr"]["levels"][str(l)] = two_part(x, ep, n_ep, 0.5, f"{tag}/RVR/L{l}", cal_seeds=cal_seeds, ref_flags=rvr_ref_flags)
    # ---- the manipulations
    for name in MANIPS:
        q = wt["q"][name]
        n = int(q.sum())
        rec = {"n": n, "n_episodes": int(len(set(eid[q]))), "levels": {}}
        out["manips"][name] = rec
        if n == 0:
            continue
        don = wt["donor"].get(name)
        fr, ac = idx_manip(name, base[q], None if don is None else base[don[q]])
        tr_m = take(fr, ac)
        Sm = window_scores(tr_m, med)
        ep, n_ep = episode_codes(eid[q])
        for l in levels:
            xm, xn = pair_x(Sm[l]["sur"], Su[l]["sur"][q]), pair_x(Sm[l]["null"], Su[l]["null"][q])
            r = {"acc": two_part(xm, ep, n_ep, 0.5, f"{tag}/{name}/L{l}/acc", cal_seeds=cal_seeds)}
            if not lean:
                r["null_acc"] = two_part(xn, ep, n_ep, 0.5, f"{tag}/{name}/L{l}/null", cal_seeds=cal_seeds)
                r["diff"] = two_part(xm - xn, ep, n_ep, 0.0, f"{tag}/{name}/L{l}/diff", cal_seeds=cal_seeds)
                r["label_ab"] = label_cell(r["acc"], r["diff"], name in NULL_COUNTS)
            r["median_score_manip"] = float(np.median(Sm[l]["sur"]))
            r["median_score_unmod"] = float(np.median(Su[l]["sur"][q]))
            rec["levels"][str(l)] = r
        if keep_traces:
            traces[name] = {"q": q, **{l: tr_m[l]["sur"] / med[l]["sur"] for l in levels}, **{f"unmod_{l}": tr_u[l]["sur"][q] / med[l]["sur"] for l in levels}}
        if median_traces:
            mtr[name] = {**{str(l): np.median(tr_m[l]["sur"] / med[l]["sur"], axis=0) for l in levels}, **{f"unmod_{l}": np.median(tr_u[l]["sur"][q] / med[l]["sur"], axis=0) for l in levels}}
    out["_traces"] = traces
    out["_mtraces"] = mtr
    return out
