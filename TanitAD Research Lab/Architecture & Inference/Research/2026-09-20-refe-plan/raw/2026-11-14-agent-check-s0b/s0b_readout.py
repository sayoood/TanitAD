"""S0-B readout (agent-check card, section 3 + FEASIBILITY_AGENTS 4.1): deterministic pose-only footprint readout of a raster,
veto, gain, capture, G-FID, paired log-cluster bootstrap, mechanical outcome row. CPU only.

FROZEN MM DECISIONS (2026-11-14, before any data):
  poses     proposal = (0,0,0) at t=0 + the 20 banked poses (t=0.2..4.0 s, ego rear-axle frame at t0), linear to 10 Hz
            (heading unwrapped) -> 41 poses, t=0.0..4.0 s.  Rear axle is the pose; the footprint centre is rear_axle_to_center
            ahead along the heading.  No A7 last-heading repair (the card says banked traj_xy/yaw as they are).
  footprint nuPlan Pacifica box: length, width, rear_axle_to_center READ from nuplan get_pacifica_parameters
            (nuplan/common/actor_state/vehicle_parameters.py:125-138 -> width 2.297, front 4.049, rear 1.127; length = front+rear,
            rear_axle_to_center = half_length - rear_length, :83-86).  Rasterised by s0b_raster.box_cells (the agents' rasteriser).
  NC-proxy  any footprint cell overlaps an occupied cell at the SAME 10 Hz step, t = 0.1..4.0 s (steps 1..40).  t=0 is NOT used;
            its overlap share is reported.
  TTC-proxy the geometric part of NAVSIM `_calculate_ttc` (pdm_scorer.py:492-498): the footprint of step t, translated along ITS
            heading by speed*delta for delta in {0, .3, .6, .9} s (future_time_idcs = arange(0,10,3)), tested against the raster at
            step t+delta/0.1; t = 0..40, except (t=0, delta=0) which is the t=0 overlap above.  speed = the proposal's own
            finite difference |p[k]-p[k-1]|/0.1 (speed[0]=speed[1]); steps with speed < 5e-3 m/s (stopped_speed_threshold) skipped.
            NOT reproduced: at-fault typing (ACTIVE_FRONT / STOPPED_TRACK / lateral, static = 0.5 vs agent = 0), is_agent_ahead/behind,
            lane / intersection / non-drivable conditions, red-light polygons, "already collided track ids", per-object identity,
            the PDM simulator's re-tracking of the proposal (we use the proposal poses), exact polygons (0.5 m cell raster).
  veto      flag = NC-proxy OR TTC-proxy on a BINARY raster, no tau.  Drop flagged, pick by the CANONICAL rule
            (sel_common.agg_torch(logits, lane=False), first index on ties) among the rest; all 64 flagged -> P6's pick.
  gain      PDMS x 100 vs P6's pick using the fit pack's `pdms` column as truth.  capture c(X) = gain(X)/O_NC+TTC on the SAME sets.
  bootstrap paired, resampling LOGS, 10,000 draws, seed 20260927; the point estimate decides, the CI is reported.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

import s0b_raster as RS

DELTA_IDX = (0, 3, 6, 9)                 # NAVSIM future_time_idcs = np.arange(0, 10, 3)
STOPPED_SPEED = 5e-3                     # NAVSIM PDMScorerConfig.stopped_speed_threshold
N_POSE = 41                              # 0.0 .. 4.0 s at 10 Hz
BOOT_DRAWS = 10_000
BOOT_SEED = 20260927


@dataclass
class FootParams:
    length: float
    width: float
    rear_axle_to_center: float
    source: str


def read_pacifica() -> FootParams:
    """Dimensions from the devkit (never hard-coded here)."""
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    vp = get_pacifica_parameters()
    return FootParams(float(vp.length), float(vp.width), float(vp.rear_axle_to_center),
                      "nuplan get_pacifica_parameters() (vehicle_parameters.py:125-138; rear_axle_to_center :83-86)")


def load_sel_common(sel_dir):
    """The canonical pick rule lives in sel_common.py (tip blob, exported next to these scripts under ref/)."""
    if str(sel_dir) not in sys.path:
        sys.path.insert(0, str(sel_dir))
    import sel_common
    return sel_common


# ----------------------------------------------------------------------------------------------- proposals
def proposals_10hz(traj_xy, yaw):
    """traj_xy (P,20,2), yaw (P,20) [t=0.2..4.0] -> (P,41,3) at 0.1 s: prepend (0,0,0), unwrap, linear."""
    traj_xy = np.asarray(traj_xy, float)
    yaw = np.asarray(yaw, float)
    P = traj_xy.shape[0]
    kx = np.concatenate([np.zeros((P, 1)), traj_xy[..., 0]], axis=1)
    ky = np.concatenate([np.zeros((P, 1)), traj_xy[..., 1]], axis=1)
    kh = np.unwrap(np.concatenate([np.zeros((P, 1)), yaw], axis=1), axis=1)
    tk = np.arange(21) * 0.2
    tq = np.arange(N_POSE) * 0.1
    out = np.empty((P, N_POSE, 3))
    for p in range(P):
        out[p, :, 0] = np.interp(tq, tk, kx[p])
        out[p, :, 1] = np.interp(tq, tk, ky[p])
        out[p, :, 2] = np.interp(tq, tk, kh[p])
    return out


def pose_speeds(poses):
    """(P,41,3) -> (P,41) finite-difference speed, speed[0] = speed[1]."""
    d = np.linalg.norm(np.diff(poses[..., :2], axis=1), axis=-1) / 0.1
    return np.concatenate([d[:, :1], d], axis=1)


# ----------------------------------------------------------------------------------------------- footprints
@dataclass
class FootTables:
    idx_nc: np.ndarray        # (P,41,K) flat cell index, padded with RS.NXY (the permanent free pad column)
    idx_ttc: np.ndarray       # (P,41,len(DELTA_IDX),K)
    speed: np.ndarray         # (P,41)
    partial_share: float      # share of (proposal, step 1..40) poses with some footprint cell outside the grid
    none_in_grid_share: float  # share with NO footprint cell in the grid
    n_cells_mean: float


def _compress(rows, cols, valid, shape_lead):
    cnt = valid.sum(axis=1)
    kmax = max(int(cnt.max()) if cnt.size else 0, 1)
    order = np.argsort(~valid, axis=1, kind="stable")[:, :kmax]
    flat = np.take_along_axis(rows * RS.NY + cols, order, axis=1)
    ok = np.take_along_axis(valid, order, axis=1)
    flat = np.where(ok, flat, RS.NXY)
    return flat.reshape(shape_lead + (kmax,)).astype(np.int32), cnt


def footprint_tables(poses, fp: FootParams) -> FootTables:
    P, T, _ = poses.shape
    hl, hw = fp.length / 2.0, fp.width / 2.0
    px, py, ps = poses[..., 0], poses[..., 1], poses[..., 2]
    cx = px + fp.rear_axle_to_center * np.cos(ps)
    cy = py + fp.rear_axle_to_center * np.sin(ps)
    speed = pose_speeds(poses)
    rows, cols, valid, partial = RS.box_cells(cx.ravel(), cy.ravel(), ps.ravel(), hl, hw)
    idx_nc, cnt = _compress(rows, cols, valid, (P, T))
    cnt = cnt.reshape(P, T)
    partial = partial.reshape(P, T)
    # TTC: shifted along the pose heading by speed * delta
    cxs, cys = [], []
    for d in DELTA_IDX:
        dl = d * 0.1
        cxs.append(cx + speed * dl * np.cos(ps))
        cys.append(cy + speed * dl * np.sin(ps))
    cxs, cys = np.stack(cxs, 2), np.stack(cys, 2)                   # (P,T,D)
    psb = np.broadcast_to(ps[..., None], cxs.shape)
    rows, cols, valid, _ = RS.box_cells(cxs.ravel(), cys.ravel(), psb.ravel(), hl, hw)
    idx_ttc, _ = _compress(rows, cols, valid, (P, T, len(DELTA_IDX)))
    sl = slice(1, N_POSE)
    return FootTables(idx_nc, idx_ttc, speed, float(partial[:, sl].mean()), float((cnt[:, sl] == 0).mean()),
                      float(cnt[:, sl].mean()))


def flags_for_raster(Rg, ft: FootTables):
    """Rg (T,NXY+1) bool gather table. -> nc_flag (P,), ttc_flag (P,), overlap_t0 (bool)."""
    T = Rg.shape[0]
    t = np.arange(1, N_POSE)
    nc = Rg[t[None, :, None], ft.idx_nc[:, 1:, :]].any(axis=(1, 2))        # steps 1..40, SAME step
    P = ft.idx_nc.shape[0]
    ttc = np.zeros(P, dtype=bool)
    ts = np.arange(N_POSE)
    for j, d in enumerate(DELTA_IDX):
        occ = Rg[np.minimum(ts + d, T - 1)[None, :, None], ft.idx_ttc[:, :, j, :]].any(axis=2)   # (P,41)
        use = ft.speed >= STOPPED_SPEED
        if d == 0:
            use = use.copy()
            use[:, 0] = False                                                 # (t=0, delta=0) = the t=0 overlap, reported apart
        ttc |= (occ & use).any(axis=1)
    overlap_t0 = bool(Rg[0, ft.idx_nc[0, 0]].any())
    return nc, ttc, overlap_t0


# ----------------------------------------------------------------------------------------------- pick / gain
def veto_pick(rule, flag, p6_pick):
    """rule (S,64) canonical aggregate, flag (S,64) bool, p6_pick (S,). Drop flagged, canonical pick among the rest (first index
    on ties); ALL flagged -> P6's pick (fallback)."""
    r = np.where(flag, -np.inf, rule)
    pk = r.argmax(axis=1)
    allf = flag.all(axis=1)
    return np.where(allf, p6_pick, pk), allf


def set_gains(rule, pdms, p6_pick, flag_nc, flag_ttc):
    """Per-set gains (PDMS x 100 vs P6's pick) for the combined / NC-only / TTC-only veto + all-flagged mask."""
    a = np.arange(len(rule))
    base = pdms[a, p6_pick]
    out = {}
    for name, fl in (("both", flag_nc | flag_ttc), ("nc_only", flag_nc), ("ttc_only", flag_ttc)):
        pk, allf = veto_pick(rule, fl, p6_pick)
        out[name] = 100.0 * (pdms[a, pk] - base)
        if name == "both":
            out["all_flagged"] = allf
    return out


def oracle_gains(rule, pdms, nc, ttc, p6_pick):
    """O_NC+TTC / O_NC / O_TTC: the TRUE veto, same code path as the arms (flags = truth)."""
    return set_gains(rule, pdms, p6_pick, nc == 0, ttc == 0)


def gfid(flag, nc, ttc, rule):
    """precision / recall of `flag` vs truth (nc==0)|(ttc==0), over all proposals and over P6's top-5 by the canonical rule."""
    truth = (nc == 0) | (ttc == 0)
    top5 = np.argsort(-rule, axis=1, kind="stable")[:, :5]
    m5 = np.zeros_like(flag)
    np.put_along_axis(m5, top5, True, axis=1)
    out = {}
    for name, m in (("all", np.ones_like(flag)), ("top5", m5)):
        f, t = flag[m], truth[m]
        tp = int((f & t).sum()); fp_ = int((f & ~t).sum()); fn = int((~f & t).sum())
        out[name] = {"tp": tp, "fp": fp_, "fn": fn, "n": int(m.sum()), "truth_pos": int(t.sum()),
                     "precision": tp / (tp + fp_) if tp + fp_ else None, "recall": tp / (tp + fn) if tp + fn else None}
    return out


# ----------------------------------------------------------------------------------------------- C-SHUF
def derangement(n, rng):
    """Random permutation of range(n) with NO fixed point (n >= 2), by rejection."""
    if n < 2:
        raise ValueError("derangement needs n >= 2")
    ident = np.arange(n)
    while True:
        p = rng.permutation(n)
        if not (p == ident).any():
            return p


def shuffle_partners(logs, seed=20261114):
    """partner[i] = another set of the SAME log (derangement per log, logs in sorted order, one rng); -1 where the log has one set."""
    rng = np.random.default_rng(seed)
    logs = np.asarray(logs)
    partner = np.full(len(logs), -1, dtype=np.int64)
    for lg in sorted(set(logs.tolist())):
        ii = np.nonzero(logs == lg)[0]
        if len(ii) < 2:
            continue
        partner[ii] = ii[derangement(len(ii), rng)]
    return partner


# ----------------------------------------------------------------------------------------------- bootstrap
def cluster_bootstrap(logs, values, n_draws=BOOT_DRAWS, seed=BOOT_SEED, counts=None):
    """Paired log-cluster bootstrap. values: {name: per-set array (S,)} (sums are what is resampled). Returns, per draw, the
    totals {name: (B,)} and the resampled set count (B,) so a ratio of totals is taken draw by draw. SAME draws for every name."""
    logs = np.asarray(logs)
    ul = sorted(set(logs.tolist()))
    L = len(ul)
    pos = {l: i for i, l in enumerate(ul)}
    li = np.array([pos[l] for l in logs.tolist()])
    n_l = np.bincount(li, minlength=L).astype(float)
    sums = {k: np.bincount(li, weights=np.asarray(v, float), minlength=L) for k, v in values.items()}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, L, size=(n_draws, L))
    cnt = np.zeros((n_draws, L))
    for b in range(n_draws):
        cnt[b] = np.bincount(idx[b], minlength=L)
    tot = {k: cnt @ s for k, s in sums.items()}
    return tot, cnt @ n_l


def ci(x, lo=2.5, hi=97.5):
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, lo)), float(np.percentile(x, hi))] if len(x) else [None, None]


def column_report(logs, gains, o_gain, n_draws=BOOT_DRAWS, seed=BOOT_SEED):
    """gains {arm: per-set gain (S,)}; o_gain per-set oracle gain. -> {arm: {gain, gain_ci, capture, capture_ci}} + denominator."""
    vals = dict(gains)
    vals["__O__"] = o_gain
    tot, n = cluster_bootstrap(logs, vals, n_draws, seed)
    out = {"n_sets": int(len(logs)), "n_logs": int(len(set(np.asarray(logs).tolist()))),
           "O_NC+TTC": {"gain": float(np.mean(o_gain)), "ci": ci(tot["__O__"] / n)}}
    for arm, g in gains.items():
        O = float(np.mean(o_gain))
        out[arm] = {"gain": float(np.mean(g)), "gain_ci": ci(tot[arm] / n),
                    "capture": float(np.mean(g) / O) if O != 0 else None,
                    "capture_ci": ci(tot[arm] / np.where(tot["__O__"] == 0, np.nan, tot["__O__"]))}
    return out


# ----------------------------------------------------------------------------------------------- outcome row
def outcome_row(col, cid_status, cshuf_n=None):
    """The card's committed table, mechanically. col = column_report(...) output (+ 'C-REG-*' gains in col['_ctrl']).
    cid_status: 'PASS' | 'FAIL' | 'NOT RUN'. Returns (row, reasons)."""
    why = []
    cs = col.get("C-SHUF")
    if cs is not None and cs.get("gain") is not None:
        if cs["gain"] > 0:
            why.append(f"C-SHUF gain {cs['gain']:+.4f} > 0")
        if cs["capture"] is not None and cs["capture"] >= 0.10:
            why.append(f"C-SHUF capture {cs['capture']:.4f} >= 0.10")
    for k in ("C-REG-zero", "C-REG-one"):
        if col[k]["gain"] != 0.0:
            why.append(f"{k} gain {col[k]['gain']!r} != 0.0")
    if cid_status == "FAIL":
        why.append("C-ID < 0.99")
    if why:
        return "INVALID", why
    if col["R-true"]["capture"] is None:
        return "NOT RUN", ["denominator is zero"]
    ct, c0 = col["R-true"]["capture"], col["R-t0"]["capture"]
    if ct < 0.50:
        row = "S0B-READOUT"
    elif c0 >= 0.25:
        row = "S0B-GO"
    else:
        row = "S0B-STATIC"
    if cid_status != "PASS":
        why.append(f"C-ID {cid_status}: row is provisional")
    return row, why
