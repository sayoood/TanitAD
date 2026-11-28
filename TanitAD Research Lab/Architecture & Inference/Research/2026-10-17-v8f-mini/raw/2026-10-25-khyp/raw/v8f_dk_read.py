"""v8F-mini LONGITUDINAL DISTANCE KEEPING -- the half of the binding LONGITUDINAL family that every v8F-mini result called "not instrumented".

  python v8f_dk_read.py --eval-dir <runs/v8f_mini_full/eval> --out dk_read.json [--lead-block B1.npz ...] [--perwindow-out pw.npz]

WHY THIS FILE EXISTS. The binding four-family rule says LONGITUDINAL = speed accuracy AND distance keeping (headway / time-gap / TTC to the
lead agent). The programme built that instrument for this very 141-clip, 2,399-window grid on 2026-09-02 (BACKLOG R1, `D-REFAV1-LEAD-BLOCK`):
`taniteval/tools/build_lead_block_b1.py` -> the banked per-frame lead block (`b1_eval_lead_block.npz`, `obstacle.offline`, 147 v7.2 EVAL clips) ->
`taniteval/lead_metrics.py`. It was wired into `refav1_arm.py` only. v8F-mini never called it: BUILT, TESTED, UNREACHABLE FROM ITS CALLER.
This reader is the missing call site. It defines NO new metric: every number below is `lead_metrics.distance_keeping` /
`distance_keeping_by_speed` / `paired_distance_keeping` / `lead_viol` / `paired_lead_viol` on the arm's `pred` path, scored against the SAME logged
lead future as the human (GT) path.

FRAMES AND UNITS (each established from source, quote with any number):
  * `pred`, `traj30` and the GT are metres in the WINDOW-ORIGIN EGO FRAME at cache step t (x forward, y left), 30 waypoints at t+0.2 s .. t+6.0 s
    (`v8f_mini_data.gt_future` / `window_kin`: `fk.gt_waypoints(grid, ts, 1..30)`, grid = raw poses[0::2]; `DT = 0.2`). `pred` is scored against that
    GT in `v8f_mini_eval.traj_metrics` (`np.linalg.norm(pred - gt, axis=-1)`), so pred and GT share frame and units BY CONSTRUCTION; this reader
    additionally REFUSES a pred whose ||pred - gt|| differs from the run's saved `ade6` (the family reader's guard) and a pred whose first-step
    displacement is not ~ v0 * dt (a units / axes guard).
  * the lead block (`build_lead_block_b1.py` docstring, `meta_json.conventions`): lead CENTRE in the same window-origin ego frame at t0, x forward,
    y left, metres; `ts_rel_s = 0.2 * (1..K)`; `speeds` = ego speed at t0 (m/s); rows keyed (clip_id, RAW 10 Hz frame), cache step t <-> frame 2t
    (the refav1 loader's rule; `v8f_mini_data.window_kin`: k = 2 * ts). gap = along - size_x/2 = rig origin to the lead's REAR face.
  * JOIN GUARDS (the same three `refav1_arm.join_lead_block` applies, re-stated here because that function is bound to the dump-file format):
    (1) block `ts_rel_s[:K] == dt * (1..K)` to 1e-6 s; (2) the eval's v0 equals the block speed at every joined row to 1e-3 m/s -- the LABEL-FREE
    proof the (clip, frame) mapping is right; (3) a duplicate (clip, frame) row, or a grid window with no row, is a refusal (not a silent NO_LABEL).
  * KINEMATIC-CONTRACT CONTROL (a known value): `distance_keeping` of the HUMAN path recomputed here must reproduce the block's own `gt_*` columns
    (computed from the labeler's 10 Hz poses). MEASURED on the banked block: median |d headway| 0.019 m; a frame or unit error is metres. REFUSED above
    `CONTRACT_MEDIAN_TOL_M`. Only checkable when the horizon used equals the block's K (the gt_* columns are over the block's K steps).

DEFINITIONS (from `lead_metrics`, unchanged): headway = min over the horizon steps at which the lead is AHEAD and inside the |lat| < 2 m corridor of the
predicted path's own local heading, of the gap; time-gap = headway / v0, UNDEFINED below 0.5 m/s (NaN, not clamped); min-TTC = min over closing steps
(closing rate > 0.1 m/s), CAPPED at 30 s when never closing -- a CENSORED observation, so `n_closing` rides with every TTC mean. LEAD-VIOL = share of
lead windows whose plan's min time-gap is < 1.0 s while the human's is >= 1.0 s (its registered at-risk denominator, and the LEAD-PROGRESS control that
stops it being met by crawling, are reported as lead_metrics defines them).

WHAT A WINDOW CAN BE. The block labels every window LEAD / NO_LEAD / NOT_STRAIGHT / NO_LABEL (`lead_source`). Only LEAD windows enter a mean. NO_LEAD
(road clear within 80 m), NOT_STRAIGHT (the ego's own future leaves the straight corridor: a bend hides the lead) and NO_LABEL (no obstacle.offline, or
the horizon leaves the labelled span) are COUNTED with their reasons and are NEVER scored as free flow. A LEAD window whose ARM path never lies inside
the corridor behind the lead has no value for that arm (`lead_lost`), counted per arm.

ESTIMATOR. Per-arm means: `ci.episode_cluster_bootstrap`. Every delta: `ci.paired_episode_cluster_bootstrap` over the 141 EPISODES, B = 2,000, on the
windows where BOTH sides are finite (n_used / n_a_only / n_b_only reported). It answers ONLY "would another draw of EPISODES say this?" It is blind to
training-run variance (H-ESTIM-SEED-1): the seed floor F = |mean_s0 - mean_s1| per arm and metric rides beside every delta, and `exceeds_2F_both_seeds` is
the programme's own reading rule. No distance-keeping bar is registered for v8F-mini, so NOTHING here carries PASS / FAIL. Direction: a positive delta
means MORE gap / time / TTC than the comparator, not "better"; the better-or-worse reading is the distance to the HUMAN (`vs_human`) and LEAD-VIOL.

OPEN LOOP. The lead's logged future does not react to the arm (tier "T1-class open loop", the stamp the source evals carry): an arm that closes on the lead
is scored against where the lead WAS, not against how it would have braked. That is the instrument's contract; it is a caveat, not a flaw to patch here.

Refusals (rc 2, message on stderr): a missing eval / lead-block / P10 file; a lead block without the per-frame keys; a horizon grid mismatch; a grid
window with no block row; a speed-proof, kinematic-contract, ade or first-step-units failure; a pred of the wrong shape; a non-finite pred; arm files
not on one eid grid; a P10 that does not reproduce floors_perwindow.npz; an identity control (arm vs itself) that does not read exactly +0 [0, 0].
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
PKG = HERE.parents[1]
try:
    REPO = HERE.parents[5]
except IndexError:                                    # pragma: no cover - a shallow scratch layout
    REPO = HERE.parent
RAW = PKG / "raw"
LEAD_BLOCK_DEFAULT = REPO / "TanitAD Research Lab" / "Benchmarks & Evals" / "Research" / "2026-09-02-b1-eval-lead-block" / "raw" / "b1_eval_lead_block.npz"
#: build_lead_block_b1.py --k 30 on the SAME inputs (6 s horizon); landed beside this reader's result, see RESULT_DK.md
LEAD_BLOCK_K30 = RAW / "2026-10-22-dk" / "b1_eval_lead_block_k30.npz"
P10_DEFAULT = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-refav1-hj-mini" / "raw" / "p10_yaw4512" / "p10_eval_pred6.npy"

SCHEMA = "v8f-dk-read-v1"
TIER = "T1-class open loop (the stamp the source evals carry): the arm is conditioned on its OWN held upper-layer outputs, never on GT; the lead's logged future does not react to it"
ARMS, SEEDS = ("M1", "M2b"), (0, 1)
#: (a, b) -> a delta of a - b. K-1's pairing (every arm - P10) and WM-1's (M2b - M1 per seed), seed-matched.
PAIRS = [("M2b_s0", "P10"), ("M2b_s1", "P10"), ("M1_s0", "P10"), ("M1_s1", "P10"), ("M2b_s0", "M1_s0"), ("M2b_s1", "M1_s1")]
DK_KEYS = ("headway_min_m", "time_gap_min_s", "min_ttc_s")
STATES = ("LEAD", "NO_LEAD", "NOT_STRAIGHT", "NO_LABEL")
B = 2000
DT = 0.2
HORIZON = 30
LEAD_SPEED_TOL_MPS = 1e-3          # refav1_arm.LEAD_SPEED_TOL_MPS: float32 round-tripping only
LEAD_TS_TOL_S = 1e-6               # refav1_arm.LEAD_TS_TOL_S
TOL_ADE = 1e-5                     # v8f_mini_full_report.TOL_ADE
CONTRACT_MEDIAN_TOL_M = 0.25       # MEASURED 0.019 m on the banked block; 13x margin; a frame / unit error is metres
CONTRACT_MIN_OVERLAP = 0.95        # of the block's finite gt headways, the recomputed human must also be finite on this share
FIRST_STEP_RATIO = (0.2, 5.0)      # median (pred_x[0] / (v0 * dt)) over windows with v0 > 2 m/s. Wide on purpose: it catches x100 / x1000 unit slips and
#                                    swapped axes (ratio ~ 0), NOT a ft/m slip (3.28) and NOT a slow or fast plan; the saved-ade6 guard owns frame / sign errors
FIRST_STEP_MIN_V0 = 2.0
MIN_STRATUM_N = 30


class Refusal(Exception):
    """A guard fired: the read is over the wrong windows / units and would still produce a plausible number."""


def refuse(msg: str):
    raise Refusal(msg)


# ------------------------------------------------------------------------------------------------ imports
def _paths():
    for p in (REPO / "stack", REPO / "taniteval"):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _lm():
    _paths()
    from taniteval import lead_metrics as lm
    return lm


def _ci():
    _paths()
    from taniteval import ci
    return ci


def code_provenance() -> dict:
    """Where the instrument was imported FROM. The venv's editable `tanitad` points at a dead G: tree; a stale import reads like a result."""
    lm = _lm()
    out = {"lead_metrics": os.path.abspath(lm.__file__)}
    ci = _ci()
    out["ci"] = os.path.abspath(ci.__file__)
    for k, p in out.items():
        if p.replace("\\", "/").upper().startswith("G:/"):
            refuse(f"{k} was imported from {p}: the dead G: tree. Put the repo's stack and taniteval on PYTHONPATH.")
    return out


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ the lead block
_BLOCK_REQUIRED = ("clip_id", "frame", "ts_rel_s", "leads", "lead_lens", "speeds", "state")


def load_block(path) -> dict:
    """The per-frame B1 block -> plain dict + ``_idx`` {(clip, frame): row} + ``_meta``. REFUSES a missing file, a positional block (no
    clip_id / frame) and a duplicate (clip, frame) row (the join would be ambiguous)."""
    path = str(path)
    if not os.path.exists(path):
        refuse(f"lead block {path} does not exist: nothing to score distance keeping against. Build it with taniteval/tools/build_lead_block_b1.py "
               f"(--k 30 for a 6 s horizon) or point --lead-block at the banked 2026-09-02-b1-eval-lead-block/raw/b1_eval_lead_block.npz")
    with np.load(path, allow_pickle=True) as z:
        blk = {k: z[k] for k in z.files}
    miss = [k for k in _BLOCK_REQUIRED if k not in blk]
    if miss:
        refuse(f"lead block {path} is missing {miss}: it is not a per-frame B1 block (a positional block cannot be joined to a (clip, frame) grid)")
    n = int(np.asarray(blk["leads"]).shape[0])
    cid = np.asarray(blk["clip_id"]).astype(str).reshape(-1)
    fr = np.asarray(blk["frame"]).astype(np.int64).reshape(-1)
    if cid.shape[0] != n or fr.shape[0] != n or np.asarray(blk["state"]).reshape(-1).shape[0] != n:
        refuse(f"lead block {path}: clip_id {cid.shape[0]} / frame {fr.shape[0]} rows for {n} lead rows: corrupt")
    idx = {}
    for i, (c, f) in enumerate(zip(cid.tolist(), fr.tolist())):
        if (c, f) in idx:
            refuse(f"lead block {path}: duplicate row for ({c}, frame {f})")
        idx[(c, f)] = i
    meta = {}
    if "meta_json" in blk:
        try:
            meta = json.loads(np.asarray(blk["meta_json"], dtype=np.uint8).tobytes().decode("utf-8"))
        except Exception as e:                         # noqa: BLE001
            meta = {"unreadable": f"{type(e).__name__}: {e}"}
    blk["_idx"], blk["_meta"], blk["_path"] = idx, meta, path
    blk["_k"] = int(np.asarray(blk["leads"]).shape[1])
    return blk


def join_block(blk, clip, frame, v0, *, k, dt=DT, speed_tol=LEAD_SPEED_TOL_MPS) -> dict:
    """Lead rows for the grid's windows IN GRID ORDER, keyed (clip, frame). Only LEAD rows keep lead data."""
    ts = np.asarray(blk["ts_rel_s"], np.float64).reshape(-1)
    want = np.arange(1, int(k) + 1, dtype=np.float64) * float(dt)
    if int(k) > blk["_k"] or ts.size < int(k) or float(np.max(np.abs(ts[:k] - want))) > LEAD_TS_TOL_S:
        refuse(f"lead block horizon grid {np.round(ts[:min(ts.size, int(k))], 4).tolist()} != the eval's {np.round(want, 4).tolist()} (dt {dt} s, K {k}); "
               f"a lead track at other instants must not be truncated or resampled onto this path")
    if "dt_s" in blk and abs(float(np.asarray(blk["dt_s"]).reshape(-1)[0]) - float(dt)) > 1e-9:
        refuse(f"lead block dt_s {float(np.asarray(blk['dt_s']).reshape(-1)[0])} != the eval's {dt}")
    clip, frame = np.asarray(clip).astype(str), np.asarray(frame).astype(np.int64)
    rows = np.array([blk["_idx"].get((c, int(f)), -1) for c, f in zip(clip.tolist(), frame.tolist())], np.int64)
    if (rows < 0).any():
        bad = np.flatnonzero(rows < 0)
        absent_clips = sorted(set(clip[bad].tolist()) - {c for c, _ in blk["_idx"]})
        refuse(f"GRID MISMATCH: {int(bad.size)} of {rows.size} eval windows have no row in the lead block "
               f"(e.g. clip {clip[bad[0]]} frame {int(frame[bad[0]])}; {len(absent_clips)} clips absent from the block altogether). "
               f"The block must cover every window of the eval grid; a partial join would score a different sample than the ADE it sits beside.")
    v0 = np.asarray(v0, np.float64).reshape(-1)
    bsp = np.asarray(blk["speeds"], np.float64).reshape(-1)[rows]
    dmax = float(np.max(np.abs(v0 - bsp)))
    if not dmax <= speed_tol:
        refuse(f"LABEL-FREE SPEED PROOF FAILED: max |eval v0 - block speed at the joined row| = {dmax:.4g} m/s > {speed_tol}. The (clip, frame) mapping is wrong; "
               f"refusing to place another clip's traffic on these windows")
    state = np.asarray(blk["state"]).astype(str).reshape(-1)[rows]
    is_lead = state == "LEAD"
    leads = np.full((rows.size, int(k), 2), np.nan)
    lens = np.full(rows.size, np.nan)
    leads[is_lead] = np.asarray(blk["leads"], np.float64)[rows[is_lead]][:, :int(k)]
    lens[is_lead] = np.asarray(blk["lead_lens"], np.float64).reshape(-1)[rows[is_lead]]
    out = {"rows": rows, "state": state, "leads": leads, "lead_lens": lens, "speeds": v0, "is_lead": is_lead,
           "speed_proof_max_mps": dmax, "k": int(k), "dt": float(dt),
           "counts": {s: int((state == s).sum()) for s in STATES}}
    unk = sorted(set(state.tolist()) - set(STATES))
    if unk:
        refuse(f"lead block carries states {unk} outside {STATES}")
    for kk in DK_KEYS:
        col = f"gt_{kk}"
        out[col] = (np.asarray(blk[col], np.float64).reshape(-1)[rows] if col in blk else None)
    out["block_k"] = blk["_k"]
    return out


# ------------------------------------------------------------------------------------------------ guards on the arm's paths
def check_pred(name, pred, gt, v0, k, saved_ade6=None):
    """Shape / finiteness / frame / units guards on one arm's ``pred`` [N, 30, 2]."""
    pred = np.asarray(pred)
    if pred.ndim != 3 or pred.shape[0] != gt.shape[0] or pred.shape[1] < k or pred.shape[2] != 2:
        refuse(f"{name}: pred is {pred.shape}, expected ({gt.shape[0]}, >= {k}, 2): not this eval grid")
    if not np.isfinite(pred[:, :k]).all():
        refuse(f"{name}: pred holds non-finite waypoints within the first {k} steps")
    if saved_ade6 is not None:
        n = min(pred.shape[1], gt.shape[1])
        ade = np.linalg.norm(pred[:, :n].astype(np.float64) - gt[:, :n], axis=-1).mean(1)
        dev = float(np.abs(ade - np.asarray(saved_ade6, np.float64)).max())
        if not dev <= TOL_ADE:
            refuse(f"{name}: ||pred - gt|| differs from its saved ade6 by {dev:.3g} m (> {TOL_ADE}): the rebuilt ground truth is not this run's grid / frame / units")
    m = np.asarray(v0, np.float64) > FIRST_STEP_MIN_V0
    if m.sum() >= 10:
        r = float(np.median(pred[m, 0, 0].astype(np.float64) / (np.asarray(v0, np.float64)[m] * DT)))
        if not FIRST_STEP_RATIO[0] <= r <= FIRST_STEP_RATIO[1]:
            refuse(f"{name}: median first-step forward displacement / (v0 * dt) = {r:.3f}, outside {FIRST_STEP_RATIO}: pred is not metres in the (x forward) "
                   f"window-origin frame at 0.2 s steps (units or axes error)")


def contract_check(dk_h, J) -> dict:
    """Human path recomputed here vs the block's own gt_* columns (a KNOWN VALUE). Skipped, with the reason, when K != the block's K."""
    if J["k"] != J["block_k"]:
        return {"status": "SKIPPED", "reason": f"horizon used K={J['k']} != the block's K={J['block_k']}: its gt_* columns span {J['block_k']} steps"}
    if J["gt_headway_min_m"] is None:
        return {"status": "SKIPPED", "reason": "the block carries no gt_headway_min_m column"}
    a, b = np.asarray(dk_h["headway_min_m"], np.float64), J["gt_headway_min_m"]
    nb = int(np.isfinite(b).sum())
    both = np.isfinite(a) & np.isfinite(b)
    if nb and both.sum() < CONTRACT_MIN_OVERLAP * nb:
        refuse(f"KINEMATIC-CONTRACT CHECK FAILED: the recomputed human path has a finite headway on {int(both.sum())} of the block's {nb} GT windows "
               f"(< {CONTRACT_MIN_OVERLAP:.0%}): the lead is not in the same frame as the path")
    d = np.abs(a[both] - b[both])
    med = float(np.median(d)) if d.size else 0.0
    if not med <= CONTRACT_MEDIAN_TOL_M:
        refuse(f"KINEMATIC-CONTRACT CHECK FAILED: median |headway(recomputed human) - block gt_headway_min_m| = {med:.3f} m > {CONTRACT_MEDIAN_TOL_M} m "
               f"(MEASURED 0.019 m on the banked block): a join, frame or unit error, not a model result")
    out = {"status": "OK", "n_both": int(both.sum()), "n_block_finite": nb, "n_recomputed_only": int((np.isfinite(a) & ~np.isfinite(b)).sum()),
           "median_abs_headway_diff_m": round(med, 5), "p95_abs_headway_diff_m": round(float(np.percentile(d, 95)), 4) if d.size else None,
           "max_abs_headway_diff_m": round(float(d.max()), 4) if d.size else None, "tol_median_m": CONTRACT_MEDIAN_TOL_M}
    for kk in ("time_gap_min_s", "min_ttc_s"):
        bb = J[f"gt_{kk}"]
        if bb is not None:
            aa = np.asarray(dk_h[kk], np.float64)
            bo = np.isfinite(aa) & np.isfinite(bb)
            out[f"median_abs_{kk}_diff"] = round(float(np.median(np.abs(aa[bo] - bb[bo]))), 5) if bo.any() else None
    return out


# ------------------------------------------------------------------------------------------------ per-arm and paired reads
def _mean_ci(vals, eid, n_boot, seed):
    v = np.asarray(vals, np.float64)
    ok = np.isfinite(v)
    if not ok.any():
        return {"n": 0, "status": "NOT-APPLICABLE", "reason": "no finite value"}
    r = _ci().episode_cluster_bootstrap(v[ok], eid[ok], reduce="mean", n_boot=n_boot, seed=seed)
    r["n"] = int(ok.sum())
    return r


def _reasons(dk, J, key):
    """Why a window has no value for ``key``: the block's state, and for LEAD windows what the arm did."""
    st, v = J["state"], np.asarray(dk[key], np.float64)
    have = np.isfinite(v)
    r = {s: int((st == s).sum()) for s in STATES if s != "LEAD"}
    lead = J["is_lead"]
    r["lead_window_but_path_never_in_corridor"] = int((lead & ~np.isfinite(np.asarray(dk["headway_min_m"], np.float64))).sum())
    if key == "time_gap_min_s":
        r["standstill_v0_below_0p5_mps"] = int((lead & np.isfinite(np.asarray(dk["headway_min_m"], np.float64)) & ~have).sum())
    r["n_with_value"], r["n_windows"] = int(have.sum()), int(have.size)
    return r


def arm_block(name, dk, hdk, J, gt_k, path_k, eid, n_boot, seed):
    lm = _lm()
    ci = _ci()
    out = {"n_windows": int(eid.size), "n_lead_windows": int(J["is_lead"].sum()), "window_states": dict(J["counts"]), "metrics": {}, "no_value_reasons": {}, "vs_human": {}}
    for kk in DK_KEYS:
        m = _mean_ci(dk[kk], eid, n_boot, seed)
        if kk == "min_ttc_s":
            fin = np.isfinite(np.asarray(dk[kk], np.float64))
            m["n_closing"] = int(dk.get("n_closing", 0))
            m["ttc_cap_s"] = float(dk.get("ttc_cap_s", 30.0))
            m["censoring_note"] = dk.get("censoring_note")
            m["n_censored_at_cap"] = int((fin & (np.asarray(dk[kk], np.float64) >= m["ttc_cap_s"] - 1e-9)).sum())
        out["metrics"][kk] = m
        out["no_value_reasons"][kk] = _reasons(dk, J, kk)
        a, h = np.asarray(dk[kk], np.float64), np.asarray(hdk[kk], np.float64)
        both = np.isfinite(a) & np.isfinite(h)
        vh = {"n_both": int(both.sum()), "n_arm_only": int((np.isfinite(a) & ~np.isfinite(h)).sum()), "n_human_only": int((np.isfinite(h) & ~np.isfinite(a)).sum())}
        if both.any():
            vh["signed_arm_minus_human"] = ci.paired_episode_cluster_bootstrap(a[both], h[both], eid[both], n_boot=n_boot, seed=seed)
            vh["abs_error"] = ci.episode_cluster_bootstrap(np.abs(a[both] - h[both]), eid[both], reduce="mean", n_boot=n_boot, seed=seed)
        else:
            vh["status"] = "NOT-APPLICABLE"
        out["vs_human"][kk] = vh
    lv = lm.lead_viol(dk, hdk, eid, lead_valid=J["is_lead"], n_boot=n_boot, seed=seed,
                      plan_arc=lm.path_arc_length(path_k), human_arc=lm.path_arc_length(gt_k))
    pw = lv.pop("_per_window")
    keep = ("status", "reason", "definition", "n_lead_windows", "n_viol", "share", "ci", "n_at_risk", "share_of_at_risk", "share_of_at_risk_ci",
            "n_human_tg_undefined", "n_human_tg_below_thr", "n_plan_lost_lead", "share_incl_lost_lead", "lead_progress_ratio", "lead_progress_ratio_clipped")
    out["lead_viol"] = {k: lv[k] for k in keep if k in lv}
    pr = dict(lv.get("progress") or {})
    pr.pop("_per_window", None)
    out["lead_viol"]["progress"] = {k: pr.get(k) for k in ("status", "n", "n_at_risk", "lead_progress_ratio", "lead_progress_ratio_clipped", "meets_min_progress",
                                                         "n_below_min", "n_human_arc_degenerate", "reason") if k in pr}
    out["by_speed"] = lm.distance_keeping_by_speed(dk, J["speeds"], eid, states=J["state"], n_boot=min(n_boot, 1000), seed=seed)
    return out, {"_per_window": pw, **lv}


def _seed_floor(arms_out):
    F = {}
    for arm in ARMS:
        F[arm] = {}
        for kk in DK_KEYS:
            a, b = (arms_out[f"{arm}_s{s}"]["metrics"][kk] for s in SEEDS)
            F[arm][kk] = abs(a["mean"] - b["mean"]) if "mean" in a and "mean" in b else None
        a, b = (arms_out[f"{arm}_s{s}"]["lead_viol"].get("share_of_at_risk") for s in SEEDS)
        F[arm]["lead_viol_share_of_at_risk"] = abs(a - b) if a is not None and b is not None else None
    return F


def pair_block(a_name, b_name, dkS, lvS, absE, J, eid, F, n_boot, seed):
    lm = _lm()
    ci = _ci()
    out = {"a_minus_b": f"{a_name} - {b_name}", "dk": lm.paired_distance_keeping(dkS[a_name], dkS[b_name], eid, names=(a_name, b_name), n_boot=n_boot, seed=seed)["metrics"],
           "abs_error_vs_human": {}}
    for kk in DK_KEYS:
        a, b = absE[a_name][kk], absE[b_name][kk]
        ok = np.isfinite(a) & np.isfinite(b)
        out["abs_error_vs_human"][kk] = ({"n_used": int(ok.sum()), **ci.paired_episode_cluster_bootstrap(a[ok], b[ok], eid[ok], n_boot=n_boot, seed=seed),
                                          "reading": "positive = the first arm sits FARTHER from the human's own value than the second"} if ok.any()
                                         else {"n_used": 0, "status": "NOT-APPLICABLE"})
    pl = lm.paired_lead_viol(lvS[a_name], lvS[b_name], eid, names=(a_name, b_name), n_boot=n_boot, seed=seed)
    out["lead_viol"] = {"all_lead_windows": {k: pl.get(k) for k in ("n_lead_windows", "n_viol_a", "n_viol_b", "delta", "lo", "hi", "separated", "n_windows", "n_episodes", "estimator")},
                        "at_risk": pl.get("at_risk"), "progress": pl.get("progress")}
    # the programme's rule for a lever effect: the delta must also clear 2F at BOTH seeds; a single paired pair cannot show that, so each row carries
    # the floor of each ARM it contains (an arm-vs-P10 row has only the arm's F; P10 is deterministic).
    fl = {}
    for nm in (a_name, b_name):
        arm = nm.split("_s")[0]
        if arm in F:
            fl[arm] = F[arm]
    out["seed_floor_F_of_arms_in_pair"] = fl
    return out


def two_f_flags(pairs_out, F):
    """exceeds_2F_both_seeds per (arm contrast, metric): the paired delta separated and |delta| > 2F at BOTH seeds. F of the ARM(S) in the contrast."""
    flags = {}
    for contrast in (("M2b", "P10"), ("M1", "P10"), ("M2b", "M1")):
        key = f"{contrast[0]}_minus_{contrast[1]}"
        flags[key] = {}
        for kk in DK_KEYS:
            ok = True
            for s in SEEDS:
                a = f"{contrast[0]}_s{s}"
                b = "P10" if contrast[1] == "P10" else f"{contrast[1]}_s{s}"
                r = pairs_out[f"{a}__minus__{b}"]["dk"][kk]
                if r.get("status") != "OK" or not r.get("separated"):
                    ok = False
                    break
                floor = max(F[c][kk] or 0.0 for c in contrast if c in F)
                if not abs(r["delta"]) > 2.0 * floor:
                    ok = False
                    break
            flags[key][kk] = bool(ok)
    return flags


def ttc_stride2_sensitivity(preds, gt_k, J, eid, k, dt, n_boot, seed, pairs=PAIRS) -> dict:
    """SENSITIVITY, not a metric: min-TTC is a min over per-step closing rates, so a rough path (the learned arms' mean |step speed change| is ~2.2x
    the human's, `2026-10-20-lon-rough/NOTE.md`) reads a lower TTC from jitter alone. The SAME `lead_metrics.distance_keeping`, called on every other
    waypoint (0.4 s spacing, `dt` = 0.4, the lead track subsampled identically), halves the jitter's weight. If an arm-vs-P10 TTC delta shrinks here it
    was roughness; if it persists it is a real closing. Needs k >= 4."""
    lm, ci = _lm(), _ci()
    if k < 4:
        return {"status": "SKIPPED", "reason": f"horizon K={k} < 4: fewer than 2 points at 0.4 s spacing"}
    sl = slice(1, k, 2)
    leads2, dt2 = J["leads"][:, sl], 2.0 * dt
    dk = {}
    out = {"_what": "min-TTC on every other waypoint (0.4 s spacing, dt=0.4); headway / time-gap are NOT re-read here", "dt_s": dt2, "arms": {}, "pairs": {}}
    for nm, p in {**preds, "human": gt_k}.items():
        pk = np.asarray(p, np.float64)[:, :k][:, sl]
        dk[nm] = lm.distance_keeping(pk, leads2, J["lead_lens"], J["speeds"], dt2)
        t = _mean_ci(dk[nm]["min_ttc_s"], eid, n_boot, seed)
        t["n_closing"] = int(dk[nm].get("n_closing", 0))
        out["arms"][nm] = t
    for a, b in pairs:
        if a in dk and b in dk:
            out["pairs"][f"{a}__minus__{b}"] = lm.paired_distance_keeping(dk[a], dk[b], eid, names=(a, b), n_boot=n_boot, seed=seed)["metrics"]["min_ttc_s"]
    return out


def read_dk(*, preds: dict, gt, eid, clip, frame, v0, block, k, dt=DT, n_boot=B, seed=0, saved_ade6=None, p10_floor_ade6=None, pairs=PAIRS) -> dict:
    """The whole read for one block / horizon. ``preds``: name -> pred [N, 30, 2] (names M1_s0 .. M2b_s1, P10)."""
    lm = _lm()
    prov = code_provenance()
    gt = np.asarray(gt, np.float64)
    eid = np.asarray(eid)
    if eid.shape[0] != gt.shape[0]:
        refuse(f"eid has {eid.shape[0]} entries for {gt.shape[0]} windows")
    if dt != DT:
        refuse(f"dt {dt} != {DT}")
    for nm, p in preds.items():
        check_pred(nm, p, gt, v0, k, (saved_ade6 or {}).get(nm))
    if p10_floor_ade6 is not None and "P10" in preds:
        dev = float(np.abs(np.linalg.norm(np.asarray(preds["P10"], np.float64) - gt, axis=-1).mean(1) - np.asarray(p10_floor_ade6, np.float64)).max())
        if not dev <= TOL_ADE:
            refuse(f"the P10 predictions do not reproduce floors_perwindow.npz's P10_refit__ade6 (max dev {dev:.3g} m)")
    J = join_block(block, clip, frame, v0, k=k, dt=dt)
    hdk = lm.distance_keeping(gt[:, :k], J["leads"], J["lead_lens"], J["speeds"], dt)
    contract = contract_check(hdk, J)
    gt_k = gt[:, :k]
    arms_out, dkS, lvS, absE = {}, {}, {}, {}
    for nm, p in preds.items():
        pk = np.asarray(p, np.float64)[:, :k]
        dk = lm.distance_keeping(pk, J["leads"], J["lead_lens"], J["speeds"], dt)
        dkS[nm] = dk
        arms_out[nm], lvS[nm] = arm_block(nm, dk, hdk, J, gt_k, pk, eid, n_boot, seed)
        absE[nm] = {kk: np.abs(np.asarray(dk[kk], np.float64) - np.asarray(hdk[kk], np.float64)) for kk in DK_KEYS}
    human = {"n_windows": int(eid.size), "n_lead_windows": int(J["is_lead"].sum()),
             "metrics": {kk: _mean_ci(hdk[kk], eid, n_boot, seed) for kk in DK_KEYS},
             "no_value_reasons": {kk: _reasons(hdk, J, kk) for kk in DK_KEYS}, "kinematic_contract": contract}
    human["metrics"]["min_ttc_s"]["n_closing"] = int(hdk.get("n_closing", 0))
    have_floor = all(f"{a}_s{s}" in arms_out for a in ARMS for s in SEEDS)
    F = _seed_floor(arms_out) if have_floor else {}
    # the identity control: an arm against itself must read EXACTLY +0 [0, 0] on every metric, or the paired instrument is broken
    ident = {}
    ref = next(iter(preds))
    c = lm.paired_distance_keeping(dkS[ref], dkS[ref], eid, names=(ref, ref), n_boot=n_boot, seed=seed)["metrics"]
    for kk in DK_KEYS:
        r = c[kk]
        ident[kk] = {"delta": r.get("delta"), "lo": r.get("lo"), "hi": r.get("hi"), "n_used": r.get("n_used")}
        if r.get("status") == "OK" and not (r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0):
            refuse(f"IDENTITY CONTROL FAILED: {ref} - {ref} reads {r['delta']} [{r['lo']}, {r['hi']}] on {kk}, not +0 [0, 0]: the paired instrument is broken")
    lvi = lm.paired_lead_viol(lvS[ref], lvS[ref], eid, names=(ref, ref), n_boot=n_boot, seed=seed)
    if lvi.get("status") == "OK" and not (lvi["delta"] == 0.0 and lvi["lo"] == 0.0 and lvi["hi"] == 0.0):
        refuse(f"IDENTITY CONTROL FAILED: LEAD-VIOL {ref} - {ref} reads {lvi['delta']} [{lvi['lo']}, {lvi['hi']}], not +0 [0, 0]")
    ident["lead_viol"] = {"delta": lvi.get("delta"), "lo": lvi.get("lo"), "hi": lvi.get("hi")}
    pairs_out = {}
    for a, b in pairs:
        if a in preds and b in preds:
            pairs_out[f"{a}__minus__{b}"] = pair_block(a, b, dkS, lvS, absE, J, eid, F, n_boot, seed)
    out = {"horizon": {"k": int(k), "dt_s": dt, "seconds": round(k * dt, 3)},
           "lead_block": {"path": block["_path"], "sha256": sha256(block["_path"]), "block_k": block["_k"], "built_utc": block["_meta"].get("built_utc"),
                          "tool": block["_meta"].get("tool"), "n_rows": int(block["_idx"].__len__())},
           "join": {"key": "(clip_id, RAW frame 2t)", "n_windows": int(eid.size), "n_episodes": int(np.unique(eid).size), "window_states": J["counts"],
                    "speed_proof_max_mps": J["speed_proof_max_mps"], "speed_proof_tol_mps": LEAD_SPEED_TOL_MPS},
           "human_reference": human, "arms": arms_out, "seed_floor_F": F, "pairs": pairs_out,
           "identity_control": {"arm": ref, **ident},
           "ttc_stride2_sensitivity": ttc_stride2_sensitivity(preds, gt_k, J, eid, k, dt, n_boot, seed),
           "code": prov}
    if have_floor:
        out["exceeds_2F_both_seeds"] = two_f_flags(pairs_out, F) if all(f"{a}__minus__{b}" in pairs_out for a, b in PAIRS) else None
    perwindow = {"eid": eid, "state": J["state"], "is_lead": J["is_lead"], "speeds": J["speeds"]}
    for kk in DK_KEYS:
        perwindow[f"human__{kk}"] = np.asarray(hdk[kk], np.float64)
        for nm in preds:
            perwindow[f"{nm}__{kk}"] = np.asarray(dkS[nm][kk], np.float64)
    return {"read": out, "perwindow": perwindow}


# ------------------------------------------------------------------------------------------------ inputs
def ground_truth(a):
    """-> dict(gt [N,30,2], eid, clip, frame, v0, dt, provenance). ``--gt-npz`` (gt, eid, clip, frame, v0, dt) replaces the rebuild; tests use it."""
    if a.gt_npz:
        if not os.path.exists(a.gt_npz):
            refuse(f"--gt-npz {a.gt_npz} does not exist")
        z = np.load(a.gt_npz, allow_pickle=True)
        need = ("gt", "eid", "clip", "frame", "v0", "dt")
        miss = [x for x in need if x not in z.files]
        if miss:
            refuse(f"--gt-npz is missing {miss}")
        return {"gt": z["gt"].astype(np.float64), "eid": z["eid"], "clip": z["clip"].astype(str), "frame": z["frame"].astype(np.int64),
                "v0": z["v0"].astype(np.float64), "dt": float(z["dt"]), "prov": {"source": "gt-npz (TEST INPUT)", "path": a.gt_npz}}
    for p in (a.eval_grid, a.eval_kin, a.eval_join):
        if not os.path.exists(p):
            refuse(f"eval input {p} does not exist")
    _paths()
    from tanitad.data import v8f_mini_data as D
    grid = json.load(open(a.eval_grid, encoding="utf-8"))
    X = D.eval_context(grid, a.eval_kin, D.load_join(a.eval_join), kin_mode="yaw")
    order = list(grid["order"])
    return {"gt": X["gt"].astype(np.float64), "eid": X["eid"], "clip": np.array([order[e] for e in X["eid"]]), "frame": 2 * np.asarray(X["t"], np.int64),
            "v0": np.asarray(X["kin"][:, 0], np.float64), "dt": float(D.DT),
            "prov": {"source": "v8f_mini_data.eval_context(kin_mode='yaw') over the repo's eval inputs; frame = 2t, v0 = kin[:, 0]",
                     "files": {os.path.basename(p): sha256(p) for p in (a.eval_grid, a.eval_kin, a.eval_join)}}}


def load_preds(a, eid):
    preds, saved, prov = {}, {}, {}
    for arm in ARMS:
        for s in SEEDS:
            tag = f"{arm}_s{s}"
            pz = os.path.join(a.eval_dir, f"eval_{tag}.npz")
            if not os.path.exists(pz):
                refuse(f"eval_{tag}.npz is missing from {a.eval_dir}")
            z = np.load(pz)
            if "pred" not in z.files or "eid" not in z.files:
                refuse(f"eval_{tag}.npz carries no pred / eid")
            if not np.array_equal(z["eid"], eid):
                refuse(f"eval_{tag}.npz is not on the rebuilt eval grid (eid differs)")
            preds[tag] = z["pred"].astype(np.float64)
            if "ade6" in z.files:
                saved[tag] = np.asarray(z["ade6"], np.float64)
            prov[tag] = {"path": pz, "sha256": sha256(pz)}
    if not os.path.exists(a.p10_pred):
        refuse(f"the P10 predictions {a.p10_pred} do not exist")
    preds["P10"] = np.load(a.p10_pred).astype(np.float64)
    prov["P10"] = {"path": str(a.p10_pred), "sha256": sha256(a.p10_pred)}
    fl = os.path.join(a.eval_dir, "floors_perwindow.npz")
    floor_ade = np.load(fl)["P10_refit__ade6"] if os.path.exists(fl) else None
    return preds, saved, floor_ade, prov


def _clean(o):
    """JSON-safe: numpy scalars / arrays -> python; NaN / inf -> None (strict JSON has neither)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.generic):
        return _clean(o.item())
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    return o


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", default=str(RAW / "full" / "eval"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--perwindow-out", help="npz of the per-window headway / time-gap / TTC of every arm and the human")
    ap.add_argument("--lead-block", action="append", help="repeatable; default = the banked B1 block (K=10, 2 s) plus raw/2026-10-22-dk/b1_eval_lead_block_k30.npz (K=30, 6 s) when that file exists")
    ap.add_argument("--horizon-k", type=int, help="steps of pred to score (default = each block's K)")
    ap.add_argument("--p10-pred", default=str(P10_DEFAULT))
    ap.add_argument("--n-boot", type=int, default=B)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-grid", default=str(RAW / "eval_grid_windows.json"))
    ap.add_argument("--eval-kin", default=str(RAW / "eval141_kin.npz"))
    ap.add_argument("--eval-join", default=str(RAW / "join_eval141.npz"))
    ap.add_argument("--gt-npz", help="TEST INPUT: npz with gt, eid, clip, frame, v0, dt in place of the rebuild")
    a = ap.parse_args(argv)
    try:
        G = ground_truth(a)
        preds, saved, floor_ade, pprov = load_preds(a, G["eid"])
        blocks = a.lead_block or [str(LEAD_BLOCK_DEFAULT)] + ([str(LEAD_BLOCK_K30)] if LEAD_BLOCK_K30.exists() else [])
        reads, perwindow = {}, {}
        for bp in blocks:
            blk = load_block(bp)
            k = int(a.horizon_k or blk["_k"])
            if k > HORIZON:
                refuse(f"--horizon-k {k} > {HORIZON}")
            res = read_dk(preds=preds, gt=G["gt"], eid=G["eid"], clip=G["clip"], frame=G["frame"], v0=G["v0"], block=blk, k=k, dt=G["dt"], n_boot=a.n_boot,
                          seed=a.seed, saved_ade6=saved, p10_floor_ade6=floor_ade)
            key = f"k{k}"
            if key in reads:
                refuse(f"two lead blocks give the same horizon {key}")
            reads[key] = res["read"]
            perwindow.update({f"{key}__{n}": v for n, v in res["perwindow"].items()})
    except Refusal as e:
        print(f"V8F DK READ REFUSED: {e}", file=sys.stderr)
        return 2
    except (OSError, KeyError, ValueError) as e:
        print(f"V8F DK READ INPUT ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    out = {"schema": SCHEMA, "tier": TIER,
           "estimator": f"ci.episode_cluster_bootstrap (means) / ci.paired_episode_cluster_bootstrap (deltas), B={a.n_boot}, resampling EPISODES; answers 'another draw of "
                        f"episodes' only; training-seed variance enters only through F = |s0 - s1| per arm",
           "n_windows": int(G["eid"].size), "n_episodes": int(np.unique(G["eid"]).size), "ground_truth": G["prov"], "inputs": pprov,
           "no_bar_registered": "no distance-keeping bar is registered for v8F-mini: nothing here is PASS / FAIL", "reads": reads}
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(_clean(out), f, indent=1, default=str)
    if a.perwindow_out:
        np.savez_compressed(a.perwindow_out, **perwindow)
    for key, r in reads.items():
        print(f"[{key}] states {r['join']['window_states']}  speed proof {r['join']['speed_proof_max_mps']:.2e} m/s  contract {r['human_reference']['kinematic_contract']['status']}")
        for nm, ab in r["arms"].items():
            m = ab["metrics"]
            print(f"  {nm:7s} n_hw {m['headway_min_m']['n']:4d}  headway {m['headway_min_m'].get('mean')}  time-gap {m['time_gap_min_s'].get('mean')}  "
                  f"TTC {m['min_ttc_s'].get('mean')} (closing {m['min_ttc_s'].get('n_closing')})  LEAD-VIOL at-risk {ab['lead_viol'].get('share_of_at_risk')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
