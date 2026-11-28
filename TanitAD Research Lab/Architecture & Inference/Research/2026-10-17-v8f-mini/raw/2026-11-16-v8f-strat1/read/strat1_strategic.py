"""strat1_strategic.py -- STRATEGIC route following (PREREG_V8F_STRAT1 section 4.4, READ_PLAN 2.3-2.4) and route-setting RC-A rows (section 4.2).
Tier: T0 at the window's first tick (fresh ages; S's tick = NOW); a T1 per-tick row beside.  Evidence class: MEASURED (ours, zero training).
Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B 2,000, seed 0 -- "another draw of EPISODES" only; "another TRAINING run" is the second seed.

S_CKPT (imported, not copied: strategic_cells.ckpt_cell / side_table, audit commit 0c58187e) is 1 when S's checkpoint heading at arc a is on the announced side by >= theta (25 deg; SLIGHT 10),
scored where d_text + 20 m <= a.  Deciding arc 80 m.  Ground truth per window at the table row eval_off[ci] + t: side from the navtok class (UTURN none), d_text = navtext_nav eval_nav_d
(the countdown), text flag != 0 -> NaN in every cell.  Skill = real - donor (paired); real alone is never quoted for a text-fed arm.
Outcomes (card section 6, per seed pair): S-FOLLOWS = real-donor > 0 separated AND real-best-constant > 0 separated, at both seeds; S-PARTIAL = real-donor separated at both seeds, not above the
best constant at >= 1 seed; S-IGNORES = real-donor not separated at >= 1 seed.  Bands with < 30 windows or < 10 episodes are n/a with their n.
Usage:  python strat1_strategic.py --t0 DIR --out OUT.json [--t1 DIR] [--smoke]     (DIR holds eval_STRAT1-<arm>_<real|donor|zero>.npz from run_t0_strat1.py)
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strat1_lib as L  # noqa: E402
import strategic_cells as SC  # noqa: E402  (copy of raw/2026-11-16-v8f-strategic-audit/strategic_cells.py, sha256 in READ_PINS)

ARCS = (30, 50, 80)
DECIDE = 2                       # index of 80 m
MIN_W, MIN_E = 30, 10
VARIANTS = ("real", "donor", "zero")


# ------------------------------------------------------------------------------------------------------------------------------ ground truth
def ground(navtok, navtext, join, ci, t):
    """per-window ground truth.  navtok / navtext / join: dicts of arrays (npz).  Windows are (ci, t); rows are eval_off + t."""
    off = np.asarray(navtext["eval_off"], np.int64)
    if not (np.array_equal(off, np.asarray(navtok["eval_off"])) and np.array_equal(off, np.asarray(join["off"])) and np.array_equal(navtext["eval_clip"], join["clip"])):
        raise RuntimeError("navtok / navtext / join disagree on clip order or offsets: the tables are misaligned")
    r = L.rows_of(off, ci, t)
    sd, th = SC.side_table(navtok["vocab"])
    cls = np.asarray(navtok["eval_cls_id"], np.int64)[r]
    flag = np.asarray(navtok["eval_flag"])[r]
    text_ok = flag == 0
    side = np.where(text_ok, sd[cls], 0).astype(np.int8)
    theta = np.where(text_ok, th[cls], np.nan)
    d = np.asarray(navtext["eval_nav_d"], np.float64)[r]
    d = np.where(side != 0, d, np.nan)
    return {"row": r, "side": side, "theta": theta, "d_text": d, "d_raw": np.asarray(navtok["eval_dist_m"], np.float64)[r], "text_ok": text_ok,
            "rc": np.asarray(join["rc"])[r], "rc_ok": np.asarray(join["rc_ok"], bool)[r], "ci": np.asarray(ci), "t": np.asarray(t)}


def route_cells(ckpt, G, complete_m=20.0, d=None):
    """S_CKPT [N, 3] (NaN = not scorable) from S's checkpoints in the NOW frame (ckpt[..., 2] = heading, rad)."""
    old = SC.COMPLETE_M
    SC.COMPLETE_M = float(complete_m)
    try:
        return SC.ckpt_cell(np.asarray(ckpt, np.float64)[..., 2], ARCS, G["side"], G["theta"], G["d_text"] if d is None else d, G["text_ok"], G["rc_ok"])
    finally:
        SC.COMPLETE_M = old


def best_const_cells(G, ref):
    """the best constant predictor of the side: always the majority side among the windows scorable in `ref` [N, 3] (per arc).  1 where the window's side is the majority, else 0; NaN elsewhere."""
    out = np.full(ref.shape, np.nan)
    maj = {}
    for m in range(ref.shape[1]):
        ok = np.isfinite(ref[:, m])
        if not ok.any():
            continue
        s = G["side"][ok]
        maj[m] = 1 if (s == 1).sum() >= (s == -1).sum() else -1
        out[ok, m] = (G["side"][ok] == maj[m]).astype(float)
    return out, maj


def rcA_errors(ckpt, G):
    """RC-A checkpoint errors (checkpoint_errors, taniteval.v8f_consistency) per window: pos_err_m, head_err_deg [N, 3] (NaN where rc is invalid)"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("strat1_cons", L.TREE + "/taniteval/taniteval/v8f_consistency.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.checkpoint_errors(np.asarray(ckpt, np.float64), np.asarray(G["rc"], np.float64), G["rc_ok"])


RC_CELLS = tuple(f"{k}@{a}" for k in ("pos_err_m", "head_err_deg") for a in ARCS)


def rc_cells(err):
    """{cell: [N]} of the six RC-A checkpoint-error cells (lower is better) from rcA_errors()"""
    return {f"{k}@{a}": np.asarray(err[k][:, m], np.float64) for k in ("pos_err_m", "head_err_deg") for m, a in enumerate(ARCS)}


def route_setting(CI, arm, ref, floors, eid, nb=L.NB):
    """S1 - C0 on the six RC-A cells (card 4.2), the cost / gain rule of the card with the floor |C0_rep - C0_s0| (obd_read.cell_delta, imported).  cost = separated worse AND |delta| > floor"""
    sys.path.insert(0, os.environ.get("STRAT1_OBD_DIR", HERE + "/theory/obd/read"))
    import obd_read as OR
    return {c: OR.cell_delta(CI, arm[c], ref[c], eid, False, None if floors is None else floors.get(c), nb) for c in RC_CELLS}


# ------------------------------------------------------------------------------------------------------------------------------ counts, rows
def n_block(cell, eid):
    """n per arc: windows and episodes scorable; n/a when below the STRATEGIC_PLAN section 3 rule"""
    out = []
    for m in range(cell.shape[1]):
        ok = np.isfinite(cell[:, m])
        nw, ne = int(ok.sum()), int(len(np.unique(np.asarray(eid)[ok])))
        out.append({"arc_m": ARCS[m], "n_windows": nw, "n_episodes": ne, "n_a": bool(nw < MIN_W or ne < MIN_E)})
    return out


def level(CI, cell, eid):
    """mean S_CKPT per arc with an episode-cluster interval (bootstrap of the arm against a zero reference is not a paired claim: the mean is reported with its n, no CI)"""
    out = []
    for m in range(cell.shape[1]):
        ok = np.isfinite(cell[:, m])
        out.append(float(cell[ok, m].mean()) if ok.any() else None)
    return out


def contrast_rows(CI, a, b, eid):
    """paired a - b per arc; n/a below the n rule (n printed)"""
    rows = []
    nb = n_block(np.where(np.isfinite(a) & np.isfinite(b), a, np.nan), eid)
    for m in range(a.shape[1]):
        if nb[m]["n_a"]:
            rows.append({"arc_m": ARCS[m], "status": "n/a", "reason": f"below {MIN_W} windows or {MIN_E} episodes", "n_windows": nb[m]["n_windows"], "n_episodes": nb[m]["n_episodes"]})
        else:
            r = L.paired(CI, a[:, m], b[:, m], eid)
            r["arc_m"] = ARCS[m]
            rows.append(r)
    return rows


def arm_block(CI, cells, G, eid, c0_real=None):
    """one arm at one seed: cells = {variant: [N,3]}; returns levels, n, and the paired contrasts"""
    real = cells["real"]
    bc, maj = best_const_cells(G, real)
    blk = {"n": n_block(real, eid), "level": {v: level(CI, cells[v], eid) for v in cells}, "best_constant": {"majority_side": {str(ARCS[m]): int(s) for m, s in maj.items()},
           "level": level(CI, bc, eid)}}
    blk["real_minus_donor"] = contrast_rows(CI, real, cells["donor"], eid) if "donor" in cells else None
    blk["real_minus_zero"] = contrast_rows(CI, real, cells["zero"], eid) if "zero" in cells else None
    blk["real_minus_best_constant"] = contrast_rows(CI, real, bc, eid)
    if c0_real is not None:
        blk["minus_C0_real_nav"] = contrast_rows(CI, real, c0_real, eid)
    return blk


def outcome(per_seed):
    """per_seed {seed: arm_block of S1} -> (label, reason).  The deciding arc is 80 m."""
    rows = {}
    for s, b in per_seed.items():
        rd, rb = b["real_minus_donor"][DECIDE], b["real_minus_best_constant"][DECIDE]
        if rd.get("status") != "OK" or rb.get("status") != "OK":
            return "NOT-READ", f"seed {s}: the 80 m row is n/a (n windows {rd.get('n_windows')}, episodes {rd.get('n_episodes')})"
        rows[s] = (bool(rd["separated"] and rd["delta"] > 0), bool(rb["separated"] and rb["delta"] > 0))
    if len(rows) < 2:
        return "NOT-YET", "the card needs both seeds"
    if not all(r[0] for r in rows.values()):
        return "S-IGNORES", "real - donor is not separated above 0 at >= 1 seed"
    if all(r[1] for r in rows.values()):
        return "S-FOLLOWS", "real - donor and real - best constant are separated above 0 at both seeds"
    return "S-PARTIAL", "real - donor separated at both seeds, not above the best constant at >= 1 seed"


NEXT = {"S-FOLLOWS": "the 120 m checkpoint card (6-30 s band; PI go)",
        "S-PARTIAL": "the 120 m checkpoint card; inside 80 m the scene already shows the turn, so the nav's value lies beyond the vision horizon",
        "S-IGNORES": "zero-training check first: S1 with nav zeroed vs donor on RC-A error (does S read nav at all?); then the 120 m card",
        "NOT-YET": "read both seeds", "NOT-READ": "name the n that blocks the 80 m row; S-IGNORES then reports the window count"}


# ------------------------------------------------------------------------------------------------------------------------------ controls (also run by the tests)
def k7(G):
    """HOLD -> 0.0; the announced side x 40 deg -> 1.0; the mirrored side -> 0.0.  Returns {name: mean S_CKPT at 80 m} over scorable windows."""
    n = len(G["side"])
    def mk(psi):
        ck = np.zeros((n, 3, 3))
        ck[..., 2] = np.asarray(psi, np.float64)[:, None]
        return ck
    out = {}
    for name, psi in (("hold", np.zeros(n)), ("announced_x40deg", np.radians(40.0) * G["side"]), ("mirrored", -np.radians(40.0) * G["side"])):
        c = route_cells(mk(psi), G)[:, DECIDE]
        out[name] = float(np.nanmean(c)) if np.isfinite(c).any() else None
    return out


def k8(G, donor_idx):
    """a synthetic arm that COPIES its nav input (heading = the side of the nav it is fed x 40 deg): real reads 1.0, donor reads the donor's agreement with the window's own side"""
    n = len(G["side"])
    def ck(side):
        c = np.zeros((n, 3, 3))
        c[..., 2] = (np.radians(40.0) * side.astype(float))[:, None]
        return c
    real, donor = route_cells(ck(G["side"]), G)[:, DECIDE], route_cells(ck(G["side"][donor_idx]), G)[:, DECIDE]
    return {"real": float(np.nanmean(real)), "donor": float(np.nanmean(donor)), "expected_donor": float(np.mean(G["side"][donor_idx][np.isfinite(real)] == G["side"][np.isfinite(real)]))}


def k9(ticks, t0_ckpt, ci_t, atol=1e-6):
    """T1 tick-0 `s_ckpt` against the T0 real-nav `ckpt` of the same window.  ticks: npz dict; t0_ckpt [N,3,3]; ci_t: {(ci, t): index into the T0 arrays}.  -> (max abs diff, n compared, ok)"""
    m = np.asarray(ticks["tick"]) == 0
    idx = [ci_t[(int(c), int(t))] for c, t in zip(np.asarray(ticks["ci"])[m], np.asarray(ticks["t0"])[m])]
    d = np.abs(np.asarray(ticks["s_ckpt"], np.float64)[m] - np.asarray(t0_ckpt, np.float64)[idx])
    mx = float(d.max()) if d.size else float("nan")
    return mx, int(m.sum()), bool(np.isfinite(mx) and mx <= atol)


def t1_row(ticks, G, ci_t, complete_m=20.0):
    """S_CKPT per tick from the T1 dump (real nav, own actions): side = the text class at the window's t0 (held), d = nav_in[3] x 100 where nav_in[4] = 1, headings s_ckpt_now.  -> cell [n_tick_rows, 3] and the rows' window ids"""
    w = np.asarray([ci_t[(int(c), int(t))] for c, t in zip(ticks["ci"], ticks["t0"])])
    nav = np.asarray(ticks["nav_in"], np.float64)
    d = np.where(nav[:, 4] == 1, nav[:, 3] * 100.0, np.nan)
    Gt = {k: np.asarray(G[k])[w] for k in ("side", "theta", "text_ok", "rc_ok")}
    d = np.where(Gt["side"] != 0, d, np.nan)
    old = SC.COMPLETE_M
    SC.COMPLETE_M = float(complete_m)
    try:
        cell = SC.ckpt_cell(np.asarray(ticks["s_ckpt_now"], np.float64)[..., 2], ARCS, Gt["side"], Gt["theta"], d, Gt["text_ok"], Gt["rc_ok"])
    finally:
        SC.COMPLETE_M = old
    return cell, w, np.asarray(ticks["tick"])


# ------------------------------------------------------------------------------------------------------------------------------ driver
def load_inputs(smoke_grid=None):
    join = dict(np.load(L.pinned("join_eval141.npz"), allow_pickle=False))
    navtok = dict(np.load(L.pinned("navtok_table.npz"), allow_pickle=False))
    navtext = dict(np.load(L.pinned("navtext_nav.npz"), allow_pickle=False))
    ci, t = L.grid_windows(smoke_grid or L.pinned("eval_grid_windows.json"), join["clip"])
    return join, navtok, navtext, ci, t, ci.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", required=True)
    ap.add_argument("--t1", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", default="", help="smoke only: a reduced grid json (the T0 evals were run on it)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--n-boot", type=int, default=L.NB)
    a = ap.parse_args()
    CI = L.load_ci()
    join, navtok, navtext, ci, t, eid = load_inputs(a.grid or None)
    G = ground(navtok, navtext, join, ci, t)
    rep = {"tool": "strat1_strategic.py", "tier": "T0 at the window's first tick (fresh ages; real / donor / zero nav); T1 per-tick row beside", "evidence_class": "MEASURED (ours; T0 evals, zero training)",
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap, B={a.n_boot}, seed 0 ({CI.SOURCE}); 'another draw of EPISODES' only",
           "constants": {"COMPLETE_M": SC.COMPLETE_M, "theta_deg": SC.THETA_DEG, "arcs_m": list(ARCS), "deciding_arc_m": ARCS[DECIDE]},
           "inputs_sha256": {k: L.PINS[k] for k in ("navtok_table.npz", "navtext_nav.npz", "join_eval141.npz")}, "arms": {}, "controls": {}}
    ci_t = {(int(c), int(tt)): i for i, (c, tt) in enumerate(zip(ci, t))}
    rc_arrays = {}
    def load(tag, var):  # noqa: E306
        p = f"{a.t0}/eval_STRAT1-{tag}_{var}.npz"
        z = dict(np.load(p, allow_pickle=True))
        if not (np.array_equal(z["win_ci"], ci) and np.array_equal(z["win_t"], t)):
            raise RuntimeError(f"{p}: the evaluator's windows differ from the grid")
        return z
    for tag in L.ARMS:
        if not all(os.path.exists(f"{a.t0}/eval_STRAT1-{tag}_{v}.npz") for v in VARIANTS):
            rep["arms"][tag] = {"status": "NOT-YET", "reason": "a T0 pass is missing"}
            continue
        zs = {v: load(tag, v) for v in VARIANTS}
        cells = {v: route_cells(zs[v]["ckpt"], G) for v in VARIANTS}
        e = rcA_errors(zs["real"]["ckpt"], G)
        rc_arrays[tag] = rc_cells(e)
        rep["arms"][tag] = {"cells": cells, "sens": {
            f"complete_m_{cm}": level(CI, route_cells(zs["real"]["ckpt"], G, complete_m=cm), eid) for cm in (10, 30)}}
        raw_d = route_cells(zs["real"]["ckpt"], G, d=G["d_raw"])
        rep["arms"][tag]["sens"]["raw_text_distance"] = level(CI, raw_d, eid)
    done = {k: v for k, v in rep["arms"].items() if "cells" in v}
    for seed in a.seeds:
        c0 = done.get(f"C0_s{seed}")
        for fam in ("S1", "N1", "C0"):
            tag = f"{fam}_s{seed}"
            if tag in done:
                done[tag]["block"] = arm_block(CI, done[tag]["cells"], G, eid, None if (fam == "C0" or c0 is None) else c0["cells"]["real"])
    fl = None
    if "C0_rep" in rc_arrays and "C0_s0" in rc_arrays:
        sys.path.insert(0, HERE + "/theory/obd/read")
        import obd_read as OR
        fl = {c: OR.floor_of(rc_arrays["C0_rep"][c], rc_arrays["C0_s0"][c]) for c in RC_CELLS}
    rep["route_setting_RC-A"] = {"floor_F": fl, "tier": "T0 real nav, fresh ages", "rows": {}}
    for seed in a.seeds:
        if f"S1_s{seed}" in rc_arrays and f"C0_s{seed}" in rc_arrays:
            rows = route_setting(CI, rc_arrays[f"S1_s{seed}"], rc_arrays[f"C0_s{seed}"], fl, eid, a.n_boot)
            rep["route_setting_RC-A"]["rows"][f"S1_minus_C0_s{seed}"] = rows
    rs = rep["route_setting_RC-A"]["rows"]
    rep["route_setting_RC-A"]["cost_at_both_seeds"] = [c for c in RC_CELLS if len(rs) == 2 and all(r[c].get("cost") for r in rs.values())]
    lab = {}
    per = {s: done[f"S1_s{s}"]["block"] for s in a.seeds if f"S1_s{s}" in done}
    lab["label"], lab["reason"] = outcome(per)
    lab["next_step"] = NEXT.get(lab["label"])
    rep["outcome_route_following"] = lab
    rep["controls"]["K7"] = k7(G)
    rep["controls"]["K8"] = k8(G, np.roll(np.arange(len(eid)), 37))
    for tag in done:
        done[tag]["cells"] = {v: {"n_scorable": [int(np.isfinite(c[:, m]).sum()) for m in range(3)]} for v, c in done[tag]["cells"].items()}
    if a.t1:
        for tag in done:
            p = f"{a.t1}/ticks/ticks_STRAT1-{tag}_m10_causal_heads.npz"
            if os.path.exists(p):
                tk = dict(np.load(p, allow_pickle=True))
                zr = load(tag, "real")
                mx, n, ok = k9(tk, zr["ckpt"], ci_t)
                row = {"K9": {"max_abs_diff": mx, "n_windows": n, "ok": ok}}
                if ok and tag.startswith("S1"):
                    cell, w, tick = t1_row(tk, G, ci_t)
                    row["S_CKPT_per_tick"] = {str(k): level(CI, cell[tick == k], eid[w][tick == k]) for k in np.unique(tick)}
                else:
                    row["T1_row"] = "not quoted" if not ok else "text arm only"
                done[tag]["T1_beside"] = row
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    print("WROTE", a.out, "|", lab["label"], "|", lab["reason"])


if __name__ == "__main__":
    main()
