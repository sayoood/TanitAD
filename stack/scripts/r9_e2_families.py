#!/usr/bin/env python3
"""r9_e2_families.py -- the PAIRED four-family readers for the refcv9 E2 judge (``r9_e2_verdict.py`` clauses (d) and (e)).  CPU only, no GPU, no new estimator.

    python r9_e2_families.py --mode families --a-dump <K2 seed-S dump> --b-dump <K0 seed-S dump> --dev-split DEV_HOLD_SPLIT.json --out families_s0.json
    python r9_e2_families.py --mode camblind --a-dump <K2 seed-S dump> --b-dump <K2-camera-blind seed-S dump> --dev-split DEV_HOLD_SPLIT.json --out camblind_s0.json

A "dump" is the directory ``refcv3_arm.run_dump`` writes under the refcv8 battery roll (``taniteval/tools/refcv3_arm.py:120-125``): ``manifest.json``
(``episodes[{file_index, clip_id, ...}]``) + ``ep<fi:03d>.npz`` {``g`` GT [N,K,2], ``os`` (the planner arm, T1 self-action open loop), ``ws`` [N], ...}.  ONE call
reads ONE inference seed; the judge wants seeds 0 AND 1 as two separate inputs, so run it once per seed (the planner samples: H-ESTIM-SEED-1 third variance).

WHAT IT DOES (and refuses):
  * windows are keyed ``(clip_id, ws)`` and paired by ``taniteval.ci.align_windows`` -- NEVER by ``ep###`` (a file index is arm-local).  A window present in one arm
    and not the other REFUSES (no silent intersection); a duplicate key REFUSES (``align_windows``); a GT (`g`) disagreement between the arms REFUSES (two different
    label clocks would pair different futures).
  * DEV only (``DEV_HOLD_SPLIT.json``, ``r9_e1_eval.load_window_list(path, 'dev')``; a CLIP split, key ``sha256(clip_id)[:12]``).  Windows of non-DEV clips are DROPPED and
    counted (``--non-dev refuse`` makes any of them a refusal).  HOLD is never scored.  No DEV window left REFUSES.
  * rows = the refcv8 REGISTERED metric table (``pair_read_r8.METRICS``: names and directions), computed per window with ``taniteval.four_families._seq_geometry`` and the
    programme's own tactical labeller (``tanitad.refs.refc_tactical.factor_from_kinematics`` via ``four_families.maneuver_kinematics``) -- the SAME geometry ``t1_eval.analyze``
    uses (``t1_eval.py:540-560``), not a re-derivation.  Each row = ``ci.paired_episode_cluster_bootstrap(a, b, eid, B 2000, seed 0)``: delta = A - B, A = the first arm
    (K2).  A window whose value is non-finite in either arm (e.g. a stopped window has no heading) is dropped from THAT row for both arms and counted
    (``n_dropped_nonfinite``).  A row with no finite window is OMITTED and listed in ``omitted`` (never faked); a family left with no rows reads NOT-COMPUTABLE in the judge.
  * STRATEGIC: ``{"status": "PAUSED", ...}`` (PI 2026-09-19, PI_DECISION_QUEUE ITEM 25) -- never a pass, never omitted.

THE WAYPOINT CLOCK (fixed 2026-11-12, MM 4a3d05 -- the v1 reader got it wrong).  ``_seq_geometry(wp, dt)`` needs the spacing BETWEEN the supplied waypoints
(``four_families._DT_CONTRACT``).  The battery dump publishes it: ``manifest.json`` ``grid.dt_s`` (the E2 dumps: ``grid`` 2s, k 4, ``dt_s`` 0.5, instants
0.5-2.0 s), and the registered refcv8 reader takes it from there (``pair_read_r8.py``: ``dt = float(man["grid"]["dt_s"])``).  v1 hard-coded ``DT = 0.1`` and
read the 0.5 s dumps at 0.1 s: speed and yaw rate x5, accel x25, the tangent-validity gate (``MIN_DS_MPS * dt``) 5x too low, ``LON_speed_within_0p5``
an effective 0.1 m/s band, and the tactical labeller fed 5x speeds.  Its synthetic test fixture was a dense 0.1 s grid with no ``dt_s``, so it matched the
constant and never the real dumps.  NOW: ``dt`` = the dump's ``grid.dt_s`` (no dump without it is read; both arms must agree), and a PLAUSIBILITY GUARD
measures the GT's own spacing -- median ``|g[:, 0]| / v0`` over windows with ``v0 > 2 m/s`` must be within 15 % of ``dt`` (MEASURED on the E2 K2 seed-0
dump: 0.5034 s over 2,184 moving windows for ``dt_s`` 0.5; the v1 reading would be off by a factor 5).  Fewer than 20 moving windows, or no ``v0`` in the
dump: the guard reads NOT-CHECKED and says so in the output (never a silent pass).

PRIMARY rows for the camera-blind (W4) read -- ONE per family, ``PRIMARY`` below.  They are the FIRST NR-headline row of each family in the refcv8 registered table
(``pair_read_r8.METRICS``, the rows SHIP-R8 reads under AMENDMENT 3, ``_ship_r8_verdict_base.py:clause4/clause5``; REFCV9_DESIGN 5.2 W4 says "``ade_m`` and each family's
primary row" without naming them).  The MM may re-name them at sealing: ``--primary FAMILY=row_id`` overrides, and the full row table is always written beside.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

SCHEMA = "r9e2-families-v2"           # v2: dt from the dump's grid.dt_s (v1 read 0.5 s dumps at a hard-coded 0.1 s)
DT_CHECK_MIN_V0_MPS = 2.0              # windows slower than this carry no usable |g0| / v0 ratio
DT_CHECK_REL_TOL = 0.15                # MEASURED real-data offset 0.7 % (0.5034 s vs 0.5); the v1 defect is a factor 5
DT_CHECK_MIN_N = 20
#: (family, row_id, lower_is_better) -- names and directions copied from the refcv8 registered table (pair_read_r8.METRICS) + the TACTICAL third row t1_eval's four_families carries.
ROWS = (("ADE", "ade_m", True), ("ADE", "fde_m", True),
        ("LONGITUDINAL", "LON_speed_mae_mps", True), ("LONGITUDINAL", "LON_along_mae_m", True), ("LONGITUDINAL", "LON_along_abs_final_m", True),
        ("LONGITUDINAL", "LON_speed_within_0p5", False), ("LONGITUDINAL", "LON_accel_mae_mps2", True),
        ("LATERAL", "LAT_heading_mae_deg", True), ("LATERAL", "LAT_curvature_mae_1pm_valid", True), ("LATERAL", "LAT_yaw_rate_mae_radps_valid", True),
        ("LATERAL", "LAT_cross_mae_m", True), ("LATERAL", "LAT_cross_abs_final_m", True),
        ("TACTICAL", "TAC_traj_lat_correct", False), ("TACTICAL", "TAC_traj_lon_correct", False), ("TACTICAL", "TAC_traj_man5_correct", False))
PRIMARY = {"ade_m": "ade_m", "LONGITUDINAL": "LON_speed_mae_mps", "LATERAL": "LAT_heading_mae_deg", "TACTICAL": "TAC_traj_lat_correct"}
FAMILIES = ("LONGITUDINAL", "LATERAL", "TACTICAL")
STRATEGIC = {"status": "PAUSED", "reason": "PAUSED (PI 2026-09-19, PI_DECISION_QUEUE ITEM 25): the strategic layer is switched off until tactical and operative drive with "
                                           "high quality; never a pass, never omitted, never extended to the other families."}
TIER = "T1 (self-action open loop on logged eval139 frames; the `os` arm of the battery dump)"
ESTIMATOR = "taniteval.ci.paired_episode_cluster_bootstrap, B 2000, seed 0, over EPISODES (clips); delta = A - B"


def _fail(msg: str):
    raise SystemExit(f"[r9-e2-families] REFUSED: {msg}")


def sha12(clip_id) -> str:
    return hashlib.sha256(str(clip_id).encode()).hexdigest()[:12]


def window_set_sha256(keys) -> str:
    """sha256 over the sorted ``(clip_id, ws)`` keys of the PAIRED window set -- byte-identical to ``r9_e2_distkeep.window_set_sha256``, so the distance-keeping merge
    (``r9_e2_distkeep.merge_into``) checks that both readers scored the SAME windows (a families.json without it made that check vacuous)."""
    h = hashlib.sha256()
    for c, w in sorted((str(c), int(w)) for c, w in keys):
        h.update(f"{c}\t{w}\n".encode())
    return h.hexdigest()


def _ci():
    from taniteval import ci
    return ci


def load_dev(path) -> set:
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import r9_e1_eval
    dev = set(r9_e1_eval.load_window_list(path, "dev"))
    if not dev:
        _fail(f"{path}: the DEV list is empty")
    return dev


def grid_dt(man: dict, where) -> float:
    """The waypoint spacing the dump publishes (``manifest.json`` ``grid.dt_s``).  REFUSES when absent: the spacing is not assumed (THE WAYPOINT CLOCK)."""
    try:
        dt = float((man.get("grid") or {})["dt_s"])
    except (KeyError, TypeError, ValueError):
        _fail(f"{where}: manifest.json publishes no grid.dt_s -- the spacing between the waypoints is unknown, and assuming one rescales every rate row "
              f"(four_families._DT_CONTRACT); refusing")
    if not (np.isfinite(dt) and dt > 0):
        _fail(f"{where}: manifest.json grid.dt_s = {dt!r} is not a positive spacing")
    return dt


def dt_check(G: np.ndarray, v0, dt: float, where) -> dict:
    """PLAUSIBILITY GUARD: the GT's own spacing (median ``|g[:, 0]| / v0`` over moving windows) must agree with the published ``dt`` within
    ``DT_CHECK_REL_TOL``; REFUSES a contradiction, reads NOT-CHECKED (stated) when the dump carries no ``v0`` or too few moving windows."""
    if v0 is None:
        return {"status": "NOT-CHECKED", "reason": "the dump carries no v0", "n": 0}
    v0 = np.asarray(v0, dtype=np.float64).reshape(-1)
    m = np.isfinite(v0) & (v0 > DT_CHECK_MIN_V0_MPS)
    n = int(m.sum())
    if n < DT_CHECK_MIN_N:
        return {"status": "NOT-CHECKED", "reason": f"{n} window(s) with v0 > {DT_CHECK_MIN_V0_MPS} m/s (< {DT_CHECK_MIN_N})", "n": n}
    med = float(np.median(np.linalg.norm(G[m, 0], axis=-1) / v0[m]))
    rel = abs(med / dt - 1.0)
    if rel > DT_CHECK_REL_TOL:
        _fail(f"{where}: grid.dt_s {dt} contradicts the GT's own spacing -- median |g[:, 0]| / v0 = {med:.4f} s over {n} moving windows "
              f"({rel:.1%} off, tolerance {DT_CHECK_REL_TOL:.0%}); reading it would rescale every rate row")
    return {"status": "PASS", "median_g0_over_v0_s": round(med, 6), "rel_offset": round(rel, 6), "n": n, "tol": DT_CHECK_REL_TOL}


def load_dump(dump_dir, arm: str = "os") -> dict:
    """{keys: [(clip_id, ws)], P [N,K,2], G [N,K,2], dt_s, dt_check} from a refcv3_arm dump directory."""
    d = Path(dump_dir)
    try:
        man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        _fail(f"{d}: no readable manifest.json ({e})")
    dt = grid_dt(man, d)
    keys, P, G, V0 = [], [], [], []
    for e in man.get("episodes") or []:
        z = np.load(d / f"ep{int(e['file_index']):03d}.npz")
        for k in ("g", arm, "ws"):
            if k not in z.files:
                _fail(f"{d}/ep{int(e['file_index']):03d}.npz has no key {k!r} (keys {z.files})")
        ws = np.asarray(z["ws"]).reshape(-1)
        if z["g"].shape[0] != ws.size or z[arm].shape != z["g"].shape:
            _fail(f"{d}: episode {e.get('clip_id')} arm/GT/ws shapes disagree")
        keys += [(str(e["clip_id"]), int(w)) for w in ws]
        P.append(np.asarray(z[arm], dtype=np.float64)[..., :2])
        G.append(np.asarray(z["g"], dtype=np.float64)[..., :2])
        V0.append(np.asarray(z["v0"], dtype=np.float64).reshape(-1) if "v0" in z.files and np.asarray(z["v0"]).size == ws.size else None)
    if not keys:
        _fail(f"{d}: the dump holds no windows")
    Gc = np.concatenate(G)
    v0 = None if any(v is None for v in V0) else np.concatenate(V0)
    return {"keys": keys, "P": np.concatenate(P), "G": Gc, "dir": str(d), "arm": arm, "dt_s": dt, "dt_check": dt_check(Gc, v0, dt, d)}


def restrict_dev(dump: dict, dev: set, non_dev: str = "drop") -> tuple[dict, dict]:
    keep = [i for i, (c, _) in enumerate(dump["keys"]) if sha12(c) in dev]
    n_non = len(dump["keys"]) - len(keep)
    if n_non and non_dev == "refuse":
        _fail(f"{dump['dir']}: {n_non} window(s) are not DEV windows (--non-dev refuse); HOLD must not be rolled into a DEV read")
    if not keep:
        _fail(f"{dump['dir']}: no DEV window in the dump")
    ix = np.asarray(keep)
    out = {"keys": [dump["keys"][i] for i in keep], "P": dump["P"][ix], "G": dump["G"][ix], "dir": dump["dir"], "arm": dump["arm"], "dt_s": dump["dt_s"],
           "dt_check": dump["dt_check"]}
    return out, {"n_total": len(dump["keys"]), "n_dev": len(keep), "n_non_dev_dropped": n_non}


def window_rows(P: np.ndarray, G: np.ndarray, dt: float) -> dict:
    """{row_id: float64 [N]} per-window values (NaN = no value on that window).  Geometry = ``four_families._seq_geometry`` (not re-derived).
    ``dt`` = the spacing between the waypoints (the dump's ``grid.dt_s``); deliberately NO default (THE WAYPOINT CLOCK)."""
    if not (dt and np.isfinite(dt) and dt > 0):
        _fail(f"window_rows: dt {dt!r} is not a positive waypoint spacing")
    import torch
    from taniteval import four_families as ff
    from tanitad.refs.refc_tactical import COLLAPSE_TABLE, factor_from_kinematics
    p, g = torch.as_tensor(P).float(), torch.as_tensor(G).float()
    Pg, Gg = ff._seq_geometry(p, dt), ff._seq_geometry(g, dt)
    nan = float("nan")
    out = {"ade_m": torch.linalg.norm(p - g, dim=-1).mean(1), "fde_m": torch.linalg.norm(p[:, -1] - g[:, -1], dim=-1)}
    sp, al, cr = Pg["speed"] - Gg["speed"], Pg["along"] - Gg["along"], Pg["cross"] - Gg["cross"]
    out["LON_speed_mae_mps"], out["LON_along_mae_m"] = sp.abs().mean(1), al.abs().mean(1)
    out["LON_along_abs_final_m"], out["LON_speed_within_0p5"] = al[:, -1].abs(), (sp.abs() <= 0.5).float().mean(1)
    out["LON_accel_mae_mps2"] = (Pg["accel"] - Gg["accel"]).abs().mean(1) if Pg["accel"].shape[1] else torch.full((p.shape[0],), nan)
    out["LAT_cross_mae_m"], out["LAT_cross_abs_final_m"] = cr.abs().mean(1), cr[:, -1].abs()

    def masked_mean(x, m):
        n = m.sum(1)
        return torch.where(n > 0, (x * m).sum(1) / n.clamp_min(1), torch.full_like(x[:, 0], nan))
    both = Pg["valid"] & Gg["valid"]
    dh = (Pg["heading"] - Gg["heading"] + math.pi) % (2 * math.pi) - math.pi
    out["LAT_heading_mae_deg"] = masked_mean(dh.abs() * 180.0 / math.pi, both)
    pv = Pg["pair_valid"] & Gg["pair_valid"]          # tangent-less steps carry no curvature / yaw rate (RETR-2026-09-26-YAWMASK)
    out["LAT_curvature_mae_1pm_valid"] = masked_mean((Pg["curvature"] - Gg["curvature"]).abs(), pv)
    out["LAT_yaw_rate_mae_radps_valid"] = masked_mean((Pg["yaw_rate"] - Gg["yaw_rate"]).abs(), pv)
    dyp, dvp, v0p, v1p, _ = ff.maneuver_kinematics(p, dt)
    dyg, dvg, v0g, v1g, _ = ff.maneuver_kinematics(g, dt)
    latp, lonp = factor_from_kinematics(dyp, dvp, v0p, v1p)
    latg, long_ = factor_from_kinematics(dyg, dvg, v0g, v1g)
    tbl = torch.as_tensor(COLLAPSE_TABLE, dtype=torch.long)
    out["TAC_traj_lat_correct"], out["TAC_traj_lon_correct"] = (latp == latg).float(), (lonp == long_).float()
    out["TAC_traj_man5_correct"] = (tbl[latp, lonp] == tbl[latg, long_]).float()
    return {k: v.detach().cpu().numpy().astype(np.float64) for k, v in out.items()}


def align(a: dict, b: dict, gt_tol: float = 1e-4) -> tuple[dict, dict, list, dict]:
    """Pair two DEV-restricted dumps on (clip_id, ws).  A window-set mismatch or a GT disagreement REFUSES."""
    ci = _ci()
    ia, ib = list(range(len(a["keys"]))), list(range(len(b["keys"])))
    try:
        xa, xb, eid, rep = ci.align_windows(a["keys"], ia, b["keys"], ib)
    except ValueError as e:
        _fail(f"window pairing: {e}")
    if rep["n_only_a"] or rep["n_only_b"]:
        _fail(f"window-set mismatch after DEV restriction: {rep['n_only_a']} window(s) only in A ({a['dir']}), {rep['n_only_b']} only in B ({b['dir']}); "
              f"the arms must be rolled on the SAME windows (never silently intersected)")
    xa, xb = np.asarray(xa), np.asarray(xb)
    gd = float(np.abs(a["G"][xa] - b["G"][xb]).max())
    if not gd <= gt_tol:
        _fail(f"the two arms' GT (`g`) differ by up to {gd:.6g} m on the paired windows (tol {gt_tol}): different label clocks / corpora; pairing them would compare different futures")
    return ({"P": a["P"][xa], "G": a["G"][xa]}, {"P": b["P"][xb], "G": b["G"][xb]}, eid,
            {**rep, "gt_max_abs_diff_m": gd, "window_set_sha256": window_set_sha256([a["keys"][i] for i in xa])})


def paired_rows(ra: dict, rb: dict, eid: list, n_boot: int = 2000, seed: int = 0) -> tuple[dict, dict]:
    ci = _ci()
    rows, omitted = {}, {}
    for fam, rid, lower in ROWS:
        a, b = ra[rid], rb[rid]
        ok = np.isfinite(a) & np.isfinite(b)
        if not ok.any():
            omitted[rid] = "no window with a finite value in both arms"
            continue
        e_ok = [e for e, k in zip(eid, ok) if k]
        r = ci.paired_episode_cluster_bootstrap(a[ok], b[ok], e_ok, n_boot=n_boot, seed=seed)
        row = {"delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "separated": bool(r["separated"]), "better": "lower" if lower else "higher",
               "n": int(r["n_windows"]), "n_episodes": int(r["n_episodes"]), "a_mean": float(a[ok].mean()), "b_mean": float(b[ok].mean()),
               "n_dropped_nonfinite": int((~ok).sum()), "family": fam, "estimator": r["estimator"], "tier": TIER}
        if r.get("degenerate"):
            row["degenerate"] = True
        rows[rid] = row
    return rows, omitted


def build(a_dir, b_dir, *, mode: str = "families", dev_split=None, dev=None, arm_a: str = "os", arm_b: str = "os", non_dev: str = "drop",
          n_boot: int = 2000, seed: int = 0, primary: dict | None = None, labels=("K2", "K0")) -> dict:
    if mode not in ("families", "camblind"):
        _fail(f"unknown mode {mode!r}")
    dev = set(dev) if dev is not None else (load_dev(dev_split) if dev_split else None)
    if not dev:
        _fail("no DEV list: pass --dev-split (the E1 read's pin) -- an un-restricted read could score HOLD")
    A, ca = restrict_dev(load_dump(a_dir, arm_a), dev, non_dev)
    B, cb = restrict_dev(load_dump(b_dir, arm_b), dev, non_dev)
    if A["dt_s"] != B["dt_s"]:
        _fail(f"the arms publish different waypoint spacings: {A['dir']} grid.dt_s {A['dt_s']} vs {B['dir']} grid.dt_s {B['dt_s']}")
    dt = A["dt_s"]
    pa, pb, eid, rep = align(A, B)
    rows, omitted = paired_rows(window_rows(pa["P"], pa["G"], dt), window_rows(pb["P"], pb["G"], dt), eid, n_boot, seed)
    meta = {"schema": SCHEMA, "mode": mode, "labels": f"{labels[0]} - {labels[1]}", "tier": TIER, "estimator": ESTIMATOR, "delta": "A - B (A = " + labels[0] + ")",
            "windows": "DEV (DEV_HOLD_SPLIT.json); HOLD is never read", "n_windows_paired": rep["n_common"], "window_set_sha256": rep["window_set_sha256"], "pairing": rep, "a": {**ca, "dir": A["dir"]},
            "b": {**cb, "dir": B["dir"]}, "omitted_rows": omitted, "n_boot": n_boot, "boot_seed": seed,
            "dt_s": dt, "dt_source": "manifest.json grid.dt_s (both arms equal)", "dt_check": {"a": A["dt_check"], "b": B["dt_check"]},
            "caveat": "H-ESTIM-SEED-1: the paired interval answers another draw of EPISODES, not another training run and not another INFERENCE draw; one call = one inference seed."}
    if mode == "families":
        by = {f: {rid: r for rid, r in rows.items() if r["family"] == f} for f in FAMILIES}
        return {**meta, "rows": {**by, "STRATEGIC": dict(STRATEGIC)}, "ade_m": rows.get("ade_m"), "beside_ade_fde": rows.get("fde_m"), "STRATEGIC": dict(STRATEGIC)}
    prim = {**PRIMARY, **(primary or {})}
    sel, why = {}, {}
    for key, rid in prim.items():
        if rid in rows:
            sel[key] = rows[rid]
            why[key] = rid
        else:
            why[key] = f"{rid}: NOT COMPUTED ({omitted.get(rid, 'unknown row id')})"
    return {**meta, "rows": sel, "primary_rows": why, "primary_cite": "first NR-headline row of each family in the refcv8 registered table (pair_read_r8.METRICS; SHIP-R8 "
            "clauses 4/5); REFCV9_DESIGN 5.2 W4 names 'ade_m and each family's primary row' -- the MM may re-name at sealing (--primary)", "all_rows": rows,
            "STRATEGIC": dict(STRATEGIC)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=("families", "camblind"), required=True)
    ap.add_argument("--a-dump", required=True, help="K2's dump dir (A in A - B)")
    ap.add_argument("--b-dump", required=True, help="K0's dump dir (families) / K2 camera-blind's dump dir (camblind)")
    ap.add_argument("--dev-split", required=True)
    ap.add_argument("--arm-a", default="os")
    ap.add_argument("--arm-b", default="os")
    ap.add_argument("--non-dev", choices=("drop", "refuse"), default="drop")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--primary", action="append", default=[], metavar="FAMILY=row_id")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    prim = dict(p.split("=", 1) for p in a.primary)
    try:
        doc = build(a.a_dump, a.b_dump, mode=a.mode, dev_split=a.dev_split, arm_a=a.arm_a, arm_b=a.arm_b, non_dev=a.non_dev, n_boot=a.boot, seed=a.seed, primary=prim,
                    labels=("K2", "K0") if a.mode == "families" else ("K2", "K2-camera-blind"))
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 2
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=1, default=float) + "\n", encoding="utf-8")
    print(f"[r9-e2-families] {a.mode}: {len(doc['rows'])} row groups, n_windows_paired={doc['n_windows_paired']} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
