#!/usr/bin/env python3
"""r9_e2_distkeep.py -- the LONGITUDINAL distance-keeping rows (headway / time-gap / TTC to the lead agent) for the refcv9 E2 judge, K2 vs K0.  CPU only, no GPU.

    python r9_e2_distkeep.py --a-dump <K2 seed-S dump> --b-dump <K0 seed-S dump> --dev-split DEV_HOLD_SPLIT.json --lead-block <b1 lead block .npz> \
        --out distkeep_s0.json [--merge-into families_s0.json]

WHY.  CLAUDE.md (binding, "EVERY EVAL REPORTS FOUR METRIC FAMILIES"): LONGITUDINAL = target-speed accuracy AND distance-keeping.  ``r9_e2_families.py`` writes the speed /
along-track rows only; its docstring flags distance-keeping as a work item.  This is that work item, built ONLY from the programme's own instruments (no new estimator):
``taniteval.lead_metrics.distance_keeping`` per window, ``refav1_arm.join_lead_block`` + ``refcv3_arm.lead_block_common_grid`` for the join (the same path the refcv7/refcv8
panels use, ``r7_panel.py:190``), ``taniteval.ci.paired_episode_cluster_bootstrap`` for every interval.

WHERE THE GT LEAD COMES FROM.  The roll's per-window dump (``ep###.npz`` {g, os, ws, v0}) carries NO lead data.  The lead is a post-hoc join of the banked B1 EVAL lead block
(``build_lead_block_b1.py``; obstacle.offline cuboids already in the window-origin ego frame, x fwd / y left, rig origin, ``gap`` = origin -> lead REAR face) keyed on
``(clip_id, RAW frame = ws + provider_to_raw_frame_offset)``, with the label-free speed proof (dump ``v0`` == block ``speeds`` to 1e-3 m/s).  A SPEED_MISMATCH episode REFUSES the run
(a wrong frame map, not an absence).  Only windows in state ``LEAD`` carry a value: NO_LEAD / NOT_STRAIGHT / NO_LABEL (and a clip with no block row) are NOT-COMPUTABLE per window,
COUNTED by state, never 0 and never free flow.

GRID.  The roll scores 0.5/1.0/1.5/2.0 s (``--grid 2s``); the banked blocks sit on a 0.2 s grid, so the SHARED instants are 1.0 s and 2.0 s only (dt 1.0).  The result then carries
``grid_complete: false`` -- a coarse min over 2 instants.  A block built with ``--dt 0.5 --k 4`` makes all four instants scoreable (HANDOFF has the command).

ROWS (family LONGITUDINAL, ids ``LON_dk_*``; delta = A - B, A = K2; ``better`` = the direction in which delta is an improvement), all per LEAD window of the PREDICTED ego path
(arm ``os``) against the logged lead future, via ``lead_metrics.distance_keeping``:
  LON_dk_headway_min_m          higher   min over the horizon of the gap to the lead (rig origin -> rear face); the lead must stay in the PREDICTED path's own corridor
  LON_dk_time_gap_min_s         higher   headway_min / v_ego(t0); undefined (dropped) below 0.5 m/s
  LON_dk_min_ttc_s              higher   min TTC from the arm's own closing rate; CENSORED at 30 s (a mean over censored data)
  LON_dk_headway_abs_err_m      lower    |plan headway_min - human headway_min| (the GT ego path scored against the SAME lead future)
  LON_dk_ttc_viol_rate          lower    share of windows with min TTC < 1.5 s (``rl/rewards.py:563`` ``ttc_min_s`` default)
  LON_dk_lead_viol_at_risk      lower    registered LEAD-VIOL (SPEC_REFCV8 s8.7, PI 2026-10-11 reading): plan time gap < 1.0 s on lead windows where the HUMAN keeps >= 1.0 s
  LON_dk_lead_progress_shortfall lower   the Master-Mind PROGRESS CONTROL on the same at-risk windows: max(0, 1 - plan arc / human arc); 0 at human-level progress or more, so a planner that "keeps distance" by CRAWLING reads worse while over-progress (already caught by the headway / LEAD-VIOL rows) is not penalised twice
A row is computed on windows where BOTH arms are finite (``n_dropped_nonfinite``); a row with no such window, or fewer than ``--min-n`` (default
``lead_metrics.MIN_STRATUM_N`` = 30) windows, is written as an UNDEFINED placeholder {undefined, reason, n} -- the judge's ``row_read`` then reads it NOT-COMPUTABLE
(clause (d) NOT-COMPUTABLE => NO-GO), never a pass and never a 0.

PAIRING.  DEV only (``DEV_HOLD_SPLIT.json``, same clip key as ``r9_e2_families``); windows keyed ``(clip_id, ws)`` via ``ci.align_windows`` -- a window present in one arm only
REFUSES; a GT (`g`) or lead-view disagreement between the arms REFUSES.  ``--merge-into families.json`` adds the rows under ``rows.LONGITUDINAL`` and REFUSES unless that file was
built from the SAME arms, labels and window count (and window-set sha256 when it carries one).
H-ESTIM-SEED-1: the interval answers another draw of EPISODES only; one call = one inference seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

SCHEMA = "r9e2-distkeep-v1"
FAMILY = "LONGITUDINAL"
LEAD_STATE = "LEAD"
TTC_VIOL_S = 1.5
TIER = "T1 (self-action open loop on logged eval139 frames; the `os` arm of the battery dump) scored against the logged lead future"
ESTIMATOR = "taniteval.ci.paired_episode_cluster_bootstrap, B 2000, seed 0, over EPISODES (clips); delta = A - B"
#: (row_id, better).  Directions: headway / time gap / TTC higher = more distance kept; errors and violation rates lower.
ROWS = (("LON_dk_headway_min_m", "higher"), ("LON_dk_time_gap_min_s", "higher"), ("LON_dk_min_ttc_s", "higher"), ("LON_dk_headway_abs_err_m", "lower"),
        ("LON_dk_ttc_viol_rate", "lower"), ("LON_dk_lead_viol_at_risk", "lower"), ("LON_dk_lead_progress_shortfall", "lower"))
CAVEAT = ("H-ESTIM-SEED-1: the paired interval answers another draw of EPISODES, not another training run and not another INFERENCE draw; one call = one inference seed. "
          "A TTC mean is over data censored at 30 s; a plan that leaves the lead's corridor loses the lead (its window is dropped from that row, counted in n_*_lost_lead).")


def _fail(msg: str):
    raise SystemExit(f"[r9-e2-distkeep] REFUSED: {msg}")


def _fam():
    import r9_e2_families as fam           # sibling: load_dump / sha12 / load_dev (the E2 reader's own loaders)
    return fam


def _tools():
    """(refav1_arm, refcv3_arm, lead_metrics, ci): the programme's join + metric modules, imported from the tree's taniteval/tools."""
    import taniteval
    tdir = str(Path(taniteval.__file__).resolve().parents[1] / "tools")
    if tdir not in sys.path:
        sys.path.insert(0, tdir)
    import refav1_arm as ra
    import refcv3_arm as rc
    from taniteval import ci, lead_metrics as lm
    return ra, rc, lm, ci


def window_set_sha256(keys) -> str:
    h = hashlib.sha256()
    for c, w in sorted((str(c), int(w)) for c, w in keys):
        h.update(f"{c}\t{w}\n".encode())
    return h.hexdigest()


def load_arm(dump_dir, block_path, arm: str = "os") -> dict:
    """Dump + lead join for ONE arm: keys, P, G, and the lead view (leads [N,K',2], lens, speeds, state) on the instants the dump and the block share."""
    ra, rc, lm, _ = _tools()
    d = _fam().load_dump(dump_dir, arm)
    man = json.loads((Path(dump_dir) / "manifest.json").read_text(encoding="utf-8"))
    grid = man.get("grid") or {}
    if "dt_s" not in grid or "k" not in grid:
        _fail(f"{dump_dir}: manifest.json carries no grid.dt_s / grid.k")
    dt, k = float(grid["dt_s"]), int(grid["k"])
    if d["P"].shape[1] != k:
        _fail(f"{dump_dir}: manifest grid.k={k} but the dump path has {d['P'].shape[1]} steps")
    raw_off = int(((man.get("corpus") or {}).get("frames") or {}).get("provider_to_raw_frame_offset", 0))
    view, idx, meta, info = rc.lead_block_common_grid(str(block_path), dt, k)
    if view is None:
        _fail(f"lead block {block_path}: {info.get('reason') or info}")
    eps = man.get("episodes") or []
    with tempfile.TemporaryDirectory(prefix="r9e2_dk_join_") as tmp:
        files, sman = [], {"episodes": []}
        for i, e in enumerate(eps):
            z = np.load(Path(dump_dir) / f"ep{int(e['file_index']):03d}.npz")
            if "v0" not in z.files:
                _fail(f"{dump_dir}/ep{int(e['file_index']):03d}.npz has no `v0`: the label-free speed proof of the lead join cannot run")
            f = os.path.join(tmp, f"ep{i:03d}.npz")
            np.savez(f, ws=np.asarray(z["ws"]).astype(np.int64).reshape(-1) + raw_off, v0=np.asarray(z["v0"], dtype=np.float64).reshape(-1))
            files.append(f)
            sman["episodes"].append({"file_index": i, "clip_id": e["clip_id"]})
        lead = ra.join_lead_block(files, sman, view, idx, k=info["k"], dt=info["dt_s"], frame_of_t=lambda t: t)
    cov = lead.pop("coverage")
    bad = {n: c for n, c in cov["episodes"].items() if c.get("status") == ra.LEAD_EP_SPEED}
    if bad:
        _fail(f"{dump_dir}: lead join speed proof FAILED on {len(bad)} episode(s) (first: {next(iter(bad.values())).get('reason')}) -- a wrong frame map, not an absence")
    if len(lead["state"]) != len(d["keys"]):
        _fail(f"{dump_dir}: lead join has {len(lead['state'])} windows, the dump {len(d['keys'])}")
    return {**d, "leads": np.asarray(lead["leads"], dtype=np.float64), "lens": np.asarray(lead["lead_lens"], dtype=np.float64),
            "speeds": np.asarray(lead["speeds"], dtype=np.float64), "state": np.asarray(lead["state"]).astype(str), "cols": list(info["dump_cols"]),
            "dt": float(info["dt_s"]), "instants_s": list(info["instants_s"]), "grid_k": k, "coverage": cov, "block": str(block_path), "block_meta": meta}


def restrict_dev(a: dict, dev: set, non_dev: str = "drop") -> tuple[dict, dict]:
    fam = _fam()
    keep = [i for i, (c, _) in enumerate(a["keys"]) if fam.sha12(c) in dev]
    n_non = len(a["keys"]) - len(keep)
    if n_non and non_dev == "refuse":
        _fail(f"{a['dir']}: {n_non} window(s) are not DEV windows (--non-dev refuse)")
    if not keep:
        _fail(f"{a['dir']}: no DEV window in the dump")
    ix = np.asarray(keep)
    out = {k_: (v[ix] if isinstance(v, np.ndarray) and v.shape[:1] == (len(a["keys"]),) else v) for k_, v in a.items()}
    out["keys"] = [a["keys"][i] for i in keep]
    return out, {"n_total": len(a["keys"]), "n_dev": len(keep), "n_non_dev_dropped": n_non}


def align_pair(a: dict, b: dict, gt_tol: float = 1e-4) -> tuple[dict, dict, list, dict]:
    """Pair two DEV-restricted arms on (clip_id, ws).  A window-set mismatch, a GT disagreement or a lead-view disagreement REFUSES."""
    _, _, _, ci = _tools()
    try:
        xa, xb, eid, rep = ci.align_windows(a["keys"], list(range(len(a["keys"]))), b["keys"], list(range(len(b["keys"]))))
    except ValueError as e:
        _fail(f"window pairing: {e}")
    if rep["n_only_a"] or rep["n_only_b"]:
        _fail(f"window-set mismatch after DEV restriction: {rep['n_only_a']} window(s) only in A ({a['dir']}), {rep['n_only_b']} only in B ({b['dir']}); "
              f"the arms must be rolled on the SAME windows (never silently intersected)")
    xa, xb = np.asarray(xa), np.asarray(xb)
    gd = float(np.abs(a["G"][xa] - b["G"][xb]).max())
    if not gd <= gt_tol:
        _fail(f"the two arms' GT (`g`) differ by up to {gd:.6g} m on the paired windows (tol {gt_tol}): different label clocks / corpora")
    if a["state"][xa].tolist() != b["state"][xb].tolist() or not np.allclose(a["speeds"][xa], b["speeds"][xb], atol=1e-3, equal_nan=True) \
            or not np.allclose(a["leads"][xa], b["leads"][xb], atol=1e-6, equal_nan=True) or a["cols"] != b["cols"]:
        _fail("the two arms' lead views (state / speeds / lead tracks / shared instants) differ on the paired windows: the same GT lead is the premise of the pairing")
    pick = lambda s, ix: {"P": s["P"][ix], "G": s["G"][ix], "leads": s["leads"][ix], "lens": s["lens"][ix], "speeds": s["speeds"][ix], "state": s["state"][ix],  # noqa: E731
                          "cols": s["cols"], "dt": s["dt"]}
    return pick(a, xa), pick(b, xb), eid, {**rep, "gt_max_abs_diff_m": gd, "window_set_sha256": window_set_sha256([a["keys"][i] for i in xa])}


def window_values(w: dict) -> tuple[dict, dict]:
    """({row_id: float64 [N] (NaN = no value on that window)}, extras) for ONE arm.  Geometry and every threshold come from ``taniteval.lead_metrics``."""
    _, _, lm, _ = _tools()
    cols, dt = w["cols"], w["dt"]
    Pc, Gc = w["P"][:, cols], w["G"][:, cols]
    is_lead = w["state"] == LEAD_STATE
    nan = np.full(len(is_lead), np.nan)
    dkp = lm.distance_keeping(Pc, w["leads"], w["lens"], w["speeds"], dt)
    dkg = lm.distance_keeping(Gc, w["leads"], w["lens"], w["speeds"], dt)
    pick = lambda dk, key: np.where(is_lead, np.asarray(dk[key], dtype=np.float64), nan)  # noqa: E731
    hp, tgp, ttp = pick(dkp, "headway_min_m"), pick(dkp, "time_gap_min_s"), pick(dkp, "min_ttc_s")
    hg, tgg = pick(dkg, "headway_min_m"), pick(dkg, "time_gap_min_s")
    at_risk = lm._at_risk_mask(is_lead, tgg, lm.LEAD_VIOL_TG_S)
    parc, harc = lm.path_arc_length(Pc), lm.path_arc_length(Gc)
    with np.errstate(invalid="ignore", divide="ignore"):
        prog_ok = at_risk & np.isfinite(parc) & (harc >= lm.MIN_HUMAN_ARC_M)
        ratio = np.where(prog_ok, parc / np.where(harc > 0, harc, np.nan), np.nan)      # the progress CONTROL (lead_metrics.lead_progress semantics)
        vals = {"LON_dk_headway_min_m": hp, "LON_dk_time_gap_min_s": tgp, "LON_dk_min_ttc_s": ttp, "LON_dk_headway_abs_err_m": np.abs(hp - hg),
                "LON_dk_ttc_viol_rate": np.where(np.isfinite(ttp), (ttp < TTC_VIOL_S).astype(np.float64), np.nan),
                "LON_dk_lead_viol_at_risk": np.where(at_risk, lm._viol_flags(tgp, tgg).astype(np.float64), np.nan),
                "LON_dk_lead_progress_shortfall": np.where(np.isfinite(ratio), np.clip(1.0 - ratio, 0.0, None), np.nan)}
    extras = {"n_lead_windows": int(is_lead.sum()), "n_lost_lead": int((is_lead & ~np.isfinite(hp)).sum()), "n_at_risk": int(at_risk.sum()),
              "n_closing": int(np.asarray(dkp.get("n_closing", 0)).sum()) if dkp.get("status") == "OK" else 0,
              "mean_progress_ratio": (float(np.nanmean(ratio)) if np.isfinite(ratio).any() else None), "n_progress_windows": int(np.isfinite(ratio).sum()),
              "progress_min_required": lm.LEAD_PROGRESS_MIN}
    return vals, extras


def paired_rows(va: dict, vb: dict, eid: list, *, n_boot: int = 2000, seed: int = 0, min_n: int | None = None) -> tuple[dict, dict]:
    _, _, lm, ci = _tools()
    min_n = lm.MIN_STRATUM_N if min_n is None else int(min_n)
    rows, omitted = {}, {}
    for rid, better in ROWS:
        a, b = va[rid], vb[rid]
        ok = np.isfinite(a) & np.isfinite(b)
        n = int(ok.sum())
        why = None
        if n == 0:
            why = "no LEAD window with a finite value in both arms (lead-less / standstill / lane-leaving windows are NOT-COMPUTABLE, never 0)"
        elif n < min_n:
            why = f"UNPOWERED: {n} jointly-finite window(s) < {min_n} (lead_metrics.MIN_STRATUM_N)"
        if why:
            omitted[rid] = why
            rows[rid] = {"undefined": True, "reason": why, "better": better, "n": n, "family": FAMILY, "tier": TIER}
            continue
        e_ok = [e for e, k in zip(eid, ok) if k]
        r = ci.paired_episode_cluster_bootstrap(a[ok], b[ok], e_ok, n_boot=n_boot, seed=seed)
        row = {"delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "separated": bool(r["separated"]), "better": better, "n": int(r["n_windows"]), "n_episodes": int(r["n_episodes"]),
               "a_mean": float(a[ok].mean()), "b_mean": float(b[ok].mean()), "n_dropped_nonfinite": int((~ok).sum()), "family": FAMILY, "estimator": r["estimator"], "tier": TIER}
        if r.get("degenerate"):
            row["degenerate"] = True
        rows[rid] = row
    return rows, omitted


def build(a_dir, b_dir, *, dev_split=None, dev=None, lead_block, arm_a: str = "os", arm_b: str = "os", non_dev: str = "drop", n_boot: int = 2000, seed: int = 0,
          min_n: int | None = None, labels=("K2", "K0")) -> dict:
    fam = _fam()
    dev = set(dev) if dev is not None else (fam.load_dev(dev_split) if dev_split else None)
    if not dev:
        _fail("no DEV list: pass --dev-split (an un-restricted read could score HOLD)")
    if not lead_block or not Path(lead_block).exists():
        _fail(f"lead block {lead_block!r} does not exist")
    A, ca = restrict_dev(load_arm(a_dir, lead_block, arm_a), dev, non_dev)
    B, cb = restrict_dev(load_arm(b_dir, lead_block, arm_b), dev, non_dev)
    pa, pb, eid, rep = align_pair(A, B)
    va, xa = window_values(pa)
    vb, xb = window_values(pb)
    rows, omitted = paired_rows(va, vb, eid, n_boot=n_boot, seed=seed, min_n=min_n)
    ra, _, _, _ = _tools()
    counts = {s: int((pa["state"] == s).sum()) for s in ra._LEAD_STATES}
    return {"schema": SCHEMA, "mode": "distkeep", "labels": f"{labels[0]} - {labels[1]}", "tier": TIER, "estimator": ESTIMATOR, "delta": "A - B (A = " + labels[0] + ")",
            "windows": "DEV (DEV_HOLD_SPLIT.json); HOLD is never read", "n_windows_paired": rep["n_common"], "window_set_sha256": rep["window_set_sha256"], "pairing": rep,
            "a": {**ca, "dir": A["dir"], **xa}, "b": {**cb, "dir": B["dir"], **xb},
            "lead": {"block": str(lead_block), "block_sha256": ra._sha256(str(lead_block)), "instants_s": A["instants_s"], "dt_s": A["dt"], "dump_grid_k": A["grid_k"],
                     "grid_complete": len(A["instants_s"]) == A["grid_k"], "state_counts_paired_dev": counts, "n_lead_windows": counts[LEAD_STATE],
                     "join_key": "(clip_id, RAW frame = ws + provider_to_raw_frame_offset); label-free speed proof dump v0 == block speeds (1e-3 m/s)",
                     "not_computable_windows": {s: counts[s] for s in counts if s != LEAD_STATE},
                     "grid_note": (None if len(A["instants_s"]) == A["grid_k"] else
                                   f"COARSE: scored on the {len(A['instants_s'])} instant(s) {A['instants_s']} s the dump grid shares with the block; rebuild the block at the dump's own dt/k")},
            "rows": {FAMILY: rows}, "omitted_rows": omitted, "n_boot": n_boot, "boot_seed": seed, "caveat": CAVEAT}


def merge_into(families_path, doc: dict, out=None, check_dirs: bool = True) -> dict:
    """Add ``doc``'s rows under ``rows.LONGITUDINAL`` of a ``r9_e2_families.py`` families.json.  REFUSES unless it was built from the SAME arms / labels / windows."""
    p = Path(families_path)
    try:
        fj = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        _fail(f"--merge-into {p}: unreadable ({e})")
    if not isinstance(fj, dict) or not isinstance(fj.get("rows"), dict):
        _fail(f"--merge-into {p}: not a families.json (no `rows` object)")
    if fj.get("mode") not in (None, "families"):
        _fail(f"--merge-into {p}: mode {fj.get('mode')!r} is not 'families'")
    if fj.get("labels") not in (None, doc["labels"]):
        _fail(f"--merge-into {p}: labels {fj.get('labels')!r} != {doc['labels']!r}")
    if fj.get("n_windows_paired") != doc["n_windows_paired"]:
        _fail(f"window-set mismatch: families.json pairs {fj.get('n_windows_paired')} windows, distance-keeping pairs {doc['n_windows_paired']}")
    if fj.get("window_set_sha256") not in (None, doc["window_set_sha256"]):
        _fail("window-set mismatch: families.json window_set_sha256 differs from the distance-keeping window set")
    if check_dirs:
        for side in ("a", "b"):
            fd = (fj.get(side) or {}).get("dir")
            if fd is not None and os.path.normcase(os.path.abspath(str(fd))) != os.path.normcase(os.path.abspath(str(doc[side]["dir"]))):
                _fail(f"arm {side.upper()} dump differs: families.json {fd!r} vs distance-keeping {doc[side]['dir']!r}")
    lon = fj["rows"].setdefault(FAMILY, {})
    clash = [r for r in doc["rows"][FAMILY] if r in lon and not str(r).startswith("LON_dk_")]
    if clash:
        _fail(f"row id(s) {clash} already exist in families.json LONGITUDINAL")
    lon.update(doc["rows"][FAMILY])
    fj["distance_keeping_merge"] = {"schema": SCHEMA, "window_set_sha256": doc["window_set_sha256"], "lead": doc["lead"], "omitted_rows": doc["omitted_rows"],
                                    "rows_added": sorted(doc["rows"][FAMILY])}
    dst = Path(out) if out else p
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_text(json.dumps(fj, indent=1, default=float) + "\n", encoding="utf-8")
    os.replace(tmp, dst)
    return fj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a-dump", required=True, help="K2's dump dir (A in A - B)")
    ap.add_argument("--b-dump", required=True, help="K0's dump dir")
    ap.add_argument("--dev-split", required=True)
    ap.add_argument("--lead-block", required=True, help="banked B1 EVAL lead block (.npz), per-frame, keyed (clip_id, frame)")
    ap.add_argument("--arm-a", default="os")
    ap.add_argument("--arm-b", default="os")
    ap.add_argument("--non-dev", choices=("drop", "refuse"), default="drop")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-n", type=int, default=None, help="minimum jointly-finite windows per row (default lead_metrics.MIN_STRATUM_N = 30)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge-into", default=None, help="a r9_e2_families.py families.json: add the rows under rows.LONGITUDINAL (in place)")
    a = ap.parse_args(argv)
    try:
        doc = build(a.a_dump, a.b_dump, dev_split=a.dev_split, lead_block=a.lead_block, arm_a=a.arm_a, arm_b=a.arm_b, non_dev=a.non_dev, n_boot=a.boot, seed=a.seed, min_n=a.min_n)
        if a.merge_into:
            merge_into(a.merge_into, doc)
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 2
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=1, default=float) + "\n", encoding="utf-8")
    n_def = sum(1 for r in doc["rows"][FAMILY].values() if r.get("undefined"))
    print(f"[r9-e2-distkeep] {len(doc['rows'][FAMILY])} rows ({n_def} UNDEFINED), n_windows_paired={doc['n_windows_paired']}, lead windows={doc['lead']['n_lead_windows']}, "
          f"grid_complete={doc['lead']['grid_complete']} -> {a.out}" + (f"; merged into {a.merge_into}" if a.merge_into else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
