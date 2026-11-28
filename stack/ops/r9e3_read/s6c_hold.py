"""s6c_hold.py -- refcv8's S6c off-drivable rate restricted to HOLD (or DEV, or ALL as a control), from the UNCHANGED FlyWheels ``mapsel_score.py``.

NEW file (E3-S6C-HOLD, 2026-11-08).  A sibling of ``stack/ops/r9e2_read/s6c_dev.py`` (which is NOT edited; its pure helpers are imported).  Differences:
  * the refcv8 trace holds ALL windows, so it is FILTERED to the requested subset instead of refused (s6c_dev refuses any HOLD window);
  * ``--subset hold`` keeps a window only if its clip is in HOLD and not in DEV (HOLD-only membership by construction); a window in NEITHER split REFUSES in every mode;
  * ``--subset dev`` is the same code on DEV (the "DEV beside it" read); ``--subset all`` keeps every window and is the real-data CONTROL: it must reproduce the banked
    MAPSEL_RESULT.json BASE off_drivable rate (0.17917, n 3198) exactly, or the HOLD number is not admissible;
  * the per-window BASE off_drivable vector is captured from the scorer's first ``paired_cell`` call (S6c, off_drivable, BASE as the second argument) and checked against the
    scorer's own rate / n, so an episode-cluster bootstrap (taniteval ci.episode_cluster_bootstrap, B 2000, seed 0, over EPISODES) can be reported.
TIER: T1 self-action open loop on logged eval139 frames (not driving performance).  ESTIMATOR: point = mean over S6c windows of the BASE (emitted) plan's off-drivable flag;
interval = episode-cluster bootstrap ("another draw of EPISODES" only: not another training run, not another inference run -- one checkpoint, one inference seed).  LOWER is better.
This tool reports a number; it does not compare it to any bar.  NEVER run on a refcv9 arm (HOLD discipline).  mapsel_score's DER / tau / donor quantities are computed on the
subset trace and are not read here.

    python s6c_hold.py --mapsel-code <.../2026-10-25-refcv9-mapsel/code> [--l2-code <.../2026-10-25-refcv9-l2/code>] --split DEV_HOLD_SPLIT.json --subset hold --seed 0 --trace T.npz --rast R --out OUT --read-out R.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_SD = HERE.parent / "r9e2_read"
if str(_SD) not in sys.path:
    sys.path.insert(0, str(_SD))
import s6c_dev as SD  # noqa: E402  (pure helpers only: sha12, load_dev_hold, clip_split, restrict_rows, load_module)

SUBSETS = ("hold", "dev", "all")
TIER = "T1 self-action open loop on logged eval139 frames (not driving performance)"
ESTIMATOR = ("point: S6c off_drivable rate of the emitted (native) plan, BASE arm, over the subset's S6c windows (mapsel_score.py map_compliance.S6c.arms.BASE.off_drivable); "
             "interval: episode-cluster bootstrap (taniteval ci.episode_cluster_bootstrap, B 2000, seed 0) over EPISODES = another draw of episodes only, "
             "NOT another training run, NOT another inference run (one checkpoint, one inference seed)")
GATING = SD.GATING
BOOT_B, BOOT_SEED = 2000, 0


def load_ci():
    try:
        import taniteval.ci as ci
        return ci
    except Exception:
        return SD.load_module("tanitad_ci_file", HERE.parents[2] / "taniteval" / "taniteval" / "ci.py")


def subset_mask(clip_index, uuids, dev, hold, subset):
    """-> keep bool[n].  REFUSES an unknown subset, a window in NEITHER split, and an empty result.  'hold' = HOLD and not DEV (by construction)."""
    if subset not in SUBSETS:
        raise SystemExit(f"[s6c-hold] unknown subset {subset!r}")
    isdev, ishold = SD.clip_split(clip_index, uuids, dev, hold)
    neither = ~isdev & ~ishold
    if neither.any():
        raise SystemExit(f"[s6c-hold] refusing: {int(neither.sum())} window(s) are in NEITHER split")
    keep = {"hold": ishold & ~isdev, "dev": isdev & ~ishold, "all": np.ones(len(isdev), bool)}[subset]
    if not keep.any():
        raise SystemExit(f"[s6c-hold] the {subset} filter kept no window (empty set)")
    return keep


def cut_trace(tr, keep):
    """Cut every per-window array of the loaded trace (first axis == n); rebuild idx / n.  meta is kept as banked."""
    n = tr["n"]
    keep = np.asarray(keep, bool)
    if len(keep) != n:
        raise SystemExit(f"[s6c-hold] keep length {len(keep)} != trace n {n}")
    out = {}
    for k, v in tr.items():
        if k in ("idx", "meta", "n"):
            continue
        out[k] = v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n else v
    out["idx"] = {(int(e), int(w)): j for j, (e, w) in enumerate(zip(out["e_i"], out["ws"]))}
    out["meta"], out["n"] = tr["meta"], int(keep.sum())
    return out


def drop_mask(e, w, drop):
    """bool[n]: True where the (clip, ws) window is in the explicit drop set."""
    return np.array([(int(a), int(b)) in drop for a, b in zip(e, w)], bool) if drop else np.zeros(len(e), bool)


def zero_byte_rasters(rast_dir, e_i, ws):
    """(clip, ws) keys of the trace windows whose raster file is 0 bytes (a corrupt bank: np.load raises EOFError on them)."""
    out = set()
    for e, w in zip(e_i.tolist(), ws.tolist()):
        f = Path(rast_dir) / f"e{int(e):03d}_w{int(w):04d}.npz"
        if f.exists() and f.stat().st_size == 0:
            out.add((int(e), int(w)))
    return out


def make_wraps(M, C, S2R, uuids, dev, hold, subset, census, rast_dir=None):
    """``rast_dir`` (dev / all CONTROL modes only) additionally drops windows whose raster file is 0 bytes, from the trace and from the S6 / S2 rows alike; the dropped keys are
    recorded in the census.  subset 'hold' REFUSES it: a HOLD read never drops a window."""
    if rast_dir is not None and subset == "hold":
        raise SystemExit("[s6c-hold] refusing: the HOLD read never drops a window (no --drop-zero-byte-rasters)")
    drop = set()

    def load_trace(path, *a, **kw):
        tr = M._orig_load_trace(path, *a, **kw)
        keep = subset_mask(tr["e_i"], uuids, dev, hold, subset)
        if rast_dir is not None:
            drop.update(zero_byte_rasters(rast_dir, tr["e_i"], tr["ws"]))
            census["dropped_zero_byte_raster_windows"] = sorted(drop)
            keep = keep & ~drop_mask(tr["e_i"], tr["ws"], drop)
        census["trace_windows_all"], census["trace_windows_kept"] = int(len(keep)), int(keep.sum())
        return cut_trace(tr, keep)

    def load_eval_pf(seed, *a, **kw):
        E = C._orig_load_eval_pf(seed, *a, **kw)
        keep = subset_mask(E["clip"], uuids, dev, hold, subset) & ~drop_mask(E["clip"], E["ws"], drop)
        out = SD.restrict_rows(E, keep, SD.S6_ROW_KEYS, SD.S2_LIST_KEYS)
        census["S6_rows_all"], census["S6_rows_kept"] = int(len(keep)), int(keep.sum())
        census["_s6_clip"] = np.asarray(out["clip"]).copy()
        return out

    def load_panel(seed, *a, **kw):
        P = S2R._orig_load_panel(seed, *a, **kw)
        keep = subset_mask(P["clip"], uuids, dev, hold, subset) & ~drop_mask(P["clip"], P["ws"], drop)
        out = SD.restrict_rows(P, keep, SD.PANEL_ROW_KEYS)
        census["S2_rows_all"], census["S2_rows_kept"] = int(len(keep)), int(keep.sum())
        return out
    return load_trace, load_eval_pf, load_panel


def make_paired_capture(PR, census):
    orig = PR.paired_cell

    def paired_cell(x, base, eid, *a, **kw):
        if "base_vec" not in census:        # first call = S6c / first non-BASE arm / off_drivable: `base` is the BASE S6c off_drivable per-window vector
            census["base_vec"] = np.asarray(base, float).copy()
            census["base_eid"] = np.asarray(list(eid)).copy()
        return orig(x, base, eid, *a, **kw)
    return orig, paired_cell


def run_scorer(M, C, S2R, PR, *, dev, hold, subset, argv, rast_dir=None):
    """Install the subset wraps + the capture on the scorer's loaders and run its UNCHANGED main().  Returns (rc, census)."""
    census = {}
    uuids = M.D.clip_uuids()[0]
    M._orig_load_trace, C._orig_load_eval_pf, S2R._orig_load_panel = M.load_trace, C.load_eval_pf, S2R.load_panel
    M.load_trace, C.load_eval_pf, S2R.load_panel = make_wraps(M, C, S2R, uuids, dev, hold, subset, census, rast_dir)
    orig_pc, PR.paired_cell = make_paired_capture(PR, census)
    sys.argv = [str(M.__file__)] + list(argv)
    try:
        rc = M.main()
    finally:                                   # restore, so a second run in the same process wraps the ORIGINALS
        M.load_trace, C.load_eval_pf, S2R.load_panel, PR.paired_cell = M._orig_load_trace, C._orig_load_eval_pf, S2R._orig_load_panel, orig_pc
    return rc, census


def summarize(res, census, *, subset, ci=None):
    """mapsel_score ``res`` + the captured vector -> the subset's S6c read.  REFUSES a failing gating control, an empty / non-finite set, or a captured vector that does not
    reproduce the scorer's own rate and n (the capture control)."""
    ctl = (res or {}).get("controls", {})
    for k in GATING:
        if k not in ctl or ctl[k].get("pass") is not True:
            raise SystemExit(f"[s6c-hold] refusing: control {k} is {'absent' if k not in ctl else 'FAILING'} ({ctl.get(k)})")
    try:
        blk = res["map_compliance"]["S6c"]
        row = blk["arms"]["BASE"]["off_drivable"]
    except (KeyError, TypeError):
        raise SystemExit("[s6c-hold] refusing: result has no map_compliance.S6c.arms.BASE.off_drivable")
    rate, n = row.get("rate"), row.get("n")
    if not (isinstance(rate, (int, float)) and math.isfinite(rate)) or not n or int(n) <= 0:
        raise SystemExit(f"[s6c-hold] refusing: empty / non-finite S6c set (rate={rate!r}, n={n!r})")
    if "base_vec" not in census:
        raise SystemExit("[s6c-hold] refusing: the per-window BASE vector was not captured")
    v, eid = census["base_vec"], census["base_eid"]
    fin = np.isfinite(v)
    if int(fin.sum()) != int(n) or not math.isclose(float(v[fin].mean()), float(rate), rel_tol=0, abs_tol=1e-12):
        raise SystemExit(f"[s6c-hold] refusing: captured vector (n={int(fin.sum())}, mean={float(v[fin].mean()) if fin.any() else None}) != scorer (n={n}, rate={rate})")
    boot = (ci or load_ci()).episode_cluster_bootstrap(v[fin], eid[fin], reduce="mean", n_boot=BOOT_B, seed=BOOT_SEED, dp=6)
    return {"subset": subset, "rate": float(rate), "n": int(n), "n_episodes": int(len(np.unique(eid[fin]))), "n_windows_compliant": int(blk.get("n_windows_compliant", -1)),
            "n_windows_with_gt": int(blk.get("n_windows_with_gt", -1)), "bootstrap": boot, "controls": {k: bool(v_.get("pass")) for k, v_ in ctl.items() if isinstance(v_, dict) and "pass" in v_},
            "hard_controls_pass": res.get("hard_controls_pass"), "dev_filter": {k: v_ for k, v_ in census.items() if not k.startswith(("base_", "_s6"))}}


def real_data_control(all_read, banked_path):
    """The ALL-windows read must reproduce the banked MAPSEL_RESULT.json BASE off_drivable rate and n EXACTLY (float equality)."""
    b = json.loads(Path(banked_path).read_text(encoding="utf-8"))["map_compliance"]["S6c"]["arms"]["BASE"]["off_drivable"]
    ok = bool(all_read["rate"] == b["rate"] and all_read["n"] == b["n"])
    return {"pass": ok, "banked_rate": b["rate"], "banked_n": b["n"], "all_rate": all_read["rate"], "all_n": all_read["n"], "abs_diff": abs(all_read["rate"] - b["rate"]),
            "banked": str(banked_path), "banked_sha256": hashlib.sha256(Path(banked_path).read_bytes()).hexdigest()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapsel-code", required=True)
    ap.add_argument("--l2-code", default=None)
    ap.add_argument("--split", required=True)
    ap.add_argument("--subset", choices=SUBSETS, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--rast", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop-zero-byte-rasters", action="store_true", help="dev / all CONTROL modes only: drop windows whose raster file is 0 bytes (recorded); refused with --subset hold")
    ap.add_argument("--read-out", required=True, help="the subset read JSON (rate, n, interval)")
    a, rest = ap.parse_known_args(sys.argv[1:] if argv is None else argv)
    bad = [x for x in rest if x in ("--allow-partial", "--controls-only")]
    if bad:
        raise SystemExit(f"[s6c-hold] refusing {bad}: dry-run switches never feed a read")
    dev, hold = SD.load_dev_hold(a.split)
    for p in (a.l2_code, a.mapsel_code):
        if p:
            sys.path.insert(0, str(p))
    M = SD.load_module("mapsel_score", Path(a.mapsel_code) / "mapsel_score.py")
    import l2_common as C                      # noqa: E402
    import l2_oracle_s2_rows as S2R            # noqa: E402
    PR = C.stack()["PR"]
    rc, census = run_scorer(M, C, S2R, PR, dev=dev, hold=hold, subset=a.subset,
                            argv=["--seed", str(a.seed), "--trace", a.trace, "--rast", a.rast, "--out", a.out] + rest,
                            rast_dir=a.rast if a.drop_zero_byte_rasters else None)
    if rc not in (0, None):
        raise SystemExit(f"[s6c-hold] mapsel_score exited {rc}")
    res = json.loads((Path(a.out) / "MAPSEL_RESULT.json").read_text(encoding="utf-8"))
    read = summarize(res, census, subset=a.subset)
    read.update({"infer_seed": int(a.seed), "split": str(a.split), "split_sha256": hashlib.sha256(Path(a.split).read_bytes()).hexdigest(), "trace": str(a.trace),
                 "trace_sha256": hashlib.sha256(Path(a.trace).read_bytes()).hexdigest(), "scorer_sha256": hashlib.sha256(Path(M.__file__).read_bytes()).hexdigest(),
                 "wrapper": Path(__file__).name, "tier": TIER, "estimator": ESTIMATOR, "better": "lower"})
    np.savez_compressed(Path(a.out) / f"base_vec_{a.subset}.npz", v=census["base_vec"], eid=census["base_eid"], clip=census.get("_s6_clip", np.zeros(0)))
    Path(a.read_out).write_text(json.dumps(read, indent=1), encoding="utf-8")
    print(f"[s6c-hold] subset={a.subset} rate={read['rate']:.6f} n={read['n']} episodes={read['n_episodes']} -> {a.read_out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
