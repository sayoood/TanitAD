"""s6c_dev.py -- the E2 judge's clause (c) input: the planner's S6c sidewalk rate on the DEV clips, from the UNCHANGED FlyWheels ``mapsel_score.py``.

NEW file (E2-S6C, 2026-11-04).  ``mapsel_score.main()`` is exec'd as is; the DEV restriction is applied BEFORE its rate / CI by wrapping what ``main`` loads:
  * ``l2_common.load_eval_pf(seed)`` (the S6 rows) and ``l2_oracle_s2_rows.load_panel(seed)`` (the S2 rows) are row-filtered to the DEV clips (clip_index -> uuid ->
    sha12 in DEV_HOLD_SPLIT.json; ``mapsel_data.clip_uuids()`` is the clip_index map).  Both must be filtered or ``main`` refuses ("windows missing from the trace").
  * ``load_trace`` is wrapped: a trace holding ANY HOLD clip (or a clip in neither split) REFUSES -- HOLD windows are never scored, not even in pass 2.
Then the flat JSON the judge reads is written (``rate`` = ``map_compliance.S6c.arms.BASE.off_drivable.rate``, ``n``, ``tier``, ``estimator``) -- REFUSING an empty set, a
non-finite rate, or a failing K_BASE control (``picks["BASE"] == tr["sel"]``: the BASE arm IS the emitted pick) / K_RAST control.  ``--allow-partial`` / ``--controls-only`` are
never accepted (dry-run switches).  LOWER is better.  Scored on ONE inference seed's roll: the planner SAMPLES, so seeds 0 and 1 are two separate runs (H-ESTIM-SEED-1).
CAVEAT (inherited, not changed): mapsel_score stages the BASE plan through refcv8's ``refcv8_retime_deploy_gains_pf.json`` retime (l2_common.deploy_table) for BOTH arms.
UNVERIFIED end to end: needs the real trace + rasters + the dev-box banked refcv8 eval artifacts; ``main`` is exercised only against fakes (test_r9e2_s6c.py).

    python s6c_dev.py --mapsel-code <.../2026-10-25-refcv9-mapsel/code> [--l2-code <.../2026-10-25-refcv9-l2/code>] --split DEV_HOLD_SPLIT.json --seed 0 --trace T.npz --rast R --out OUT
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

TIER = "T1 self-action open loop"
ESTIMATOR = ("point: S6c off_drivable rate of the emitted (native) plan, BASE arm, DEV clips only (mapsel_score.py map_compliance.S6c.arms.BASE.off_drivable; "
             "no interval -- one inference seed)")
GATING = ("K_BASE_arm_equals_emitted_pick", "K_RAST_all_windows_have_a_raster")
S6_ROW_KEYS = ("g", "v0", "os", "r7", "ws", "clip", "eid", "compliant", "plan8", "prior8", "stage_banked", "rows", "logits", "lon_v7", "lat_v7", "v0_extras")
S2_LIST_KEYS = ("logits_s2", "traj_s2", "sel_s2", "ex_ws", "ex_clip")      # S2 roll order: `rows` indexes them, so they are left whole
PANEL_ROW_KEYS = ("g", "v0", "bat_os", "echo", "ws", "clip", "eid", "compliant", "plan8", "prior8", "stage_banked")


def sha12(clip_id) -> str:
    return hashlib.sha256(str(clip_id).encode()).hexdigest()[:12]            # == refcv7_loader.sha12 == mapsel_data.sha12


def load_dev_hold(path):
    j = json.loads(Path(path).read_text(encoding="utf-8"))
    dev, hold = {str(x).lower() for x in j["DEV"]["sha12"]}, {str(x).lower() for x in j["HOLD"]["sha12"]}
    if not dev or not hold or dev & hold:
        raise SystemExit(f"[s6c-dev] {path}: DEV / HOLD empty or not disjoint")
    return dev, hold


def clip_split(clip_index, uuids, dev, hold):
    """clip_index array -> (is_dev bool[n], is_hold bool[n]).  An index outside the uuid list REFUSES."""
    ci = np.asarray(clip_index, np.int64)
    if len(ci) and (ci.min() < 0 or ci.max() >= len(uuids)):
        raise SystemExit(f"[s6c-dev] clip_index outside the {len(uuids)} eval clips: [{ci.min()}, {ci.max()}]")
    sh = [sha12(uuids[int(c)]) for c in ci]
    return np.array([s in dev for s in sh], bool), np.array([s in hold for s in sh], bool)


def restrict_rows(d, keep, row_keys, list_keys=(), strict_len=None):
    """A copy of dict ``d`` with every per-row key cut to ``keep``.  REFUSES a missing row key, a wrong length, an unknown ndarray of the row length (it would be left misaligned)
    or an empty result."""
    keep = np.asarray(keep, bool)
    n = len(keep)
    missing = [k for k in row_keys if k not in d]
    if missing:
        raise SystemExit(f"[s6c-dev] the banked population lacks the per-row key(s) {missing}: the DEV filter cannot cut it")
    bad = [k for k in row_keys if len(d[k]) != n]
    if bad:
        raise SystemExit(f"[s6c-dev] per-row key(s) {bad} are not length {n}")
    unknown = [k for k, v in d.items() if k not in row_keys and k not in list_keys and isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n]
    if unknown:
        raise SystemExit(f"[s6c-dev] unknown array key(s) {unknown} have the row length {n}: not cut, would misalign -- extend the key list")
    if not keep.any():
        raise SystemExit("[s6c-dev] the DEV filter kept no row (empty set)")
    return {k: (v[keep] if k in row_keys else v) for k, v in d.items()}


def make_load_eval_pf(orig, uuids, dev, hold, census):
    def load_eval_pf(seed, *a, **kw):
        E = orig(seed, *a, **kw)
        isdev, _ = clip_split(E["clip"], uuids, dev, hold)
        out = restrict_rows(E, isdev, S6_ROW_KEYS, S2_LIST_KEYS)
        census["S6_rows_all"], census["S6_rows_dev"] = int(len(isdev)), int(isdev.sum())
        return out
    return load_eval_pf


def make_load_panel(orig, uuids, dev, hold, census):
    def load_panel(seed, *a, **kw):
        P = orig(seed, *a, **kw)
        isdev, _ = clip_split(P["clip"], uuids, dev, hold)
        out = restrict_rows(P, isdev, PANEL_ROW_KEYS)
        census["S2_rows_all"], census["S2_rows_dev"] = int(len(isdev)), int(isdev.sum())
        return out
    return load_panel


def make_load_trace(orig, uuids, dev, hold, census):
    def load_trace(path, *a, **kw):
        tr = orig(path, *a, **kw)
        isdev, ishold = clip_split(tr["e_i"], uuids, dev, hold)
        if ishold.any() or not isdev.all():
            raise SystemExit(f"[s6c-dev] refusing: the trace holds {int(ishold.sum())} HOLD and {int((~isdev & ~ishold).sum())} non-DEV windows -- roll DEV windows only")
        census["trace_windows"] = int(len(isdev))
        return tr
    return load_trace


def build_flat(res, *, census, split_path, seed, extra=None):
    """mapsel_score's ``res`` -> the judge's flat JSON.  REFUSES: a missing / failing gating control, an empty or non-finite rate, an empty S6c population."""
    ctl = (res or {}).get("controls", {})
    for k in GATING:
        if k not in ctl or ctl[k].get("pass") is not True:
            raise SystemExit(f"[s6c-dev] refusing: control {k} is {'absent' if k not in ctl else 'FAILING'} ({ctl.get(k)})")
    if ctl["K_BASE_arm_equals_emitted_pick"].get("share") != 1.0:
        raise SystemExit("[s6c-dev] refusing: K_BASE share != 1.0 (BASE arm != the emitted pick)")
    try:
        blk = res["map_compliance"]["S6c"]
        row = blk["arms"]["BASE"]["off_drivable"]
    except (KeyError, TypeError):
        raise SystemExit("[s6c-dev] refusing: result has no map_compliance.S6c.arms.BASE.off_drivable")
    rate, n = row.get("rate"), row.get("n")
    if not (isinstance(rate, (int, float)) and math.isfinite(rate)) or not n or int(n) <= 0 or int(blk.get("n_windows_with_gt", 0)) <= 0:
        raise SystemExit(f"[s6c-dev] refusing: empty / non-finite S6c set (rate={rate!r}, n={n!r}, n_windows_with_gt={blk.get('n_windows_with_gt')!r})")
    return {"rate": float(rate), "n": int(n), "tier": TIER, "estimator": ESTIMATOR, "arm": "BASE (emitted native plan, no post-hoc selection)", "better": "lower",
            "n_windows_compliant": int(blk.get("n_windows_compliant", -1)), "n_windows_with_gt": int(blk["n_windows_with_gt"]), "infer_seed": int(seed),
            "split": str(split_path), "split_sha256": hashlib.sha256(Path(split_path).read_bytes()).hexdigest(), "dev_filter": dict(census),
            "controls": {k: bool(v.get("pass")) for k, v in ctl.items() if isinstance(v, dict) and "pass" in v}, "hard_controls_pass": res.get("hard_controls_pass"), **(extra or {})}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    M = importlib.util.module_from_spec(spec)
    sys.modules[name] = M
    spec.loader.exec_module(M)
    return M


def run_scorer(M, C, S2R, *, dev, hold, argv):
    """Install the DEV wraps on the scorer's loaders and run its UNCHANGED main().  Returns (rc, census)."""
    census = {}
    uuids = M.D.clip_uuids()[0]
    C.load_eval_pf = make_load_eval_pf(C.load_eval_pf, uuids, dev, hold, census)
    S2R.load_panel = make_load_panel(S2R.load_panel, uuids, dev, hold, census)
    M.load_trace = make_load_trace(M.load_trace, uuids, dev, hold, census)
    sys.argv = [str(M.__file__)] + list(argv)
    return M.main(), census


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapsel-code", required=True)
    ap.add_argument("--l2-code", default=None)
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--rast", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--flat-out", default=None)
    a, rest = ap.parse_known_args(sys.argv[1:] if argv is None else argv)
    bad = [x for x in rest if x in ("--allow-partial", "--controls-only")]
    if bad:
        raise SystemExit(f"[s6c-dev] refusing {bad}: dry-run switches never feed a judge input")
    dev, hold = load_dev_hold(a.split)
    for p in (a.l2_code, a.mapsel_code):
        if p:
            sys.path.insert(0, str(p))
    M = load_module("mapsel_score", Path(a.mapsel_code) / "mapsel_score.py")
    import l2_common as C                      # noqa: E402  (the scorer imports them inside main(): our attribute wraps are what it then sees)
    import l2_oracle_s2_rows as S2R            # noqa: E402
    rc, census = run_scorer(M, C, S2R, dev=dev, hold=hold, argv=["--seed", str(a.seed), "--trace", a.trace, "--rast", a.rast, "--out", a.out] + rest)
    if rc not in (0, None):
        raise SystemExit(f"[s6c-dev] mapsel_score exited {rc}")
    res = json.loads((Path(a.out) / "MAPSEL_RESULT.json").read_text(encoding="utf-8"))
    flat = build_flat(res, census=census, split_path=a.split, seed=a.seed,
                      extra={"trace": str(a.trace), "trace_sha256": hashlib.sha256(Path(a.trace).read_bytes()).hexdigest(),
                             "scorer_sha256": hashlib.sha256(Path(M.__file__).read_bytes()).hexdigest(), "wrapper": Path(__file__).name})
    out = Path(a.flat_out) if a.flat_out else Path(a.out) / "S6C_DEV.json"
    out.write_text(json.dumps(flat, indent=1), encoding="utf-8")
    print(f"[s6c-dev] rate={flat['rate']:.6f} n={flat['n']} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
