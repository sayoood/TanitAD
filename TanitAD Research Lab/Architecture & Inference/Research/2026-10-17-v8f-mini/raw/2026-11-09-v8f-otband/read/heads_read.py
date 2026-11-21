"""OTBAND TACTICAL-family 'T's own decision reads' (PREREG_V8F_OTBAND.md, section Read, TACTICAL bullet). No bar applies; rows are reported.

Wrapper over the registered reader t1hd_read.py (sha256 3c6d9f79..., NOT edited, NOT copied). Functions called from it:
  setup_imports   - puts the READ-ONLY code tree on sys.path (tanitad, taniteval, v8f_mini_eval, v8f_t1_eval)
  Ctx             - join, grid windows, native tracks, clip store
  load_ticks      - one --dump-ticks file -> arrays (asserts the m = 10 tick layout)
  analyze_run     - matches each tick's T pose to the logged row r'_T and builds the label cells (lat, lon, goal) at r'_T, on-route only
  t_rows          - the registered per-run rows (ticks>=1) - (tick 0) for T-LAT-F1 / T-LON-F1 / goal macro-F1
Why a wrapper: t1hd_read.main() needs the HACC `real` run layout (chain.out, four runs, banked T0 dir); analyze_run / t_rows do not.
dump dirs are not read (analyze_run's dump_dir only feeds the X1 executed-track check, not the T cells).

Rows per arm: T-LAT-F1, T-LON-F1 (macro-F1, v8f_mini_eval._f1_reducer via t1hd_read.f1_red), goal macro-F1 (tactical_metrics recipe, >0.5),
constraints (26 regression cells, per-field MAE where the GT cell is finite; the GT `cons` of the join at r'_T), each at tick 0, ticks>=1, all ticks.
Paired differences (taniteval.ci.paired_episode_cluster_bootstrap, B=2,000, seed 0, 141 episodes): OB-BAND - OB-CTL (same seed), arm - DAGOACC (same seed),
OB-BAND_rep - OB-BAND_s0. The estimator answers 'would another draw of EPISODES say this?' only; training-seed variance is the second seed / the replicate.
Tier: T1-kin (self-action open loop), nav dead-reckoned, T sample-and-hold at the tool's ages. Window sets of two arms are asserted identical before pairing.
Because r'_T (the matched logged row) depends on each arm's own driven pose, the label cells differ per arm; pairing is on window position and episode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

READER_DIR = "E:/Projects/TanitAD-artifacts/video5/ref"
READER_SHA_PREFIX = "3c6d9f79"
TREE = "F:/TanitAD-scratch/otband_read/tree"
INP = "E:/Projects/TanitAD-artifacts/sfresh_inputs"
RAW = ("E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw")
BASE_DIR = "E:/Projects/TanitAD-artifacts/v8f-dagoacc-t1"
NEW_DIR = "F:/TanitAD-scratch/otband_read/t1"
SUFFIX = "m10_causal_heads"
GROUPS = ("tick0", "ticks_ge1", "all")
MIN_FREE_GB = 4.0


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def free_gb():
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("ml", ctypes.c_ulong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                    ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong)]
    m = MS()
    m.l = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ap / 2 ** 30


def paths_for(tag):
    """(ticks, done-file or None) for a tag. DAGOACC_* are the banked baseline (done file in BASE_DIR); everything else lives under NEW_DIR."""
    if tag.startswith("DAGOACC_"):
        d = BASE_DIR
    else:
        d = NEW_DIR
    return (f"{d}/ticks/ticks_{tag}_{SUFFIX}.npz", f"{d}/done_{tag}_{SUFFIX}.txt")


def seed_of(tag):
    return tag.split("_")[-1] if "_" in tag else ""


def partner(tag, kind):
    """kind 'ctl': OB-BAND_sN -> OB-CTL_sN; 'base': OB-*_sN -> PARITY_DAGOACC_sN (README_READ.md amendment 1: DAG-OACC dumped in this
    read's environment), PARITY_DAGOACC_sN -> DAGOACC_sN (the environment check against the banked run); 'rep': OB-BAND_rep -> OB-BAND_s0."""
    s = seed_of(tag)
    if kind == "ctl":
        return f"OB-CTL_{s}" if tag.startswith("OB-BAND_s") else None
    if kind == "base":
        return f"PARITY_DAGOACC_{s}" if (tag.startswith("OB-") and s.startswith("s")) else (f"DAGOACC_{s}" if tag.startswith("PARITY_DAGOACC_") else None)
    if kind == "rep":
        return "OB-BAND_s0" if tag == "OB-BAND_rep" else None
    return None


def jenc(x):
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


def grp_slices(a2):
    """[N,3] array -> {group: flattened 1-D array, eid repeat count}."""
    return {"tick0": (a2[:, :1].reshape(-1), 1), "ticks_ge1": (a2[:, 1:].reshape(-1), 2), "all": (a2.reshape(-1), 3)}


def build_arm(H, ctx, tag, n_boot):
    """Everything per arm: analyze_run + the registered t_rows + the per-cell arrays the arm-level and paired rows use."""
    path, _ = paths_for(tag)
    tt = H.load_ticks(path)
    R = H.analyze_run(ctx, tt, None, tag, n_boot, dump_dir=None)
    ci = tt["ci"]
    N = tt["N"]
    arm = {"tag": tag, "N": N, "ci": ci, "t0": tt["t0"], "ticks_sha256": sha(path), "R": R, "tt": tt}
    arm["registered_t_rows"] = H.t_rows(R, ci, n_boot, tag)       # (ticks>=1) - (tick 0), same windows: the registered rows, unmodified
    # F1 encodings [N,3]
    enc = {}
    for hd, n in (("lat", H.NCLS["lat"]), ("lon", H.NCLS["lon"])):
        enc[hd] = H.E0._f1_encoded(R["cells"][hd]["true"], R["cells"][hd]["pred"], n)
    arm["enc"] = enc
    # goal table: flattened [N*3, 22]
    g = R["cells"]["goal"]
    arm["goal"] = {"gy": g["gy"].reshape(-1, 22), "gw": g["gw"].reshape(-1, 22), "pr": g["prob"].reshape(-1, 22), "valid": g["gw"].any(-1)}
    # constraints: abs error per field [N,3,26], NaN where the GT cell is undefined / window unlabelled / off-route
    J = ctx.J
    jT = H.jrow(ctx, ci, R["rT"])
    hasT = J["has"][jT] & (R["dT"] <= H.OFF_ROUTE_M)
    gtc = J["cons"][jT].astype(np.float64)                       # [N,3,26]
    pc = tt["t_cons_pred"].astype(np.float64)
    err = np.abs(pc - gtc)
    err = np.where(hasT[..., None] & np.isfinite(gtc), err, np.nan)
    arm["cons_err"] = err
    return arm


def f1_stats(H, enc2, ci, n, n_boot):
    out = {}
    red = H.f1_red(n)
    for g, (v, rep) in grp_slices(enc2).items():
        eid = np.repeat(ci, rep)
        r = H.EB(v.astype(np.float64), eid, reduce=red, n_boot=n_boot, seed=0)
        out[g] = {"f1": r["mean"], "lo": r["lo"], "hi": r["hi"], "n_cells": int((v >= 0).sum()), "n_episodes": r["n_episodes"], "estimator": r["estimator"], "n_boot": r["n_boot"]}
    return out


def goal_reducer(tbl):
    gy, gw, pr = tbl

    def red(v):
        v = np.rint(np.asarray(v, np.float64)).astype(np.int64)
        v = v[v >= 0]
        if not len(v):
            return float("nan")
        f1 = []
        for t in range(22):
            m = gw[v, t]
            if m.sum() == 0 or gy[v][m, t].sum() == 0:
                f1.append(np.nan)
                continue
            p = pr[v][m, t] > 0.5
            y = gy[v][m, t]
            tp, fp, fn = int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum())
            f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
        return float(np.nanmean(f1)) if np.isfinite(f1).any() else float("nan")
    return red


def goal_cells(arm):
    N = arm["N"]
    cid = np.arange(N * 3).reshape(N, 3)
    return np.where(arm["goal"]["valid"], cid, -1)


def stat_cons(arm, ci, field_names, n_boot, H):
    out = {}
    e = arm["cons_err"]
    for g in GROUPS:
        sl = {"tick0": e[:, :1], "ticks_ge1": e[:, 1:], "all": e}[g]
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pw = np.nanmean(sl, axis=1)                      # [N,26] per-window mean over the group's ticks
        d = {}
        for q, nm in enumerate(field_names):
            x = pw[:, q]
            ok = np.isfinite(x)
            if not ok.any():
                d[nm] = {"n_windows": 0, "reason": "no finite GT cell for this field"}
                continue
            r = H.EB(x, ci, reduce="mean", n_boot=n_boot, seed=0)
            d[nm] = {"mae": r["mean"], "lo": r["lo"], "hi": r["hi"], "n_windows": int(ok.sum())}
        out[g] = d
    return out


def paired_rows(H, A, B, ci, field_names, n_boot, label):
    """A minus B on the same windows/episodes. Returns rows per head and group."""
    assert (A["ci"] == B["ci"]).all() and (A["t0"] == B["t0"]).all(), "window lists differ"
    rows = {}
    pb = lambda a, b, eid, red, **k: _pb(H, a, b, eid, red, n_boot, **k)
    for hd, nm in (("lat", "T-LAT-F1"), ("lon", "T-LON-F1")):
        n = H.NCLS[hd]
        red = H.f1_red(n)
        ga, gb = grp_slices(A["enc"][hd]), grp_slices(B["enc"][hd])
        for g in GROUPS:
            a, rep = ga[g]
            b, _ = gb[g]
            rows[f"{nm} [{g}]"] = pb(a, b, np.repeat(ci, rep), red, higher_is_better=True, n_cells_a=int((a >= 0).sum()), n_cells_b=int((b >= 0).sum()))
    # goal: cell tables of the two arms stacked; B's cell ids offset so one reducer reads both
    N = A["N"]
    off = N * 3
    gy = np.concatenate([A["goal"]["gy"], B["goal"]["gy"]])
    gw = np.concatenate([A["goal"]["gw"], B["goal"]["gw"]])
    pr = np.concatenate([A["goal"]["prob"] if "prob" in A["goal"] else A["goal"]["pr"], B["goal"]["pr"]])
    red = goal_reducer((gy, gw, pr))
    ca, cb = goal_cells(A), goal_cells(B)
    cb = np.where(cb >= 0, cb + off, -1)
    for g in GROUPS:
        a = {"tick0": ca[:, :1], "ticks_ge1": ca[:, 1:], "all": ca}[g]
        b = {"tick0": cb[:, :1], "ticks_ge1": cb[:, 1:], "all": cb}[g]
        rep = a.shape[1]
        rows[f"goal macro-F1 [{g}]"] = pb(a.reshape(-1), b.reshape(-1), np.repeat(ci, rep), red, higher_is_better=True,
                                          n_cells_a=int((a >= 0).sum()), n_cells_b=int((b >= 0).sum()))
    # constraints per field (MAE, lower is better) and the mean of per-field MAEs is NOT formed (units differ)
    for g in GROUPS:
        ea = {"tick0": A["cons_err"][:, :1], "ticks_ge1": A["cons_err"][:, 1:], "all": A["cons_err"]}[g]
        eb = {"tick0": B["cons_err"][:, :1], "ticks_ge1": B["cons_err"][:, 1:], "all": B["cons_err"]}[g]
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pa, pbb = np.nanmean(ea, axis=1), np.nanmean(eb, axis=1)
        for q, nm in enumerate(field_names):
            x, y = pa[:, q], pbb[:, q]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() == 0:
                rows[f"CONS {nm} MAE [{g}]"] = {"n_windows": 0, "reason": "no window with a finite GT cell and a finite prediction in both arms"}
                continue
            x, y = np.where(ok, x, np.nan), np.where(ok, y, np.nan)
            rows[f"CONS {nm} MAE [{g}]"] = pb(x, y, ci, "mean", higher_is_better=False, n_windows=int(ok.sum()))
    return rows


def _pb(H, a, b, eid, red, n_boot, higher_is_better, **extra):
    r = H.PB(np.asarray(a, np.float64), np.asarray(b, np.float64), eid, n_boot=n_boot, seed=0, reduce=red)
    out = {"delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "separated": bool(r["separated"]), "n_episodes": r["n_episodes"],
           "estimator": r["estimator"], "n_boot": r["n_boot"], "higher_is_better": higher_is_better}
    if r.get("degenerate"):
        out["degenerate"] = True
    out.update(extra)
    return out


def arm_levels(H, arm, field_names, n_boot):
    ci = arm["ci"]
    out = {"n_windows": arm["N"], "ticks_sha256": arm["ticks_sha256"], "off_route": arm["R"]["off_route"]}
    out["T-LAT-F1"] = f1_stats(H, arm["enc"]["lat"], ci, H.NCLS["lat"], n_boot)
    out["T-LON-F1"] = f1_stats(H, arm["enc"]["lon"], ci, H.NCLS["lon"], n_boot)
    # goal macro-F1 level
    gy, gw, pr = arm["goal"]["gy"], arm["goal"]["gw"], arm["goal"]["pr"]
    red = goal_reducer((gy, gw, pr))
    cells = goal_cells(arm)
    gl = {}
    for g in GROUPS:
        v = {"tick0": cells[:, :1], "ticks_ge1": cells[:, 1:], "all": cells}[g]
        r = H.EB(v.reshape(-1).astype(np.float64), np.repeat(ci, v.shape[1]), reduce=red, n_boot=n_boot, seed=0)
        gl[g] = {"macro_f1": r["mean"], "lo": r["lo"], "hi": r["hi"], "n_cells": int((v >= 0).sum()), "n_episodes": r["n_episodes"]}
    out["goal macro-F1"] = gl
    out["CONSTRAINTS"] = stat_cons(arm, ci, field_names, n_boot, H)
    rt = arm["registered_t_rows"]
    out["registered_t_rows(ticks>=1 - tick0)"] = {k: {kk: v[kk] for kk in ("delta", "lo", "hi", "separated", "n_windows", "f1_ge1", "f1_tick0") if kk in v} for k, v in rt["decisive"].items()}
    out["registered_t_rows(ticks>=1 - tick0)"]["goal macro-F1"] = {kk: rt["beside"]["goal macro-F1"][kk] for kk in ("delta", "lo", "hi", "separated", "n_windows", "f1_ge1", "f1_tick0")}
    return out


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tags", nargs="+", required=True, help="e.g. OB-CTL_s0 OB-BAND_s0 PARITY_DAGOACC_s0 OB-BAND_rep; DAGOACC_s0/s1 are loaded as the baseline automatically")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--no-baseline", action="store_true", help="do not load DAGOACC_sN for the arm-minus-DAGOACC rows")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    # refuse a tag without a done file (the banked baseline has its own)
    for t in a.tags:
        tp, dp = paths_for(t)
        if not os.path.exists(dp):
            raise SystemExit(f"REFUSED: {t}: done file missing: {dp}")
        if not os.path.exists(tp):
            raise SystemExit(f"REFUSED: {t}: ticks file missing: {tp}")
    gb = free_gb()
    if gb < MIN_FREE_GB:
        raise SystemExit(f"REFUSED: free physical memory {gb:.1f} GB < {MIN_FREE_GB} GB")
    sys.path.insert(0, READER_DIR)
    rp = os.path.join(READER_DIR, "t1hd_read.py")
    assert sha(rp).startswith(READER_SHA_PREFIX), "t1hd_read.py is not the registered reader"
    import t1hd_read as H
    H.setup_imports(TREE)
    import tanitad
    assert os.path.abspath(tanitad.__file__).replace("\\", "/").startswith(TREE), f"tanitad imported from {tanitad.__file__}"
    D = H.D

    class A_:
        eval_join, eval_grid = f"{RAW}/join_eval141.npz", f"{RAW}/eval_grid_windows.json"
        tracks, eval_cache = f"{INP}/inputs/native_tracks.pt", f"{INP}/cache"

    ctx = H.Ctx(A_)
    field_names = list(D.CONS_FIELDS)
    wanted = list(dict.fromkeys(a.tags))
    need = list(wanted)
    if not a.no_baseline:
        for t in wanted:
            b = partner(t, "base")
            if b and b not in need:
                need.append(b)
    for t in wanted:
        for k in ("ctl", "rep"):
            p = partner(t, k)
            if p and p not in need and os.path.exists(paths_for(p)[1]):
                need.append(p)
    for t in need:
        if not os.path.exists(paths_for(t)[1]):
            raise SystemExit(f"REFUSED: {t}: done file missing: {paths_for(t)[1]}")
    arms = {}
    res = {"prereg": "PREREG_V8F_OTBAND.md section Read, TACTICAL bullet (T's own decision reads); no bar applies, rows are reported",
           "tier": "T1-kin (self-action open loop; nav dead-reckoned; T sample-and-hold at the tool's ages); m = 10, ticks 0, 1, 2",
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap, B={a.n_boot}, seed 0, over the 141 episodes; answers 'would another draw of EPISODES say this?' only "
                        "(training-seed variance: the second seed / the replicate arm; inference variance: none, T is deterministic)",
           "reader": {"path": rp, "sha256": sha(rp)}, "tree": TREE, "wrapper_sha256": sha(os.path.abspath(__file__)),
           "groups": {"tick0": "tick 0 (fresh, T0-equivalent)", "ticks_ge1": "ticks 1, 2", "all": "ticks 0, 1, 2 pooled"},
           "constraints_note": "per-field MAE of t_cons_pred against join `cons` at r'_T; undefined GT cells (NaN) are unscored; fields have different units, no pooled score is formed",
           "arms": {}, "paired": {}, "not_computable": []}
    for t in need:
        say("arm", t)
        arms[t] = build_arm(H, ctx, t, a.n_boot)
        res["arms"][t] = arm_levels(H, arms[t], field_names, a.n_boot)
    # per-field coverage notes
    for t, ar in arms.items():
        for q, nm in enumerate(field_names):
            n = int(np.isfinite(ar["cons_err"][..., q]).sum())
            if n == 0:
                res["not_computable"].append({"arm": t, "row": f"CONS {nm}", "reason": "no finite GT cell on any scored tick", "n": 0})
    pairs = []
    for t in wanted:
        for kind, nm in (("ctl", "minus OB-CTL (same seed)"), ("base", "minus DAGOACC (same seed)"), ("rep", "minus OB-BAND_s0")):
            p = partner(t, kind)
            if p and p in arms and not (kind == "base" and a.no_baseline):
                pairs.append((t, p, nm))
    if "PARITY_DAGOACC_s0" in arms and "DAGOACC_s0" in arms:
        pass  # covered by partner(.., 'base')
    for t, p, nm in pairs:
        say("paired", t, "-", p)
        res["paired"][f"{t} {nm}"] = {"a": t, "b": p, "rows": paired_rows(H, arms[t], arms[p], arms[t]["ci"], field_names, a.n_boot, f"{t}-{p}")}
    # identity control: an arm minus itself must read 0 [0, 0]
    t = wanted[0]
    idr = paired_rows(H, arms[t], arms[t], arms[t]["ci"], field_names[:1], 200, "identity")
    res["identity_control"] = {"arm": t, "all_zero": bool(all(r.get("delta", 0) == 0 and r.get("lo", 0) == 0 and r.get("hi", 0) == 0 for r in idr.values())), "n_rows": len(idr)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, default=jenc)
    say("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
