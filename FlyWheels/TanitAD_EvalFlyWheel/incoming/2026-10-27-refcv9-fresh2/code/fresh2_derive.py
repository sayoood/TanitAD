"""fresh2_derive.py -- COPY of refcv9_lha0/code/lha0_derive.py (the LHA0 feature code, a0 / psi_dot0 NOT re-derived differently).  ONLY the five input/output
paths and the expected clip count are taken from the command line (`--set first` reproduces the LHA0 package's own inputs for the self-test; `--set fresh2` points at FRESH2);
the derivation, the anchor-rule search, the tolerances and every formula are unchanged.
Original docstring:
lha0_derive.py -- PREREG_LHA0_DIAG s1.2: the mandatory calibration + the causal derivation of a0 / psi_dot0 at t0.  METADATA ONLY (poses, never frames).

  1. resolve each bank window's episode by sha12 (sha256(clip_id)[:12]) against the dump's clip_sha12_by_index; cross-check the cache poses against the dump's
     decisions/ep<e_i>.npz['ep_poses'] (exact);
  2. find the anchor-frame rule k = ws + delta (ws = t + W - 1) under which bank v0_in == poses[k, 3] for ALL windows; STOP if none matches;
  3. a0 = (v[k]-v[k-1])/dt (per-clip dt from the clock sidecar), psi_dot0 = wrap(yaw[k]-yaw[k-1])/dt, 0.5 s variant, forward-difference leak contrast, nominal-dt variants;
  4. mutation check: forward vs backward a0 differ on the majority of rows.

Writes result/LHA0_CALIBRATION.json and result/lha0_features.npz (sha12 only; no clip id is written).  CPU only.  NEW file (LHA0 agent, 2026-10-26).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TREE = Path("C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58")
sys.path.insert(0, str(TREE / "stack"))
import torch  # noqa: E402
import tanitad  # noqa: E402
from tanitad.data import clip_clock as CC  # noqa: E402
from tanitad.data.v2_dataset import _scan_meta, stable_episode_id  # noqa: E402

SETS = {"first": dict(BANK="F:/l1c/bank/ztac_fresh.npz", EPDIR="F:/l1c/kit_fresh/data/refcv6-b1-416x1024-eval139", SIDECAR="F:/l1c/kit_fresh/data/clock_sidecar.jsonl",
                      DUMP="F:/l1c/tree/step48852ema_fresh/dump_s0", OUT="F:/l1c/fresh2/selftest_derive_first", N=300),
        "fresh2": dict(BANK="F:/l1c/bank/ztac_fresh2.npz", EPDIR="F:/l1c/kit_fresh2/data/refcv6-b1-416x1024-eval139", SIDECAR="F:/l1c/kit_fresh2/data/clock_sidecar.jsonl",
                       DUMP="F:/l1c/tree/step48852ema_fresh2/dump_s0", OUT="F:/l1c/fresh2/features", N=None)}
_SET = SETS[sys.argv[sys.argv.index("--set") + 1]] if "--set" in sys.argv else None
if _SET is None:
    raise SystemExit("usage: fresh2_derive.py --set first|fresh2")
BANK, EPDIR, SIDECAR, DUMP = Path(_SET["BANK"]), Path(_SET["EPDIR"]), Path(_SET["SIDECAR"]), Path(_SET["DUMP"])
ROOT = Path(_SET["OUT"])                                  # FRESH2 COPY: outputs go here (result/), never into the LHA0 package
N_CLIPS = _SET["N"] if _SET["N"] is not None else len(list(EPDIR.glob("*.v2ep.pt")))
DELTAS = tuple(range(-3, 4))
TOL_MATCH, TOL_OTHER = 4e-6, 1e-3
NOMINAL_DT = 0.1


def sha12(cid: str) -> str:
    return hashlib.sha256(cid.encode("utf-8")).hexdigest()[:12]


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def main() -> int:
    t0 = time.time()
    assert str(Path(tanitad.__file__).resolve()).startswith(str(TREE.resolve())), tanitad.__file__
    bk = np.load(BANK, allow_pickle=False)
    meta = json.loads(str(bk["meta_json"]))
    W = int(meta["window_W"])
    e_i, t = bk["e_i"].astype(np.int64), bk["t"].astype(np.int64)
    ws = t + W - 1
    v0_in = bk["v0_in"].astype(np.float64)
    n = len(e_i)
    jx = json.load(open(DUMP / "refcv6_extras.npz.json", encoding="utf-8"))
    idx2s12 = {int(k): str(v) for k, v in jx["clip_sha12_by_index"].items()}
    ex = np.load(DUMP / "refcv6_extras.npz")
    assert np.array_equal(ex["clip_index"].astype(np.int64), e_i) and np.array_equal(ex["ws"].astype(np.int64), ws), "bank (e_i, ws) != dump extras (clip_index, ws)"
    # ---- read every episode's metadata once ----
    files = sorted(EPDIR.glob("*.v2ep.pt"))
    by_s12, by_ep = {}, {}
    sc = CC.read_clip_clock_sidecar(str(SIDECAR)).table
    for p in files:
        poses, actions, eid16, n_stack, S, cid, H, Wd = _scan_meta(str(p))
        s12 = sha12(cid)
        sid = stable_episode_id(cid)
        g0, dt = sc[sid]
        by_s12[s12] = {"poses": poses.numpy().astype(np.float64), "dt": float(dt), "g0": float(g0), "n_stack": int(n_stack), "T": int(poses.shape[0]),
                       "codec_note": "metadata only"}
    assert len(by_s12) == N_CLIPS, (len(by_s12), N_CLIPS)
    missing = sorted(set(idx2s12.values()) - set(by_s12))
    assert not missing, f"{len(missing)} dump clips have no episode file"
    # ---- cross-check against the dump's own ep_poses (exact) ----
    chk_eq = {}
    for ci_, s12 in idx2s12.items():
        d = np.load(DUMP / "decisions" / f"ep{ci_:03d}.npz")
        chk_eq[ci_] = bool(np.array_equal(d["ep_poses"].astype(np.float64), by_s12[s12]["poses"]))
    ep_poses_all_equal = bool(all(chk_eq.values()))
    # ---- 1. anchor-frame rule ----
    maxdiff = {}
    for dlt in DELTAS:
        md = 0.0
        oob = 0
        for i in range(n):
            e = by_s12[idx2s12[int(e_i[i])]]
            k = int(ws[i]) + dlt
            if k < 0 or k >= e["T"]:
                oob += 1
                continue
            md = max(md, abs(float(e["poses"][k, 3]) - float(v0_in[i])))
        maxdiff[dlt] = {"max_abs_diff": md, "out_of_range_windows": oob}
    match = [d for d in DELTAS if maxdiff[d]["out_of_range_windows"] == 0 and maxdiff[d]["max_abs_diff"] <= TOL_MATCH]
    others_ok = all(maxdiff[d]["max_abs_diff"] > TOL_OTHER or maxdiff[d]["out_of_range_windows"] > 0 for d in DELTAS if d not in match)
    cal = {"tool": "lha0_derive.py", "prereg": "PREREG_LHA0_DIAG.md s1.2", "n_windows": int(n), "window_W": W, "ws_rule": "ws = t + W - 1 (== dump extras ws, asserted)",
           "pose_columns": ["x_east_m", "y_north_m", "yaw_rad", "v_mps"], "pose_coordinate": "_scan_meta poses = cache poses[n_stack-1:]",
           "anchor_rule_search": {f"k=ws{d:+d}": maxdiff[d] for d in DELTAS}, "tol_match": TOL_MATCH, "tol_other_min": TOL_OTHER,
           "matching_deltas": match, "decisions_ep_poses_equal_cache_poses_all": ep_poses_all_equal,
           "bank_e_i_equals_dump_clip_index": True}
    if len(match) != 1 or not others_ok or not ep_poses_all_equal:
        cal["verdict"] = "STOP: no unique exact anchor-frame rule (or ep_poses cross-check failed)"
        (ROOT / "result").mkdir(exist_ok=True)
        (ROOT / "result" / "LHA0_CALIBRATION.json").write_text(json.dumps(cal, indent=1), encoding="utf-8")
        print("ZZCALZZ STOP", json.dumps({str(k): v for k, v in maxdiff.items()}), flush=True)
        return 2
    dstar = match[0]
    cal["anchor_rule"] = f"k = ws + ({dstar}) ; bank v0_in == poses[k,3] for all {n} windows"
    cal["anchor_rule_max_abs_diff"] = maxdiff[dstar]["max_abs_diff"]
    # ---- 2. derive ----
    a0 = np.full(n, np.nan); a0_nom = np.full(n, np.nan); a0_05 = np.full(n, np.nan); a0_fwd = np.full(n, np.nan)
    psi = np.full(n, np.nan); psi_nom = np.full(n, np.nan); dtw = np.zeros(n); vk = np.zeros(n)
    s12w = np.empty(n, dtype="U12")
    drops = {"a0_k-1": 0, "a0_05_k-5": 0, "a0_fwd_k+1": 0}
    for i in range(n):
        s12 = idx2s12[int(e_i[i])]
        e = by_s12[s12]
        v, yaw, dt, T = e["poses"][:, 3], e["poses"][:, 2], e["dt"], e["T"]
        k = int(ws[i]) + dstar
        s12w[i], dtw[i], vk[i] = s12, dt, v[k]
        if k - 1 >= 0:
            a0[i] = (v[k] - v[k - 1]) / dt
            a0_nom[i] = (v[k] - v[k - 1]) / NOMINAL_DT
            psi[i] = wrap(yaw[k] - yaw[k - 1]) / dt
            psi_nom[i] = wrap(yaw[k] - yaw[k - 1]) / NOMINAL_DT
        else:
            drops["a0_k-1"] += 1
        if k - 5 >= 0:
            a0_05[i] = (v[k] - v[k - 5]) / (5.0 * dt)
        else:
            drops["a0_05_k-5"] += 1
        if k + 1 < T:
            a0_fwd[i] = (v[k + 1] - v[k]) / dt
        else:
            drops["a0_fwd_k+1"] += 1
    fin = np.isfinite(a0) & np.isfinite(a0_fwd)
    frac_differ = float((np.abs(a0_fwd[fin] - a0[fin]) > 1e-6).mean())
    dts = np.array([by_s12[s]["dt"] for s in sorted(by_s12)])
    cal.update({
        "dt_rule": "per-clip dt_s of clock_sidecar.jsonl (the cache's own clock); nominal 0.1 only as the B-nominal-dt sensitivity arm",
        "dt_s_min_med_max": [float(dts.min()), float(np.median(dts)), float(dts.max())],
        "drops": drops, "n_rows_finite_a0_psi_fwd": int(fin.sum()),
        "mutation_forward_vs_backward_a0": {"frac_rows_differ_gt_1e-6": frac_differ, "majority_required": 0.5, "pass": bool(frac_differ > 0.5)},
        "corr_a0_bwd_fwd": float(np.corrcoef(a0[fin], a0_fwd[fin])[0, 1]),
        "summary": {k: {"mean": float(np.nanmean(x)), "sd": float(np.nanstd(x)), "min": float(np.nanmin(x)), "max": float(np.nanmax(x))}
                    for k, x in (("a0", a0), ("a0_05", a0_05), ("a0_fwd", a0_fwd), ("psi_dot0", psi), ("v_k", vk))},
        "verdict": "CALIBRATED" if frac_differ > 0.5 else "INVALID: forward==backward on the majority of rows",
        "wall_s": round(time.time() - t0, 1),
    })
    (ROOT / "result").mkdir(exist_ok=True)
    (ROOT / "result" / "LHA0_CALIBRATION.json").write_text(json.dumps(cal, indent=1), encoding="utf-8")
    np.savez(ROOT / "result" / "lha0_features.npz", e_i=e_i, t=t, ws=ws, sha12=s12w, v_pose=vk, a0=a0, a0_nominal_dt=a0_nom, a0_05=a0_05, a0_fwd=a0_fwd, psi_dot0=psi,
             psi_dot0_nominal_dt=psi_nom, dt=dtw, anchor_delta=np.array([dstar]), bank_sha256=np.array([sha256_file(BANK)]))
    print(f"ZZCALZZ {cal['verdict']} rule={cal['anchor_rule']} maxdiff={cal['anchor_rule_max_abs_diff']:.3e} frac_fwd_neq_bwd={frac_differ:.4f} drops={drops}", flush=True)
    return 0 if cal["verdict"] == "CALIBRATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
