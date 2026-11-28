"""The join check (label times and geometry) -- 5 hand-checked eval windows + the whole-grid speed/clock check.

For each of 5 windows (clip i, cache step t, raw frame k = 2t):
  (a) CLOCK  : the release's stored ``now_s`` equals ``grid_start_s + k * dt_s`` of the clip table (1e-9) -- the join key.
  (b) BUILDER: ``build_v9_labels.rc_fields`` re-run from the local ego-log parquet at that ``now_s`` reproduces the
               release's RC-A {30, 50, 80} (x, y, psi) to 1e-6 -- i.e. the geometry in the file IS the builder's.
  (c) POSES  : the ego-frame point at 30 / 50 / 80 m of the HUMAN path read from the v2ep poses (unsmoothed) lies within
               a few metres of RC-A (the route checkpoint is the sigma = 8 m smoothed path) -- an independent geometry read.
  (d) SPEED  : release ``v_now_ms`` vs the v2ep pose speed.
Whole grid: |v_now_ms - pose speed| over every window of the 139 labelled episodes.
Writes raw/join_check.json (these numbers are pinned in stack/tests/test_v8f_mini.py).
"""
import json
import os
import sys

import numpy as np

REPO = "E:/Projects/TanitAD"
sys.path.insert(0, f"{REPO}/stack")
sys.path.insert(0, f"{REPO}/stack/scripts")
sys.path.insert(0, f"{REPO}/taniteval")
from tanitad.data import v8f_mini_data as D            # noqa: E402
from tanitad.data.v9_labels import load_v9_release     # noqa: E402
import build_v9_labels as B                            # noqa: E402

PKG = f"{REPO}/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
REL = "C:/Users/Admin/qland/work/refcv8_hf_v9/release/v9_labels_eval139.npz"
EGO = "C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo/{}.parquet"

rel = load_v9_release(REL, expect_md5="6b5c7f207cffc3b7eb3cd527fd433599")
grid = json.load(open(f"{PKG}/raw/eval_grid_windows.json"))
z = np.load(f"{PKG}/raw/eval141_kin.npz")
off = np.concatenate([[0], np.cumsum(z["n"])])
zr = np.load(REL)
cl = {k[6:]: zr[k] for k in zr.files if k.startswith("clip__")}
sid_row = {int(s): i for i, s in enumerate(cl["sid"])}

picks = [(3, 35), (40, 47), (77, 59), (100, 23), (130, 39)]
out = {"release": REL, "windows": [], "whole_grid": {}}
for ci, t in picks:
    cid = grid["order"][ci]
    sid = D.stable_episode_id(cid)
    if sid not in sid_row:
        out["windows"].append({"clip": cid, "t": t, "skipped": "clip not in v9 eval139"})
        continue
    k = 2 * t
    r0, n, k0 = rel.clips[sid]
    row = r0 + (k - k0)
    R = rel.rows
    g0, dt = float(cl["grid_start_s"][sid_row[sid]]), float(cl["dt_s"][sid_row[sid]])
    now_rel = float(R["now_s"][row])
    w = {"clip": cid, "t": t, "k": k, "now_s_release": now_rel, "now_s_clock": g0 + k * dt}
    w["clock_abs_diff"] = abs(now_rel - w["now_s_clock"])
    # (b) the builder, from the ego log
    log = B.load_log(EGO.format(cid))
    G = B.clip_geo(log)
    rc = B.rc_fields(G.tr, G.log, G.paths, G.turns, now_rel, G.s_nat)
    w["rc_release"] = {v: [float(R[f"rc_{v}_{c}"][row]) for c in ("x", "y", "psi")] for v in D.RC_VARIANTS}
    w["rc_builder"] = {v: [float(rc[f"rc_{v}_{c}"]) for c in ("x", "y", "psi")] for v in D.RC_VARIANTS}
    w["rc_builder_vs_release_maxabs"] = max(abs(a - b) for v in D.RC_VARIANTS
                                           for a, b in zip(w["rc_release"][v], w["rc_builder"][v]))
    # (c) the human path from v2ep poses, ego frame at frame k
    P = z["poses"][off[ci]:off[ci + 1]].astype(np.float64)
    d = P[k:, :2] - P[k, :2]
    c_, s_ = np.cos(-P[k, 2]), np.sin(-P[k, 2])
    ego = np.stack([d[:, 0] * c_ - d[:, 1] * s_, d[:, 0] * s_ + d[:, 1] * c_], 1)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ego, axis=0), axis=1))])
    pts = {}
    for v, A in zip(D.RC_VARIANTS, D.CKPT_ARCS):
        if arc[-1] >= A:
            j = int(np.searchsorted(arc, A))
            pts[v] = [float(ego[j, 0]), float(ego[j, 1])]
    w["human_path_points"] = pts
    w["rc_vs_human_path_m"] = {v: float(np.hypot(w["rc_release"][v][0] - p[0], w["rc_release"][v][1] - p[1]))
                               for v, p in pts.items() if int(R[f"rc_{v}_valid"][row]) == 1}
    # (d) speed
    w["v_now_ms"] = float(R["v_now_ms"][row])
    w["pose_speed"] = float(P[k, 3])
    out["windows"].append(w)

# whole grid
diffs = []
for ci, cid in enumerate(grid["order"]):
    sid = D.stable_episode_id(cid)
    if sid not in rel.clips:
        continue
    r0, n, k0 = rel.clips[sid]
    P = z["poses"][off[ci]:off[ci + 1]].astype(np.float64)
    for t in grid["windows"][cid]:
        diffs.append(float(rel.rows["v_now_ms"][r0 + 2 * t - k0]) - P[2 * t, 3])
diffs = np.array(diffs)
out["whole_grid"] = {"n_windows": int(len(diffs)), "v_now_minus_pose_speed": {
    "mean": float(diffs.mean()), "abs_max": float(np.abs(diffs).max()), "abs_p99": float(np.percentile(np.abs(diffs), 99))}}
json.dump(out, open(f"{PKG}/raw/join_check.json", "w"), indent=1)
for w in out["windows"]:
    print({k: w[k] for k in w if k in ("clip", "t", "clock_abs_diff", "rc_builder_vs_release_maxabs", "rc_vs_human_path_m")})
print(out["whole_grid"])
