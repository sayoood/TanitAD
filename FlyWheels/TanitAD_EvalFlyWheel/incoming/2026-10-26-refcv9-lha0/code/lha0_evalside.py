"""lha0_evalside.py -- METADATA-ONLY check for the LH-A0 arm draft: does the EVAL139 side carry the pose fields a0 / psi_dot0 need, and can they be derived with the SAME rule?

  * the eval dump's decisions/ep<c>.npz['ep_poses'] [199,4] = (x, y, yaw, v), the same slice `_scan_meta` returns (verified exact for the 300 fresh episodes in lha0_derive.py);
  * the anchor rule k = ws on the EVAL bank (4,754 windows): max |v0_in - ep_poses[k,3]|;
  * per-clip dt from E:/refcv8_eval_kit/data/refcv8_clip_clock_sidecar_ts33.jsonl (sid = stable_episode_id(clip_id), via the battery windows json): coverage of the 139 eval clips;
  * a0 / psi_dot0 / 0.5 s / forward-contrast derivation for the 4,754 eval windows with the same formulas; drops; mutation fraction; a0 / psi summary next to the fresh summary.
No frames are read.  Thor is not touched.  Writes result/LHA0_EVAL_SIDE_CHECK.json and result/lha0_features_eval139.npz (sha12 only).  NEW file (LHA0 agent, 2026-10-26).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TREE = Path("C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58")
sys.path.insert(0, str(TREE / "stack"))
from tanitad.data import clip_clock as CC  # noqa: E402
from tanitad.data.v2_dataset import stable_episode_id  # noqa: E402

ED = Path("C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf/dump_s0")
BANK = Path("C:/Users/Admin/refcv8_pf/refcv9_l1/bank/ztac_eval_s0.npz")
SIDECAR = Path("E:/refcv8_eval_kit/data/refcv8_clip_clock_sidecar_ts33.jsonl")
WINDOWS = Path("C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf/windows_s0.json")


def sha12(c):
    return hashlib.sha256(c.encode("utf-8")).hexdigest()[:12]


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def main() -> int:
    bk = np.load(BANK, allow_pickle=False)
    W = int(json.loads(str(bk["meta_json"]))["window_W"])
    e_i, ws = bk["e_i"].astype(np.int64), bk["t"].astype(np.int64) + W - 1
    v0 = bk["v0_in"].astype(np.float64)
    jx = json.load(open(ED / "refcv6_extras.npz.json", encoding="utf-8"))
    idx2s12 = {int(k): str(v) for k, v in jx["clip_sha12_by_index"].items()}
    wins = json.load(open(WINDOWS, encoding="utf-8"))
    s12_sid = {sha12(c): stable_episode_id(c) for c in wins}
    table = CC.read_clip_clock_sidecar(str(SIDECAR)).table
    poses = {int(os.path.basename(f)[2:5]): np.load(f)["ep_poses"].astype(np.float64) for f in sorted(glob.glob(str(ED / "decisions" / "ep*.npz")))}
    n_clip = len(idx2s12)
    cov = {c: (s12_sid.get(s) in table) for c, s in idx2s12.items()}
    md = {dl: max(abs(poses[int(c)][int(w) + dl, 3] - x) for c, w, x in zip(e_i, ws, v0)) for dl in (-1, 0, 1)}
    n = len(e_i)
    a0, a05, afw, psi = (np.full(n, np.nan) for _ in range(4))
    dts = np.full(n, np.nan)
    s12w = np.empty(n, dtype="U12")
    for i in range(n):
        c, k = int(e_i[i]), int(ws[i])
        s = idx2s12[c]
        s12w[i] = s
        sid = s12_sid[s]
        if sid not in table:
            continue
        dt = float(table[sid][1])
        dts[i] = dt
        p = poses[c]
        a0[i] = (p[k, 3] - p[k - 1, 3]) / dt
        psi[i] = wrap(p[k, 2] - p[k - 1, 2]) / dt
        a05[i] = (p[k, 3] - p[k - 5, 3]) / (5 * dt)
        if k + 1 < len(p):
            afw[i] = (p[k + 1, 3] - p[k, 3]) / dt
    fin = np.isfinite(a0) & np.isfinite(psi)
    fin2 = fin & np.isfinite(afw)
    cal_f = json.load(open(ROOT / "result" / "LHA0_CALIBRATION.json", encoding="utf-8"))["summary"]
    summ = lambda x: {"mean": round(float(np.nanmean(x)), 4), "sd": round(float(np.nanstd(x)), 4), "min": round(float(np.nanmin(x)), 3), "max": round(float(np.nanmax(x)), 3)}
    out = {"tool": "lha0_evalside.py", "status": "metadata only (eval dump decisions ep_poses + clip-clock sidecar); no frames; Thor not touched",
           "eval139_episode_caches_on_dev_box": "NO: E:/refcv8_eval_kit carries sidecars / manifests' md5 / windows / v9 labels / ckpt only; the *.v2ep.pt live at /home/nvidia/data/refcv6-b1-416x1024-eval139 on Thor (ckpt config argv). "
                                                "The same (x, y, yaw, v) fields ARE on the dev box as the eval dump's decisions/ep*.npz ep_poses [199, 4] (== _scan_meta poses[n_stack-1:], proven exact for the 300 fresh episodes).",
           "n_eval_clips": n_clip, "ep_poses_files": len(poses), "ep_poses_shape_set": sorted({tuple(p.shape) for p in poses.values()}),
           "clock_sidecar_coverage": {"clips_with_dt": int(sum(cov.values())), "clips_without_dt": int(n_clip - sum(cov.values())), "windows_without_dt": int((~np.isfinite(dts)).sum())},
           "dt_s_min_med_max": [round(float(np.nanmin(dts)), 6), round(float(np.nanmedian(dts)), 6), round(float(np.nanmax(dts)), 6)],
           "anchor_rule_on_eval139": {f"k=ws{dl:+d}": {"max_abs_diff": float(md[dl])} for dl in md}, "n_windows": int(n), "min_ws": int(ws.min()),
           "rows_with_finite_a0_psi": int(fin.sum()), "rows_dropped": int((~fin).sum()),
           "mutation_fwd_neq_bwd_frac": float((np.abs(afw[fin2] - a0[fin2]) > 1e-6).mean()),
           "eval_summary": {"a0": summ(a0), "psi_dot0": summ(psi), "a0_05": summ(a05)}, "fresh_summary_for_comparison": {k: cal_f[k] for k in ("a0", "psi_dot0")}}
    (ROOT / "result" / "LHA0_EVAL_SIDE_CHECK.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    np.savez(ROOT / "result" / "lha0_features_eval139.npz", e_i=e_i, ws=ws, sha12=s12w, a0=a0, a0_05=a05, a0_fwd=afw, psi_dot0=psi, dt=dts)
    print("ZZEVALSIDEZZ", json.dumps({k: out[k] for k in ("clock_sidecar_coverage", "anchor_rule_on_eval139", "rows_dropped", "mutation_fwd_neq_bwd_frac", "eval_summary")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
