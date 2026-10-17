"""The label-join report: exact coverage of the v9 releases over the refav1 cache grid (train 4,572 / eval 141).

Reads only local files (the v2 manifests, the v9 releases, the speed sidecars). Writes raw/label_join_report.json.
"""
import json
import sys

import numpy as np
import torch

REPO = "E:/Projects/TanitAD"
sys.path.insert(0, f"{REPO}/stack")
sys.path.insert(0, f"{REPO}/taniteval")
from tanitad.data import v8f_mini_data as D          # noqa: E402
from tanitad.data.v9_labels import load_v9_release   # noqa: E402

PKG = f"{REPO}/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
M4572 = "C:/Users/Admin/tanitad-data/add203/in/train4572_v2manifest.pt"
REL = {"f63ece41 (4,369-clip release)": ("C:/Users/Admin/qland/work/refcv8_hf_v9/release/v9_labels_train.npz",
                                          "f63ece410b725febb8a5242cf2b01d3c", "D:/Projects/TanitAD-artifacts/v9labels/inputs/lead_train.npz"),
       "83e30bd3 (4,572-clip release)": ("C:/Users/Admin/qland/hold_v9train4572/work/release_train4572/v9_labels_train.npz",
                                          "83e30bd3791f0767707eade5f4671759",
                                          "C:/Users/Admin/qland/hold_v9train4572/work/out/lead_train4572.npz")}
SPEED_TR = "C:/Users/Admin/qland/landed_hold_datacov/raw/refcv6_speed_max_v8_train.jsonl"
SPEED_EV = "C:/Users/Admin/qland/landed_hold_datacov/raw/refcv6_speed_max_v8_eval.jsonl"

m = torch.load(M4572, weights_only=False)
clips = [(str(c), (len(p) + 2 + 1) // 2) for c, p in zip(m["clip_id"], m["poses"])]       # +2: manifest row i = frame i + 2
grid = json.load(open(f"{PKG}/raw/eval_grid_windows.json"))
out = {"train_clips": len(clips), "frames_per_clip_cache_steps": [int(min(n for _, n in clips)), int(max(n for _, n in clips))]}
out["train_eval_disjoint"] = D.assert_disjoint([c for c, _ in clips], grid["order"])
spd = D.load_speed_sidecar(SPEED_TR)
out["speed_sidecar_train"] = {"rows": len(spd), "clips_in_cache_with_row": int(sum(c in spd for c, _ in clips)),
                              "valid": int(sum(v[2] for v in spd.values()))}
out["speed_sidecar_eval"] = {"rows": len(D.load_speed_sidecar(SPEED_EV)),
                             "grid_clips_with_row": int(sum(c in D.load_speed_sidecar(SPEED_EV) for c in grid["order"]))}
out["releases"] = {}
n_win_all = sum(max(n - D.HORIZON - 3, 0) for _, n in clips)
out["train_windows_6s_complete_total"] = int(n_win_all)
for name, (npz, md5, lead) in REL.items():
    rel = load_v9_release(npz, expect_md5=md5)
    J = D.build_label_join(clips, rel, lead_npz=lead, speed=spd)
    rep = D.join_report(J)
    cov_w = 0
    for ci, (c, n) in enumerate(clips):
        o = int(J["off"][ci])
        t = np.arange(3, n - D.HORIZON)
        cov_w += int(J["has"][o + t].sum())
    R = rel.rows
    tok = np.asarray(R["nav_token"]).astype(int)
    dy = np.abs(np.asarray(R["nav_dyaw_next_deg"], float))
    valid = np.asarray(R["nav_args_valid"], float) == 1
    turn = (tok > 0) & valid & np.isfinite(dy)
    out["releases"][name] = {
        "md5": md5, "release_clips": len(rel.clips), "cache_clips_with_rows": rep["n_clips_with_v9_rows"],
        "cache_clips_without_rows": len(rep["n_clips_without"]), "train_windows_with_row": cov_w,
        "train_window_coverage": cov_w / n_win_all, "rc_ok_rate_A30_A50_A80": rep["rc_ok_rate"],
        "goal_base_rate_labelled_steps": rep["goal_base_rate"],
        "nav_turn_dyaw_deg_abs": {"n": int(turn.sum()), "median": float(np.median(dy[turn])),
                                  "p10": float(np.percentile(dy[turn], 10)), "p90": float(np.percentile(dy[turn], 90))},
        "lead_rows_in_join": int(np.isfinite(J["lead_gap"]).sum())}
out["eval_overlap"] = {"grid_episodes": 141, "grid_windows": int(sum(len(v) for v in grid["windows"].values()))}
rel_e = load_v9_release("C:/Users/Admin/qland/work/refcv8_hf_v9/release/v9_labels_eval139.npz",
                        expect_md5="6b5c7f207cffc3b7eb3cd527fd433599")
have = [D.stable_episode_id(c) in rel_e.clips for c in grid["order"]]
out["eval_overlap"].update({"episodes_with_v9_rows": int(sum(have)), "episodes_without": [c for c, h in zip(grid["order"], have) if not h],
                            "windows_with_v9_rows": int(sum(len(grid["windows"][c]) for c, h in zip(grid["order"], have) if h)),
                            "windows_without": int(sum(len(grid["windows"][c]) for c, h in zip(grid["order"], have) if not h)),
                            "v9_eval_episodes_not_in_grid": 0})
json.dump(out, open(f"{PKG}/raw/label_join_report.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "releases"}, indent=1))
for n, r in out["releases"].items():
    print(n, {k: v for k, v in r.items() if k != "goal_base_rate_labelled_steps"})
