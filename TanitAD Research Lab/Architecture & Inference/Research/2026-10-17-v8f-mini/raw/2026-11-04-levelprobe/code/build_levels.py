#!/usr/bin/env python3
"""build_levels.py -- the level files of PREREG_LEVELPROBE2.md (B1, B2): L_T0 / MEAN6 / FUT1 / FUT3 from the LOGGED measured speed `p10f[:, 0]` of each eval-141 window's clip (one step = 0.2 s).
Writes F:/TanitAD-scratch/levelprobe_scratch/levels/L_*.npz (clip_index, ws, level) and raw/levels_meta.json; appends the sha256 lines to FROZEN_SHA2.txt.   python -I code/build_levels.py"""
import hashlib
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
TREE = "F:/TanitAD-scratch/levelprobe_scratch/tree_dag"
sys.path.insert(0, f"{TREE}/stack")
sys.path.insert(0, f"{TREE}/taniteval")
import numpy as np  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402

ART = "E:/Projects/TanitAD-artifacts"
PK = f"{ART}/dagland/levelprobe"
OUTD = "F:/TanitAD-scratch/levelprobe_scratch/levels"
DATA = f"{ART}/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
fz = open(f"{PK}/FROZEN_SHA2.txt", encoding="utf-8").read()
assert fz.startswith("PREREG_LEVELPROBE2.md sha256 " + hashlib.sha256(open(f"{PK}/PREREG_LEVELPROBE2.md", "rb").read()).hexdigest()), "REFUSED: card 2 does not hash to FROZEN_SHA2.txt"
assert abs(D.DT - 0.2) < 1e-12 and abs(D.DT_NATIVE * 2 - D.DT) < 1e-12
tracks = D.load_native_tracks(f"{ART}/sfresh_inputs/inputs/native_tracks.pt")
J = D.load_join(f"{DATA}/join_eval141.npz")
store = D.ClipStore(f"{ART}/sfresh_inputs/cache", "eval", lru=32, grid="pool4x10", kin_mode="yaw", tracks=tracks)
wins = D.eval_grid_windows(f"{DATA}/eval_grid_windows.json", J)
assert len(wins) == 2399
ci_a, ws_a = np.array([w[0] for w in wins], np.int64), np.array([w[1] for w in wins], np.int64)
z = np.load(f"{ART}/dagland/kappa_t1/raw/arrays/SREP_DAGOACC_s0.npz", allow_pickle=True)
assert np.array_equal(ci_a, z["clip_index"]) and np.array_equal(ws_a, z["ws"]), "grid order differs from the rig's window order"
L = {k: np.zeros(len(wins)) for k in ("T0", "MEAN6", "FUT1", "FUT3")}
vv = 0.0
by = {}
for i, (ci, t0) in enumerate(zip(ci_a, ws_a)):
    by.setdefault(int(ci), []).append((i, int(t0)))
for ci, items in by.items():
    base = store.get(str(J["clip"][ci]))
    sp = np.asarray(base["p10f"], np.float64)[:, 0]
    for i, t0 in items:
        assert bool(base["gt_ok"][t0]) and t0 + 30 < len(sp), (ci, t0, len(sp))
        L["T0"][i], L["FUT1"][i], L["FUT3"][i], L["MEAN6"][i] = sp[t0], sp[t0 + 5], sp[t0 + 15], sp[t0 + 1:t0 + 31].mean()
        vv = max(vv, float(abs(sp[t0] - float(np.asarray(base["v"])[t0]))) if np.isfinite(np.asarray(base["v"])[t0]) else 0.0)
for k, v in L.items():
    assert np.isfinite(v).all() and (v >= 0).all(), k
os.makedirs(OUTD, exist_ok=True)
meta = {"n_windows": len(wins), "n_episodes": int(len(np.unique(ci_a))), "step_s": D.DT, "max_abs_p10f0_minus_base_v_at_t0": vv, "L_T0_equals_rig_v0_max_abs": float(np.abs(L["T0"] - z["v0"]).max()), "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "files": {}}
lines = []
for k, v in L.items():
    p = f"{OUTD}/L_{k}.npz"
    np.savez(p, clip_index=ci_a, ws=ws_a, level=v)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    meta["files"][k] = {"path": p, "sha256": h, "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
    lines.append(f"level_file {p} sha256 {h}\n")
meta["means_of_differences_vs_T0"] = {k: float((L[k] - L["T0"]).mean()) for k in ("MEAN6", "FUT1", "FUT3")}
json.dump(meta, open(f"{PK}/raw/levels_meta.json", "w", encoding="utf-8", newline="\n"), indent=1)
open(f"{PK}/FROZEN_SHA2.txt", "a", encoding="utf-8", newline="\n").write(f"levels built {meta['built']} by code/build_levels.py (sha256 {hashlib.sha256(open(__file__, 'rb').read()).hexdigest()}); before any rollout of this card\n" + "".join(lines))
print(json.dumps(meta, indent=1))
