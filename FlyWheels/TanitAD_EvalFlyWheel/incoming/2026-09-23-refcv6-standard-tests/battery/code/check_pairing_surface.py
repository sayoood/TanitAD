"""MODEL-FREE: can refcv6 (416x1024 kit cache) be scored on the IDENTICAL windows and GT futures
as the banked refcv4b / refcv5-v2 dumps (256x640 B1 eval cache)?  No model is loaded.

Checks, each written to raw/pairing_surface.json:
  C1  clip sets: kit-139 vs the 141-clip 256x640 cache vs the dumps' manifests (by clip id).
  C2  per common clip: provider `poses` [T,4] and `actions` [T,2] -- equal across the two caches?
  C3  per banked window (clip, ws): is `t = ws - (W-1)` a window of the kit dataset's index,
      and does the GT recomputed from the KIT poses through the trainer's own
      `refb_labels.waypoint_targets` equal the banked `g` exactly? (the positive control)
  C4  the three banked baseline dumps agree with EACH OTHER on (clip, ws, g) (model-free arms
      `ha`, `ha0`, `ha0_ext` bit-identical) -- the surface is one surface.
Raw clip ids never leave this process: the JSON carries sha12 only.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refcv6_loader as L  # noqa: E402

L.bootstrap()
import torch  # noqa: E402
import refb_labels  # noqa: E402
from tanitad.data.v2_dataset import build_v2_providers, load_or_build_manifest  # noqa: E402

KIT_EVAL = str(L.KIT / "data/refcv6-b1-416x1024-eval139")
B1_256 = "D:/Projects/TanitAD-artifacts/refcv5cmp/data/eval"
DUMPS = {
    "refcv4b": "C:/Users/Admin/refcv5cmp/out/refcv4b_devbox_dump",
    "refcv5v2_s0": "C:/Users/Admin/refcv5cmp/out/refcv5-v2_dump",
    "refcv5v2_s1": "C:/Users/Admin/refcv5cmp/out/refcv5-v2-seed1_dump",
}
HORIZONS = (5, 10, 15, 20, 30, 40, 50, 60)
W = 8
MAX_H_EXT = 60


def providers(d):
    man = load_or_build_manifest(d, verbose=False)
    eps = build_v2_providers([d], lru_size=2, verbose=False)
    cids = [str(c) for c in man["clip_id"]]
    assert len(cids) == len(eps)
    return dict(zip(cids, eps)), man


def main():
    t0 = time.time()
    out: dict = {"what": __doc__.strip().splitlines()[0], "kit": KIT_EVAL, "b1_256x640": B1_256,
                 "dumps": DUMPS}
    kit, kman = providers(KIT_EVAL)
    b1, bman = providers(B1_256)
    out["C1"] = {"n_kit": len(kit), "n_b1_256": len(b1), "n_common": len(set(kit) & set(b1)),
                 "only_kit": sorted(L.sha12(c) for c in set(kit) - set(b1)),
                 "only_b1_256": sorted(L.sha12(c) for c in set(b1) - set(kit)),
                 "n_stack_kit": sorted(set(int(x) for x in kman["n_stack"])),
                 "n_stack_b1": sorted(set(int(x) for x in bman["n_stack"]))}
    # ---- C2 poses / actions equality ------------------------------------------------- #
    c2 = {"n_equal_poses": 0, "n_equal_actions": 0, "n_len_differ": 0, "max_abs_pose_diff": 0.0,
          "max_abs_action_diff": 0.0, "differ": []}
    common = sorted(set(kit) & set(b1))
    for c in common:
        pk, pb = kit[c].poses, b1[c].poses
        ak, ab = kit[c].actions, b1[c].actions
        if pk.shape != pb.shape:
            c2["n_len_differ"] += 1
            c2["differ"].append({"sha12": L.sha12(c), "poses_kit": list(pk.shape),
                                 "poses_b1": list(pb.shape)})
            continue
        dp = float((pk.double() - pb.double()).abs().max())
        da = float((ak.double() - ab.double()).abs().max())
        c2["max_abs_pose_diff"] = max(c2["max_abs_pose_diff"], dp)
        c2["max_abs_action_diff"] = max(c2["max_abs_action_diff"], da)
        c2["n_equal_poses"] += int(dp == 0.0)
        c2["n_equal_actions"] += int(da == 0.0)
        if dp != 0.0 or da != 0.0:
            c2["differ"].append({"sha12": L.sha12(c), "max_pose": dp, "max_action": da})
    out["C2"] = c2
    # ---- C3/C4 banked window grid ------------------------------------------------------ #
    per_dump = {}
    for tag, d in DUMPS.items():
        man = json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))
        rows = {}
        for e in man["episodes"]:
            z = np.load(os.path.join(d, f"ep{e['file_index']:03d}.npz"))
            rows[e["clip_id"]] = {"ws": z["ws"].astype(np.int64), "g": z["g"],
                                  "ha": z["ha"], "ha0": z["ha0"], "ha0_ext": z["ha0_ext"]}
        per_dump[tag] = {"rows": rows, "grid": man["grid"], "stride": man["grid"].get("window_stride")}
    c4 = {}
    ref = per_dump["refcv4b"]["rows"]
    for tag in ("refcv5v2_s0", "refcv5v2_s1"):
        oth = per_dump[tag]["rows"]
        same_clips = set(ref) == set(oth)
        n_ws_eq = sum(1 for c in ref if c in oth and np.array_equal(ref[c]["ws"], oth[c]["ws"]))
        g_eq = all(np.array_equal(ref[c]["g"], oth[c]["g"]) for c in ref if c in oth)
        mf_eq = all(np.array_equal(ref[c][k], oth[c][k]) for c in ref if c in oth
                    for k in ("ha", "ha0", "ha0_ext"))
        c4[tag] = {"same_clip_set": same_clips, "n_clips_ws_equal": n_ws_eq, "g_bit_equal": g_eq,
                   "model_free_arms_bit_equal": mf_eq}
    out["C4_vs_refcv4b"] = c4
    # the kit window index, by the trainer's own window contract (V3Dataset.index)
    c3 = {"n_windows_banked_common": 0, "n_ws_not_a_kit_window": 0, "n_g_exact": 0,
          "max_abs_g_diff": 0.0, "n_future_invalid": 0}
    slots = [0, 1, 2, 3]
    hz = torch.tensor(HORIZONS)
    for c in common:
        if c not in ref:
            continue
        ep = kit[c]
        T = int(ep.poses.shape[0])
        ws = ref[c]["ws"]
        g_b = ref[c]["g"]
        for j, w0 in enumerate(ws):
            c3["n_windows_banked_common"] += 1
            t = int(w0) - (W - 1)
            # V3Dataset windows: t in [0, T - W - max_horizon]  (max_horizon 20)
            if t < 0 or t > T - W - 20:
                c3["n_ws_not_a_kit_window"] += 1
                continue
            pose_last = ep.poses[t + W - 1].float()[None]
            idx = torch.arange(t + W, t + W + MAX_H_EXT)
            if int(idx[max(HORIZONS[s] for s in slots) - 1]) > T - 1:
                c3["n_future_invalid"] += 1
            fut = ep.poses[idx.clamp(max=T - 1)].float()[None]
            g = refb_labels.waypoint_targets(pose_last, fut, HORIZONS)[:, slots].numpy()[0]
            d = float(np.abs(g - g_b[j]).max())
            c3["max_abs_g_diff"] = max(c3["max_abs_g_diff"], d)
            c3["n_g_exact"] += int(d == 0.0)
    out["C3"] = c3
    out["wall_s"] = round(time.time() - t0, 1)
    dst = HERE.parent / "raw" / "pairing_surface.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dst, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "C2"}, indent=1)[:4000])
    print("C2 summary", {k: v for k, v in c2.items() if k != "differ"}, "n_differ", len(c2["differ"]))


if __name__ == "__main__":
    main()
