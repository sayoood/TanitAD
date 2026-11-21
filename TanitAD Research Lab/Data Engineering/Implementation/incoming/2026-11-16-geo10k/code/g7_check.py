"""GEO10K step 7: REACHABILITY + coverage of the built labels from the trainer's own loaders (v8f_mini_data.build_label_join,
load_speed_sidecar, lead_label_arrays, load_native_tracks), and a side-by-side with the existing corpus's release."""
import json, sys
import numpy as np
sys.path.insert(0, "/root/geo10k/tree/stack")
from tanitad.data import v8f_mini_data as D
from tanitad.data.v9_labels import load_v9_release
OUT = "/root/geo10k/out"


def share(J, rel_has=True):
    has = J["has"]
    nav = J["nav"]
    tok = np.argmax(nav[:, :3], 1)
    n = int(has.sum())
    d = {"n_rows": int(len(has)), "frac_has_v9_row": float(has.mean()),
         "nav_FOLLOW": float((tok[has] == 0).mean()), "nav_TURN_L": float((tok[has] == 1).mean()), "nav_TURN_R": float((tok[has] == 2).mean()),
         "nav_args_valid": float(nav[has, 4].mean()), "rc_A30_ok": float(J["rc_ok"][has, 0].mean()), "rc_A80_ok": float(J["rc_ok"][has, 2].mean()),
         "lat_cls_labelled": float((J["lat_cls"][has] != D.IGNORE).mean()), "lon_cls_labelled": float((J["lon_cls"][has] != D.IGNORE).mean()),
         "goal_w_any": float((J["goal_w"][has] != 0).mean()), "lead_valid": float((J["lead_valid"][has] == 1).mean()),
         "lead_gap_finite": float(np.isfinite(J["lead_gap"][has]).mean()), "limit_ok_clips": float(J["limit_ok"].mean()),
         "cons_finite": float(np.isfinite(J["cons"][has]).mean())}
    return d


rep = {}
# ---- the 10k
man = [json.loads(l) for l in open("/root/datascale/corpus_manifest_10k.jsonl")]
nf = {m["clip"]: int(m["n_frames"]) for m in man}
sel = [l.strip() for l in open("/root/datascale/selection/clips_10k.txt") if l.strip()]
clips_nf = [(c, nf[c]) for c in sel]
rel = load_v9_release(f"{OUT}/v9_labels_new.npz")
speed = D.load_speed_sidecar(f"{OUT}/speed_sidecar_new10k.jsonl") if len(sys.argv) > 1 and sys.argv[1] == "speed" else None
J = D.build_label_join(clips_nf, rel, lead_npz=f"{OUT}/lead_new10k.npz", speed=speed)
rep["new10k"] = share(J)
rep["new10k"]["n_clips_missing_labels"] = int(J["n_clips_missing_labels"])
rep["new10k"]["n_clips"] = len(clips_nf)
LL = D.lead_label_arrays(J, f"{OUT}/lead_new10k.npz")
rep["new10k"]["lead_head"] = {k: LL[k] for k in ("n_rows", "n_has", "n_near", "n_clips", "n_clips_with_labels")}
tr = D.load_native_tracks(f"{OUT}/native_tracks_new10k.pt")
rep["new10k"]["native_tracks"] = {"n": len(tr), "all_selected_present": all(c in tr for c in sel)}
# the cache's own pose_grid must equal the native track on even frames (what ClipStore enforces): n of rows checked, expected rows count
bad = [c for c, n in clips_nf if (tr[c].shape[0] + 1) // 2 != n and tr[c].shape[0] // 2 + 1 != n]
rep["new10k"]["n_clips_where_cache_n_frames_not_ceil_half_native_rows"] = len(bad)
# ---- existing corpus through the SAME code, with the 4,369 release + existing lead
man0 = D.read_manifest("/root/data/cache_pod", "train")
rel0 = load_v9_release("/root/geo10k/ref/tanitad-refcv8-v9-labels/release/v9_labels_train.npz")
J0 = D.build_label_join(man0, rel0, lead_npz="/root/geo10k/ref/tanitad-refcv8-v9-labels/inputs/lead_train.npz")
rep["existing_4572_through_same_code"] = share(J0)
rep["existing_4572_through_same_code"]["n_clips_missing_labels"] = int(J0["n_clips_missing_labels"])
rep["existing_4572_through_same_code"]["n_clips"] = len(man0)
json.dump(rep, open(f"{OUT}/reach_and_coverage.json", "w"), indent=1)
print(json.dumps(rep, indent=1)[:3500])
print("ZZ GEO10K-G7 DONE ZZ")
