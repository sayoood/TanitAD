"""Post-eval analysis for flagship-v4-fromscratch @ step 15000.

(a) lateral/longitudinal decomposition (taniteval.lateral, ego frame:
    axis0 = along-track, axis1 = cross-track)
(b) PAIRED episode-cluster bootstrap vs deployed v1 (flagship-30k).

ALIGNMENT: the two window dumps label episodes differently (v4 stores the raw
episode_id, v1 used sequential 0..39), so a naive label-equality test wrongly
rejects the pairing. Admissibility is established on the DATA instead: gt and cv
bit-identical, speed max-dev 0.0, identical contiguous block sizes, and a clean
v1->v4 label bijection => the same windows in the same order and the same
episode partition. The v1 grouping is then used for both arms.
Every interval names its estimator; overlapping_holdout_se is never used.
"""
import itertools, json, sys
import torch

sys.path.insert(0, "/root/taniteval")
from taniteval import ci as _ci          # noqa: E402
from taniteval import lateral as _lat    # noqa: E402

V4W = "/root/v4eval/results/windows_flagship-v4-fromscratch-15k.pt"
V1W = "/root/taniteval/results/windows_flagship-30k.pt"
OUT = "/root/v4eval/results/flagship-v4-fromscratch-15k_lateral_and_paired.json"

w4 = torch.load(V4W, map_location="cpu", weights_only=False)
w1 = torch.load(V1W, map_location="cpu", weights_only=False)

e4 = [str(x) for x in w4["eid"]]
e1 = [str(x) for x in w1["eid"]]

def blocks(e):
    return [len(list(g)) for _k, g in itertools.groupby(e)]

gt_dev = float((w4["gt"].float() - w1["gt"].float()).abs().max())
cv_dev = float((w4["cv"].float() - w1["cv"].float()).abs().max())
spd_dev = float((w4["speed"].float() - w1["speed"].float()).abs().max())
bij = {}
bij_ok = True
for a, b in zip(e1, e4):
    if bij.setdefault(a, b) != b:
        bij_ok = False
same_blocks = blocks(e4) == blocks(e1)
aligned = (gt_dev == 0.0 and cv_dev == 0.0 and spd_dev == 0.0
           and same_blocks and bij_ok
           and w4.get("wp_steps") == w1.get("wp_steps"))

EID = e1   # identical partition; one canonical grouping for both arms

def ade(w):
    return (w["pred"].float() - w["gt"].float()).norm(dim=-1).mean(1).numpy()

a4, a1 = ade(w4), ade(w1)

res = {
    "arm": "flagship-v4-fromscratch @ step 15000 (MID-TRAINING, 15k of 30k)",
    "reference": "flagship-30k = deployed v1, step 29999 (COMPLETE)",
    "evidence_class": "MEASURED (ours; artifacts = windows_*.pt on tanitad-eval)",
    "n_windows": int(len(a4)), "n_episodes": int(len(set(EID))),
    "alignment_check": {
        "gt_max_abs_deviation_m": gt_dev,
        "cv_max_abs_deviation_m": cv_dev,
        "speed_max_abs_deviation_mps": spd_dev,
        "wp_steps_identical": w4.get("wp_steps") == w1.get("wp_steps"),
        "episode_block_sizes_identical": same_blocks,
        "v1_to_v4_label_bijection": bij_ok,
        "eid_label_strings_identical": e4 == e1,
        "paired_test_admissible": bool(aligned),
        "note": ("label strings DIFFER (v4 stores the raw episode_id, v1 used "
                 "sequential 0..39) but the underlying windows and episode "
                 "partition are provably identical, so the pairing is valid. "
                 "Verified on the data, not assumed."),
    },
    "ade_0_2s_single_arm": {
        "v4_fromscratch_15k": _ci.episode_cluster_bootstrap(a4, EID),
        "v1_flagship_30k": _ci.episode_cluster_bootstrap(a1, EID),
    },
    "v1_registry_pin": {
        "registry_full_set_ade_0_2s": 0.4271,
        "recomputed_here": round(float(a1.mean()), 4),
        "matches": bool(abs(float(a1.mean()) - 0.4271) < 5e-4),
        "note": ("independent recomputation of the deployed v1 from its own "
                 "persisted windows; matching 0.4271 validates this analysis "
                 "path against MODEL_REGISTRY 1.2"),
    },
}

if aligned:
    d = _ci.paired_episode_cluster_bootstrap(a4, a1, EID)
    d["_orientation"] = ("a - b = v4(15k) - v1(30k) on ADE@2s; LOWER is better, "
                         "so POSITIVE = v4 is BEHIND v1")
    res["paired_v4_15k_minus_v1"] = d
else:
    res["paired_v4_15k_minus_v1"] = {"SKIPPED": "windows not aligned"}

for tag, w in (("v4_15k", w4), ("v1_reference", w1)):
    try:
        res[f"lateral_block_{tag}"] = _lat.from_sparse_windows(w, mode="ego")
    except Exception as ex:
        res[f"lateral_block_{tag}"] = {"FAILED": f"{type(ex).__name__}: {ex}"}

with open(OUT, "w") as f:
    json.dump(res, f, indent=2, default=str)

skip = ("lateral_block_v4_15k", "lateral_block_v1_reference")
print(json.dumps({k: v for k, v in res.items() if k not in skip},
                 indent=2, default=str))
for tag in ("v4_15k", "v1_reference"):
    b = res[f"lateral_block_{tag}"]
    print(f"\n=== lateral {tag} ===")
    if "FAILED" in b:
        print(b["FAILED"]); continue
    print("verdict:", b.get("verdict"))
    print("energy_share:", json.dumps(b.get("energy_share"), default=str))
    agg = b.get("dense_aggregate", {})
    for k in ("along_abs", "cross_abs", "cross_peak", "cross_peak_p90"):
        if k in agg:
            print(f"  {k}: {json.dumps(agg[k], default=str)}")
print(f"\n[analyze] -> {OUT}")
