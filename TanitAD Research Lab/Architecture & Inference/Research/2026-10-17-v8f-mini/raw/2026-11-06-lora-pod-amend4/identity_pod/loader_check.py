# M2b's own data loader (tanitad.data.v8f_mini_data) opens the pod cache; CPU only. Usage: loader_check.py <cache_root>
import json, sys
from pathlib import Path
import torch
from tanitad.data import v8f_mini_data as V
root = Path(sys.argv[1]); box2 = Path("/root/identity_pod/cache_c")
res = {}
for split in ("train", "eval"):
    man = V.read_manifest(root, split)
    st = V.ClipStore(root, split)
    for clip, n in man:
        d = st.get(clip)
        assert d["feats"].shape == (n, 40, 1024), d["feats"].shape
        ref = torch.load(box2 / split / f"{clip}.pt", map_location="cpu", weights_only=True)
        mine = torch.load(root / split / "clips" / f"{clip}.pt", map_location="cpu", weights_only=True)
        res[clip[:8]] = {"split": split, "n_frames": n, "feats_shape": list(d["feats"].shape), "feats_dtype": str(d["feats"].dtype),
                         "keys_equal_box2": sorted(mine) == sorted(ref), "meta_keys_equal_box2": sorted(mine["meta"]) == sorted(ref["meta"]),
                         "meta_equal_box2": mine["meta"] == ref["meta"],
                         "kin_fields_equal_box2": {k: bool(torch.allclose(mine[k].double(), ref[k].double(), rtol=0, atol=0, equal_nan=True)) for k in ("v", "a", "steer", "yaw_rate", "pose_grid", "kin3", "p10", "mean")},
                         "max_abs_diff_box2": {k: float(torch.nan_to_num(mine[k].double() - ref[k].double()).abs().max()) for k in ("yaw_rate", "mean")}}
rows = {s: [json.loads(l) for l in open(root / s / "manifest.jsonl")] for s in ("train", "eval")}
b2 = {"train": json.loads(open("/root/identity_pod/box2_manifest_row_train.json").read()), "eval": json.loads(open("/root/identity_pod/box2_manifest_row_eval.json").read())} if Path("/root/identity_pod/box2_manifest_row_train.json").exists() else {}
res["manifest_row_keys_equal_box2"] = {s: (sorted(rows[s][0]) == sorted(b2[s])) for s in b2}
print(json.dumps(res, indent=1))
