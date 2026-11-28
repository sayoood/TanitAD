"""Exercise refcv7_loader.build_eval_dataset(..., with_perception_targets=True) on the REAL eval kit:
the agent join (train pad from the record), the 10 cm FINE store at the declared extent, the 3-D join,
VIS-1 on the eval split. model=None (no checkpoint needed: the camera-coverage check is skipped and
recorded as such). Writes C:/lgt/r7ldr_scratch/perc_targets_check.json. Never prints clip ids."""
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["REFCV6_REPO"] = "C:/lgt/r7ldr"
os.environ["REFCV6_KIT"] = "D:/refcv6_eval_kit"
os.environ.pop("REFCV6_REMAP_OVERRIDES", None)
for p in ("C:/lgt/r7ldr/stack", "C:/lgt/r7ldr/stack/scripts", "C:/lgt/r7ldr/taniteval"):
    sys.path.insert(0, p)
import importlib.util  # noqa: E402

import torch  # noqa: E402

spec = importlib.util.spec_from_file_location("refcv7_loader_perc", "C:/lgt/r7ldr/stack/tanitad/eval/refcv7_loader.py")
L = importlib.util.module_from_spec(spec)
sys.modules["refcv7_loader_perc"] = L
spec.loader.exec_module(L)
tr = L.trainer()
from tanitad.refs import refc_v3 as v3  # noqa: E402

c6 = json.load(open("D:/refcv6_eval_kit/ckpt_final/config.json", encoding="utf-8"))
config = json.load(open("C:/lgt/r7ldr_gate/standin_config_refcv7_intended_argv.json", encoding="utf-8"))
config.pop("_STANDIN", None)
# the TRAIN split's pad: refcv6-r101-s0's stamp (same join file, same train corpus); stand-in, recorded
config["agent_join_stats"] = {"train": {"agent_pad": int(c6["agent_join_stats"]["train"]["agent_pad"])}}
args, argv, arec = L.parse_args(config)
tr._read_anchor_artifact(args)
cfg = tr._pin_trainer_cfg(v3.refc_v3_sized_config(args.size, hier=args.arm == "hier"), args)
t0 = time.time()
e_ds, e_eps, rec = L.build_eval_dataset(None, cfg, args, config, with_perception_targets=True)
out = {"build_s": round(time.time() - t0, 1), "n_episodes": rec["n_episodes"], "n_windows": rec["n_windows"],
       "record_keys": sorted(rec)}
for k in ("map_fine", "agent_join", "join3d", "vis1"):
    v = rec.get(k)
    if isinstance(v, dict):
        out[k] = {kk: vv for kk, vv in v.items()
                  if kk in ("n_windows", "n_clips", "frac_ok", "verdict", "extent", "n_stack", "pad",
                            "pad_source", "load_s", "n_lines", "n_agents", "split", "sidecar_n_clips",
                            "frac_windows_labelled", "n_windows_labelled")}
        if k == "vis1":
            out[k] = {"eval": {kk: vv for kk, vv in v["eval"].items() if kk != "sidecar_sha256"},
                      "sidecar_sha256_16": v["sidecar"].get("sha256", "")[:16] if isinstance(v.get("sidecar"), dict) else None}
idx = L.inrun_eval_perm(e_ds, 2, 2)
items = [e_ds[i] for i in idx[:3]]
b = torch.utils.data.default_collate(items)
want = ("map_fine", "map_fine_label", "map_ep", "agent_box", "agent_valid", "agent_vis_full",
        "agent_vis_px", "agent_vis_known", "tac_goal_y", "tac_goal_w", "v_max_ms", "pose_hist")
out["batch_keys_present"] = {k: (k in b) for k in want}
out["batch_shapes"] = {k: list(b[k].shape) for k in want if k in b and torch.is_tensor(b[k])}
out["map_fine_labelled"] = [bool(x) for x in b["map_fine_label"]] if "map_fine_label" in b else None
out["n_agents_valid"] = [int(x) for x in b["agent_valid"].sum(-1)] if "agent_valid" in b else None
out["n_vis_known"] = [int(x) for x in b["agent_vis_known"].sum(-1)] if "agent_vis_known" in b else None
out["stand_in"] = "agent_pad from refcv6-r101-s0's config.json (the refcv7 train pad is the Thor record's)"
json.dump(out, open("C:/lgt/r7ldr_scratch/perc_targets_check.json", "w", encoding="utf-8"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str)[:3500])
