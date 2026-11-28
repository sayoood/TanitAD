"""The eval-time LOSS through the refcv7 loader's model + eval dataset (with perception targets) on ONE
real eval window, full size, CPU: the carriers G-EVAL does not compare (`_tac_goal_pos_weight`,
`_tac_goal_class_mask`, `_map_hires_class_weight`, `_vis1`) are READ here by the trainer's own
`compute_losses_v3` (eval mode, no grad). The reference model is the real train()'s (sentinel data).
Writes C:/lgt/r7ldr_scratch/eval_loss_check.json. Never prints clip ids."""
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["REFCV6_REPO"] = "C:/lgt/r7ldr"
os.environ["REFCV6_KIT"] = "D:/refcv6_eval_kit"
os.environ.pop("REFCV6_REMAP_OVERRIDES", None)
for p in ("C:/lgt/r7ldr/stack", "C:/lgt/r7ldr/stack/scripts", "C:/lgt/r7ldr/taniteval"):
    sys.path.insert(0, p)
import importlib.util  # noqa: E402

import torch  # noqa: E402

import launch_gate as LG  # noqa: E402

OUT = Path("C:/lgt/r7ldr_scratch/evalloss")
OUT.mkdir(parents=True, exist_ok=True)
argv = json.load(open("C:/Users/Admin/qland/work/refcv7/probe_cost/launch_argv_intended.json", encoding="utf-8"))
MODEL_INPUTS = ("--anchors", "--agent-rig-extrinsics", "--nav-compliance-tau-file", "--map-hires-class-weights")
loc, i = [], 0
while i < len(argv):
    a = argv[i]
    if a == "--trunk-compile":
        i += 1
        continue
    loc.append(a)
    j = i + 1
    while j < len(argv) and not argv[j].startswith("--"):
        v = argv[j]
        if a == "--out":
            v = str(OUT / "out")
        elif v.startswith("/home/nvidia/data"):
            v = (v.replace("/home/nvidia/data", "D:/refcv6_eval_kit/data") if a in MODEL_INPUTS
                 else str(OUT / "__unread__" / a.strip("-")))
        loc.append(v)
        j += 1
    i = j
T = LG.load_trainer(type("C", (), {"tree": "C:/lgt/r7ldr", "prof": {"trainer": "stack/scripts/refc_v3_train.py"}})())
cap = LG.run_trainer_until(T, loc, "model")
ck = OUT / "ckpt.pt"
torch.save({"model": cap["model"].state_dict(), "step": 0}, ck)
del cap
c6 = json.load(open("D:/refcv6_eval_kit/ckpt_final/config.json", encoding="utf-8"))
config = json.load(open("C:/lgt/r7ldr_gate/standin_config_refcv7_intended_argv.json", encoding="utf-8"))
config.pop("_STANDIN", None)
config["agent_join_stats"] = {"train": {"agent_pad": int(c6["agent_join_stats"]["train"]["agent_pad"])}}
spec = importlib.util.spec_from_file_location("refcv7_loader_evalloss", "C:/lgt/r7ldr/stack/tanitad/eval/refcv7_loader.py")
L = importlib.util.module_from_spec(spec)
sys.modules["refcv7_loader_evalloss"] = L
spec.loader.exec_module(L)
tr = L.trainer()
model, cfg, args, rec = L.build_model(config, str(ck), device="cpu", strict=True)
t0 = time.time()
e_ds, e_eps, drec = L.build_eval_dataset(model, cfg, args, config, with_perception_targets=True)
res = {"dataset_s": round(time.time() - t0, 1), "camera_coverage": drec.get("camera_coverage")}
perm = L.inrun_eval_perm(e_ds, 1, 16)
# the first window of the in-run eval's fixed set that carries a VIS-1-known agent AND a 10 cm label
pick = None
for idx in perm:
    it = e_ds[idx]
    if bool(it["map_fine_label"]) and int(it["agent_vis_known"].sum()) > 0:
        pick = idx
        break
res["window"] = {"perm_position": perm.index(pick) if pick is not None else None}
b = torch.utils.data.default_collate([e_ds[pick]])
model.eval()
t1 = time.time()
with torch.no_grad():
    losses = tr.compute_losses_v3(model, b, "cpu", mode=getattr(args, "mode", "diffusion"))
res["loss_s"] = round(time.time() - t1, 1)
scal = {k: float(v) for k, v in losses.items() if torch.is_tensor(v) and v.ndim == 0}
res["loss"] = scal.get("loss")
res["loss_finite"] = bool(math.isfinite(scal.get("loss", float("nan"))))
res["n_scalar_terms"] = len(scal)
res["non_finite_terms"] = sorted(k for k, v in scal.items() if not math.isfinite(v))
res["terms_of_interest"] = {k: scal[k] for k in sorted(scal) if any(
    t in k for t in ("map_hires", "box3d", "agent", "tac", "vis", "n_pos")) and "_iou_" not in k
    and "_inter_" not in k and "_union_" not in k}
res["carriers"] = {"tac_goal_pos_weight_len": None if model._tac_goal_pos_weight is None
                   else int(model._tac_goal_pos_weight.numel()),
                   "tac_goal_class_mask_sum": None if model._tac_goal_class_mask is None
                   else float(model._tac_goal_class_mask.sum()),
                   "map_hires_class_weight": [round(float(x), 4) for x in model._map_hires_class_weight],
                   "vis1": bool(model._vis1)}
json.dump(res, open("C:/lgt/r7ldr_scratch/eval_loss_check.json", "w", encoding="utf-8"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str)[:4000])
try:
    ck.unlink()
except OSError:
    pass
