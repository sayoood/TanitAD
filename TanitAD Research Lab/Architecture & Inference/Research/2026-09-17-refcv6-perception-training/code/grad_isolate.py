"""PER-HEAD GRADIENT REACH, ISOLATED -- the discriminating control.

⛔ WHY THE TRAINER'S OWN `ga_trunk` IS NOT SUFFICIENT. In a joint run the
planner loss ALSO reaches the trunk, so a non-zero `ga_trunk` is consistent with
a perception head that reaches nothing -- it is exactly the shape of evidence
that let `tac_goal_tok_head` sit at `grad_abs_sum` 0 for 40,284 steps while
every other number looked healthy. The sufficient form backwards ONE term at a
time and reads the trunk after each.

Four arms, same batch, same weights, gradients zeroed between:

  planner_only : the trainer's own loss with both perception weights 0.0
  map_only     : `loss_map` alone
  box3d_only   : `loss_box3d` alone
  both         : the joint loss

⭐ And a DELIBERATE REGRESSION per head: `fmap_s16.detach()`. If the branch is
really reading the trunk, detaching the seam must drive `ga_trunk` to EXACTLY
0.0 while the head's own gradient stays non-zero. A probe that reads the same
number either way is measuring nothing.
"""
import io
import json
import sys
from pathlib import Path

import torch

WT = r"C:/Users/Admin/tanitad-wt-perctrain"
for p in (WT + "/stack", WT, WT + "/taniteval", WT + "/stack/scripts"):
    sys.path.insert(0, p)

import refc_v3_train as T                                        # noqa: E402
from tanitad.models import refcv6_perception_branch as PB        # noqa: E402

OUT = Path(sys.argv[1])
ARGV = [
    "--arm", "hier", "--size", "tiny", "--out", str(OUT.parent / "_gradprobe"),
    "--device", "cpu", "--seed", "0", "--steps", "1", "--batch", "2",
    "--workers", "0",
    "--v2-cache", r"D:/Projects/TanitAD-artifacts/v2ep-eval139-256x1024cyl",
    "--image-hw", "256", "1024", "--v2-lru", "2",
    "--trunk", "timm", "--trunk-name", "resnet34.a1_in1k",
    "--trunk-in-channels", "9",
    "--agent-join", r"D:/Projects/TanitAD-artifacts/b1-agent-join-3d-20260917/"
                    r"b1eval_agents_3d.jsonl.xz",
    "--agent-join-verify", "off", "--agents", "head", "--w-agent", "1.0",
    "--agent-queries", "16", "--agent-pad", "32",
    "--agent-rig-camera", "extrinsics",
    "--agent-rig-extrinsics",
    r"D:/Projects/TanitAD-artifacts/refcv5v2_final/extrinsics141.json",
    "--map-gt-root", r"D:/Projects/TanitAD-artifacts/sam3-maps-eval",
    "--map-lru", "2", "--join3d",
    r"D:/Projects/TanitAD-artifacts/b1-agent-join-3d-20260917/"
    r"b1eval_agents_3d.jsonl.xz",
    "--w-map", "1.0", "--w-box3d", "1.0",
]
args = T.build_parser().parse_args(ARGV)
cfg = T._pin_trainer_cfg(T.v3.refc_v3_sized_config(args.size, hier=True), args)

from tanitad.data.v2_dataset import build_v2_providers                # noqa: E402
from train_p8_occupancy import JoinFileReader                         # noqa: E402

eps = build_v2_providers([args.v2_cache[0]], lru_size=2)[:4]
ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20)
rd = JoinFileReader(args.agent_join,
                    episode_ids={int(e.episode_id) for e in eps},
                    with_rates=True, with_track_ids=True)
ds.enable_agent_join(rd, pad=32)
clip_of, n_stack = T._clip_table_for_caches([args.v2_cache[0]])
from tanitad.data import agent_cuboid_gt as ACG                       # noqa: E402
from tanitad.data import perception_targets as PT                     # noqa: E402

ds.enable_map_gt(PT.MapGTStore(Path(args.map_gt_root), max_open=2),
                 clip_of, n_stack)
ds.enable_join3d(ACG.open_join3d(args.join3d, clips=set(clip_of.values())))

model = T.v3.RefCV3Model(cfg)
model._w_agent, model._w_bev_aux, model._bev_shuffle = 1.0, 0.0, False
model._w_tac_goal = 0.0
model._tac_goal_pos_weight = model._tac_goal_class_mask = None
model._w_u0, model._w_goal_point = 0.0, 0.0
model._rig_camera, _ = T._build_rig_camera(cfg, args)
model._w_map, model._w_box3d = 1.0, 1.0
pcfg = PB.PerceptionBranchConfig(w_map=1.0, w_box3d=1.0)
model._perception = PB.build_perception_branch(model, pcfg)
_e, table = T._read_rig_extrinsics(args.agent_rig_extrinsics)
model._lift_bank = PB.LiftGeometryBank(table, frame=PB.frame_for_model(model),
                                       stride=pcfg.stride)

batch = torch.utils.data.default_collate([ds[0], ds[1]])
rep = {"n_windows": len(ds), "batch_keys": sorted(batch),
       "branch_params": model._perception.param_breakdown(), "arms": {}}


def zero():
    for p in model.parameters():
        p.grad = None


def reach(tag, term, detach_seam=False):
    """Backward ONE term, then read every module's grad_abs_sum."""
    zero()
    model._perception_detach_seam = detach_seam
    losses = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
    if term == "all":
        losses["loss"].backward()
    elif term == "planner":
        # the joint loss MINUS the two perception terms == the planner's own
        (losses["loss"] - losses["map"] - losses["box3d"]).backward()
    else:
        losses[term].backward()
    r = PB.grad_reach_report(model)
    rep["arms"][tag] = {
        "loss_value": float(losses[term if term in losses else "loss"]),
        "map": float(losses.get("map", float("nan"))),
        "box3d": float(losses.get("box3d", float("nan"))),
        "n_map_cells": float(losses.get("n_map_cells", -1)),
        "box3d_n_z": float(losses.get("box3d_n_z", -1)),
        **{k: v["grad_abs_sum"] for k, v in r.items()},
        **{k + "_n": v["n_params_with_grad"] for k, v in r.items()}}
    print(tag, json.dumps(rep["arms"][tag])[:400], flush=True)


reach("planner_only", "planner")
reach("map_only", "map")
reach("box3d_only", "box3d")
reach("both", "all")

# ---- the DELIBERATE REGRESSION: detach the trunk seam --------------------- #
# ⛔ Monkey-patched at the branch, not via a flag: it must not be reachable
# from argv. If `ga_trunk` does NOT fall to exactly 0.0 here, the "gradients
# reach the trunk" claim above was reading someone else's gradient.
_fwd = model._perception.forward


def _detached(fmap_s16, grid=None, valid=None):
    return _fwd(fmap_s16.detach(), grid, valid)


model._perception.forward = _detached
reach("map_only_SEAM_DETACHED", "map")
reach("box3d_only_SEAM_DETACHED", "box3d")
model._perception.forward = _fwd

a = rep["arms"]
rep["verdict"] = {
    "map_reaches_trunk": a["map_only"]["trunk"] > 0.0,
    "box3d_reaches_trunk": a["box3d_only"]["trunk"] > 0.0,
    "map_trunk_grad": a["map_only"]["trunk"],
    "box3d_trunk_grad": a["box3d_only"]["trunk"],
    "detached_map_trunk_grad": a["map_only_SEAM_DETACHED"]["trunk"],
    "detached_box3d_trunk_grad": a["box3d_only_SEAM_DETACHED"]["trunk"],
    "control_is_discriminating": bool(
        a["map_only_SEAM_DETACHED"]["trunk"] == 0.0
        and a["box3d_only_SEAM_DETACHED"]["trunk"] == 0.0
        and a["map_only_SEAM_DETACHED"]["map_head"] > 0.0
        and a["box3d_only_SEAM_DETACHED"]["box_decoder"] > 0.0),
    "heads_with_zero_grad": sorted(
        k for k in ("lift", "bev_encoder", "map_head", "box_memory",
                    "box_decoder")
        if a["both"].get(k, 0.0) == 0.0),
}
io.open(OUT, "w", encoding="utf-8").write(json.dumps(rep, indent=1))
print(json.dumps(rep["verdict"], indent=1))
