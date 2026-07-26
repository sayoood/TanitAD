"""COUNTERFACTUAL: re-select the v4-30k fan with the longitudinal selection
gate ZEROED, on the identical 881 windows.

MECHANISM UNDER TEST (all MEASURED from code + config, not assumed):
  FlagshipV15Head.select():  score = refined_logits + sel_gate * (-|v_term(i) - v_goal|)
  v_goal = clamp(vt_speed, v0 +/- sel_accel_max*2.0)      [flagship_v15.py:455-457]
  _goal_inputs sets vt_speed = v0                          [train_flagship_v4.py:172]
  => v_goal == v0 EXACTLY (sel_accel_max=2.5 => reach=5.0 m/s, clamp is a no-op)
  => the "target-speed-aware" term is a pure CONSTANT-VELOCITY preference,
     in TRAINING and at EVAL alike.
  sel_gate is LEARNED from init 0.0; MEASURED 0.1101 @15k -> 0.1580 @30k.

Setting sel_gate := 0 removes ONLY that term. The fan, the trunk, the decoder
and refined_logits are untouched (sel_gate enters after decoder()), so this is
a clean selection-only ablation on the same forward-pass proposals.

Both arms goal-mode ORACLE, so the comparison is like-for-like with the gate's
primary. Paired episode-cluster bootstrap on the SAME windows.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, "/root/v4eval/stack/scripts")
sys.path.insert(0, "/root/v4eval/stack")
sys.path.insert(0, "/root/taniteval")

import eval_flagship_v4 as E
from taniteval.ci import paired_episode_cluster_bootstrap, episode_cluster_bootstrap

CKPT = "/workspace/_v4gate/flagship-v4-fromscratch-30k/ckpt.pt"
VAL = "/root/valdata/physicalai-val-0c5f7dac3b11"
ANCH = "/root/models/flagship-v4-fromscratch-15k/flagship_v4_anchors_dense.pt"
DEV = "cuda"

cfg = E._eval_cfg(); plan = E._plan(cfg)
ds_val = E.build_val_dataset_v4(VAL, cfg, plan)
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
HCFG = "/workspace/_v4gate/flagship-v4-fromscratch-30k/config.json"  # the run's OWN sibling config
world, grounding, head, step, hcfg, goal_head = E.load_v4_from_ck(
    ck, DEV, head_config_path=HCFG, anchors_dense_path=ANCH)
del ck
import driving_diagnostic as dd

trained_gate = float(head.sel_gate.detach())
res = {"_experiment": "sel_gate counterfactual on flagship-v4-fromscratch @30k",
       "_evidence_class": "MEASURED (ours)",
       "_ckpt": CKPT, "_ckpt_step": int(step),
       "_goal_provenance": "ORACLE (route/route_graded/vt_band from ego's own future "
                           "poses). NOT a deployed-capability surface.",
       "_estimator": "paired_episode_cluster_bootstrap (taniteval/ci.py); "
                     "NEVER overlapping_holdout_se; resampling unit = episode",
       "_sel_gate_trained": trained_gate,
       "_ablation": "head.sel_gate := 0.0 (removes ONLY the longitudinal "
                    "constant-velocity selection term; fan/decoder untouched)",
       "arms": {}}

runs = {}
for tag, gate_val in (("as_trained", trained_gate), ("sel_gate_zero", 0.0)):
    with torch.no_grad():
        head.sel_gate.fill_(gate_val)
    t = time.time()
    data, diag = E.collect_planner(world, grounding, head, ds_val, DEV, dd,
                                   episodes=40, stride=8, batch=16,
                                   goal_mode="oracle", goal_head=goal_head)
    runs[tag] = (data, diag)
    res["arms"][tag] = {"sel_gate": gate_val,
                        "wallclock_s": round(time.time() - t, 1),
                        "n_windows": int(data["pred"].shape[0]),
                        "diag": diag}
    print(f"[abl] {tag}: gate={gate_val:.4f} done in {time.time()-t:.0f}s", flush=True)

dA, dB = runs["as_trained"][0], runs["sel_gate_zero"][0]
eid = [str(x) for x in dA["eid"]]
assert eid == [str(x) for x in dB["eid"]], "window misalignment"
assert torch.allclose(dA["gt"], dB["gt"], atol=1e-6), "gt mismatch"

ade = lambda d: (d["pred"] - d["gt"]).norm(dim=-1).mean(1).numpy()
miss = lambda d: ((d["pred"] - d["gt"]).norm(dim=-1)[:, -1] > 2.0).float().numpy()
def ac(d):
    r = d["pred_dense"] - d["gt_dense"]
    return r[..., 0].abs().mean(1).numpy(), r[..., 1].abs().mean(1).numpy()

aA, aB = ade(dA), ade(dB)
alA, crA = ac(dA); alB, crB = ac(dB)

res["harness_check"] = {
    "as_trained_ade_0_2s": round(float(aA.mean()), 4),
    "gate_published_30k_oracle": 0.6423,
    "reproduces_within_0.001": abs(float(aA.mean()) - 0.6423) <= 0.001,
    "_rule": "if the as-trained arm does not reproduce the gate number, the "
             "ablation arm is not quotable either",
}
res["point_estimates"] = {
    "ade_0_2s_as_trained": round(float(aA.mean()), 4),
    "ade_0_2s_sel_gate_zero": round(float(aB.mean()), 4),
    "miss_at_2m_as_trained": round(float(miss(dA).mean()), 4),
    "miss_at_2m_sel_gate_zero": round(float(miss(dB).mean()), 4),
    "v1_reference_ade_0_2s": 0.4271,
}
# oriented as_trained - zero, so POSITIVE = the gate HURTS
res["paired_as_trained_minus_zero"] = {
    "ade_0_2s": paired_episode_cluster_bootstrap(aA, aB, eid, n_boot=2000, seed=0),
    "miss_at_2m": paired_episode_cluster_bootstrap(miss(dA), miss(dB), eid, n_boot=2000, seed=0),
    "along_abs_dense_LONGITUDINAL": paired_episode_cluster_bootstrap(alA, alB, eid, n_boot=2000, seed=0),
    "cross_abs_dense_LATERAL": paired_episode_cluster_bootstrap(crA, crB, eid, n_boot=2000, seed=0),
}
res["singles"] = {
    "ade_0_2s_as_trained": episode_cluster_bootstrap(aA, eid, n_boot=2000, seed=0),
    "ade_0_2s_sel_gate_zero": episode_cluster_bootstrap(aB, eid, n_boot=2000, seed=0),
}
res["selection_change"] = {
    "frac_windows_selection_changed": round(float(
        (~np.isclose(aA, aB, atol=1e-6)).mean()), 4),
    "_read": "if this is ~0 the gate is inert and the ablation proves nothing",
}
torch.save({"as_trained": dA, "sel_gate_zero": dB},
           "/root/v4_selgate_ablation_windows.pt")
print(json.dumps({k: v for k, v in res.items() if k != "arms"}, indent=2))
with open("/root/v4_selgate_ablation.json", "w") as f:
    json.dump(res, f, indent=2, default=str)
