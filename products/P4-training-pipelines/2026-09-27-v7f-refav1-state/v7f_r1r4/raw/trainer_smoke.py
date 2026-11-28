"""CPU smoke of train_v6_staged.py's REAL --dry-run path with the R1/R4 flags OFF vs ON.

⚠️ WHY A DRIVER AND NOT THE CLI: the tip's ``dry_run`` builds the stack WITHOUT seeding the
global RNG (``torch.manual_seed(a.seed)`` is called in ``train()`` only), so two identical CLI
dry-runs give different losses (MEASURED here: 46.5779 vs 46.5833 at step 1, flags OFF). This
driver seeds the global RNG before each ``dry_run`` so the arms share their pre-existing
weights -- the only way an ON-vs-OFF comparison of loss terms means anything.

Synthetic tensors, tiny geometry, stage S-T, 4 Adam steps. NOTHING here is quotable as a
result; it proves the flags assemble, step, stay finite, and (zero-init) leave step 1 exactly
where the OFF arm is.
Usage: python trainer_smoke.py <stack_root> <out_dir>
"""
import argparse
import json
import sys

stack_root, out_root = sys.argv[1], sys.argv[2]
sys.path.insert(0, stack_root)
sys.path.insert(0, stack_root + "/scripts")
import torch  # noqa: E402

import train_v6_staged as T  # noqa: E402

GEOM = ["--in-channels", "3", "--frame-h", "32", "--frame-w", "32", "--enc-dim", "32",
        "--enc-depth", "1", "--enc-heads", "2", "--readout-grid", "4", "--readout-dim", "8",
        "--pred-dim", "32", "--pred-depth", "1", "--pred-heads", "2", "--window", "4",
        "--horizons", "1", "--d-tac", "32", "--d-str", "16", "--d-goal-embed", "16",
        "--adapter-hidden", "32", "--sigreg-slices", "8", "--dry-steps", "4",
        "--dry-batch", "4", "--dry-k", "12", "--seed", "0"]
ARMS = {"off": [], "r4_e2e": ["--tac-op-cond", "e2e"],
        "r4_detached": ["--tac-op-cond", "detached"], "r1a": ["--max-speed-input-v6"],
        "r1b": ["--plan-vmax-cap"],
        "all_e2e": ["--tac-op-cond", "e2e", "--max-speed-input-v6", "--plan-vmax-cap"]}

res = {}
for arm, flags in ARMS.items():
    ap = T.build_parser()
    ap.add_argument("--i-know-this-is-the-control-arm", action="store_true",
                    dest="control_arm_ack", help=argparse.SUPPRESS)
    a = ap.parse_args(["--stage", "S-T", "--out", f"{out_root}/{arm}", "--dry-run",
                       *GEOM, *flags])
    probs = T.preflight(a)
    torch.manual_seed(0)                       # the seeding the tip's dry_run lacks
    r = T.dry_run(a)
    res[arm] = {"preflight_problems": probs,
                "steps": [{k: s.get(k) for k in ("step", "loss", "plan_wta", "t1_latent",
                                                  "seam_op", "fan_oracle_ade", "gnorm",
                                                  "terms")} for s in r["steps"]],
                "n_trainable_tensors": r["n_trainable_tensors"],
                "isolation_pass": r["isolation"]["pass"],
                "param_total": r["param_report"]["total"],
                "r1_r4": {k: v for k, v in (r.get("r1_r4") or {}).items() if k != "config"}}
off = res["off"]["steps"]
for arm in ARMS:
    res[arm]["loss_bit_identical_to_OFF_per_step"] = [
        s["loss"] == o["loss"] for s, o in zip(res[arm]["steps"], off)]
print(json.dumps(res, indent=1))
