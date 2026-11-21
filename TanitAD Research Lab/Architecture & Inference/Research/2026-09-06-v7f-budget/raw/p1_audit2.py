"""P1 -- name the 25 at-init norm weights and give each one its MECHANISM.

Usage:  python p1_audit2.py <run_dir_with_config.json_and_ckpt.pt>

For every 1-D `.weight` tensor of a V6Stack rebuilt at the run's OWN
v6_config and frozen at the run's OWN stage:

    requires_grad | in-optimizer (by id) | p.grad is None after a REAL
    backward through the REAL v6_loss_step at the run's OWN loss weights

and, from the run's OWN ckpt.pt, whether the tensor is bit-exactly at init
(std == 0 and mean == 1.0). The two are then CROSS-TABULATED: a mechanism
that does not predict the checkpoint is not an explanation.

ASCII-only output. CPU only, zero GPU.
"""
import dataclasses
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "stack"))
sys.path.insert(0, str(HERE / "stack" / "scripts"))

from tanitad.config import EncoderConfig, PredictorConfig, ReadoutConfig
from tanitad.models.v6 import V6Config, V6Stack, apply_stage_freeze
from train_v6_staged import V6LossWeights, synthetic_train_batch, v6_loss_step


def build_cfg(CFG):
    v = dict(CFG["v6_config"])
    v.pop("_derived", None)
    enc = EncoderConfig(**v.pop("encoder"))
    ro = ReadoutConfig(**v.pop("readout"))
    pr = v.pop("predictor")
    pr["horizons"] = tuple(pr["horizons"])
    pred = PredictorConfig(**pr)
    fields = {f.name for f in dataclasses.fields(V6Config)}
    kw = {k: (tuple(x) if isinstance(x, list) else x)
          for k, x in v.items() if k in fields}
    return V6Config(encoder=enc, readout=ro, predictor=pred, **kw)


def build_weights(CFG):
    lw = CFG["loss_weights"]
    fields = {f.name for f in dataclasses.fields(V6LossWeights)}
    return V6LossWeights(**{k: v for k, v in lw.items() if k in fields})


def main(run_dir):
    run = Path(run_dir)
    CFG = json.loads((run / "config.json").read_text())
    LR = float(CFG["args"]["lr"]); WD = float(CFG["args"]["wd"])
    STEPS = int(CFG["args"]["steps"]); STAGE = CFG["stage"]
    O5_K = int(CFG["args"].get("o5_k", 1))
    print("=" * 78)
    print("P1 AUDIT  run=%s  stage=%s lr=%g wd=%g steps=%d"
          % (run.name, STAGE, LR, WD, STEPS))
    print("=" * 78)

    torch.manual_seed(0)
    stack = V6Stack(build_cfg(CFG))
    n_tot = sum(p.numel() for p in stack.parameters())
    ok_geo = (n_tot == CFG["param_report"]["total"])
    print("[ctrl] rebuilt params %d vs config.json %d -> %s"
          % (n_tot, CFG["param_report"]["total"],
             "IDENTICAL" if ok_geo else "MISMATCH"))
    assert ok_geo, "geometry mismatch -- refusing to audit a different model"

    rep = apply_stage_freeze(stack, STAGE)
    ok_fr = (rep["n_trainable"] == CFG["freeze"]["n_trainable"]
             and rep["n_frozen"] == CFG["freeze"]["n_frozen"])
    print("[ctrl] freeze trainable/frozen %d/%d vs config.json %d/%d -> %s"
          % (rep["n_trainable"], rep["n_frozen"],
             CFG["freeze"]["n_trainable"], CFG["freeze"]["n_frozen"],
             "IDENTICAL" if ok_fr else "MISMATCH"))
    assert ok_fr, "freeze mismatch"

    trainable = [p for p in stack.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=WD)
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}

    # ---- a REAL forward + backward at the run's own weights ---------------
    o1_k = 10
    stack.zero_grad(set_to_none=True)
    b = synthetic_train_batch(stack, batch=2, k=max(o1_k, O5_K, 2), seed=0)
    b["gt_wp"] = torch.randn(2, o1_k, 2,
                             generator=torch.Generator().manual_seed(0))
    out = v6_loss_step(stack, b, stage=STAGE, weights=build_weights(CFG),
                       o1_k=o1_k, o5_k=O5_K,
                       o5_form=CFG["args"].get("o5_form", "l1"),
                       generator=torch.Generator().manual_seed(0),
                       sigreg_generator=torch.Generator().manual_seed(0))
    out["loss"].backward()

    # ---- the checkpoint's own at-init verdict -----------------------------
    ck = torch.load(run / "ckpt.pt", map_location="cpu", weights_only=False)
    sd = ck.get("stack") or ck.get("model") or ck
    print("[ckpt] step=%s  keys=%d" % (ck.get("step", "?"), len(sd)))

    rows = []
    for n, p in stack.named_parameters():
        if p.ndim != 1 or not n.endswith(".weight"):
            continue
        t = sd.get(n)
        if t is None:
            at_init, seen = None, False
        else:
            tf = t.float()
            at_init = (float(tf.std()) == 0.0
                       and abs(float(tf.mean()) - 1.0) < 1e-9)
            seen = True
        rows.append({"name": n, "group": stack.group_of(n),
                     "numel": int(p.numel()),
                     "requires_grad": bool(p.requires_grad),
                     "in_optimizer": id(p) in opt_ids,
                     "grad_is_None": p.grad is None,
                     "ckpt_at_init": at_init, "in_ckpt": seen})

    # keys present in the ckpt but not rebuilt
    built = {r["name"] for r in rows}
    extra = [k for k, v in sd.items()
             if hasattr(v, "ndim") and v.ndim == 1 and k.endswith(".weight")
             and k not in built]
    print("[ctrl] 1-D .weight rebuilt=%d  in ckpt=%d  ckpt-only=%d"
          % (len(rows), len(rows) - sum(1 for r in rows if not r["in_ckpt"])
             + len(extra), len(extra)))
    if extra:
        print("       ckpt-only keys: %s" % extra)

    # ---- MECHANISM assignment ---------------------------------------------
    def mech(r):
        if not r["requires_grad"]:
            return "A_STAGE_FREEZE"
        if r["grad_is_None"]:
            return "B_GUARDED_OFF_no_grad"
        return "C_TRAINS"
    for r in rows:
        r["mechanism"] = mech(r)

    print("")
    print("%-46s %-12s %-22s %s" % ("name", "group", "mechanism", "ckpt@init"))
    print("-" * 100)
    for r in sorted(rows, key=lambda x: (x["mechanism"], x["name"])):
        print("%-46s %-12s %-22s %s"
              % (r["name"], r["group"], r["mechanism"],
                 "AT-INIT" if r["ckpt_at_init"] else
                 ("moved" if r["ckpt_at_init"] is False else "absent")))

    pred_init = [r for r in rows if r["mechanism"] != "C_TRAINS"]
    pred_move = [r for r in rows if r["mechanism"] == "C_TRAINS"]
    obs_init = [r for r in rows if r["ckpt_at_init"] is True]
    print("")
    print("PREDICTED at-init (A+B) = %d   OBSERVED at-init in ckpt = %d"
          % (len(pred_init), len(obs_init)))
    print("PREDICTED trains   (C)  = %d   OBSERVED moved in ckpt   = %d"
          % (len(pred_move), len(rows) - len(obs_init)))
    wrong = [r["name"] for r in rows
             if (r["mechanism"] != "C_TRAINS") != (r["ckpt_at_init"] is True)]
    print("DISAGREEMENTS between mechanism and checkpoint: %d %s"
          % (len(wrong), wrong if wrong else ""))
    byg = {}
    for r in pred_init:
        byg.setdefault(r["mechanism"], []).append(r["name"])
    for k in sorted(byg):
        print("  %s : %d" % (k, len(byg[k])))

    (HERE / ("p1_audit_%s.json" % run.name)).write_text(
        json.dumps({"run": run.name, "stage": STAGE, "lr": LR, "wd": WD,
                    "steps": STEPS, "rows": rows, "ckpt_only": extra},
                   indent=1))
    print("[bank] wrote p1_audit_%s.json" % run.name)


if __name__ == "__main__":
    main(sys.argv[1])
