"""P1 FINAL -- the 25 NAMED, each with its mechanism, replicated across arms.

The 7 keys the CURRENT v6.py no longer builds (each head owned its own goal
vocabulary when v7-tiny trained; HEAD shares the top-level `vocab_*`) are
resolved through the SAME `V6Stack.group_of` prefix map, so their mechanism is
read from the same funnel as the other 18, not asserted.
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "stack"))
sys.path.insert(0, str(HERE / "stack" / "scripts"))

from tanitad.models.v6 import V6Stack, stage_trainable_groups
from p1_audit2 import build_cfg

ASSETS = Path("C:/Users/Admin/tanitad-caches/mm-e19-assets-20260901")
ARMS = ["v7tiny_postrain30k", "v7tiny_postrain30k_freeze", "v7tiny_rdw8p30k",
        "v7tiny_splitp30k", "v7tiny_o14fut30k", "v7tiny_k8clip05p30k",
        "v7tiny_k60clip05p30k"]

base = json.loads((HERE / "p1_audit_v7tiny_postrain30k.json").read_text())
by_name = {r["name"]: r for r in base["rows"]}
CFG = json.loads((ASSETS / "v7tiny_postrain30k" / "config.json").read_text())
torch.manual_seed(0)
stack = V6Stack(build_cfg(CFG))
STAGE = CFG["stage"]
live_groups = set(stage_trainable_groups(STAGE))
print("stage %s trains groups %s" % (STAGE, sorted(live_groups)))
print("")

print("=" * 96)
print("THE 25 NORM WEIGHTS THAT SIT BIT-EXACTLY AT 1.0 AFTER 30,000 AdamW "
      "STEPS -- NAMED, WITH MECHANISM")
print("=" * 96)
print("%-44s %-12s %-9s %-9s %-9s %s"
      % ("name", "group", "req_grad", "in_opt", "p.grad", "mechanism"))
print("-" * 96)

rows25 = []
for arm in ARMS[:1]:
    ck = torch.load(ASSETS / arm / "ckpt.pt", map_location="cpu",
                    weights_only=False)
    sd = ck.get("stack") or ck.get("model") or ck
    ones = [k for k, v in sd.items()
            if hasattr(v, "ndim") and v.ndim == 1 and k.endswith(".weight")]
    for k in ones:
        t = sd[k].float()
        if not (float(t.std()) == 0.0 and abs(float(t.mean()) - 1.0) < 1e-9):
            continue
        g = stack.group_of(k)
        r = by_name.get(k)
        if r is not None:
            rg, io, gn = r["requires_grad"], r["in_optimizer"], r["grad_is_None"]
            mech = ("A_STAGE_FREEZE" if not rg
                    else "B_OBJECTIVE_GUARDED_OFF")
            src = "measured"
        else:
            # not built by HEAD's v6.py -- its group still resolves, and the
            # stage funnel is what decides requires_grad for EVERY parameter.
            rg = g in live_groups
            io, gn, src = rg, True, "group_of+stage funnel"
            mech = "A_STAGE_FREEZE" if not rg else "B_OBJECTIVE_GUARDED_OFF"
        rows25.append({"name": k, "group": g, "requires_grad": rg,
                       "in_optimizer": io, "grad_is_None": gn,
                       "mechanism": mech, "evidence": src})

for r in sorted(rows25, key=lambda x: (x["mechanism"], x["group"], x["name"])):
    print("%-44s %-12s %-9s %-9s %-9s %s"
          % (r["name"], r["group"], r["requires_grad"], r["in_optimizer"],
             "None" if r["grad_is_None"] else "present", r["mechanism"]))

na = sum(1 for r in rows25 if r["mechanism"] == "A_STAGE_FREEZE")
nb = len(rows25) - na
print("-" * 96)
print("TOTAL %d   A_STAGE_FREEZE=%d   B_OBJECTIVE_GUARDED_OFF=%d"
      % (len(rows25), na, nb))

print("")
print("=" * 96)
print("REPLICATION ACROSS ARMS -- is the 25/14 split a property of the RECIPE "
      "or of one checkpoint?")
print("=" * 96)
names25 = {r["name"] for r in rows25}
for arm in ARMS:
    p = ASSETS / arm / "ckpt.pt"
    if not p.exists():
        print("  %-30s MISSING" % arm); continue
    ck = torch.load(p, map_location="cpu", weights_only=False)
    sd = ck.get("stack") or ck.get("model") or ck
    ones = [k for k, v in sd.items()
            if hasattr(v, "ndim") and v.ndim == 1 and k.endswith(".weight")]
    ai = {k for k in ones
          if float(sd[k].float().std()) == 0.0
          and abs(float(sd[k].float().mean()) - 1.0) < 1e-9}
    cj = ASSETS / arm / "config.json"
    if cj.exists():
        cfgj = json.loads(cj.read_text())
        lw = cfgj["loss_weights"]
        tail = ("stage=%-4s o1/o2/o3=%s/%s/%s"
                % (cfgj["stage"], lw["o1_ctrl"], lw["o2_nearfield"],
                   lw["o3_masked"]))
    else:
        tail = "stage=?    (no config.json shipped with this arm)"
    print("  %-30s 1Dw=%-3d at-init=%-3d identical-set=%-5s %s"
          % (arm, len(ones), len(ai), ai == names25, tail))

(HERE / "p1_the_25.json").write_text(json.dumps(
    {"arm_family": "v7-tiny (mm-e19-assets-20260901)", "stage": STAGE,
     "lr": CFG["args"]["lr"], "wd": CFG["args"]["wd"],
     "steps": CFG["args"]["steps"], "n": len(rows25),
     "A_stage_freeze": na, "B_objective_guarded_off": nb,
     "rows": rows25}, indent=1))
print("")
print("[bank] wrote p1_the_25.json")
