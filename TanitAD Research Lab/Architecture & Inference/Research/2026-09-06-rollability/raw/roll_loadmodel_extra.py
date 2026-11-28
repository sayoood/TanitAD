"""D-ROLL-1 -- the AUTHORITATIVE acceptance path: `refcv3_arm.load_model`.

Not a re-implementation: this calls the instrument's own STRICT loader, the one
that refused before the fix. ASCII-only prints.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

WT = r"C:\Users\Admin\tanitad-wt"
for p in (os.path.join(WT, "stack"), WT):
    if p not in sys.path:
        sys.path.insert(0, p)

_p = os.path.join(WT, "taniteval", "tools", "refcv3_arm.py")
_spec = importlib.util.spec_from_file_location("refcv3_arm_for_roll", _p)
arm = importlib.util.module_from_spec(_spec)
sys.modules["refcv3_arm_for_roll"] = arm
_spec.loader.exec_module(arm)

CKPTS = [
    ("refcv4b@40284", r"C:\Users\Admin\refcv4b_final\ckpt_40284_FINAL.pt"),
    ("refcv4b-egodrop@5000", r"C:\Users\Admin\refcv4b_egodrop\pull\ckpt_5000.pt"),
    ("refcv4b-egodrop@9500", r"C:\Users\Admin\refcv4b_egodrop\pull\ckpt_9500.pt"),
    ("refcv3-rlmin@40284", r"C:\Users\Admin\rl_refcv3_min\base\ckpt_step40284_frozen.pt"),
    ("refcv3-ol@30000", r"C:\Users\Admin\run_refcv3_ol\ckpt\ckpt_30000.pt"),
    ("refcv3-viz@40284", r"C:\Users\Admin\run_refcv3_viz\ckpt\ckpt_step40284_frozen.pt"),
]

out, bad = [], 0
for name, ck in CKPTS:
    try:
        model, cfg, targs, prov = arm.load_model(ck, device="cpu")
    except SystemExit as ex:
        bad += 1
        out.append({"arm": name, "status": "REFUSED", "msg": str(ex)[:700]})
        print("[%-20s] REFUSED  %s" % (name, str(ex)[:180]))
        continue
    except Exception as ex:                                   # noqa: BLE001
        bad += 1
        out.append({"arm": name, "status": "ERROR",
                    "error": f"{type(ex).__name__}: {ex}"[:400],
                    "tb": traceback.format_exc()[-900:]})
        print("[%-20s] ERROR    %s" % (name, str(ex)[:180]))
        continue
    sd = prov["state_dict_load"]
    # same-breath NON-ZERO control: a real model with real weights was returned.
    n_par = sum(p.numel() for p in model.parameters())
    rec = {"arm": name, "status": "LOADED", "step": prov.get("step"),
           "missing_keys": sd["missing_keys"],
           "unexpected_keys": sd["unexpected_keys"],
           "tolerated_inert_buffers": sd["tolerated_inert_buffers"],
           "n_missing": len(sd["missing_keys"]),
           "n_unexpected": len(sd["unexpected_keys"]),
           "param_breakdown_has_tacgoal":
               "tac_goal_tok_head" in prov["param_breakdown"],
           "param_total": prov["param_breakdown"]["total"],
           "tac_vocab_version": prov["tac_vocab_version"],
           "cfg_tac_goal_tok_head": bool(prov["cfg"].get("tac_goal_tok_head")),
           "control_n_params": n_par,
           "control_cross_checks": sorted(prov["config_cross_checks"].keys())}
    out.append(rec)
    print("[%-20s] LOADED   step=%-6s missing=%d(%s) unexpected=%d "
          "total=%d tacgoal_line=%s n_params=%d"
          % (name, rec["step"], rec["n_missing"],
             "inert" if rec["tolerated_inert_buffers"] else "-",
             rec["n_unexpected"], rec["param_total"],
             rec["param_breakdown_has_tacgoal"], n_par))
    del model
    sys.stdout.flush()

dst = r"C:\Users\Admin\_roll\roll_loadmodel_extra.json"
with open(dst, "w", encoding="utf-8") as fh:
    json.dump({"checkpoints": out, "n_refused": bad}, fh, indent=1)
print("WROTE", dst, "n_refused =", bad)
raise SystemExit(0 if bad == 0 else 1)
