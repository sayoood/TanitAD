"""Read the LEARNED longitudinal selection gate (`sel_gate`) out of the v4
checkpoints at 15k and 30k.

WHY: FlagshipV15Head.select() scores the refined fan as
    score = refined_logits + sel_gate * (-|v_term(i) - v_goal|)
with `sel_gate` a LEARNED scalar initialised to ZERO (V15Config.sel_gate_init=0.0).
It is computed into out["telemetry"] every step but the trainer's row-writer
filters it out (train_flagship_v4.py:693-703), so it never reached train_log.jsonl.
The checkpoint is the only surviving record.

CPU only. No GPU load.
"""
import json, torch

CK = {"15k": "/workspace/models/flagship-v4-fromscratch-15k/ckpt_step15000.pt",
      "30k": "/workspace/_v4gate/flagship-v4-fromscratch-30k/ckpt.pt"}

WANT = ("sel_gate", "vtarget", "route_emb", "route_graded", "ego_null")
out = {"_evidence_class": "MEASURED (ours; v4 checkpoints on tanitad-eval)",
       "_why": "sel_gate is the learned scale on the longitudinal selection term; "
               "init 0.0. Its trained value is itself the measurement (design note, "
               "flagship_v15.py:439-449).",
       "arms": {}}

for tag, p in CK.items():
    ck = torch.load(p, map_location="cpu", weights_only=False)
    rec = {"ckpt": p, "step": ck.get("step"), "top_keys": sorted(ck.keys())[:20]}
    sd = None
    for k in ("head", "head_state", "model_head", "planner"):
        if k in ck and isinstance(ck[k], dict):
            sd = ck[k]; rec["state_dict_key"] = k; break
    if sd is None:
        for k, v in ck.items():
            if isinstance(v, dict) and any("sel_gate" in kk for kk in v):
                sd = v; rec["state_dict_key"] = k; break
    if sd is None:
        rec["ERR"] = "no head state_dict found"; out["arms"][tag] = rec; continue
    hits = {k: v for k, v in sd.items() if any(w in k for w in WANT)}
    rec["matched_params"] = {}
    for k, v in hits.items():
        if torch.is_tensor(v):
            rec["matched_params"][k] = {
                "shape": list(v.shape), "numel": int(v.numel()),
                "value": (float(v.reshape(-1)[0]) if v.numel() == 1 else None),
                "abs_mean": float(v.float().abs().mean()),
                "l2": float(v.float().norm()),
            }
    out["arms"][tag] = rec

# the headline comparison
try:
    def gate(t):
        for k, v in out["arms"][t]["matched_params"].items():
            if k.endswith("sel_gate") or k == "sel_gate":
                return k, v["value"]
        return None, None
    k15, g15 = gate("15k"); k30, g30 = gate("30k")
    out["HEADLINE_sel_gate"] = {
        "param_15k": k15, "sel_gate_15k": g15,
        "param_30k": k30, "sel_gate_30k": g30,
        "init_value": 0.0,
        "delta_15k_to_30k": (None if (g15 is None or g30 is None) else round(g30 - g15, 6)),
        "_read": ("At EVAL the harness overwrites vt_speed with the OBSERVED v0 "
                  "(eval_flagship_v4.py:78), so v_goal == v0 and the penalty becomes "
                  "-|v_term(i) - v0| -- a CONSTANT-VELOCITY preference. A sel_gate that "
                  "GREW during training therefore makes the EVAL-time selector "
                  "progressively more constant-velocity-biased while the TRAINING "
                  "objective (which sees the real minted vt_speed) keeps improving."),
    }
except Exception as e:
    out["HEADLINE_sel_gate"] = {"ERR": str(e)}

print(json.dumps(out, indent=2))
with open("/root/v4_sel_gate.json", "w") as f:
    json.dump(out, f, indent=2)
