"""probe_ckpt_o_interface.py -- CPU, read-only. For each of the four registered checkpoints: the model config's `o_interface` (the wrapper REFUSES a substitution on `cut`: O would read a constant), plus the
config fields the substitution depends on (`cons_mu` / `cons_sd` exist; code dim 60). No forward pass, no rollout.

    python probe_ckpt_o_interface.py
"""
import json
import sys

import torch

ART = "E:/Projects/TanitAD-artifacts"
CK = {"CTRL_s0": f"{ART}/v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt", "CTRL_s1": f"{ART}/v8f-dagoacc/M2b_s1_dagoacc/ckpt_last.pt",
      "REP_s0": f"{ART}/v8f-dag-orc-oacc/M2b_s0_dagoaccorcrep/ckpt_last.pt", "REP_s1": f"{ART}/v8f-dag-orc-oacc/M2b_s1_dagoaccorcrep/ckpt_last.pt"}
out = {}
for r, p in CK.items():
    ck = torch.load(p, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    sd = ck.get("model") or ck.get("state_dict") or {}
    keys = [k for k in sd if "cons_mu" in k or "cons_sd" in k]
    out[r] = {"o_interface": cfg.get("o_interface", "full (field absent)"), "top_keys": sorted(ck.keys())[:12], "cons_buffers": keys[:4],
              "cons_mu_len": [int(sd[k].numel()) for k in keys if "cons_mu" in k][:1]}
    del ck
print(json.dumps(out, indent=1))
sys.exit(0 if all(v["o_interface"] != "cut" for v in out.values()) else 1)
