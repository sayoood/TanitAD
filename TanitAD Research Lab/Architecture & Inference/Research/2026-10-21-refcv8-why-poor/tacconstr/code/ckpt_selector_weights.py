"""What does the TRAINED refcv8 selector weight?  Reads the EMA-48,852 checkpoint's r8_sel / r8_rc_gate scalars (CPU, mmap, read-only).
FactorizedSelection (refcv8_conditioning.py:471-499) scores a candidate  beta_lat log p(lat) + beta_lon log p(lon) + gamma_prog sat_prog + gamma_head sat_head,
all four ZERO-INIT: a value away from 0 was moved by training.  sat_prog = -|log1p(path length) - log1p(predicted prog6)| (<= 0; range up to log1p(120)=4.80),
sat_head = -|wrap(terminal heading - predicted psi_term)| / pi (<= 0; range [-1, 0])."""
import json, math
import torch
ck = torch.load("E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt", map_location="cpu", mmap=True, weights_only=False)
sd = ck["model"]
out = {"ckpt": "E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt", "step": int(ck["step"]), "kind": ck.get("kind")}
for k in ("beta_lat", "beta_lon", "gamma_prog", "gamma_head"):
    out[k] = float(sd[f"core.decoder.r8_sel.{k}"])
out["r8_rc_gate"] = float(sd["core.decoder.r8_rc_gate"])
out["max_swing_of_each_term_on_the_score"] = {
    "gamma_prog_x_sat_prog (sat range up to log1p(120))": abs(out["gamma_prog"]) * math.log1p(120.0),
    "gamma_head_x_sat_head (sat range 1)": abs(out["gamma_head"]) * 1.0,
    "beta_lat_x_log_p (log p floored at log 1e-4 = -9.21)": abs(out["beta_lat"]) * 9.2103,
    "beta_lon_x_log_p (same floor)": abs(out["beta_lon"]) * 9.2103}
out["r8_mod_proj_nonzero"] = {k: float(v.float().abs().mean()) for k, v in sd.items() if "r8_mod.proj" in k and k.endswith("weight")}
out["note"] = ("log p is floored at -9.21 so the beta terms' maximum swing is an UPPER bound that only a near-zero-probability hypothesis reaches; typical |log p| of the top hypothesis is ~0.1-1. "
               "The weights say what training moved, not how many metres the term is worth.")
json.dump(out, open("C:/Users/Admin/refcv8_pf/tacconstr/CKPT_SELECTOR_WEIGHTS.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "note"}, indent=1))
