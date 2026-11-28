import sys, torch
sys.path.insert(0, "refe")
from model import REFeConfig
c = REFeConfig()
print({k: getattr(c, k) for k in ("dec_width","dec_heads","dec_depth","horizon_steps","traj_dim","n_score_components","score_dec_depth")})
sd = torch.load("D:/Projects/TanitAD-artifacts/refe-final-2026-10-01/model_final.pt", map_location="cpu", weights_only=False, mmap=True)
print(type(sd), list(sd.keys())[:5] if isinstance(sd, dict) else "")
m = sd["model"] if "model" in sd else sd
ks = [k for k in m if k.split(".")[0] in ("score_q_mlp","score_dec","score_head")]
print(len(ks)); 
for k in ks[:6]: print(k, tuple(m[k].shape), m[k].dtype)
print([k for k in m if k.startswith("score_head")], {k: tuple(m[k].shape) for k in m if k.startswith("score_head")})
print(sd.get("meta", {}).get("n_score_components") if isinstance(sd.get("meta"), dict) else None)
