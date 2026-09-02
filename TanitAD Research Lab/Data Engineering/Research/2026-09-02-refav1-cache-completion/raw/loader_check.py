"""Consumer-side check: the rebuilt fp8 entry must open through the LIVE
loader (`tanitad.data.refav1_loader.RefAV1Windows`) at the live run's window
config, pass its ceil(T_ep/2) grid refusal, and yield finite windows."""
import json
import shutil
from pathlib import Path

import torch

from tanitad.data.refav1_loader import RefAV1Windows

REB = Path("C:/Users/Admin/refav1_probe/rebuild")
CLIP = "16d325e9-dbfe-439c-a0ed-4ff8850ad2e2"
cache = REB / "cache_one"
eps = REB / "eps_one"
cache.mkdir(exist_ok=True)
eps.mkdir(exist_ok=True)
shutil.copyfile(REB / f"{CLIP}.pt", cache / f"{CLIP}.pt")
shutil.copyfile(REB / f"{CLIP}.v2ep.pt", eps / f"{CLIP}.v2ep.pt")

# the live run's config (sup_refav1_v2.sh + RefAV1Config defaults:
# op_window 4, op_steps 30, str_dt 3.0, str_ext_steps 2)
ld = RefAV1Windows(cache, eps, op_window=4, op_steps=30, str_dt=3.0,
                   str_ext_steps=2, lru=4, seed=0)
rep = {"episodes": ld.names, "T_cache": ld._T, "n_windows": len(ld),
       "reach": ld.reach, "clip_id": ld.clip_id}
b = ld.batch(4)
rep["batch"] = {k: (list(v.shape) + [str(v.dtype)] if torch.is_tensor(v) else v)
                for k, v in b.items()}
rep["feats_finite"] = bool(torch.isfinite(b["feats"]).all())
rep["future_finite"] = bool(torch.isfinite(b["future_feats"]).all())
rep["feats_mean_abs"] = float(b["feats"].abs().mean())
rep["v0"] = [round(float(x), 3) for x in b["v0"]]
rep["actions_a_range"] = [float(b["actions"][..., 0].min()), float(b["actions"][..., 0].max())]
rep["actions_kappa_range"] = [float(b["actions"][..., 1].min()), float(b["actions"][..., 1].max())]
rep["verdict"] = ("LOADER ACCEPTS the rebuilt entry" if rep["feats_finite"] and rep["future_finite"]
                  else "LOADER REJECTS")
print(json.dumps(rep, indent=1))
(REB / "loader_check.json").write_text(json.dumps(rep, indent=1))
print("banked ->", REB / "loader_check.json")
