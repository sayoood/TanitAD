import os, sys, json, dataclasses
from pathlib import Path
os.environ.setdefault("TANITEVAL_STACK_OVERRIDE", "/root/models/assess-20260719/stack-v2b")
sys.path.insert(0, "/root/taniteval")
import torch
from taniteval.registry import MODELS
from taniteval import loaders

e = [m for m in MODELS if m["key"] == "refb-v2-30k"][0]
print("ENTRY:", e["key"], "arch=", e["arch"], "strict=", e.get("strict"),
      "speed_input=", e.get("speed_input"), "yaw_input=", e.get("yaw_input"))
ck = torch.load(e["ckpt"], map_location="cpu", weights_only=False)
print("CKPT top-level keys:", list(ck.keys()))
print("CKPT step:", ck.get("step"))

import tanitad
print("tanitad from:", tanitad.__file__)
from tanitad.refs.refb import RefBModel, refb_config


def apply(o, d):
    for k, v in d.items():
        if not hasattr(o, k):
            continue
        cur = getattr(o, k)
        if dataclasses.is_dataclass(cur) and isinstance(v, dict):
            apply(cur, v)
        elif isinstance(v, (int, float, bool, str)) or v is None:
            try:
                object.__setattr__(o, k, type(cur)(v) if cur is not None else v)
            except Exception:
                object.__setattr__(o, k, v)
        elif isinstance(v, list) and not dataclasses.is_dataclass(cur):
            object.__setattr__(o, k, tuple(v) if isinstance(cur, tuple) else v)


cfg = refb_config()
cj = Path(e["ckpt"]).parent / "config.json"
apply(cfg, json.loads(cj.read_text()).get("cfg", {}))
m = RefBModel(cfg)
rep = m.load_state_dict(ck["model"], strict=False)
print("MISSING keys (n=%d):" % len(rep.missing_keys), list(rep.missing_keys))
print("UNEXPECTED keys (n=%d):" % len(rep.unexpected_keys), list(rep.unexpected_keys))
nparam = sum(p.numel() for p in m.parameters())
print("model params:", nparam)

try:
    L = loaders.load(e, "cuda")
    print("LOADER(strict) OK: step=", L["step"], "feed=", L["feed"],
          "traj_capable=", L["traj_capable"], "state_dim=", L["state_dim"])
except Exception as ex:
    import traceback
    print("LOADER STRICT FAILED:", type(ex).__name__, str(ex)[:500])
    traceback.print_exc()
