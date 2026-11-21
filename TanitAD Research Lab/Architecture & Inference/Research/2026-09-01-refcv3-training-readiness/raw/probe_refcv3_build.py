"""Build RefCV3 at the registered rungs and COUNT params; then attempt the
B1-native 256x640 non-square build to MEASURE what blocks it."""
import json
import sys
import traceback

sys.path.insert(0, r"C:/Users/Admin/tanitad-wt/stack")
import torch
from tanitad.refs import refc
from tanitad.refs import refc_v3 as v3

out = {}

# 1) registered rungs, hier + flat — construct and COUNT
for size in ("small", "xl"):
    for hier in (True, False):
        torch.manual_seed(0)
        m = v3.RefCV3Model(v3.refc_v3_sized_config(size, hier=hier))
        bd = v3.param_breakdown_v3(m)
        out[f"{size}/{'hier' if hier else 'flat'}"] = bd
        del m

# 2) the dominance delta at small (the C122 pin)
delta = v3.config_delta(v3.refc_v3_sized_config("small", hier=True),
                        v3.refc_v3_sized_config("small", hier=False))
out["delta_small"] = {k: [repr(a), repr(b)] for k, (a, b) in delta.items()}

# 3) non-square 256x640 (B1-native geometry) — what actually breaks?
cfg = v3.refc_v3_sized_config("small", hier=True)
cfg.core.encoder = refc.CNNEncoderConfig(
    in_channels=9, image_size=256, image_width=640,
    base_width=cfg.core.encoder.base_width, blocks=cfg.core.encoder.blocks)
try:
    m = v3.RefCV3Model(cfg)
    out["nonsquare_build"] = "BUILT OK"
    # if it built, try a forward
    try:
        m.eval()
        with torch.no_grad():
            o = m(torch.rand(1, cfg.core.window, 9, 256, 640),
                  v0=torch.tensor([3.0]))
        out["nonsquare_forward"] = {k: list(v.shape) for k, v in o.items()
                                    if torch.is_tensor(v)}
    except Exception:
        out["nonsquare_forward_error"] = traceback.format_exc(limit=4)
except Exception:
    out["nonsquare_build_error"] = traceback.format_exc(limit=4)

print(json.dumps(out, indent=1, default=str))
