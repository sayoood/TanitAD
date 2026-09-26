"""Does --equalize-bottom-rows 43 reach the refcv6 trunk? Built exactly as the battery loader / trainer build it
(parse the run's own config.json argv -> tr._pin_trainer_cfg -> refc.build_encoder), CPU, no checkpoint."""
import json, os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "C:/Users/Admin/ev6_battery/code")
import refcv6_loader as L          # noqa: E402
import torch                       # noqa: E402
import tanitad                     # noqa: E402
print("tanitad from:", tanitad.__file__)
config = json.load(open("D:/refcv6_eval_kit/ckpt/config.json", encoding="utf-8"))
tr = L.trainer()
print("trainer from:", tr.__file__)
args, argv, arec = L.parse_args(config, None)
print("args.equalize_bottom_rows =", getattr(args, "equalize_bottom_rows", "MISSING"))
from tanitad.refs import refc_v3 as v3, refc   # noqa: E402
cfg = tr._pin_trainer_cfg(v3.refc_v3_sized_config(args.size, hier=args.arm == "hier"), args)
enc_cfg = cfg.core.encoder
print("cfg.core.encoder.trunk_equalize_bottom_rows =", getattr(enc_cfg, "trunk_equalize_bottom_rows", "MISSING"),
      "| declared field:", "trunk_equalize_bottom_rows" in getattr(type(enc_cfg), "__dataclass_fields__", {}))
enc = refc.build_encoder(enc_cfg)
trunk = next(m for m in enc.modules() if type(m).__name__ == "TimmResNetTrunk")
print("trunk.cfg.equalize_bottom_rows =", getattr(trunk.cfg, "equalize_bottom_rows", "MISSING"))
K = int(enc_cfg.in_channels) // 3
x = torch.rand(1, 3 * K, 416, 1024)
y = trunk.normalise(x)
print("after one normalise: norm_calls =", trunk.norm_calls, "equalize_calls =", getattr(trunk, "equalize_calls", 0),
      "| bottom-43 rows zeroed in [0,1] domain:", bool(torch.allclose(y[..., -43:, :], ((torch.zeros_like(x[..., -43:, :]) - trunk._mean) / trunk._std), atol=1e-6)))
