import sys, torch
sys.path.insert(0, "/root/TanitAD/stack")
from tanitad.config import flagship4b_config
from tanitad.models.fourbrain import WorldModel

CK = sys.argv[1] if len(sys.argv) > 1 else "/root/models/flagship-speed/ckpt.pt"
cfg = flagship4b_config()
# speedjerk = v0 as 3rd action channel
object.__setattr__(cfg.predictor, "action_dim", 3)
if getattr(cfg, "tactical_pred", None) is not None:
    object.__setattr__(cfg.tactical_pred, "action_dim", 3)
model = WorldModel(cfg)
ck = torch.load(CK, map_location="cpu", weights_only=True)
missing, unexpected = model.load_state_dict(ck["model"], strict=False)
print("load: missing=%d unexpected=%d" % (len(missing), len(unexpected)))
if missing:
    print("  missing e.g.:", missing[:5])
if unexpected:
    print("  unexpected e.g.:", unexpected[:5])
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(dev).eval()
W = model.predictor.cfg.window
print("window:", W, "tactical waypoint_horizons:", cfg.tactical_policy.waypoint_horizons)
# dummy inference: encode -> strategic -> tactical -> waypoints
frames = torch.rand(1, W, cfg.encoder.in_channels, cfg.encoder.image_size,
                    cfg.encoder.image_size, device=dev)
with torch.no_grad():
    states = model.encode_window(frames)
    print("states:", tuple(states.shape))
    nav = torch.zeros(1, dtype=torch.long, device=dev)
    ctx = model.strategic_policy(states, nav)["ctx"]
    print("ctx:", tuple(ctx.shape))
    out = model.tactical_policy(states, ctx)
    print("tactical out keys:", list(out.keys()))
    wp = out["waypoints"]
    print("waypoints type:", type(wp).__name__)
    if isinstance(wp, dict):
        print("  wp keys:", list(wp.keys()))
        for k, v in wp.items():
            print("   ", k, tuple(v.shape), v[0].cpu().numpy().round(2).tolist())
    else:
        print("  wp shape:", tuple(wp.shape))
print("SMOKE_OK step=%s" % ck.get("step"))
