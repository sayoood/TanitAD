"""TanitEval — REAL forward profiling: encode + predictor + the frozen encoders
REF-A runs at inference. Latency (ms), throughput (fps/wps), peak VRAM. Not
guessed — measured via the models' actual forward API on the eval-pod A40."""
import json
import sys
import torch

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")
from taniteval.registry import MODELS
from taniteval import loaders, profiling

device = "cuda"
B, W = 8, 8


def prof(fn):
    try:
        with torch.no_grad():
            return profiling.time_forward(fn, warmup=5, iters=20, device=device)
    except Exception as e:
        return {"err": f"{type(e).__name__}: {str(e)[:80]}"}


out = {}
for m in MODELS:
    if m["key"] == "refb":
        continue
    L = loaders.load(m)
    model, sd = L["model"], L["state_dim"]
    ad = 3 if m.get("speed_input") else 2
    r = {}
    # trained in-model encoder (flagship only; REF-A's is frozen+external)
    if getattr(model, "encoder", None) is not None:
        x = torch.randn(B, 9, 256, 256, device=device)
        p = prof(lambda: model.encode(x))
        if "latency_ms" in p:
            p["throughput_fps"] = round(B * 1000 / p["latency_ms"], 1)
        r["encode_trained_vit"] = p
    else:
        # REF-A: the temporal adapter (frozen features -> state), d_dino-wide
        feat = torch.randn(B, W, 256, m.get("d_dino", 768), device=device)
        p = prof(lambda: model.encode_window(feat) if hasattr(model, "encode_window")
                 else model.encode(feat))
        r["adapter"] = p
    # shared operative predictor (the world-model step)
    states = torch.randn(B, W, sd, device=device)
    acts = torch.randn(B, W, ad, device=device)
    for call in (lambda: model.predictor(states, acts, intent=None),
                 lambda: model.predictor(states, acts)):
        p = prof(call)
        if "err" not in p:
            r["predictor"] = p
            break
    else:
        r["predictor"] = p
    out[m["key"]] = r
    print(m["key"], json.dumps(r), flush=True)

# frozen encoders — REF-A's REAL per-frame inference cost (external to the ckpt)
try:
    from transformers import Dinov2Model, IJepaModel
    for name, mk in (("DINOv2-B/14", lambda: Dinov2Model.from_pretrained("facebook/dinov2-base")),
                     ("I-JEPA-H/14", lambda: IJepaModel.from_pretrained("facebook/ijepa_vith14_1k"))):
        enc = mk().to(device).eval()
        x = torch.randn(B, 3, 224, 224, device=device)
        p = prof(lambda: enc(pixel_values=x))
        if "latency_ms" in p:
            p["throughput_fps"] = round(B * 1000 / p["latency_ms"], 1)
        out["frozen_" + name] = p
        print("frozen", name, json.dumps(p), flush=True)
        del enc; torch.cuda.empty_cache()
except Exception as e:
    out["frozen_err"] = str(e)[:120]

json.dump(out, open("/root/taniteval/results/forward_profile.json", "w"), indent=2)
print("WROTE forward_profile.json", flush=True)
