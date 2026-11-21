"""Extract the anchor vocabulary buffer from REF-C checkpoints (CPU only)."""
import json, sys, torch

out = {}
for arm, p in (("refc-base-30k", "/home/nvidia/models/refc-base/ckpt.pt"),
               ("refc-xl-30k",   "/home/nvidia/models/refc-xl/ckpt.pt")):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("state_dict", ck))
    keys = [k for k in sd if "anchor" in k.lower()]
    print(arm, "anchor-ish keys:", keys, flush=True)
    a = None
    for k in keys:
        v = sd[k]
        if torch.is_tensor(v) and v.dim() == 3 and v.shape[-1] == 2:
            a = v.float().contiguous(); print("  ->", k, tuple(v.shape), flush=True)
            break
    if a is None:
        print("  !! no anchor buffer found; step=", ck.get("step")); continue
    out[arm] = a
    print(f"  step={ck.get('step')} shape={tuple(a.shape)}", flush=True)
    del ck, sd
torch.save(out, "/home/nvidia/refc_anchor_vocab.pt")
print(json.dumps({k: list(v.shape) for k, v in out.items()}))
