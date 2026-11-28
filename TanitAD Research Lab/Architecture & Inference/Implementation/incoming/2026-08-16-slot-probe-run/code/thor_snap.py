"""Make an fp16 weights-only snapshot of the LIVE v6F ckpt, on Thor, CPU-only.

⚠️ Thor is TRAINING. This is deliberately memory-light: the source is opened
with mmap=True (lazy, page-cache backed, reclaimable) and each tensor is cast
one at a time, so the resident cost is the ~0.67 GB output, not the 3.53 GB
input. No GPU is touched. It writes the SAME layout train_v6_staged already
emits ({"model", "_meta", "_fp16_weights_only"}), so the probe's reader needs
no special case.
"""
import json, os, torch, sys
src = "/home/nvidia/experiments/v6F-SW-30k/ckpt.pt"
dst = "/home/nvidia/v6F_snap_fp16.pt"
ck = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
step = int(ck.get("step", -1))
cfg = ck.get("config")
if cfg is None:
    cfg = json.load(open("/home/nvidia/experiments/v6F-SW-30k/config.json"))
sd = ck["stack"]
out = {}
for k in list(sd.keys()):
    v = sd[k]
    out[k] = v.detach().to(torch.float16).clone() if v.is_floating_point() else v.detach().clone()
torch.save({"model": out, "_meta": {"step": step, "config": cfg},
            "_fp16_weights_only": True}, dst)
print("SNAPOK step=%d n=%d bytes=%d" % (step, len(out), os.path.getsize(dst)))
