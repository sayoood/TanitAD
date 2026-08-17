"""Snapshot the LIVE v6F ckpt to fp16, on Thor, CPU-only, mmap'd (measured safe 2026-08-16)."""
import json, os, torch
src = "/home/nvidia/experiments/v6F-SW-30k/ckpt.pt"
ck = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
step = int(ck.get("step", -1))
cfg = ck.get("config") or json.load(open("/home/nvidia/experiments/v6F-SW-30k/config.json"))
sd = ck["stack"]
out = {}
for k in list(sd.keys()):
    v = sd[k]
    out[k] = v.detach().to(torch.float16).clone() if v.is_floating_point() else v.detach().clone()
dst = "/home/nvidia/ckpt_snaps/v6F_sw_step%06d.fp16.pt" % step
torch.save({"model": out, "_meta": {"step": step, "config": cfg}, "_fp16_weights_only": True}, dst)
print("SNAPOK step=%d n=%d bytes=%d dst=%s" % (step, len(out), os.path.getsize(dst), dst))
