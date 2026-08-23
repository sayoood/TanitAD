import json, os, torch
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
