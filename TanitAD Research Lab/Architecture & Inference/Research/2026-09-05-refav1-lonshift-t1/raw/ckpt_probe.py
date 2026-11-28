import sys, hashlib, torch

p = sys.argv[1]
d = None
for kw in ({"mmap": True, "weights_only": True}, {"weights_only": True},
           {"mmap": True, "weights_only": False}, {"weights_only": False}):
    try:
        d = torch.load(p, map_location="cpu", **kw)
        print("LOADED_WITH", kw)
        break
    except Exception as e:
        print("LOADFAIL", kw, type(e).__name__, str(e)[:120])
if d is None:
    print("ALL_LOADS_FAILED"); sys.exit(1)

print("TOPKEYS", sorted([str(k) for k in d.keys()])[:25] if isinstance(d, dict) else type(d))
step = d.get("step", d.get("global_step", "ABSENT")) if isinstance(d, dict) else "ABSENT"
print("STEP", step)

sd = None
for k in ("model", "state_dict", "model_state_dict", "ema"):
    if isinstance(d, dict) and isinstance(d.get(k), dict):
        sd = d[k]; print("SDKEY", k); break
if sd is None and isinstance(d, dict):
    cand = {k: v for k, v in d.items() if torch.is_tensor(v)}
    if cand:
        sd = cand; print("SDKEY", "<toplevel tensors>")
if sd is None:
    print("NO_STATE_DICT"); sys.exit(0)

keys = sorted(str(k) for k in sd.keys())
print("NPARAMKEYS", len(keys))
h = hashlib.sha256(); h2 = hashlib.sha256()
tot = 0.0; n = 0
for k in keys:
    v = sd[k]
    if not torch.is_tensor(v):
        continue
    h.update(k.encode()); h.update(str(tuple(v.shape)).encode()); h.update(str(v.dtype).encode())
    if v.numel() == 0:
        continue
    f = v.detach().float().reshape(-1)
    tot += float(f.sum()); n += f.numel()
    take = torch.cat([f[:8], f[-8:]])
    h2.update(k.encode())
    h2.update(b"".join(("%.6e" % float(x)).encode() for x in take))
print("SHAPE_DIGEST", h.hexdigest()[:32])
print("CONTENT_DIGEST", h2.hexdigest()[:32])
print("PARAM_SUM", "%.6f" % tot)
print("PARAM_NUMEL", n)
