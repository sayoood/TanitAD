import sys, time, torch, hashlib
p = sys.argv[1]
t0 = time.time()
d = torch.load(p, map_location="cpu", weights_only=False)
print("load_s", round(time.time() - t0, 2))
print("type", type(d).__name__)
if isinstance(d, dict):
    for k, v in d.items():
        try:
            print("  ", k, type(v).__name__, getattr(v, "dtype", ""), tuple(getattr(v, "shape", ())))
        except Exception:
            print("  ", k, repr(v)[:100])
