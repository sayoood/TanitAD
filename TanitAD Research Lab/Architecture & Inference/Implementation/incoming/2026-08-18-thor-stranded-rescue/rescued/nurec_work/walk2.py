import msgpack, json, zipfile, pickletools, sys

PATH = "/home/nvidia/nurec_work/x/volume.msgpack"
with open(PATH, "rb") as f:
    raw = f.read()
d = msgpack.unpackb(raw, raw=False, strict_map_key=False)
nre = d["nre_data"]
sd = nre["state_dict"]
cfg = nre["config"]

print("=" * 70)
print("CONFIG / layers / background  (19 keys)")
print(json.dumps(cfg["layers"]["background"], indent=1, default=str)[:4000])
print("=" * 70)
print("CONFIG / layers / road  (20 keys)")
print(json.dumps(cfg["layers"]["road"], indent=1, default=str)[:4000])
print("=" * 70)
print("CONFIG / renderer")
print(json.dumps(cfg["renderer"], indent=1, default=str)[:3000])

print("=" * 70)
print("FIRST 24 BYTES OF EACH GAUSSIAN ARRAY (looking for npy/dtype headers)")
for k, v in sd.items():
    if ".gaussians_nodes." in k and isinstance(v, (bytes, bytearray)) and len(v) > 0:
        print(f"{len(v):>12d}  {v[:24].hex()}  {k}")

print("=" * 70)
print("SCALAR-ISH ARRAYS (<=64 bytes) raw hex")
for k, v in sd.items():
    if isinstance(v, (bytes, bytearray)) and 0 < len(v) <= 64:
        print(f"{len(v):>5d} {v.hex()}  {k}")

print("=" * 70)
print("ALL state_dict keys not already shown (non-gaussians_nodes) with sizes")
for k, v in sd.items():
    if ".gaussians_nodes." not in k:
        t = type(v).__name__
        n = len(v) if isinstance(v, (bytes, bytearray, dict, list)) else v
        print(f"{t:>6s} {n}  {k}")
