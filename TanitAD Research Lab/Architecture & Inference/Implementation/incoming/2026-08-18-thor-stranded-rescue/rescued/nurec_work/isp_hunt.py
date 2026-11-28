"""Probe: find the trained PPISP (exposure/vignetting/colour/CRF) parameters
inside volume.nurec's msgpack. Case-insensitive key search across the WHOLE tree."""
import msgpack, re, sys, json
import numpy as np

PATH = "/home/nvidia/nurec_work/x/volume.msgpack"
with open(PATH, "rb") as f:
    raw = f.read()
obj = msgpack.unpackb(raw, raw=False, strict_map_key=False)
nre = obj["nre_data"]
print("nre_data keys:", list(nre.keys()))
for k in nre.keys():
    v = nre[k]
    print(f"  {k}: {type(v).__name__}", (f"n={len(v)}" if hasattr(v, "__len__") else v))

PAT = re.compile(r"isp|exposure|vignet|crf|tonemap|tone_map|white_?bal|awb|gamma|srgb|"
                 r"color|colour|ccm|response|radiom|photom|post_?proc|camera_model|"
                 r"affine|gain|bias|brightness|contrast", re.I)

hits = []
def walk(v, path):
    if isinstance(v, dict):
        for k, vv in v.items():
            ks = str(k)
            p = f"{path}/{ks}"
            if PAT.search(ks):
                hits.append((p, vv))
            walk(vv, p)
    elif isinstance(v, list):
        for i, vv in enumerate(v[:4]):
            walk(vv, f"{path}[{i}]")

walk(nre, "")
print(f"\n=== {len(hits)} key hits ===")
def desc(v):
    if isinstance(v, dict): return f"dict(n={len(v)}) keys={list(v.keys())[:12]}"
    if isinstance(v, list): return f"list(n={len(v)}) head={v[:6]}"
    if isinstance(v, (bytes, bytearray)): return f"bytes(len={len(v)})"
    return f"{type(v).__name__} {v!r}"[:200]
for p, v in hits:
    print(f"{p}: {desc(v)}")

# --- state_dict: enumerate ALL keys that are NOT .gaussians_nodes.* ------------
sd = nre["state_dict"]
print(f"\n=== state_dict has {len(sd)} entries ===")
non_gauss = [k for k in sd if not str(k).startswith(".gaussians_nodes.")]
print(f"--- {len(non_gauss)} entries outside .gaussians_nodes.* ---")
for k in sorted(non_gauss):
    v = sd[k]
    print(f"  {k}: {desc(v)}")

print("\n--- ALL state_dict keys matching the ISP pattern ---")
for k in sorted(sd):
    if PAT.search(str(k)):
        print(f"  {k}: {desc(sd[k])}")
