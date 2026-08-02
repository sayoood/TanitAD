"""Scan DINO feature dirs for corrupt/truncated .pt files; delete them so
dino_precompute regenerates them. Reports which real files (not symlinks) are bad."""
import glob, os, torch
bad = []
for d in ("comma2k19-train", "comma2k19-val", "physicalai-train", "physicalai-val"):
    for f in glob.glob(f"/workspace/dino_feats/{d}-*dinov2-b14/ep_*.pt"):
        try:
            torch.load(f, map_location="cpu", weights_only=True)
        except Exception as e:
            bad.append(f)
            print(f"CORRUPT {f}: {type(e).__name__}", flush=True)
print(f"\n{len(bad)} corrupt files")
for f in bad:
    os.remove(f)
    print(f"removed {f}")
print("SCAN_DONE")
