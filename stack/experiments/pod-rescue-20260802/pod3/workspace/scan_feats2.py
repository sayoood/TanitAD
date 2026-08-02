"""Definitive corrupt-feature scan: resolve every mix symlink to its real
target, check (a) size sane, (b) torch.load works, (c) has feats_fp16. Remove
bad SOURCE files so dino_precompute regenerates them. Reports source dir counts."""
import glob, os, torch
targets = set()
for mdir in ("/workspace/mixfeats/mix-train-v1", "/workspace/mixfeats/mix-val-v1"):
    for lnk in glob.glob(f"{mdir}/ep_*.pt"):
        targets.add(os.path.realpath(lnk))
print(f"{len(targets)} unique real feature files behind the mix")
bad = []
for f in sorted(targets):
    try:
        sz = os.path.getsize(f)
        if sz < 100_000:                       # truncated (normal >> 1MB)
            raise RuntimeError(f"tiny {sz}B")
        d = torch.load(f, map_location="cpu", weights_only=True)
        if "feats_fp16" not in d:
            raise RuntimeError("no feats_fp16")
    except Exception as e:
        bad.append(f)
        print(f"CORRUPT {f}: {type(e).__name__} {e}", flush=True)
print(f"\n{len(bad)} corrupt")
for f in bad:
    os.remove(f)
print("SCAN2_DONE removed", len(bad))
