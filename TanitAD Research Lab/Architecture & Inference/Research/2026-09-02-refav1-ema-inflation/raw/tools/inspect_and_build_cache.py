"""Inspect the 20-clip slice and build a self-contained --cache dir for refa_v1_train.py.

- prints dtype/shape of one fp8 feature file, keys of one v2ep, index.json geometry
- checks the 20 clip_ids against the eval labels blob
- builds C:/Users/Admin/tsc2/cache/<uuid>.pt as NTFS hard links to the slice files
  (no copy, no modification of the slice dir) + an index.json carrying the slice's
  geometry/parity fields and exactly the 20 episode names.
"""
import gzip, json, os, sys
from pathlib import Path
import torch

SLICE = Path(r"C:/Users/Admin/refav1_eval_slice")
FP8 = SLICE / "fp8"
EPS = SLICE / "eps"
OUT = Path(r"C:/Users/Admin/tsc2/cache")
LABELS = Path(r"C:/Users/Admin/tanitad-wt/_s2build/release/v72/s2_labels_v7.2_eval.jsonl.gz")

names = sorted(p.stem for p in FP8.glob("*.pt"))
print("n fp8 files:", len(names))
assert len(names) == 20, names

# --- one feature file
f0 = torch.load(FP8 / f"{names[0]}.pt", map_location="cpu", weights_only=True)
print("feat type:", type(f0))
if torch.is_tensor(f0):
    print("feat dtype/shape:", f0.dtype, tuple(f0.shape))
    x = f0[:2].float()
    print("feat sample mean/std/absmax:", float(x.mean()), float(x.std()), float(x.abs().max()))
else:
    print("feat keys:", list(f0.keys()) if hasattr(f0, "keys") else f0)

# --- all files: T_c
Ts = {}
for nm in names:
    t = torch.load(FP8 / f"{nm}.pt", map_location="cpu", weights_only=True, mmap=True)
    Ts[nm] = tuple(t.shape) if torch.is_tensor(t) else None
print("shapes:", sorted(set(Ts.values())))

# --- one episode
e0 = torch.load(EPS / f"{names[0]}.v2ep.pt", map_location="cpu", weights_only=False)
print("v2ep keys:", sorted(e0.keys()))
print("poses:", tuple(e0["poses"].shape), "actions:", tuple(e0["actions"].shape),
      "clip_id:", e0.get("clip_id"), "codec:", e0.get("codec"))
bad = []
clip_ids = {}
for nm in names:
    o = torch.load(EPS / f"{nm}.v2ep.pt", map_location="cpu", weights_only=False)
    T_ep = int(o["poses"].shape[0])
    want = -(-T_ep // 2)
    clip_ids[nm] = o.get("clip_id")
    if Ts[nm] is None or Ts[nm][0] != want:
        bad.append((nm, Ts[nm], T_ep, want))
print("grid mismatches:", bad)
print("clip_id == stem for all:", all(clip_ids[nm] == nm for nm in names))

# --- index.json geometry
idx = json.loads((SLICE / "index.json").read_text(encoding="utf-8"))
print("index keys:", sorted(idx.keys()))
print("index n episodes:", len(idx.get("episodes", [])))
print("geometry:", idx.get("geometry"))
print("parity_key:", idx.get("parity_key"), "skip_hash:", idx.get("skip_hash"))
for k in idx:
    if k != "episodes":
        print("  index[%s] =" % k, json.dumps(idx[k])[:300])
print("slice names in index episodes:", sum(nm in set(idx.get("episodes", [])) for nm in names), "/ 20")

# --- labels blob
recs = []
with gzip.open(LABELS, "rt", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line:
            recs.append(json.loads(line))
print("labels blob records:", len(recs))
blob_clips = {r.get("clip_id") for r in recs}
hit = [nm for nm in names if clip_ids[nm] in blob_clips]
print("slice clips found in eval blob:", len(hit), "/ 20")
print("record keys (first):", sorted(recs[0].keys()))
print("first record (trunc):", json.dumps(recs[0])[:600])

# --- build the cache dir with hard links
OUT.mkdir(parents=True, exist_ok=True)
for nm in names:
    dst = OUT / f"{nm}.pt"
    src = FP8 / f"{nm}.pt"
    if dst.exists():
        if dst.stat().st_size == src.stat().st_size:
            continue
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError as e:
        print("hardlink failed, copying:", nm, e)
        import shutil
        shutil.copyfile(src, dst)
new_idx = {k: v for k, v in idx.items() if k != "episodes"}
new_idx["episodes"] = names
new_idx["derived_from"] = str(SLICE / "index.json")
new_idx["note"] = ("E-ARCH-TSC-2 20-clip EVAL slice, hard-linked from refav1_eval_slice/fp8; "
                   "geometry/parity fields copied verbatim from the slice index.json")
(OUT / "index.json").write_text(json.dumps(new_idx, indent=1), encoding="utf-8")
print("cache dir:", OUT, "files:", len(list(OUT.glob("*.pt"))))
print("OK")
