"""P4 -- v2 corpus train-readiness smoke (READ-ONLY on the corpus).

Runs the EXACT data path ``scripts/train_flagship4b.py`` takes under ``--v2-cache``
(lines ~330-350): ``build_v2_providers`` -> ``_wrap`` (FlagshipWindowDataset) ->
``DataLoader``, at the real flagship config, and reports:

  A  load_compressed() vs the lazy loader: byte-identical frames/poses/actions
     on a random sample of REAL built clips (pins the loader against the builder).
  B  provider construction over the WHOLE shard: manifest build time + RSS.
  C  window contract: key set / shapes / dtypes of a real batch.
  D  throughput: windows/s and peak RSS at a realistic worker count.

Uses a repo-synced SHADOW stack (``--stack``) so a drifted pod checkout cannot
decide the answer.
"""
from __future__ import annotations
import argparse, json, os, random, resource, sys, time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--stack", required=True, help="repo-synced stack root")
ap.add_argument("--cache", nargs="+", required=True)
ap.add_argument("--pod-stack", default="/workspace/TanitAD/stack")
ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--v2-lru", type=int, default=64)
ap.add_argument("--batches", type=int, default=40)
ap.add_argument("--sample", type=int, default=8)
ap.add_argument("--config", default="flagship4b")
a = ap.parse_args()

sys.path.insert(0, a.stack)
sys.path.insert(1, os.path.join(a.stack, "scripts"))

from tanitad.config import flagship4b_config, flagship4b_reduced_config  # noqa: E402
from tanitad.data.v2_dataset import (build_v2_providers,                 # noqa: E402
                                     decode_full_episode)
from tanitad.train.flagship_losses import horizon_plan                   # noqa: E402
import train_flagship4b as T4                                            # noqa: E402

R = {"stack": a.stack, "cache": a.cache, "host": os.uname().nodename,
     "torch": torch.__version__,
     "v2_dataset_from": sys.modules["tanitad.data.v2_dataset"].__file__,
     "train_flagship4b_from": T4.__file__}


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def rss_tree_gb():
    tot = 0
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            with open(f"/proc/{p}/statm") as f:
                tot += int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except OSError:
            pass
    return tot / 1024 ** 3


# ---- A: load_compressed (builder) vs the lazy loader (trainer) --------------
sys.path.append(os.path.join(a.pod_stack, "scripts"))     # v2_compressed lives here
A = {"checked": 0, "frames_identical": 0, "poses_identical": 0,
     "actions_identical": 0, "mismatch": []}
try:
    import v2_compressed as VC                                          # noqa: E402
    A["v2_compressed_from"] = VC.__file__
    files = sorted(f for f in os.listdir(a.cache[0]) if f.endswith(".v2ep.pt"))
    random.Random(20260725).shuffle(files)
    for fn in files[:a.sample]:
        p = os.path.join(a.cache[0], fn)
        ref = VC.load_compressed(p)                # the BUILDER's own reader
        lazy = decode_full_episode(p)              # the TRAINER's reader
        fo = bool(torch.equal(ref.frames, lazy.frames))
        po = bool(torch.equal(ref.poses.float(), lazy.poses))
        ao = bool(torch.equal(ref.actions.float(), lazy.actions))
        A["checked"] += 1
        A["frames_identical"] += fo
        A["poses_identical"] += po
        A["actions_identical"] += ao
        if not (fo and po and ao):
            A["mismatch"].append({"file": fn, "frames": fo, "poses": po,
                                  "actions": ao})
        A.setdefault("shapes", []).append(
            {"file": fn, "frames": list(ref.frames.shape),
             "maneuvers": list(ref.maneuvers.shape) if ref.maneuvers is not None else None,
             "man_hist": torch.bincount(ref.maneuvers, minlength=5).tolist()
             if ref.maneuvers is not None else None})
        del ref, lazy
except Exception as e:                                                  # noqa: BLE001
    A["error"] = f"{type(e).__name__}: {str(e)[:300]}"
R["A_builder_vs_loader"] = A

# ---- B: providers over the WHOLE shard -------------------------------------
cfg = (flagship4b_config() if a.config == "flagship4b"
       else flagship4b_reduced_config())
plan = horizon_plan(cfg, op_fwd_k=4, tac_fwd_k=16, str_fwd_k=20)
t0 = time.time()
providers = build_v2_providers(a.cache, lru_size=a.v2_lru, verbose=True)
R["B_providers"] = {
    "n_providers": len(providers), "build_seconds": round(time.time() - t0, 1),
    "rss_after_manifest_gb": round(rss_gb(), 3),
    "frames_shape_0": list(providers[0].frames.shape),
    "poses_shape_0": list(providers[0].poses.shape),
    "actions_shape_0": list(providers[0].actions.shape),
    "T_out_sum": int(sum(int(p.frames.shape[0]) for p in providers)),
    "channels": int(providers[0].frames.shape[1]),
    "image_size": int(providers[0].frames.shape[-1])}

ds = T4._wrap(providers, cfg, plan, cfg.encoder.in_channels)
R["B_providers"]["window"] = int(cfg.predictor.window)
R["B_providers"]["max_horizon"] = int(plan.max_horizon)
R["B_providers"]["maneuver_h"] = int(plan.maneuver_h)
R["B_providers"]["labels_v2"] = bool(cfg.v2_labels)
R["B_providers"]["n_windows"] = len(ds)

# ---- C: window contract -----------------------------------------------------
it = ds[len(ds) // 3]
R["C_window_contract"] = {
    "n_keys": len(it), "keys": sorted(it),
    "spec": {k: ([list(v.shape), str(v.dtype)] if torch.is_tensor(v)
                 else [type(v).__name__, v]) for k, v in sorted(it.items())}}

# ---- D: throughput ----------------------------------------------------------
from torch.utils.data import DataLoader                                  # noqa: E402
dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, drop_last=True,
                num_workers=a.workers, persistent_workers=bool(a.workers))
t0, n, first_batch = time.time(), 0, None
peak_tree = 0.0
for b in dl:
    if first_batch is None:
        first_batch = {k: (list(v.shape) if torch.is_tensor(v) else str(v))
                       for k, v in sorted(b.items())}
        t0 = time.time()                              # exclude worker warm-up
        continue
    n += 1
    if n % 10 == 0:
        peak_tree = max(peak_tree, rss_tree_gb())
    if n >= a.batches:
        break
el = time.time() - t0
R["D_throughput"] = {
    "workers": a.workers, "batch_size": a.batch_size, "v2_lru": a.v2_lru,
    "batches_timed": n, "seconds": round(el, 2),
    "batches_per_s": round(n / el, 3), "windows_per_s": round(n * a.batch_size / el, 1),
    "rss_self_peak_gb": round(rss_gb(), 3),
    "rss_process_tree_peak_gb": round(peak_tree, 2),
    "batch_shapes": first_batch}
del dl

json.dump(R, open(a.out, "w"), indent=2)
print(json.dumps(R, indent=2)[:9000], flush=True)
print(f"[p4] wrote {a.out}", flush=True)
