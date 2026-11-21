"""P4b -- builder-vs-loader equivalence on a real sample + worker scaling with a
SUBTREE-ONLY RSS measurement (p4's tree RSS counted the whole pod). READ-ONLY."""
from __future__ import annotations
import argparse, json, os, random, resource, sys, time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--stack", required=True)
ap.add_argument("--cache", nargs="+", required=True)
ap.add_argument("--pod-stack", default="/workspace/TanitAD/stack")
ap.add_argument("--out", required=True)
ap.add_argument("--sample", type=int, default=24)
ap.add_argument("--worker-grid", default="4,8,16")
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--batches", type=int, default=40)
ap.add_argument("--v2-lru", type=int, default=64)
a = ap.parse_args()

sys.path.insert(0, a.stack)
sys.path.insert(1, os.path.join(a.stack, "scripts"))
from tanitad.config import flagship4b_config                             # noqa: E402
from tanitad.data.v2_dataset import (build_v2_providers,                 # noqa: E402
                                     decode_full_episode)
from tanitad.train.flagship_losses import horizon_plan                   # noqa: E402
import train_flagship4b as T4                                            # noqa: E402
from torch.utils.data import DataLoader                                  # noqa: E402

R = {"host": os.uname().nodename, "torch": torch.__version__, "cache": a.cache}


def _kids(pid, acc):
    acc.append(pid)
    try:
        for t in os.listdir(f"/proc/{pid}/task"):
            with open(f"/proc/{pid}/task/{t}/children") as f:
                for c in f.read().split():
                    _kids(int(c), acc)
    except OSError:
        pass
    return acc


def subtree_rss_gb():
    tot = 0
    for p in set(_kids(os.getpid(), [])):
        try:
            with open(f"/proc/{p}/statm") as f:
                tot += int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except OSError:
            pass
    return tot / 1024 ** 3


# ---- A: the BUILDER's reader vs the TRAINER's reader, on real clips --------
sys.path.append(os.path.join(a.pod_stack, "scripts"))
A = {"checked": 0, "frames_identical": 0, "poses_identical": 0,
     "actions_identical": 0, "mismatch": [], "maneuver_field": []}
import v2_compressed as VC                                               # noqa: E402
A["v2_compressed_from"] = VC.__file__
files = sorted(f for f in os.listdir(a.cache[0]) if f.endswith(".v2ep.pt"))
random.Random(20260725).shuffle(files)
for fn in files[:a.sample]:
    p = os.path.join(a.cache[0], fn)
    ref = VC.load_compressed(p)
    lazy = decode_full_episode(p)
    fo = bool(torch.equal(ref.frames, lazy.frames))
    po = bool(torch.equal(ref.poses.float(), lazy.poses))
    ao = bool(torch.equal(ref.actions.float(), lazy.actions))
    A["checked"] += 1
    A["frames_identical"] += fo
    A["poses_identical"] += po
    A["actions_identical"] += ao
    if not (fo and po and ao):
        A["mismatch"].append({"file": fn, "frames": fo, "poses": po, "actions": ao})
    m = ref.maneuvers
    if A["checked"] <= 3:
        A["maneuver_field"].append(
            {"file": fn, "shape": None if m is None else list(m.shape),
             "dtype": None if m is None else str(m.dtype),
             "min": None if m is None else float(m.min()),
             "max": None if m is None else float(m.max()),
             "frames_shape": list(ref.frames.shape)})
    del ref, lazy
R["A_builder_vs_loader"] = A
print(json.dumps(A, indent=1), flush=True)

# ---- D: worker scaling with subtree-only RSS -------------------------------
cfg = flagship4b_config()
plan = horizon_plan(cfg, op_fwd_k=4, tac_fwd_k=16, str_fwd_k=20)
t0 = time.time()
providers = build_v2_providers(a.cache, lru_size=a.v2_lru, verbose=True)
R["manifest_seconds_warm"] = round(time.time() - t0, 2)
ds = T4._wrap(providers, cfg, plan, cfg.encoder.in_channels)
R["n_windows"] = len(ds)
R["window_bytes_est_mb"] = round(
    (cfg.predictor.window + plan.max_horizon) * 9 * 256 * 256 * 4 / 1e6, 1)

R["D_scaling"] = []
base_rss = subtree_rss_gb()
for w in [int(x) for x in a.worker_grid.split(",")]:
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, drop_last=True,
                    num_workers=w, persistent_workers=bool(w))
    it = iter(dl)
    next(it)                                       # warm-up batch, not timed
    t0, peak = time.time(), base_rss
    n = 0
    for _ in range(a.batches):
        next(it)
        n += 1
        if n % 8 == 0:
            peak = max(peak, subtree_rss_gb())
    el = time.time() - t0
    row = {"workers": w, "batches": n, "seconds": round(el, 2),
           "windows_per_s": round(n * a.batch_size / el, 1),
           "mb_per_s_frames": round(n * a.batch_size * R["window_bytes_est_mb"] / el, 0),
           "subtree_rss_peak_gb": round(peak, 2),
           "rss_self_gb": round(resource.getrusage(
               resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 2)}
    R["D_scaling"].append(row)
    print(json.dumps(row), flush=True)
    del it, dl

json.dump(R, open(a.out, "w"), indent=2)
print(f"[p4b] wrote {a.out}", flush=True)
