"""Measure (a) full-read throughput and (b) whether mmap=True lets us read poses only."""
import sys, time, hashlib, os, glob
import torch

cache = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
files = sorted(glob.glob(os.path.join(cache, "ep_*.pt")))[:n]

# --- mmap probe: read ONLY poses ---
t0 = time.time()
ok_mmap = True
try:
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False, mmap=True)
        p = d["poses"].contiguous().numpy().tobytes()
        h = hashlib.sha256(p).hexdigest()
        del d
except Exception as e:
    ok_mmap = False
    print("MMAP FAILED:", type(e).__name__, str(e)[:200])
t_mmap = time.time() - t0
print(f"mmap poses-only: {n} eps in {t_mmap:.2f}s  -> {t_mmap/max(n,1)*1000:.0f} ms/ep  ok={ok_mmap}")

# --- full read ---
tot = 0
t0 = time.time()
for f in files:
    d = torch.load(f, map_location="cpu", weights_only=False)
    fr = d["frames_u8"]
    h = hashlib.sha256(fr.contiguous().numpy().tobytes()).hexdigest()
    tot += fr.numel()
    del d
t_full = time.time() - t0
print(f"full read+hash: {n} eps in {t_full:.2f}s -> {t_full/max(n,1):.2f} s/ep, "
      f"{tot/1e6/max(t_full,1e-9):.0f} MB/s, T-total={tot}")

# --- mmap partial frames (strided) ---
t0 = time.time()
for f in files:
    d = torch.load(f, map_location="cpu", weights_only=False, mmap=True)
    fr = d["frames_u8"]
    T = fr.shape[0]
    idxs = sorted(set([0, T // 4, T // 2, 3 * T // 4, T - 1]))
    hh = hashlib.sha256()
    for t in idxs:
        hh.update(fr[t].contiguous().numpy().tobytes())
    del d
t_str = time.time() - t0
print(f"mmap strided-5-frames: {n} eps in {t_str:.2f}s -> {t_str/max(n,1)*1000:.0f} ms/ep")
