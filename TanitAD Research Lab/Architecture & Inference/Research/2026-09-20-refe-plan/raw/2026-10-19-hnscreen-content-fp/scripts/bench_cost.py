"""COST of the content fingerprint on THIS box (CPU only; writes <= 160 MiB under C:, nothing on D:, reads one existing file on D: read-only).
usage: bench_cost.py <refe dir> <scratch dir on C:> [<big existing file to read-only hash>]"""
import hashlib
import json
import os
import shutil
import sys
import time

refe, scratch = sys.argv[1], sys.argv[2]
big = sys.argv[3] if len(sys.argv) > 3 else None
sys.path.insert(0, refe)
import gate_token as GT  # noqa: E402

MiB = 1 << 20
out = {"box": {"cpus": os.cpu_count()}}

# 1. pure sha256 throughput (CPU, in memory)
buf = os.urandom(MiB) * 256                                           # 256 MiB, random-ish
t0 = time.perf_counter()
h = hashlib.sha256()
for i in range(0, len(buf), 4 * MiB):
    h.update(buf[i:i + 4 * MiB])
dt = time.perf_counter() - t0
out["sha256_in_memory_GBps_1thread"] = round(len(buf) / dt / 1e9, 3)
del buf

# 2. a synthetic cache: 5 units of (ctx 28 MiB + fan 4 MiB) = 160 MiB, the shape of the real one (many big files + small sidecars)
d = os.path.join(scratch, "bench_cache")
shutil.rmtree(d, ignore_errors=True)
os.makedirs(os.path.join(d, "units"))
rng_block = os.urandom(MiB)
for u in range(5):
    for name, mib in ((f"u{u}.ctx.u16", 28), (f"u{u}.fan.f32", 4)):
        with open(os.path.join(d, "units", name), "wb") as f:
            for k in range(mib):
                f.write(rng_block[k:] + rng_block[:k])
    with open(os.path.join(d, "units", f"u{u}.json"), "w") as f:
        json.dump({"unit": u}, f)
with open(os.path.join(d, "manifest.json"), "w") as f:
    json.dump({"tool": "ctx_cache/1"}, f)
total = sum(os.path.getsize(os.path.join(r, fn)) for r, _, fs in os.walk(d) for fn in fs)
out["synthetic_cache_MiB"] = round(total / MiB, 1)
t0 = time.perf_counter()
GT.seal_content_manifest(d)
out["seal_seconds_threads_default"] = round(time.perf_counter() - t0, 3)
rows = {}
for thr in (1, 2, 4, 8, 16):
    os.environ[GT.HASH_THREADS_ENV] = str(thr)
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        fp = GT.content_fingerprint(d)
        best = min(best, time.perf_counter() - t0)
        assert fp["n_problems"] == 0
    rows[thr] = {"seconds": round(best, 3), "GBps": round(total / best / 1e9, 3)}
out["content_fingerprint_warm_cache_by_threads"] = rows
os.environ.pop(GT.HASH_THREADS_ENV, None)
t0 = time.perf_counter()
for _ in range(3):
    GT.fingerprint(d)
out["listing_fingerprint_seconds_each"] = round((time.perf_counter() - t0) / 3, 4)
shutil.rmtree(d, ignore_errors=True)

# 3. a REAL large file, read-only (a checkpoint on D:): the first read is as cold as this box makes it, the second is warm
if big and os.path.isfile(big):
    size = os.path.getsize(big)
    res = []
    for tag in ("first_read", "second_read"):
        t0 = time.perf_counter()
        n, sha = GT._sha256_and_size(big)
        dt = time.perf_counter() - t0
        res.append({"read": tag, "bytes": n, "seconds": round(dt, 2), "GBps": round(n / dt / 1e9, 3)})
    out["real_file_read_and_hash"] = {"path": big, "runs": res}

# 4. extrapolation to the screen's cache: 19,000 scenes x 3.75 MiB (PREREG_P6HN_SCREEN 3 / S2) + fans (< 1 %)
gib = 19000 * 3.75 / 1024
out["screen_cache_GiB_estimate"] = round(gib, 1)
for k in ("sha256_in_memory_GBps_1thread",):
    pass
best_thr = max(rows.values(), key=lambda r: r["GBps"])
out["screen_cache_one_pass_seconds_at_best_warm_rate"] = round(gib * 1.073741824e9 / (best_thr["GBps"] * 1e9), 1)
if "real_file_read_and_hash" in out:
    cold = out["real_file_read_and_hash"]["runs"][0]["GBps"]
    out["screen_cache_one_pass_seconds_at_first_read_rate"] = round(gib * 1.073741824e9 / (cold * 1e9), 1)
print(json.dumps(out, indent=1))
