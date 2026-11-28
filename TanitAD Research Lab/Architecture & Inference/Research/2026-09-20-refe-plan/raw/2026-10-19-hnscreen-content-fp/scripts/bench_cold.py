import sys, time
sys.path.insert(0, sys.argv[1])
import gate_token as GT
for p in sys.argv[2:]:
    t0 = time.perf_counter(); n, sha = GT._sha256_and_size(p); dt = time.perf_counter() - t0
    print(f"{p}: {n} bytes {dt:.1f} s {n/dt/1e6:.0f} MB/s (first read)")
