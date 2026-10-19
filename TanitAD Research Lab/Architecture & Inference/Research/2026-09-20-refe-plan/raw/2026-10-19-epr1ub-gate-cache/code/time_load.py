import sys, time
sys.path.insert(0, sys.argv[1])
import epr1ub_gate as G
t0 = time.perf_counter(); G.load_verified_ub(sys.argv[1]); first = time.perf_counter() - t0
ts = []
for _ in range(20):
    t0 = time.perf_counter(); G.load_verified_ub(sys.argv[1]); ts.append(time.perf_counter() - t0)
ts.sort()
print(f"first {first*1e3:.1f} ms; next 20: median {ts[10]*1e3:.1f} ms, min {ts[0]*1e3:.1f} ms, max {ts[-1]*1e3:.1f} ms")
