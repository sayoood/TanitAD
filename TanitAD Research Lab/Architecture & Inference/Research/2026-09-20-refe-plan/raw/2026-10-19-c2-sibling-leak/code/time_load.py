"""Cost of a repeat load_verified_pair on the fixed gate: python -B -I time_load.py <tree>   (first = cold process, then 15 repeats)"""
import os, sys, time
tree = os.path.abspath(sys.argv[1]); sys.path.insert(0, tree)
import srttc_c2_gate as CG
t0 = time.perf_counter(); CG.load_verified_pair(tree, {}); first = time.perf_counter() - t0
ts = []
for _ in range(15):
    t0 = time.perf_counter(); CG.load_verified_pair(tree, {}); ts.append(time.perf_counter() - t0)
ts.sort()
print(f"first {first*1e3:.1f} ms; next 15: median {ts[7]*1e3:.1f} ms, min {ts[0]*1e3:.1f} ms, max {ts[-1]*1e3:.1f} ms")
