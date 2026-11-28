"""check_banked_n1d.py BN -- the banked N1D inputs the N1DC chain reads exist: old-executor eval-141 dumps (follow + Z, 141 episodes each), ticks, and the six T0 passes (HF tar navrbd_read_outputs.tar, sha256 143f3b2a)."""
import os
import sys

bn = sys.argv[1]
bad = []
for s in (0, 1):
    for z in ("", "Z"):
        d = f"{bn}/dumps141/dump_NAVRBD-N1D_s{s}{z}_m10_causal_heads"
        n = len([f for f in os.listdir(d) if f.startswith("ep") and f.endswith(".npz")]) if os.path.isdir(d) else -1
        if n != 141:
            bad.append((d, n))
        if not os.path.isfile(f"{bn}/dumps141/ticks/ticks_NAVRBD-N1D_s{s}{z}_m10_causal_heads.npz"):
            bad.append((f"ticks s{s}{z}", "missing"))
    for v in ("real", "donor", "zero"):
        if not os.path.isfile(f"{bn}/out/t0/eval_NAVRBD-N1D_s{s}_{v}.npz"):
            bad.append((f"t0 s{s} {v}", "missing"))
print("banked N1D inputs:", "OK" if not bad else bad)
sys.exit(1 if bad else 0)
