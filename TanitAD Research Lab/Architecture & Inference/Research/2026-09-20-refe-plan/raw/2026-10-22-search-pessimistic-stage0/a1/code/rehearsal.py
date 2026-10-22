#!/usr/bin/env python3
"""Dress rehearsal: the WHOLE `run_all` (every arm, every control, the decision) on one full-size SYNTHETIC world with the real geometry.  It proves the pipeline runs end to end and shows what the
verdict looks like when the truth is known (kappa = 0: no session effect; kappa > 0: a planted one).  It reads no real token.

    CUDA_VISIBLE_DEVICES=-1 python rehearsal.py <kappa> <seed> [out.json]
"""
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage0 as S0  # noqa: E402
import synth  # noqa: E402

kappa, seed = float(sys.argv[1]), int(sys.argv[2])
out = sys.argv[3] if len(sys.argv) > 3 else None
w = synth.make_world(seed=seed, kappa=kappa, D=256, frac=1.0, b_hi=5.0, het=0.7)
d = S0.make_design(w)
t = time.time()
res = S0.run_all(w, d, None, log=lambda m: print(m, round(time.time() - t, 1), flush=True))
dec = res["decision"]
print("OUTCOME", dec["outcome"], dec.get("sub"), dec.get("failed_controls"), "| kappa", kappa, "seed", seed, "| seconds", round(time.time() - t, 1))
for m, v in dec["per_measure"].items():
    if v.get("available"):
        print(" ", m, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items() if k in ("gap", "point", "lo_bonf", "hi_bonf", "swap_null_nu", "seed_replicate_point", "achieved_se", "achieved_mde80", "certified")})
print(" controls", {k: v["pass"] for k, v in res["controls"].items()})
print(" planted (kappa: M1 gap, read, collapse)", {k: (round(v["M1_gap"], 4), v["planted_gap_read"], v["collapse"]["pass"]) for k, v in res["controls"]["C-II-b"]["per_kappa"].items()})
print(" SEEN - OOF (M1)", round(res["controls"]["C-II-a"]["M1_seen_minus_oof"], 4))
if out:
    json.dump(S0.jsonable(res), open(out, "w"), indent=1)
