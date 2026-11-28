#!/usr/bin/env python3
"""DATASCALE cost table from the pilot's own report. Usage: python ds_cost.py PILOT_REPORT.json > COST_DATASCALE.md
Rates are MEASURED on the pilot; the L40 column is ESTIMATED (GPU stage only; stage and build are network/CPU bound and do not change)."""
import json
import sys

r = json.load(open(sys.argv[1], encoding="utf-8"))
st, bd, en = r["stage"], r["build"], r["encode"]
rate = {"stage": st.get("clips_per_h"), "build": bd.get("clips_per_h_wall_13_workers"), "encode": en.get("clips_per_h_wall")}
L40_ENC_SPEEDUP = 1.7          # ESTIMATED: spec-sheet bf16 ratio L40 / RTX 4000 Ada, NOT measured
mb = {"mp4": st.get("mp4_MB_mean") or 0, "ego": st.get("ego_MB_mean") or 0, "obs": st.get("obs_MB_mean") or 0,
      "v2ep": bd.get("v2ep_MB_mean") or 0, "cache": en.get("cache_MB_mean") or 0}
fail = (len(st["fail"]) + len(bd["fail"])) / max(st["n"], 1)
print("# COST_DATASCALE (pilot-derived)\n")
print(f"MEASURED on the 200-clip pilot (`PILOT_REPORT.json`): stage {rate['stage']} clips/h, build {rate['build']} clips/h (13 workers), encode {rate['encode']} clips/h; failure rate {fail:.1%}.")
print(f"Per clip (MB): mp4 {mb['mp4']}, egomotion {mb['ego']}, obstacle {mb['obs']}, v2ep (transient) {mb['v2ep']}, cache (kept) {mb['cache']}.\n")
print("| N clips | pod-h sequential (RTX 4000 Ada) | pod-h pipelined (slowest stage) | pod-h pipelined, L40 encode (ESTIMATED) | transient disk if v2ep kept (GB) | kept cache = HF private (GB) | obstacle coverage (ESTIMATED from pool) |")
print("|---|---|---|---|---|---|---|")
for n, tag in ((10000, ""), (45000, ""), (133929, " (admissible ceiling under the frozen scope)"), (306151, " (NOT admissible under the frozen scope; extrapolation)")):
    seq = sum(n / v for v in rate.values() if v)
    pipe = n / min(v for v in rate.values() if v)
    enc_l40 = n / min([v for k, v in rate.items() if v and k != "encode"] + [rate["encode"] * L40_ENC_SPEEDUP])
    print(f"| {n}{tag} | {seq:.1f} | {pipe:.1f} | {enc_l40:.1f} | {n * mb['v2ep'] / 1e3:.0f} | {n * mb['cache'] / 1e3:.0f} | 97.98 % (131,219 of 133,929 in the pool, MEASURED metadata) |")
print("\nEvidence: rates and sizes MEASURED (pilot, one run, one seed of the selection; no replicate, so the rate carries unknown run-to-run noise). Extrapolation to N beyond 200 is ESTIMATED: linear in N,")
print("network and CPU contention at 8 threads / 13 workers assumed unchanged. The earlier corpus build measured 1,703 clips/h build and 2.76 s/clip encode (`/root/logs/v2ep_run.log`, `/root/logs/cache_pod.log`): INHERITED cross-check.")
