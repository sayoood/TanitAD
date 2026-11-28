#!/usr/bin/env python3
"""ESTIMATES the pod's gap after the patches, from MEASURED inputs and stated assumptions.  Nothing here is a pod measurement of the patched code: the patched code has not run on the pod.

    python pod_estimate.py --attribution attribution.json --bench fp_timing_dev_synth413k.json --out lever_estimate.json

Model (per tree+data re-measure R, seconds):
    R_before   = measured on the pod (verify #1/#2: 227 s; verify #3: ~204 s)                                        MEASURED
    L_before   = R_before - (checkpoint sha256 <= 4 s) - (code tree <= 2 s)  = the 413,184-file image listing          INFERRED (subtraction)
    c          = L_before / N     (pod cost per listed file, 0.5 ms)                                                    INFERRED
    p_old, p_new = interpreter microseconds per file of the old / new code with the filesystem taken out               MEASURED on the dev box (bench_fingerprint.py --cpu-only, simulated stat)
    on the pod:  p_old_pod = k * p_old (k = how much slower the pod's CPU is at Python than the dev box: 1 or 2),  io = c - p_old_pod   (the part that is a filesystem round trip)
    L_after(T,S) = N * (k * p_new + io / min(T, S))     T = worker threads (16), S = how many metadata requests the pod's volume serves at once (UNKNOWN: 1, 2, 4, 8, 16)
    R_after    = L_after + (code tree <= 2 s); the checkpoint hash (<= 4 s) now runs beside the listing instead of before it
Per arm (fresh path): before = 5 re-measures (gate start, finalize, three verifies); lever B removes one verify (saves R_before); lever A makes each of the 4 remaining ones cheaper by
(L_before - L_after): saved(A + B) = R_before + 4 * (L_before - L_after).
S = 1 means the volume serializes metadata: threads do not help and only the string ops do.  The honest range is therefore wide, and the pod timing run (bench_fingerprint.py on the pod,
about 5 minutes, read-only) turns S from an assumption into a measurement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attribution", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    att = json.loads(Path(a.attribution).read_text(encoding="utf-8"))
    bench = json.loads(Path(a.bench).read_text(encoding="utf-8"))
    N = att["inside_one_remeasure"]["image_listing_files"]
    R_before = att["segments"]["D_verify_400_and_403"]["per_verify_min"] * 60.0
    tree_s, hash_s = att["inside_one_remeasure"]["code_tree_upper_s"], att["inside_one_remeasure"]["checkpoint_sha256_upper_bound_s"]
    L_before = R_before - tree_s - hash_s
    c_us = 1e6 * L_before / N
    cpu = bench["cpu_only"][0]["per_file_us"]
    p_old, p_new = cpu["old"], cpu["new"][str(a.threads)] if str(a.threads) in cpu["new"] else min(cpu["new"].values())
    total_gap = att["gap_min"]
    rows = []
    for k in (1, 2):
        for S in (1, 2, 4, 8, 16):
            io_us = max(c_us - k * p_old, 0.0)
            L_after = N * (k * p_new + io_us / min(a.threads, S)) / 1e6
            R_after = L_after + tree_s
            save_B = R_before / 60.0                                          # one verify removed (MEASURED 3.4-3.8 min each)
            save_A_each = max(L_before - L_after, 0.0) / 60.0                 # every remaining re-measure gets cheaper by this much (the checkpoint hash overlap, <= 4 s, is NOT counted)
            save_AB = save_B + 4 * save_A_each
            rows.append({"k_pod_cpu_slowdown": k, "S_metadata_concurrency": S, "io_us_per_file": round(io_us, 1), "L_after_s": round(L_after, 1), "R_after_s": round(R_after, 1),
                         "saved_per_arm_min": {"B_only": round(save_B, 2), "A_per_remeasure": round(save_A_each, 2), "A_and_B": round(save_AB, 2)},
                         "gap_after_min": {"N-s0 shape (no preflight)": round(total_gap - save_AB, 1), "arms 2-8 shape (+ ~6 min preflight)": round(total_gap + 6.0 - save_AB, 1)}})
    out = {"evidence_class": "ESTIMATED (model in the module docstring; the patched code has NOT run on the pod)",
           "inputs": {"N_files": N, "R_before_s": R_before, "L_before_s_INFERRED": round(L_before, 1), "c_us_per_file_INFERRED": round(c_us, 1), "p_old_us_dev_MEASURED_under_load": p_old,
                      "p_new_us_dev_MEASURED_under_load": p_new, "threads": a.threads, "tree_upper_s": tree_s, "hash_upper_s": hash_s,
                      "pod_gap_before_min_MEASURED": total_gap},
           "scenarios": rows,
           "reading": "B_only is independent of A and of S: it removes one verify (MEASURED 3.4-3.8 min).  A_and_B depends on S, the pod volume's metadata concurrency, which only the pod can say."}
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"inputs": out["inputs"], "A_and_B_saved_per_arm_min": {f"k{r['k_pod_cpu_slowdown']}_S{r['S_metadata_concurrency']}": r["saved_per_arm_min"]["A_and_B"] for r in rows}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
