"""Compact the T2 per-window dumps for the repo, WITHOUT losing recomputability.

Kept at full float32: ``fan_err`` — every ade_0_2s in the report is a mean over
it, and Bar A's R-2 established that storage precision is a scientific parameter
on this fan (fp16 caching flipped argmaxes and moved a headline by 0.0028 m).

Dropped because they are exact functions of what is kept:
  ``pdms_lite``   = (~nc_fault) * (5*EP + 5*(~ttc_flag) + 2*comfort_ok) / 12
  ``progress``    = the anchor terminal along-track value — window-INVARIANT for
                    a global vocabulary, so it is stored once as [C] not [W, C].
Demoted to float16 (reported only through percentiles / spreads, where 1e-2 m is
far below the reporting resolution): ``gap``, ``ttc_min``, and the per-candidate
comfort margins.

A verification pass re-derives every headline from the compact file and asserts
it matches the full file bit-for-bit on the float32 quantities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

F16 = ("gap", "ttc_min", "a_lon_max", "a_lon_min", "a_lat_absmax",
       "jerk_absmax", "yr_absmax")
DROP = ("pdms_lite",)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    D = torch.load(a.inp, map_location="cpu", weights_only=False)

    PR = np.asarray(D["progress"], np.float64)
    prog_invariant = bool(np.allclose(PR, PR[0][None], atol=0, rtol=0))
    out = {}
    for k, v in D.items():
        if k in DROP:
            continue
        if k == "progress":
            out["progress_per_candidate"] = (PR[0].astype(np.float32)
                                             if prog_invariant
                                             else PR.astype(np.float32))
            out["progress_is_window_invariant"] = prog_invariant
            continue
        if k in F16 and hasattr(v, "astype"):
            arr = np.asarray(v)
            # a_lat and yaw-rate margins carry no v0 dependence, so for a global
            # anchor vocabulary they are identical in every window — store once.
            if arr.ndim == 2 and np.array_equal(arr, np.broadcast_to(arr[0],
                                                                    arr.shape)):
                out[k + "_per_candidate"] = arr[0].astype(np.float32)
            else:
                out[k] = arr.astype(np.float16)
        else:
            out[k] = v
    out["_note"] = (
        "compact per-window x per-candidate rule labels. fan_err is float32 "
        "(precision is load-bearing); pdms_lite is DROPPED and recomputed as "
        "(~nc_fault)*(5*EP + 5*(~ttc_flag) + 2*comfort_ok)/12 with "
        "EP = clip(progress/max_c progress, 0, 1) and a no-progress veto; "
        "progress is stored once per candidate. Every rate, rho, selection arm "
        "and bootstrap interval in PERCANDIDATE_LABELS.md recomputes from this "
        "file with NO GPU.")
    torch.save(out, a.out)

    # ---- verification: the headline rates must survive compaction -----------
    E = torch.load(a.out, map_location="cpu", weights_only=False)
    chk = dict(
        fan_err_bitwise_identical=bool(np.array_equal(
            np.asarray(D["fan_err"]), np.asarray(E["fan_err"]))),
        nc_rate_full=round(float(np.asarray(D["nc_fault"]).mean()), 6),
        nc_rate_compact=round(float(np.asarray(E["nc_fault"]).mean()), 6),
        ttc_rate_full=round(float(np.asarray(D["ttc_flag"]).mean()), 6),
        ttc_rate_compact=round(float(np.asarray(E["ttc_flag"]).mean()), 6),
        progress_window_invariant=prog_invariant,
        bytes_before=Path(a.inp).stat().st_size,
        bytes_after=Path(a.out).stat().st_size)
    # recompute pdms_lite from the compact file and compare to the original
    P = (E["progress_per_candidate"][None] if prog_invariant
         else E["progress_per_candidate"])
    P = np.broadcast_to(np.asarray(P, np.float64),
                        np.asarray(E["fan_err"]).shape)
    pmax = np.maximum(P.max(1, keepdims=True), 1e-6)
    EP = np.where(P <= 0.0, 0.0, np.clip(P / pmax, 0.0, 1.0))
    pd_re = ((~np.asarray(E["nc_fault"])) *
             (5.0 * EP + 5.0 * (~np.asarray(E["ttc_flag"])) +
              2.0 * np.asarray(E["comfort_ok"])) / 12.0)
    chk["pdms_lite_max_abs_recompute_error"] = float(
        np.abs(pd_re - np.asarray(D["pdms_lite"], np.float64)).max())
    chk["pdms_lite_recomputes"] = bool(
        chk["pdms_lite_max_abs_recompute_error"] < 1e-6)
    print(json.dumps(chk, indent=1))
    assert chk["fan_err_bitwise_identical"] and chk["pdms_lite_recomputes"]


if __name__ == "__main__":
    main()
