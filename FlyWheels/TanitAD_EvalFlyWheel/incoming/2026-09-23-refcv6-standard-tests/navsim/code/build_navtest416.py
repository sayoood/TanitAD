#!/usr/bin/env python3
"""The navtest (NAVSIM v1.1) frame bank at refcv6's 416x1024 (TANITAD VENV, CPU + disk).

⭐ NOTHING IS RE-IMPLEMENTED. W3's shard-streaming bank builder
(``…/2026-09-19-navsim-v1-navtest/code/build_navtest_frames.py``: receipt-verified shards only,
stream each ``.tgz`` once, extract ONLY the needed F0/L0/R0 jpgs, stitch every UNIQUE frame once,
one ``frames_sNN.npy`` per shard + one ``index.json``, carry-over for logs spanning shards,
per-token sha256[:16]) is IMPORTED by path and run with exactly two module constants re-pointed:

* ``H, W`` (the shard array's frame shape) -> 416, 1024;
* ``load_e2`` -> ``frames416.E2BF`` configured for ``FRAME_416x1024`` (E2's verbatim stitch with
  its three frame globals re-pointed — KB-256 / KG-416 in ``tests/test_frames_lift6.py``).

So the bank differs from W3's 256x640 bank (``w3_navtest_v1/frame_bank``, 12,146 tokens, 8.18 GiB)
ONLY in the output frame. Same inputs (W3's export ``navtest_inputs.json.gz``), same shards (the
32 ``openscene_sensor_test_camera_*.tgz``, 127,882,665,618 B, sha256 == HF ETag per the receipt).

    python code/build_navtest416.py --bank D:/Archive/devbox-C/navsim/exp/refcv6_navtest416/frame_bank
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import frames416 as F4  # noqa: E402

W3_BUILDER = os.path.join(F4.REPO, "FlyWheels", "TanitAD_EvalFlyWheel", "incoming",
                          "2026-09-19-navsim-v1-navtest", "code", "build_navtest_frames.py")
W3_BUILDER_D = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                "2026-09-19-navsim-v1-navtest/code/build_navtest_frames.py")
NAVTEST_INPUTS = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--inputs", default=NAVTEST_INPUTS)
    ap.add_argument("--shards", default="all")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-logs", type=int, default=0)
    a = ap.parse_args(argv)
    path = W3_BUILDER if os.path.isfile(W3_BUILDER) else W3_BUILDER_D
    w3 = F4._load("w3_build_navtest_frames", path)
    F4.assert_frame(F4.FRAME_416x1024)
    F4.configure(F4.FRAME_416x1024)
    w3.H, w3.W = 416, 1024
    w3.load_e2 = lambda: F4.E2BF
    print(f"[navtest416] W3 builder {path}; frame {F4.E2BF.FRAME.tag()}", flush=True)
    args = ["--inputs", a.inputs, "--bank", a.bank, "--shards", a.shards,
            "--threads", str(a.threads)]
    if a.max_logs:
        args += ["--max-logs", str(a.max_logs)]
    return w3.main(args)


if __name__ == "__main__":
    sys.exit(main())
