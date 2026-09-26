#!/usr/bin/env python3
"""Score ONE refcv6 seam on NAVSIM v1.1 ``navtest`` with W3's harness, IMPORTED (any python).

W3's ``run_v1.py`` (``…/2026-09-19-navsim-v1-navtest/code/``) is loaded by path and its
``cmd_score`` is called UNCHANGED — the v1.1 tree ``3e8291b``, W3's Windows wrapper (the loader
separator patch, observation-only hooks, RAM guard), W3's ``SeamAgentV1`` (token-keyed,
fingerprint-checked), the navtest metric cache (12,146 cached), ``PYTHONHASHSEED=1``, and its
guards: C1 the PDMS identity on every row, C2 log/CSV counts, C3 the token set, the seam call
count. Only W3's module global ``RAW`` is re-pointed so every artifact lands in THIS package
(``<out>/<label>/``); the devkit's scratch goes to ``w3_navtest_v1/runs/<label>`` under a
refcv6-prefixed label, so nothing of W3's is overwritten.

    python code/score_navtest6.py --label r6s1000_R6_A1_sub200 --seam <seam.npz> \
        --tokens D:/…/2026-09-19-navsim-v1-navtest/raw/A1_sub200_tokens.json
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
W3_CODE = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
           "2026-09-19-navsim-v1-navtest/code")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--seam", default="")
    ap.add_argument("--official", default="", choices=("", "STOP", "CV"),
                    help="score a MODEL-FREE control through this same driver (harness control "
                         "KH-navtest: must reproduce W3's banked CSV cell for cell)")
    ap.add_argument("--tokens", default=None, help="W3-format {'tokens', 'token_log'} subset")
    ap.add_argument("--out", default=os.path.join(PKG, "raw", "navtest"))
    a = ap.parse_args(argv)
    if not a.label.startswith("r6"):
        sys.exit("⛔ labels must start with 'r6' so the devkit scratch never collides with W3's")
    spec = importlib.util.spec_from_file_location("w3_run_v1", os.path.join(W3_CODE, "run_v1.py"))
    w3 = importlib.util.module_from_spec(spec)
    sys.modules["w3_run_v1"] = w3
    spec.loader.exec_module(w3)
    os.makedirs(a.out, exist_ok=True)
    w3.RAW = os.path.abspath(a.out)                  # where every artifact lands
    w3.PKG = PKG                                     # only used for a relpath in the counts (C: vs D:)
    if bool(a.seam) == bool(a.official):
        sys.exit("⛔ exactly one of --seam / --official")
    arm = a.official if a.official else f"SEAM:{os.path.abspath(a.seam)}"
    ns = types.SimpleNamespace(label=a.label, arm=arm,
                               tokens=a.tokens, cache_name="navtest", worker="sequential",
                               hashseed="1", record_poses=True, patch_loader=True)
    return w3.cmd_score(ns)


if __name__ == "__main__":
    sys.exit(main())
