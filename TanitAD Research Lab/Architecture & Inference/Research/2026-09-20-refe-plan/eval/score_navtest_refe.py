#!/usr/bin/env python3
"""Score ONE REFe seam on NAVSIM v1.1 ``navtest`` with W3's harness, IMPORTED (SPEC_NAVTEST).

The pattern of the EvalFlyWheel's ``score_navtest6.py``: W3's ``run_v1.py`` is loaded by path and
its ``cmd_score`` is called UNCHANGED (the v1.1 tree ``3e8291b``, W3's Windows wrapper, W3's
``SeamAgentV1`` -- token-keyed, fingerprint-checked --, the navtest metric cache, ``PYTHONHASHSEED=1``,
and its guards C1 PDMS identity / C2 counts / C3 token set). Only W3's module global ``RAW`` is
re-pointed so every artifact lands in THIS package; labels must start with ``refe`` so the devkit
scratch never collides with W3's or refcv6's.

    python eval/score_navtest_refe.py --label refe_e0_human_sub200 --seam <seam.npz> --tokens <subset.json>

``--human-seam <out.npz> --tokens <subset>`` instead WRITES the HUMAN seam (W3's exported
``human_future_poses``) for the SPEC's E-0 harness check, and exits.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
W3_CODE = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
           "2026-09-19-navsim-v1-navtest/code")
W3_EXPORT = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"


def write_human_seam(out, tokens_json):
    exp = json.load(gzip.open(W3_EXPORT, "rt", encoding="utf-8"))["tokens"]
    toks = list(exp)
    if tokens_json:
        sub = json.load(open(tokens_json, encoding="utf-8"))
        want = set(sub["tokens"] if isinstance(sub, dict) else sub)
        toks = [t for t in toks if t in want]
    poses = np.stack([np.asarray(exp[t]["human_future_poses"], dtype=np.float32) for t in toks])
    samp = exp[toks[0]]["human_future_sampling"]
    np.savez(out, token=np.array(toks), fingerprint=np.array([exp[t]["fingerprint"] for t in toks]),
             poses=poses, sampling=np.array([samp[0], samp[1]]), arm=np.array("HUMAN_from_export"))
    print(f"  wrote HUMAN seam: {len(toks)} tokens, sampling {samp} -> {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=None)
    ap.add_argument("--seam", default=None)
    ap.add_argument("--tokens", default=None, help="W3-format {'tokens', 'token_log'} subset")
    ap.add_argument("--out", default=os.path.join(PKG, "eval", "raw", "navtest"))
    ap.add_argument("--human-seam", default=None)
    a = ap.parse_args(argv)
    if a.human_seam:
        return write_human_seam(a.human_seam, a.tokens)
    if not (a.label and a.seam):
        sys.exit("--label and --seam are required")
    if not a.label.startswith("refe"):
        sys.exit("⛔ labels must start with 'refe' so the devkit scratch never collides with W3's/refcv6's")
    spec = importlib.util.spec_from_file_location("w3_run_v1", os.path.join(W3_CODE, "run_v1.py"))
    w3 = importlib.util.module_from_spec(spec)
    sys.modules["w3_run_v1"] = w3
    spec.loader.exec_module(w3)
    os.makedirs(a.out, exist_ok=True)
    w3.RAW = os.path.abspath(a.out)
    w3.PKG = PKG
    ns = types.SimpleNamespace(label=a.label, arm=f"SEAM:{os.path.abspath(a.seam)}",
                               tokens=a.tokens, cache_name="navtest", worker="sequential",
                               hashseed="1", record_poses=True, patch_loader=True)
    return w3.cmd_score(ns)


if __name__ == "__main__":
    sys.exit(main())
