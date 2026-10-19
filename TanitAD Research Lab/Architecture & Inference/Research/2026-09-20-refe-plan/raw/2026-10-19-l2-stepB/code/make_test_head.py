#!/usr/bin/env python3
"""A TEST head, NOT the L2 head: the sealed L2 recipe (`recipe_chosen.json`, sha256-pinned in the sealed prereg) fitted on ALL 400 Step A tokens with the SEALED l2lib code, seed `--seed`.
It exists only so G-REACH-STEPB and the cost smoke can run before the fresh-carve Step A reads.  It writes the same file format as `l2fresh/code/fit_final_heads.py`
(weight [6,256] f32, bias [6] f32, steps, seed, recipe_sha256, feats_sha256, n_tokens).  Reads no outcome: it fits and writes, nothing is scored.   CPU only.

    CUDA_VISIBLE_DEVICES=-1 python make_test_head.py --out <dir> [--seed 0]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

L2_CODE = os.environ.get("L2_CODE", "C:/Users/Admin/par18/l2/code")
sys.path.insert(0, L2_CODE)
import l2lib  # noqa: E402

RECIPE = "D:/Projects/TanitAD-artifacts/l2_offprop/recipe_chosen.json"
RECIPE_SHA256_PIN = "c7f5b75b0a8f25f96c00dcc379ac583fa9c80622ce5600ad4af8387159119263"        # the sealed prereg's PINS
FEATS = "D:/Projects/TanitAD-artifacts/l2_offprop/feats.npz"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    got = hashlib.sha256(open(RECIPE, "rb").read().replace(b"\r\n", b"\n")).hexdigest()
    if got != RECIPE_SHA256_PIN:
        raise SystemExit(f"recipe sha256 {got[:12]} != sealed pin {RECIPE_SHA256_PIN[:12]}")
    rec = l2lib.Recipe.from_json(RECIPE)
    D = l2lib.load_real(FEATS)
    ts = l2lib.make_trainset(D, np.arange(D.N))
    fit = l2lib.fit_head(D.tok_in.reshape(-1, D.D), D.tok_p.reshape(-1, D.D), ts, D.W6, D.b6, a.seed, rec, w_pair=1.0)
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, f"test_head_s{a.seed}.npz")
    np.savez(p, weight=fit["W"].astype(np.float32), bias=fit["b"].astype(np.float32), steps=np.array(fit["steps"]), seed=np.array(a.seed), recipe_sha256=np.array(got),
             feats_sha256=np.array(hashlib.sha256(open(FEATS, "rb").read()).hexdigest()), n_tokens=np.array(D.N), kind=np.array("TEST HEAD, not the L2 head"))
    sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
    dW = fit["W"][l2lib.EP] - np.asarray(D.W6, np.float32)[l2lib.EP]
    print("ZZTESTHEAD", p, sha, "steps", fit["steps"], "ep_row_l2_delta_vs_P2", round(float(np.linalg.norm(dW)), 4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
