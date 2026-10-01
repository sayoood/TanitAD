#!/usr/bin/env python3
"""Mutation self-test for `slow_copy_probe.py`'s gates: each deliberate regression must turn EXACTLY its gate RED,
and the unmutated inputs must pass every gate and report a number. GPU-free (`analyze --no-families`).

  none               no mutation                                  -> every gate green, ZZSLOW_OK
  s64+1e-3           one 64-set logit of the GPU dump moved 1e-3   -> C2 red (the shipped logits not reproduced)
  poses_differ       the GPU pass's pose check set to 1e-3 m       -> C2 red (today's forward != the dump's)
  unmasked!=plain    the unmasked-decoder identity set to 1e-3     -> C5 red (the re-implemented decoder != model's)
  stop_row+1e-3      the masked STOP row moved 1e-3               -> C5 red (!= stop_candidate_probe's V-mask row)
  build_copy+1e-3    one built f=0.75 copy moved 1 mm              -> C6 red (the harness scored another construction)

    python eval/selftest_slow_copy_probe.py
Prints ZZSELFTEST_OK or ZZSELFTEST_FAIL <case>; exit 0 / 1.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "slow_copy_probe.py")
WD = "D:/Projects/TanitAD/data/refe_navtest/proptable/sub200_ep015/slow_copies"
GPU_DUMP, BUILD = f"{WD}/gpu_dump.npz", f"{WD}/build.npz"


def g_s64(g, b):
    g["s64"][0, 0, 0] += 1e-3


def g_poses(g, b):
    g["traj_maxdiff"][0] = 1e-3


def g_nomask(g, b):
    g["id_nomask"][0] = 1e-3


def g_stop(g, b):
    g["m"][0, 192, 0] += 1e-3


def b_copy(g, b):
    b["nat_f075"][0, 0, 5, 0] += 1e-3


CASES = (("none", None, None),
         ("s64+1e-3", g_s64, "C2_gpu_forward_vs_dump_and_table"),
         ("poses_differ", g_poses, "C2_gpu_forward_vs_dump_and_table"),
         ("unmasked!=plain", g_nomask, "C5_masked_decoder_identities"),
         ("stop_row+1e-3", g_stop, "C5_masked_decoder_identities"),
         ("build_copy+1e-3", b_copy, "C6_shared_module_reproduces_the_harness_copies"))


def main() -> int:
    G0 = dict(np.load(GPU_DUMP, allow_pickle=False))
    B0 = dict(np.load(BUILD, allow_pickle=False))
    tmp = tempfile.mkdtemp(prefix="selftest_slow_probe_")
    bad = []
    try:
        for name, mut, expect in CASES:
            g = {k: np.array(v, copy=True) for k, v in G0.items()}
            b = {k: np.array(v, copy=True) for k, v in B0.items()}
            if mut:
                mut(g, b)
            gp, bp, op = (os.path.join(tmp, f"{name}_g.npz"), os.path.join(tmp, f"{name}_b.npz"),
                          os.path.join(tmp, f"{name}.json"))
            np.savez(gp, **g)
            np.savez(bp, **b)
            r = subprocess.run([sys.executable, PROBE, "analyze", "--gpu-dump", gp, "--build", bp, "--out", op,
                                "--boot", "200", "--no-families"], capture_output=True, text=True, cwd=HERE)
            last = [ln for ln in r.stdout.splitlines() if ln.startswith("ZZSLOW")]
            c = json.load(open(op, encoding="utf-8")).get("controls", {}) if os.path.exists(op) else {}
            red = [k for k, v in c.items() if isinstance(v, dict) and v.get("pass") is False]
            if expect is None:
                ok = red == [] and bool(last) and last[-1] == "ZZSLOW_OK"
            else:
                ok = red == [expect] and bool(last) and last[-1].startswith("ZZSLOW_FAIL")
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:16s} red={red} expected={expect} -> "
                  f"{last[-1] if last else 'no marker (rc ' + str(r.returncode) + ')'}", flush=True)
            if not ok:
                bad.append(name)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ZZSELFTEST_OK" if not bad else f"ZZSELFTEST_FAIL {' '.join(bad)}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
