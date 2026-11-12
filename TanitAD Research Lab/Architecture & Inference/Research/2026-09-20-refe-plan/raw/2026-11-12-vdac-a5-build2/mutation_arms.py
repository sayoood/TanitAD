#!/usr/bin/env python3
"""Runs every deliberate-regression arm of vdac_stage0_forward.py once and prints whether its check went RED.  Run from refe/ (beside test_vdac_stage0_forward.py):
    CUDA_VISIBLE_DEVICES=-1 python mutation_arms.py > mutation_results.txt"""
import io
import os
import sys
from contextlib import redirect_stdout

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.argv = ["x"]
sys.path.insert(0, os.getcwd())
import test_vdac_stage0_forward as T  # noqa: E402
import vdac_stage0_forward as F  # noqa: E402

with redirect_stdout(io.StringIO()):
    T.TRun.setUpClass()
E = T.TRun.E
r = T.TRun("test_trained_predictions_match_an_independent_forward")
print("mutation arms of vdac_stage0_forward.py (each must make its check fail or the tool refuse); CPU, synthetic fixtures")
bad = 0


def run(env, fn):
    global bad
    os.environ["VDAC_FWD_MUTATE"] = env
    try:
        with redirect_stdout(io.StringIO()):
            fn()
        bad += 1
        print(f"{env:28s} NOT CAUGHT  <-- FAIL")
    except BaseException as e:  # noqa: BLE001
        print(f"{env:28s} RED: {type(e).__name__}: {str(e)[:100]}".replace("\n", " "))
    finally:
        os.environ.pop("VDAC_FWD_MUTATE")


with redirect_stdout(io.StringIO()):
    r.chk_trained_predictions()
print("clean run                    check passes (no mutation)")
for env in ("skip_head_load", "skip_head_load,no_guard", "y_flip_output", "misalign_keys"):
    run(env, r.chk_trained_predictions)


def chk_init():
    rc, txt = E.run_main(os.path.join(r.td.name, "mi_" + os.environ["VDAC_FWD_MUTATE"]), "init")
    assert rc == 0, f"rc {rc}"


def chk_sha():
    a = E.argv(os.path.join(r.td.name, "ms_" + os.environ["VDAC_FWD_MUTATE"]), "head")
    a[a.index("--insample-list-sha256") + 1] = "0" * 64
    try:
        F.main(a, factory=E.factory)
    except SystemExit as e:
        assert str(e.code).startswith("REFUSED")
        return
    raise AssertionError("a wrong frame-list sha256 was accepted")


def chk_gpu():
    try:
        F.require_gpu_free("cuda", False, "t")
    except SystemExit as e:
        assert "gpu-busy" in str(e.code)
        return
    raise AssertionError("a busy GPU was accepted")


run("init_seed1", chk_init)
run("skip_list_sha", chk_sha)
F.GPU_PROBE = lambda: (["1, trainer, 1 MiB"], "")
run("skip_gpu_guard", chk_gpu)
print("ALL ARMS RED" if not bad else f"{bad} ARM(S) NOT CAUGHT")
