"""Gate C0C-ID (PREREG_T2_DOMINANCE G1) as a fast dev-box test: a thin sample of `code/c0c_gate.py`. The full gate (every dump row, 8+4 windows per seed, both seeds) is `C0C_ID_RESULT.json`.
CPU only; banked data only. Run from t2v8f/: CUDA_VISIBLE_DEVICES=-1 python -I -m pytest tests/test_t2v8f_c0c_gate.py -q"""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("T2A_STUB", "1")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import numpy as np  # noqa: E402

import pytest  # noqa: E402

import c0c_gate as G  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def _restore_t2v8f_run():
    """`c0c_gate.world()` changes `t2v8f_run` for the whole PROCESS: it rebinds `resolve_ckpt` to the two dev-box C0 checkpoints and calls `t2v8f_c0c.install` (ARMS and CKPT_FILE gain C0C_s0 / C0C_s1, `build_arm`
    is wrapped). `test_t2v8f_run.py::test_dispatch` then failed in a full-suite run (a seed-less ZERO arm in the lambda; then C0C_s0 absent from the CLI's `--arm` choices). Put the originals back when this module
    is done (AGENT O, 2026-11-06; the first full-suite run since the gate test existed)."""
    import t2v8f_run as R
    saved = {k: getattr(R, k) for k in ("resolve_ckpt", "build_arm", "ARMS")}
    ck = dict(R.CKPT_FILE)
    had = hasattr(R, "_C0C_INSTALLED")
    yield
    for k, v in saved.items():
        setattr(R, k, v)
    R.CKPT_FILE.clear()
    R.CKPT_FILE.update(ck)
    if not had and hasattr(R, "_C0C_INSTALLED"):
        del R._C0C_INSTALLED


def test_a1_arithmetic_is_exact_and_red_arms_break_it():
    for seed in (0, 1):
        a = G.arithmetic_identity(seed, stride=400)
        assert a["pass"] and a["eq_plan"] == a["rows"] and a["eq_plan_s"] == a["rows"], a
        assert all(v["verdict"] == "RED" for v in a["red"].values()), a["red"]                    # smooth off, feasible off, prior zeros: each differs from the dump on >= 1 row
    ri = G.arithmetic_identity(0, ri_dump=True, stride=400)
    assert ri["pass"], ri


def test_live_identity_exact_against_the_ri_tool_and_regressions_red():
    seed = 0
    wins, draw = G.pick_windows(seed, 1, 2)
    assert draw["n_kbox_active_added"] >= 1
    rows = [G.live_identity(seed, w) for w in wins]
    assert all(r["eq_tool"] and r["eq_tool_raw"] and r["eq_tool_s"] for r in rows), rows
    assert all(r["a2_within_tol"] for r in rows), [r["a2_max_abs_m"] for r in rows]
    assert all(not r["red_still_equal"]["smooth_off"] and not r["red_still_equal"]["prior_zeros"] for r in rows)
    kb = [r for r in rows if r["kbox_active"]]
    assert kb and all(not r["red_still_equal"]["feasible_off"] for r in kb)                      # feasible off bites where the clamp is active
    assert all(r["off_off"]["plan_equal_C0"] and r["off_off"]["extra_equal_apart_from_added"] for r in rows)


def test_ri_tool_is_the_pinned_one():
    C = G.world()["C"]
    assert C.sha256_file(G.world()["ri"].__file__) == C.RI_TOOL_SHA and os.path.getsize(G.world()["ri"].__file__) == 76704
