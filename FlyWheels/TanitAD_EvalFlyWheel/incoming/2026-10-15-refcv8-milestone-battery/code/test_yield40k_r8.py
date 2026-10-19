"""yield40k_r8.sh: the refcv7 chains + K2/K4 must not take the GPU while the step-40k EMA battery is pending.

Runs the real helper under sh against a STUB gpu_lock.py (never the real lock). Expectations are literals.
Mutation arm: with the predicate disabled the pending case must show no yield, so the pending assertion would go red.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SH = shutil.which("sh") or r"C:\Program Files\Git\usr\bin\sh.exe"
STUB = "\n".join([
    "import json, os, sys, time",
    "a = sys.argv; out = a[a.index('--out') + 1]; d = os.path.dirname(__file__)",
    "if os.path.exists(os.path.join(d, 'crash')):",
    "    json.dump({'ok': False, 'error': 'stub crash'}, open(out, 'w')); sys.exit(0)",
    "ok = os.path.exists(os.path.join(d, 'free'))",
    "if not ok:",
    "    time.sleep(float(a[a.index('--max-wait-s') + 1]) + 0.2)  # a real busy lock WAITS the try window",
    "json.dump({'ok': ok}, open(out, 'w'))",
    "",
])


def _posix(p: str) -> str:
    p = p.replace("\\", "/")
    return "/" + p[0].lower() + p[2:] if len(p) > 1 and p[1] == ":" else p


def _run(tmp, *, ckpt, alive, log, summary_after=None, kill_after=None, free=True, crash=False, mutate=False,
         max_wait=30):
    t = _posix(str(tmp))
    os.makedirs(tmp / "bat", exist_ok=True)
    (tmp / "bat" / "gpu_lock.py").write_text(STUB)
    for f, on in (("bat/free", free), ("bat/crash", crash), ("ckpt", ckpt), ("alive", alive)):
        p = tmp / f
        if on:
            p.write_text("x")
        elif p.exists():
            p.unlink()
    (tmp / "c40.log").write_text(log)
    for f in ("sum", "w.log"):
        if (tmp / f).exists():
            (tmp / f).unlink()
    bg = []
    if summary_after is not None:
        bg.append(f"( sleep {summary_after}; echo x > {t}/sum ) &")
    if kill_after is not None:
        bg.append(f"( sleep {kill_after}; rm -f {t}/alive ) &")
    script = (f"PY='{_posix(sys.executable)}'; BAT={t}/bat; Y40_CKPT={t}/ckpt; Y40_SUM={t}/sum; Y40_LOG={t}/c40.log; "
              f"Y40_SLEEP=1; Y40_TRY_S=1; . '{_posix(os.path.join(HERE, 'yield40k_r8.sh'))}'; "
              f"ps() {{ [ -f {t}/alive ] && echo 'x chain_40k_r8.sh'; }}; "
              + ("b40ema_pending() { return 1; }; " if mutate else "")
              + " ".join(bg) + f" acquire_y40 job1 {t}/lock.json {t}/w.log {max_wait}; echo RC=$?; wait")
    out = subprocess.run([SH, "-c", script], capture_output=True, text=True, timeout=120).stdout
    w = (tmp / "w.log").read_text() if (tmp / "w.log").exists() else ""
    return out.strip().splitlines()[-1], w.count("ZZY40YIELDZZ"), w.count("ZZY40RESUMEZZ")


pytestmark = pytest.mark.skipif(not os.path.exists(SH), reason="no POSIX sh")


def test_no_snapshot_acquires_at_once(tmp_path):
    assert _run(tmp_path, ckpt=False, alive=True, log="ZZC40STARTZZ\n") == ("RC=0", 0, 0)


def test_pending_yields_until_the_summary_exists(tmp_path):
    assert _run(tmp_path, ckpt=True, alive=True, log="ZZC40STARTZZ\n", summary_after=3) == ("RC=0", 1, 1)


def test_ema_failure_releases_the_yield(tmp_path):
    assert _run(tmp_path, ckpt=True, alive=True, log="ZZC40STARTZZ\nZZC40EMAFAILZZ x\n") == ("RC=0", 0, 0)


def test_an_old_failure_does_not_mask_a_new_start(tmp_path):
    assert _run(tmp_path, ckpt=True, alive=True, log="ZZC40STARTZZ\nZZC40EMAFAILZZ x\nZZC40STARTZZ\n",
                kill_after=3) == ("RC=0", 1, 1)


def test_dead_40k_chain_does_not_block(tmp_path):
    assert _run(tmp_path, ckpt=True, alive=False, log="ZZC40STARTZZ\n") == ("RC=0", 0, 0)


def test_busy_lock_waits_then_times_out_with_rc3(tmp_path):
    t0 = time.time()
    assert _run(tmp_path, ckpt=False, alive=False, log="", free=False, max_wait=4)[0] == "RC=3"
    assert time.time() - t0 >= 4          # it WAITED (several try windows), it did not refuse at once


def test_lock_tool_refusal_returns_at_once_instead_of_spinning(tmp_path):
    """The defect caught 2026-10-19: an instant ok-false (crash / probe failure) was retried until max_wait."""
    t0 = time.time()
    assert _run(tmp_path, ckpt=False, alive=False, log="", crash=True, max_wait=600)[0] == "RC=3"
    assert time.time() - t0 < 60


def test_mutation_predicate_off_does_not_yield(tmp_path):
    assert _run(tmp_path, ckpt=True, alive=True, log="ZZC40STARTZZ\n", summary_after=3, mutate=True) == ("RC=0", 0, 0)
