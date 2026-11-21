"""PREREG_V8F_BETA Amendment 1 (rung beta's eval-only gate on box 2): the queue. CPU only, < 20 s.

Run: PYTHONPATH=<repo>/stack;<repo>/taniteval;<repo>/stack/scripts pytest -q stack/tests/test_v8f_beta_box2.py

What is pinned, and the control that must go red for each:
  * the queue (CRLF) runs `--plan beta` on the v8F-mini M2b checkpoints of PREREG_V8F_MINI Amendment 4 (runs\\v8f_mini_full, cache_full,
    --kin-mode yaw, cache_full's native_tracks.pt), writes into its OWN eval directory (eval_beta: the K-1 eval's eval_M2b_s<k> files
    must never be overwritten), and passes no --fan-artifact;
  * its dry run passes the SEED through (a goto-dispatched stage once read the stage name as its first argument);
  * the argv it prints is accepted by the REAL `v8f_mini_eval.py` parser for both `run` and `compare-beta` (the call then fails on the
    missing checkpoint, i.e. AFTER argparse; an unknown flag would exit 2 inside argparse instead).
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
QUEUE = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-18-v8f-beta" / "code" / "v8f_beta_box2_queue.cmd"


@pytest.mark.skipif(not QUEUE.exists(), reason="queue not in this checkout")
def test_beta_queue_reads_the_amendment4_m2b_checkpoints_into_its_own_eval_dir():
    raw = QUEUE.read_bytes()
    assert raw.count(b"\r\n") == raw.count(b"\n") > 40, "the queue must stay CRLF"
    txt = raw.decode("utf-8")
    assert 'RUNSM=D:\\tanitad\\runs\\v8f_mini_full"' in txt and 'CACHEFULL=D:\\tanitad\\cache_full"' in txt
    assert 'set "EVB=%RUNSM%\\eval_beta"' in txt and "--out-dir %EVB%" in txt
    assert 'set "TRACKS=%CACHEFULL%\\native_tracks.pt"' in txt and "--kin-mode yaw" in txt
    assert "M2b_s%~1\\ckpt_last.pt" in txt and "--plan beta" in txt
    assert "--fan-artifact" not in txt.split("rem PI RULING")[1]           # Amendment 1: the pinned grid, checked offline
    assert "M1_s" not in txt and "M2a_s" not in txt                        # M2b only: M2a is not trained


def _dry(tmp_path, args):
    env = dict(os.environ, V8F_DRYRUN="1", REPO=str(REPO), RUNSM=str(tmp_path / "runsm"), CACHEFULL=str(tmp_path / "cf"),
               V9=str(tmp_path / "v9"))
    r = subprocess.run(f'cmd /c ""{QUEUE}" {args}"', env=env, capture_output=True, text=True, timeout=60, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1
    return lines[0]


@pytest.mark.skipif(os.name != "nt" or not QUEUE.exists(), reason="the queue is a Windows batch file")
def test_beta_queue_dry_run_passes_the_seed_and_the_real_parser_accepts_its_argv(tmp_path):
    import v8f_mini_eval as EV
    r1 = _dry(tmp_path, "run 1")
    assert "--tag M2b_s1 " in r1 and str(tmp_path / "runsm" / "M2b_s1" / "ckpt_last.pt") in r1
    assert str(tmp_path / "runsm" / "eval_beta") in r1
    c = _dry(tmp_path, "compare")
    assert "compare-beta" in c and "eval_M2b_s0.json" in c and "eval_M2b_s1.json" in c
    for line in (r1, c):
        argv = shlex.split(line, posix=False)
        argv = [x.strip('"') for x in argv[2:]]                             # drop "python" "v8f_mini_eval.py"
        with pytest.raises(FileNotFoundError):                              # past argparse: an unknown flag would be SystemExit(2)
            EV.main(argv)
