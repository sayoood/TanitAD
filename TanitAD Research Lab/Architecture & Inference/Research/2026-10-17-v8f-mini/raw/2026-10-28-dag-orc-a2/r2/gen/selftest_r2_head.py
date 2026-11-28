#!/usr/bin/env python3
"""Self-test AND mutation proof of the DAG-ORC (DAG-OACC base) chain r2: the re-run after the r1 chain stopped on K-REACH.  CPU only; nothing touches box 2, box2_ship, git (read-only `git show` on the mirror) or the network.

    python selftest_chain.py [--quiet]      writes selftest_chain_result.json; rc 0 iff every test passes AND every mutant is killed

ASSEMBLED by mk_selftest_r2.py from  (1) this head,  (2) r1's selftest_chain.py sections A (gate asserter on four real-record worlds + the token), A-mut, K (run-record / K-REACH / one-lever) VERBATIM -- here run against r2's checks module,
which carries r1's code byte for byte, so the r1 proofs transfer and are re-proved -- and  (3) the r2 tail:  C2 (the r2 chain script: dry-run, check-tree, STREAM_PATTERN, real-run refusals, tampered helpers),  X1 (the r2 predecessor rule),
X2 (the stream-record clause),  X3 (the box-2 and dev-box ASIDE, run for real with cmd.exe / bash on a scratch tree, including relaunch idempotence),  X4 (r2 invariants vs r1),  X5 (the waker).
NOT re-run here, and why: r1's section Q / R (the queue patch and its non-dry run on a stub box) test the QUEUE, which r2 does not change (r2 ships no queue). The r2 asserter is r1's + ONE clause (ego_stream_problems), so r1's asserter tests A / A-mut run against it
with the fixed trainer's field grafted into the worlds; X6 tests the new clause (rows + mutants) and the REAL records of the fixed trainer's composed run (fixtures/compose_fix_real/).
A MUTANT is a deliberate regression: a world / log / chain / lib with ONE thing wrong that the registered check must REFUSE.  A mutant that is accepted (or refused for a different reason than the one it was built to trip) is a SURVIVOR and fails the run.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent                          # .../dagorc_r2/gen
R2 = HERE.parent
R2SHIP = R2 / "box2_ship"
R1DIR = Path("E:/Projects/TanitAD-artifacts/dagorc_rebase/chain")
R1SHIP = R1DIR / "box2_ship"
INST = Path("C:/Users/Admin/box2_ship")
sys.path.insert(0, str(R1DIR))
sys.path.insert(0, str(R2SHIP))                                 # r2's checks and the (byte-identical) asserter win over r1's on the path
sys.path.insert(0, str(HERE))
import gate_dagoaccorc_r2_assert as G                   # noqa: E402  (r2's asserter: r1's + ego_stream_problems)
import v8f_dagoaccorc_r2_chain_checks as C              # noqa: E402
import world as W                                       # noqa: E402  (r1's: DAG-OACC's REAL banked records; it imports r1's asserter from r1's box2_ship, which is r1's record and stays installed)
import make_chain_r2 as MC                              # noqa: E402

W.G = G                                                 # world.py's functions (arm_argv, build, run_check) now run against r2's asserter
_WB = W.build


def _build_with_stream(root, arm, seed, mutate=None, **kw):
    """r1's world builder + the field the FIXED trainer records (levers.ego_perturb_stream: 1 on A, 0 on REP) in the grafted gate selfcheck; a mutant's ``mutate`` runs AFTER it, so it can delete / change the field."""
    def mut2(sc):
        sc["levers"]["ego_perturb_stream"] = 1 if G.ARMS[arm] > 0.0 else 0
        if mutate:
            mutate(sc)
    return _WB(root, arm, seed, mutate=mut2, **kw)


W.build = _build_with_stream

SCR = Path("F:/TanitAD-scratch/dagorc_r2_tmp/chain_selftest")
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
QUEUE = R1DIR / "stage" / PK / "code" / "v8f_mini_box2_queue.cmd"
TIPQ = R1DIR / "tip" / "queue_bf404bfe.cmd"
BASH = "C:/Program Files/Git/bin/bash.exe"
PYV = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
BOX2 = {"REPO": r"D:\tanitad\repo", "PY": r"D:\tanitad\venv\Scripts\python.exe", "CACHEFULL": r"D:\tanitad\cache_full", "V9": r"D:\tanitad\data\v9", "RUNSM": r"D:\tanitad\runs\v8f_mini_full"}
QUIET = "--quiet" in sys.argv
RESULTS, MUTANTS = [], []


def T(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    if not QUIET or not cond:
        print(("ok   " if cond else "FAIL ") + name + ("" if cond else f"  <- {str(detail)[:300]}"))


def M(name, killed, why=""):
    MUTANTS.append((name, bool(killed)))
    if not QUIET or not killed:
        print(("killed   " if killed else "SURVIVOR ") + name + ("" if killed else f"  <- {str(why)[:300]}"))


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def fresh(sub):
    d = SCR / sub
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    return d


