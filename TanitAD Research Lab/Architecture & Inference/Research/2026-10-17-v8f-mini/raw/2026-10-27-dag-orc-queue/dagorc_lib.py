"""Shared helpers of the DAG-ORC box-2 queue proofs: run the REAL queue .cmd through cmd.exe and parse what it prints. (dag2q_lib.py's form, re-pointed at dag_orc_chain/.)

``dry(...)`` is the chain's own ``dry_stage``: a one-line wrapper batch file sets V8F_DRYRUN=1 and BOX 2's paths, then ``call``s the queue, so the line printed IS the line the real run executes.

SCRATCH: every transient file (wrapper .cmd files, fake boxes, mutants) goes to ``BT`` -- DAGORC_BT, default F:/TanitAD-scratch/dagorc_work/bt -- never under dag_orc_chain/ (E: is exFAT with 1 MiB clusters and ~6 GB free).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
BT = Path(os.environ.get("DAGORC_BT") or r"F:\TanitAD-scratch\dagorc_work\bt")
BT.mkdir(parents=True, exist_ok=True)
QUEUE = Path(os.environ["DAGORC_QUEUE"]) if os.environ.get("DAGORC_QUEUE") else HERE / "stage" / PK / "code" / "v8f_mini_box2_queue.cmd"      # DAGORC_QUEUE: the mutation runner points the proofs at a MUTATED queue
TIP_QUEUE = HERE / "tip" / "queue_3b7b404e.cmd"                                                                                          # the repo tip's queue (3b7b404e): what every existing stage must still print
BOX2 = {"REPO": r"D:\tanitad\repo", "PY": r"D:\tanitad\venv\Scripts\python.exe", "CACHEFULL": r"D:\tanitad\cache_full", "V9": r"D:\tanitad\data\v9", "RUNSM": r"D:\tanitad\runs\v8f_mini_full"}
PY = r"C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
DAG2_DIR = Path(os.environ.get("DAGORC_DAG2_DIR") or r"E:\Projects\TanitAD-artifacts\v8f-dagger2")      # DAG2-0's banked run (read-only; only argv.json / summary.json / selfcheck.json / SHA256SUMS / gate/ are ever opened)
DAG_DIR = Path(os.environ.get("DAGORC_DAG_DIR") or r"E:\Projects\TanitAD-artifacts\v8f-dagger")        # DAG's banked run (checkpoint sha256 read from its SHA256SUMS.txt; the checkpoint is not opened)


def _wrapper(queue: Path, args: str, env: dict, dryrun: bool) -> str:
    lines = ["@echo off"]
    if dryrun:
        lines.append("set V8F_DRYRUN=1")
    for k, v in env.items():
        lines.append(f'set "{k}={v}"')
    lines.append(f'call "{queue}" {args}')
    lines.append("exit /b %errorlevel%")
    return "\r\n".join(lines) + "\r\n"


def cmd(args: str, *, env: dict | None = None, dryrun: bool = True, queue: Path = QUEUE, cwd=None, timeout=300):
    """(returncode, stdout lines) of ``queue args`` through cmd.exe. ``env`` is the box paths + extras; the dev box's own environment never decides what box 2 would run (BOX2 paths are the default)."""
    e = {**BOX2, **(env or {})}
    with tempfile.NamedTemporaryFile("w", suffix=".cmd", delete=False, dir=str(BT), newline="") as fh:
        fh.write(_wrapper(queue, args, e, dryrun))
        w = fh.name
    try:
        p = subprocess.run(["cmd.exe", "/c", w], capture_output=True, stdin=subprocess.DEVNULL, timeout=timeout, cwd=cwd)
    finally:
        os.unlink(w)
    out = (p.stdout or b"").decode("utf-8", "replace").replace("\r", "")
    return p.returncode, [x for x in out.split("\n") if x != ""]


class QueueStageError(RuntimeError):
    """a stage of the queue under test did not print its line (rc != 0, or not exactly one trainer line): the queue's defect, never the rig's"""


def dry(args: str, *, queue: Path = QUEUE, env=None):
    """The printed line(s) of a stage under V8F_DRYRUN (rc must be 0)."""
    rc, out = cmd(args, env=dict(env or {}), dryrun=True, queue=queue)
    if rc != 0:
        raise QueueStageError(f"{args}: rc={rc} {out}")
    return out


def trainer_line(lines):
    x = [l for l in lines if "v8f_mini_train.py" in l]
    if len(x) != 1:
        raise QueueStageError(f"expected exactly one trainer line, got {lines}")
    return x[0]


def tokens(line: str) -> list:
    """The queue's own tokenisation (chain_v8f_orc.sh ``argv_of``): double quotes group and are removed, nothing else is interpreted."""
    return [x.strip('"') for x in re.findall(r'"[^"]*"|\S+', line.strip())]


def train_argv(line: str) -> list:
    t = tokens(line)
    return t[next(k for k, x in enumerate(t) if x.lower().endswith("v8f_mini_train.py")) + 1:]


def flags(a: list) -> dict:
    """flag -> list of value lists (the asserter's own form); a flag that is missing reads ``<MISSING --x>`` through ``fl(...)`` so a dropped flag is a FAILED check, not a KeyError."""
    o, k = {}, 0
    while k < len(a):
        if a[k].startswith("--"):
            j = k + 1
            while j < len(a) and not a[j].startswith("--"):
                j += 1
            o.setdefault(a[k], []).append(a[k + 1:j])
            k = j
        else:
            k += 1
    return o


def fl(a: list, name: str):
    f = flags(a).get(name)
    return f"<MISSING {name}>" if f is None else (f[0] if len(f) == 1 else f)


def drop(x: list, names=("--out",), bare=()) -> list:
    o, k = [], 0
    while k < len(x):
        if x[k] in names:
            k += 2
        elif x[k] in bare:
            k += 1
        else:
            o.append(x[k])
            k += 1
    return o


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def banked_argv(arm_dir: str, seed: int) -> list:
    """DAG2-0's banked argv.json (arm_dir = M2b_s<S>_dag20): the reference of the ONE-LEVER rule."""
    return json.loads((DAG2_DIR / f"M2b_s{seed}_{arm_dir}" / "argv.json").read_text(encoding="utf-8"))["argv"]
