"""Shared helpers of the DAGGER2 box-2 queue proofs: run the REAL queue .cmd through cmd.exe and parse what it prints. (dagq_lib.py's form, re-pointed at dag2q/.)

``dry(...)`` is the chain's own ``dry_stage``: a one-line wrapper batch file sets V8F_DRYRUN=1 and BOX 2's paths, then ``call``s the queue, so the line printed IS the line the real run executes.

SCRATCH: every transient file (wrapper .cmd files, fake boxes, mutants) goes to ``BT`` -- DAG2Q_BT, default the session scratch directory on C: -- never under dag2q/: E: is exFAT with 1 MiB clusters, so every
loose file costs a megabyte there.
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
BT = Path(os.environ.get("DAG2Q_BT") or r"C:\Users\Admin\AppData\Local\Temp\claude\E--Projects-TanitAD\bfe78918-15e8-439b-8566-593ca3297bd8\scratchpad\dag2q_work\bt")
BT.mkdir(parents=True, exist_ok=True)
QUEUE = Path(os.environ["DAG2Q_QUEUE"]) if os.environ.get("DAG2Q_QUEUE") else HERE / "stage" / PK / "code" / "v8f_mini_box2_queue.cmd"      # DAG2Q_QUEUE: the mutation runner points the proofs at a MUTATED queue
TIP_QUEUE = HERE / "tip" / "queue_8e121ea8.cmd"                                                                                          # the repo tip's queue (8e121ea8): what every existing stage must still print
BOX2 = {"REPO": r"D:\tanitad\repo", "PY": r"D:\tanitad\venv\Scripts\python.exe", "CACHEFULL": r"D:\tanitad\cache_full", "V9": r"D:\tanitad\data\v9", "RUNSM": r"D:\tanitad\runs\v8f_mini_full"}
SHA_FAKE = "ab" * 32
PY = r"C:\Users\Admin\venvs\tanitad\Scripts\python.exe"


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


def dry(args: str, *, sha=None, queue: Path = QUEUE, env=None):
    """The printed line(s) of a stage under V8F_DRYRUN (rc must be 0); ``sha`` = DAGSHA_DRY (what the chain hashed from the pulled bank.json)."""
    e = dict(env or {})
    if sha:
        e["DAGSHA_DRY"] = sha
    rc, out = cmd(args, env=e, dryrun=True, queue=queue)
    if rc != 0:
        raise RuntimeError(f"{args}: rc={rc} {out}")
    return out


def trainer_line(lines):
    x = [l for l in lines if "v8f_mini_train.py" in l]
    if len(x) != 1:
        raise RuntimeError(f"expected exactly one trainer line, got {lines}")
    return x[0]


def tokens(line: str) -> list:
    """The queue's own tokenisation (chain_v8f_orc.sh ``argv_of``): double quotes group and are removed, nothing else is interpreted."""
    return [x.strip('"') for x in re.findall(r'"[^"]*"|\S+', line.strip())]


def train_argv(line: str) -> list:
    t = tokens(line)
    return t[next(k for k, x in enumerate(t) if x.lower().endswith("v8f_mini_train.py")) + 1:]


def flags(argv: list) -> dict:
    o, k = {}, 0
    while k < len(argv):
        if argv[k].startswith("--"):
            j = k + 1
            while j < len(argv) and not argv[j].startswith("--"):
                j += 1
            o.setdefault(argv[k], []).append(argv[k + 1:j])
            k = j
        else:
            k += 1
    return o


def argv_sha256(argv: list) -> str:
    """launch_gate.argv_sha256: sha256 of the ORDERED json array, compact separators."""
    return hashlib.sha256(json.dumps(list(argv), ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def drop_flag(argv: list, names: set, bare: set = frozenset()) -> list:
    """argv without the flags in ``names`` (each with its value tokens) and the bare flags in ``bare``."""
    o, k = [], 0
    while k < len(argv):
        if argv[k] in names:
            k += 1
            while k < len(argv) and not argv[k].startswith("--"):
                k += 1
        elif argv[k] in bare:
            k += 1
        else:
            o.append(argv[k])
            k += 1
    return o


DAG_ONLY = {"--dagger-bank", "--dagger-bank-sha256", "--dagger-seed"}
