"""Shared helpers of the DAG-OACC / FLAT-DAG box-2 queue proofs: run the REAL queue .cmd through cmd.exe and parse what it prints. (dag2q_lib.py's form, re-pointed at oaccflatq/.)

``dry(...)`` is the chain's own ``dry_stage``: a one-line wrapper batch file sets V8F_DRYRUN=1 and BOX 2's paths, then ``call``s the queue, so the line printed IS the line the real run executes.
SCRATCH: every transient file (wrapper .cmd files, fake boxes, mutants) goes to ``BT`` -- OAF_BT, default the session scratch directory on C: -- never under oaccflatq/ (E: is exFAT with 1 MiB clusters).
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
BT = Path(os.environ.get("OAF_BT") or r"C:\Users\Admin\AppData\Local\Temp\claude\E--Projects-TanitAD\bfe78918-15e8-439b-8566-593ca3297bd8\scratchpad\oaccflatq_work\bt")
BT.mkdir(parents=True, exist_ok=True)
QUEUE = Path(os.environ["OAF_QUEUE"]) if os.environ.get("OAF_QUEUE") else HERE / "stage" / PK / "code" / "v8f_mini_box2_queue.cmd"      # OAF_QUEUE: the mutation runner points the proofs at a MUTATED queue
TIP_QUEUE = HERE / "tip" / "queue_1d4f4fbb.cmd"                                                                                          # the repo tip's queue (1d4f4fbb): what every existing stage must still print
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


def dry(args: str, *, sha=None, wacc=None, queue: Path = QUEUE, env=None):
    """The printed line(s) of a stage under V8F_DRYRUN (rc must be 0); ``sha`` = DAGSHA_DRY (what the chain hashed from the pulled bank.json), ``wacc`` = WACCDAG_DRY."""
    e = dict(env or {})
    if sha:
        e["DAGSHA_DRY"] = sha
    if wacc:
        e["WACCDAG_DRY"] = wacc
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
    """The queue's own tokenisation (chain argv_of): double quotes group and are removed, nothing else is interpreted."""
    return [x.strip('"') for x in re.findall(r'"[^"]*"|\S+', line.strip())]


def train_argv(line: str) -> list:
    t = tokens(line)
    return t[next(k for k, x in enumerate(t) if x.lower().endswith("v8f_mini_train.py")) + 1:]


class _Flags(dict):
    """argv flags -> list of value lists. A MISSING flag reads as the sentinel [["<MISSING --x>"]] (so a check on a dropped flag is a FAILED check, never a KeyError that crashes the whole self-test); `in` / `.get` / `set()` are unchanged."""

    def __missing__(self, k):
        return [[f"<MISSING {k}>"]]


def flags(argv: list) -> dict:
    o, k = _Flags(), 0
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


def dispatched(queue_text: str) -> list:
    """Every dispatched stage name of a queue file (``if /i "%~1"=="X" goto :X``)."""
    return re.findall(r'(?m)^if /i "%~1"=="(\w+)" goto :\w+\r?$', queue_text)


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
