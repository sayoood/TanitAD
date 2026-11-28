#!/usr/bin/env python3
"""Self-test AND mutation proof of the DAG-ORC (DAG-OACC base) chain r2: the re-run after the r1 chain stopped on K-REACH.  CPU only; nothing touches box 2, box2_ship, git (read-only `git show` on the mirror) or the network.

    python selftest_chain.py [--quiet]      writes selftest_chain_result.json; rc 0 iff every test passes AND every mutant is killed

ASSEMBLED by mk_selftest_r2.py from  (1) this head,  (2) r1's selftest_chain.py sections A (gate asserter on four real-record worlds + the token), A-mut, K (run-record / K-REACH / one-lever) VERBATIM -- here run against r2's checks module,
which carries r1's code byte for byte, so the r1 proofs transfer and are re-proved -- and  (3) the r2 tail:  C2 (the r2 chain script: dry-run, check-tree, STREAM_PATTERN, real-run refusals, tampered helpers),  X1 (the r2 predecessor rule),
X2 (the stream-record clause),  X3 (the box-2 and dev-box ASIDE, run for real with cmd.exe / bash on a scratch tree, including relaunch idempotence),  X4 (r2 invariants vs r1),  X5 (the waker).
NOT re-run here, and why: r1's section Q / R (the queue patch and its non-dry run on a stub box) test the QUEUE, which r2 does not change (r2 ships no queue); the asserter is byte-identical to r1's (X4 pins its sha256).
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
import gate_dagoaccorc_assert as G                      # noqa: E402
import v8f_dagoaccorc_r2_chain_checks as C              # noqa: E402
import world as W                                       # noqa: E402  (r1's: DAG-OACC's REAL banked records)
import make_chain_r2 as MC                              # noqa: E402

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



FIX = R2 / "fixtures"
import inspect
LIBSRC = (R2SHIP / "oaf_dagoaccorc_r2_lib.sh").read_text(encoding="utf-8")
CSRC = (R2SHIP / "v8f_dagoaccorc_r2_chain_checks.py").read_text(encoding="utf-8")
def sh(cmdline, env=None, timeout=300):
    e = {**os.environ, "MSYS_NO_PATHCONV": "0", **(env or {})}
    try:
        p = subprocess.run([BASH, "-c", cmdline], capture_output=True, stdin=subprocess.DEVNULL, timeout=timeout, env=e)
    except subprocess.TimeoutExpired as ex:
        return 124, ((ex.stdout or b"").decode("utf-8", "replace") + (ex.stderr or b"").decode("utf-8", "replace")).replace("\r", "")
    return p.returncode, (p.stdout or b"").decode("utf-8", "replace").replace("\r", "") + (p.stderr or b"").decode("utf-8", "replace").replace("\r", "")


def bp(p):
    return str(p).replace("\\", "/")


# ====================================================================== X3. the ASIDE, run for real on a scratch tree
print("== X3: r2_aside_all (dev box rename + the box-2 cmd job) on a scratch tree, with the REAL lib text and cmd.exe")
RUNS_CMD = r"D:\tanitad\runs\v8f_mini_full"
ARMS2 = ("dagoaccorc", "dagoaccorcrep")


def put(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))


def r1_state(world, full=True):
    """The state r1 left: dev box ART and box 2's runs root (A s0 trained+evaluated, the four gates, pins). full=False: only the gates (no run, no eval)."""
    art, runs = world / "ART", world / "runs"
    put(art / "K-BUILD.txt", "k-build\n")
    put(art / "M2b_s0_dagoaccorc" / "ckpt_last.pt", os.urandom(4096))
    put(art / "M2b_s0_dagoaccorc" / "gate" / "selfcheck.json", '{"r1": true}')
    for s in (0, 1):
        for a in ARMS2:
            put(runs / f"gate_M2b-{a}_s{s}" / "selfcheck.json", f'{{"r1": "{a}{s}"}}')
            put(runs / f"gate_M2b-{a}_s{s}.log", f"gate log {a}{s}\n")
            for suf in ("", ".new", ".chk"):
                put(runs / "dagpin" / f"M2b_s{s}_{a}.train.argv.txt{suf}", f"pin {a}{s}{suf}\n")
    if full:
        put(runs / "M2b_s0_dagoaccorc" / "ckpt_last.pt", os.urandom(8192))
        put(runs / "M2b_s0_dagoaccorc" / "summary.json", '{"done": true}')
        put(runs / "M2b_s0_dagoaccorc" / "train.log", "train\n")
        for ext in ("json", "npz", "log"):
            put(runs / "eval" / f"eval_M2b-dagoaccorc_s0.{ext}", f"eval {ext}\n")
        put(runs / "eval" / "eval_M2b-dagoaccorc_s0_proj.npz", b"proj")
    # DECOYS that must never be touched
    for rel, data in (("M2b_s0/ckpt_last.pt", b"init0"), ("M2b_s1/ckpt_last.pt", b"init1"), ("M2b_s0_dagoacc/ckpt_last.pt", b"oacc"), ("dagger_bank/s0/bank.json", b"bank"), ("dagoacc/w_acc.txt", b"49.6969\n"),
                      ("dagpin/M2b_s0_dagoacc.train.argv.txt", b"oaccpin"), ("dagpin/bank_sha_dagoacc_s0.txt", b"sha"), ("eval/eval_M2b-dagoacc_s0.npz", b"oacceval"), ("gate_M2b-dagoacc_s0/selfcheck.json", b"oaccgate"),
                      ("gate_M2b-dagoacc_s0.log", b"oaccgatelog")):
        put(runs / rel, data)
    return world


def snap(root):
    """the state under test: the dev box's ART trees and box 2's runs root (the harness's own logs / jobs are not part of it)"""
    out = {}
    for p in sorted(Path(root).rglob("*")):
        rel = p.relative_to(root).as_posix()
        if not (rel.startswith("ART") or rel.startswith("runs")):
            continue
        out[rel] = "DIR" if p.is_dir() else hashlib.sha256(p.read_bytes()).hexdigest()
    return out


FAKE_SCP = SCR / "fake_scp.py"


def write_harness(world, lib_text):
    put(world / "lib.sh", lib_text.replace("\r", ""))
    put(FAKE_SCP, "import sys\nfrom pathlib import Path\nsrc, jobs, runs = sys.argv[1], sys.argv[2], sys.argv[3]\nb = Path(src).read_bytes()\n"
                  "old = rb'D:\\tanitad\\runs\\v8f_mini_full'\nassert b.count(old) == 1, b.count(old)\nPath(jobs, Path(src).name).write_bytes(b.replace(old, runs.encode()))\n")
    winruns = str(world / "runs").replace("/", "\\")
    h = """set -u
W=%(W)s
B=$W/B; L=$B/logs; mkdir -p $L $W/jobs
TAG=0000; ART=$W/ART; BOOT0=x; BUSY_TAKEN=0; CUR=$L/cur; SSH=false; SCP=fakescp
stop() { echo "ZZV8FMINI-STOP $*"; exit 1; }
source $W/lib.sh
fakescp() { "%(PY)s" "%(FS)s" "$1" "$W/jobs" '%(WR)s'; }
launch() { MSYS_NO_PATHCONV=1 cmd /c "$(cygpath -w $W/jobs/$1.cmd)" > $L/$1.log 2>&1 < /dev/null; echo "launched $1"; return 0; }
clean() { tr '\\r' '\\n' < "$L/$1.log"; }
waitrc() { local rc; rc=$(clean $1 | grep -a 'ZZJOB-END' | tail -1 | sed -n 's/.* rc=\\([-0-9]*\\).*/\\1/p'); [ "$rc" = 0 ] || { clean $1 | grep -a -i -E 'FAIL|refus|error|Traceback' | tail -8; stop "$1 rc=$rc"; }; }
r2_aside_all
echo HARNESS-END
""" % dict(W=bp(world), PY=PYV, FS=bp(FAKE_SCP), WR=winruns)
    put(world / "harness.sh", h)
    return world / "harness.sh"


def aside(world, lib_text=None):
    hs = write_harness(world, lib_text if lib_text is not None else LIBSRC)
    return sh(f'bash "{bp(hs)}"', timeout=180)


def items_present_r1(runs):
    """box-2 items of r1 that exist in the r1 state (relative, '/' form)"""
    return [rel for rel in (x.replace("\\", "/") for x in LIB_ITEMS) if (runs / rel).exists()]


def lib_items(lib_text):
    return [x.replace("\\", "/") for x in re.findall(r"^    '([^']+)'(?: \\)?$", lib_text, re.M)]


LIB_ITEMS = [x.replace("/", "\\") for x in lib_items(LIBSRC)]
T("X3-0 the lib lists 44 box-2 items: 4 run dirs, 4 gate dirs + 4 logs, 12 pin files, 20 eval files", len(LIB_ITEMS) == 44 and len({*LIB_ITEMS}) == 44, len(LIB_ITEMS))
SUFX = "_r1_kreach_stop"


def aside_checks(tag, lib_text=None, expect_mutant=False):
    """Run scenarios S1..S3 on a fresh world; return the list of failed expectations (empty = the registered behaviour)."""
    fails = []
    w = r1_state(fresh(f"aside_{tag}"))
    before = snap(w)
    present = items_present_r1(w / "runs")
    rc, out = aside(w, lib_text)
    after = snap(w)
    # S1: first launch
    n_aside = len(re.findall(r"^ASIDE-", out, re.M))
    if rc != 0 or "HARNESS-END" not in out:
        fails.append(f"S1 rc {rc}: {out[-300:]}")
    if not (w / "ART_r1_kreach_stop" / "K-BUILD.txt").exists() or (w / "ART" / "K-BUILD.txt").exists() or not (w / "ART").is_dir():
        fails.append("S1 the dev box ART was not renamed aside and re-created empty")
    for rel in present:                           # every r1 artifact renamed, content preserved byte for byte
        old, new = w / "runs" / rel, w / "runs" / (rel + SUFX)
        if old.exists() or not new.exists():
            fails.append(f"S1 {rel} not moved aside")
        elif before["runs/" + rel] != after.get("runs/" + rel + SUFX):
            fails.append(f"S1 {rel} content changed")
    for rel, h in before.items():                 # nothing was lost: every file of the r1 state is somewhere under its new name or untouched (decoys)
        if rel.startswith("ART/"):
            if after.get("ART_r1_kreach_stop/" + rel[4:]) != h:
                fails.append(f"S1 ART file {rel} lost / changed")
        elif rel.startswith("runs/") and h != "DIR":
            tail = rel[5:]
            is_item = any(tail == i or tail.startswith(i + "/") for i in present)
            if is_item:
                top = next(i for i in present if tail == i or tail.startswith(i + "/"))
                moved = "runs/" + top + SUFX + tail[len(top):]
                if after.get(moved) != h:
                    fails.append(f"S1 {rel} lost (expected {moved})")
            elif after.get(rel) != h:
                fails.append(f"S1 decoy {rel} touched")
    if not (w / "runs" / "r2_aside_done.txt").exists():
        fails.append("S1 no aside marker")
    if n_aside < 44 + 1:
        fails.append(f"S1 only {n_aside} ASIDE lines (44 box-2 items + the local one)")
    # r2 starts writing its own files
    put(w / "ART" / "r2_progress.txt", "r2\n")
    put(w / "runs" / "gate_M2b-dagoaccorc_s0" / "selfcheck.json", '{"r2": true}')
    put(w / "runs" / "M2b_s0_dagoaccorc" / "summary.json", '{"r2": true}')
    snap2 = snap(w)
    rc2, out2 = aside(w, lib_text)
    snap3 = snap(w)
    # S2: relaunch: idempotent, r2's progress untouched
    if rc2 != 0 or "ASIDE-ALREADY-DONE" not in out2 or "ASIDE-KEPT" not in out2:
        fails.append(f"S2 relaunch rc {rc2} / no ALREADY-DONE / no KEPT: {out2[-300:]}")
    if snap3 != snap2:
        fails.append("S2 the relaunch changed the tree: " + str([k for k in set(snap2) | set(snap3) if snap2.get(k) != snap3.get(k)][:4]))
    # S3: marker lost AND an r2 item with its r1 aside copy present: refused, nothing overwritten, no marker
    (w / "runs" / "r2_aside_done.txt").unlink(missing_ok=True)
    rc3, out3 = aside(w, lib_text)
    snap4 = snap(w)
    if not (rc3 != 0 and "ASIDE-FAIL" in out3 and "refusing to overwrite" in out3):
        fails.append(f"S3 not refused (rc {rc3}): {out3[-300:]}")
    if (w / "runs" / "r2_aside_done.txt").exists():
        fails.append("S3 a marker was written after a failure")
    if {k: v for k, v in snap4.items() if k != "runs/r2_aside_done.txt"} != {k: v for k, v in snap3.items() if k != "runs/r2_aside_done.txt"}:
        fails.append("S3 the failed job changed the tree: " + str([k for k in set(snap3) | set(snap4) if snap3.get(k) != snap4.get(k)][:4]))
    return fails


def aside_retry_checks(tag, lib_text=None):
    """S4: a job that died midway (half the items moved, no marker): the retry moves the rest, nothing twice."""
    fails = []
    w = r1_state(fresh(f"aside4_{tag}"))
    runs = w / "runs"
    pre = ["gate_M2b-dagoaccorc_s0", "gate_M2b-dagoaccorc_s0.log", "M2b_s0_dagoaccorc"]
    for rel in pre:
        os.rename(runs / rel, runs / (rel + SUFX))
    rc, out = aside(w, lib_text)
    if rc != 0 or not (runs / "r2_aside_done.txt").exists():
        fails.append(f"S4 retry rc {rc}: {out[-200:]}")
    for rel in pre:
        if not (runs / (rel + SUFX)).exists() or (runs / (rel + SUFX + SUFX)).exists():
            fails.append(f"S4 {rel} moved twice / lost")
    if (runs / "gate_M2b-dagoaccorc_s1").exists() or not (runs / ("gate_M2b-dagoaccorc_s1" + SUFX)).exists():
        fails.append("S4 the rest was not moved")
    return fails


f1 = aside_checks("real")
T("X3-1 the REAL lib: first launch renames EVERY r1 artifact (dev box and box 2) with content preserved and decoys untouched; a relaunch is idempotent and keeps r2's progress; a lost marker + an aside copy is REFUSED without overwriting", f1 == [], f1)
f4 = aside_retry_checks("real")
T("X3-2 the REAL lib: a job that died midway (no marker) is retried: the rest is moved, the moved items are not moved twice", f4 == [], f4)
T("X3-3 the box-2 job text contains no del / rmdir / move / copy / xcopy / robocopy / Remove-Item / rm (renames and the marker line only)",
  not re.search(r"(?i)\b(del|erase|rmdir|rd|move|copy|xcopy|robocopy|remove-item)\b|\brm\b", subprocess.run([BASH, "-c", f'source "{bp(R2SHIP)}/oaf_dagoaccorc_r2_lib.sh"; r2_aside_job_text X'], capture_output=True).stdout.decode("utf-8", "replace")))
import build_r2_lib as BL                            # noqa: E402
import build_r2_checks as BC                         # noqa: E402
T("X3-4 the :aside subroutine is DAG-OACC r2's (modulo the suffix): build_r2_lib.render() asserts it and reproduces the lib on disk byte for byte", BL.render() == LIBSRC)
T("X3-5 the r2 checks file on disk == build_r2_checks.derive(r1's file) byte for byte", BC.derive(BC.R1.read_text(encoding="utf-8")) == CSRC)

LIB_MUTS = [
    ("the eval npz are not in the item list (the queue would SKIP the eval and r2 would pull the refused run's)", lambda t: "\n".join(x for x in t.split("\n") if not re.match(r"^    'eval\\eval_M2b-dagoaccorc_s0\.npz'", x)), "1"),
    ("the run directories are not in the item list (the queue would SKIP the training)", lambda t: "\n".join(x for x in t.split("\n") if not re.match(r"^    'M2b_s0_dagoaccorc' ", x)), "1"),
    ("the pinned argv is not in the item list", lambda t: "\n".join(x for x in t.split("\n") if not re.match(r"^    'dagpin\\M2b_s0_dagoaccorc\.train\.argv\.txt' ", x)), "1"),
    ("the suffix is the OACC one (_r1_assert_stop)", lambda t: t.replace("R2_SUF=_r1_kreach_stop", "R2_SUF=_r1_assert_stop").replace('%~nx1_r1_kreach_stop', '%~nx1_r1_assert_stop').replace('"%~1_r1_kreach_stop"', '"%~1_r1_assert_stop"'), "1"),
    ("the job DELETES instead of renaming", lambda t: t.replace('ren "%~1" "%~nx1_r1_kreach_stop"', 'rmdir /s /q "%~1" 2>nul & del /q "%~1" 2>nul'), "1"),
    ("the job no longer refuses when the aside copy exists", lambda t: re.sub(r"if exist \"%~1_r1_kreach_stop\" \(echo ASIDE-FAIL[^\n]*\n", "", t, count=1), "1"),
    ("the job writes no marker (a relaunch would re-process r2's own files)", lambda t: t.replace("""    echo '>"%RS%\\r2_aside_done.txt" echo r2 aside done %DATE% %TIME%'\n""", ""), "1"),
    ("the dev-box rename is a delete", lambda t: t.replace('  mv "$d" "$n" || stop "r2 aside: cannot rename $d to $n"', '  rm -rf "$d" || stop "x"'), "1"),
    ("the dev-box aside ignores an existing aside copy (mv INTO it)", lambda t: t.replace('  [ -e "$n" ] && { echo "ASIDE-KEPT $d : the r1 copy is already aside as $n"; return 0; }\n', ""), "1"),
    ("$ART is not re-created after the rename", lambda t: t.replace('  mkdir -p "$ART" || stop "r2 aside: cannot create $ART"\n', ""), "1"),
]
for i, (name, fn, _) in enumerate(LIB_MUTS):
    mt = fn(LIBSRC)
    assert mt != LIBSRC, name
    fl = aside_checks(f"m{i}", mt) + aside_retry_checks(f"m{i}", mt)
    M(f"MX3-{i + 1:02d} the aside lib {name}: the scratch-tree scenarios catch it", len(fl) >= 1, fl)

npass, nfail = sum(1 for _, ok in RESULTS if ok), sum(1 for _, ok in RESULTS if not ok)
nkill, nsurv = sum(1 for _, ok in MUTANTS if ok), sum(1 for _, ok in MUTANTS if not ok)
print("DEBUG-X3:", npass, "tests pass,", nfail, "fail;", nkill, "mutants killed,", nsurv, "survivors")
