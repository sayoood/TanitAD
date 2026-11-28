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

SCR = Path("F:/TanitAD-scratch/dagorc_r2_tmp/chain_selftest_dbg")
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


# ====================================================================== C2. the r2 chain script
import inspect                                          # noqa: E402

print("== C2: the r2 chain script (dry-run, check-tree, STREAM_PATTERN, real-run refusals, tampered helpers)")
FIX = R2 / "fixtures"
R1LOG = FIX / "r1_chain_v8f_dagoaccorc.out"            # r1's chain log, snapshotted (the live copy is C:/Users/Admin/box2_ship/logs/chain_v8f_dagoaccorc.out)
OACC_TAIL = FIX / "oacc_r2_chain_tail.out"              # the tail of the DAG-OACC r2 chain's log (it ENDED in ZZV8FMINI-DONE)
DEFAULT_CHAIN = (R2SHIP / "chain_v8f_dagoaccorc_r2.sh").read_text(encoding="utf-8")
R1_CHAIN = (R1SHIP / "chain_v8f_dagoaccorc.sh").read_text(encoding="utf-8")
SP = "stack/scripts/v8f_mini_train.py|STUB_EGO_STREAM_SEPARATED"
SR = "levers.ego_perturb_stream=true"
SET_CHAIN = MC.render(SP, SR)


def sh(cmdline, env=None, timeout=300):
    e = {**os.environ, "MSYS_NO_PATHCONV": "0", **(env or {})}
    try:
        p = subprocess.run([BASH, "-c", cmdline], capture_output=True, stdin=subprocess.DEVNULL, timeout=timeout, env=e)
    except subprocess.TimeoutExpired as ex:
        return 124, ((ex.stdout or b"").decode("utf-8", "replace") + (ex.stderr or b"").decode("utf-8", "replace")).replace("\r", "")
    return p.returncode, (p.stdout or b"").decode("utf-8", "replace").replace("\r", "") + (p.stderr or b"").decode("utf-8", "replace").replace("\r", "")


def bp(p):
    return str(p).replace("\\", "/")


def ship2(root, chain=None, rm=(), tamper=None):
    shutil.rmtree(root, ignore_errors=True)
    (root / "tmp").mkdir(parents=True)
    for f in ("gate_dagoaccorc_assert.py", "v8f_dagoaccorc_r2_chain_checks.py", "oaf_dagoaccorc_r2_lib.sh"):
        shutil.copy(R2SHIP / f, root / f)
    (root / "chain_v8f_dagoaccorc_r2.sh").write_bytes((DEFAULT_CHAIN if chain is None else chain).encode("utf-8"))
    for f in ("oaf_lib.sh", "v8f_oaf_chain_checks.py", "v8f_guard_dagoaccorc.sh"):
        shutil.copy(INST / f, root / f)
    for f in rm:
        (root / f).unlink()
    if tamper:
        with open(root / tamper, "ab") as fh:
            fh.write(b"\n# tampered\n")
    return root


def run_in(sd, args, env=None, timeout=300, kill_after=None):
    """kill_after: coreutils `timeout N` INSIDE the shell, so a chain that waits forever is killed with its children (a python-level timeout would kill only the outer bash and leave the chain looping, holding the pipe)."""
    e = {"TMPDIR": bp(sd) + "/tmp", "OAF_SHIP_DIR": bp(sd), "DAGOACCORC_ART": bp(sd) + "/art", **(env or {})}
    pre = " ".join(f'{k}="{v}"' for k, v in e.items())
    tmo = f"timeout {kill_after} " if kill_after else ""
    return sh(f'cd "{bp(sd)}" && {pre} {tmo}bash ./chain_v8f_dagoaccorc_r2.sh {args}', timeout=timeout if not kill_after else kill_after + 60)



CSRC = (R2SHIP / "v8f_dagoaccorc_r2_chain_checks.py").read_text(encoding="utf-8")
LIBSRC = (R2SHIP / "oaf_dagoaccorc_r2_lib.sh").read_text(encoding="utf-8")
RUNS_CMD = r"D:\tanitad\runs\v8f_mini_full"
ARMS2 = ("dagoaccorc", "dagoaccorcrep")
def load_mod(src, tag):
    d = SCR / "mods" / tag
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    shutil.copy(R2SHIP / "gate_dagoaccorc_assert.py", d / "gate_dagoaccorc_assert.py")
    (d / f"chk_{tag}.py").write_text(src, encoding="utf-8", newline="\n")
    spec = importlib.util.spec_from_file_location(f"chk_{tag}", d / f"chk_{tag}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def put(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))


# ====================================================================== X4. r2 invariants vs r1
print("== X4: what r2 must NOT have changed")


def modfile(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


C1 = modfile(R1SHIP / "v8f_dagoaccorc_chain_checks.py", "r1_checks")
same_src = {n: inspect.getsource(getattr(C1, n)) == inspect.getsource(getattr(C, n)) for n in ("reached", "_count_pair", "one_lever", "findk", "argv_sha256")}
T("X4-1 K-REACH (reached, _count_pair), one_lever, findk, argv_sha256 are SOURCE-IDENTICAL to r1's checks (the bands 0.1 +- 0.02 / 0.3 +- 0.02 / overlap exactly 0 are unchanged)", all(same_src.values()) and C1.REACH_TOL == C.REACH_TOL == 0.02, same_src)
rr_src1, rr_src2 = inspect.getsource(C1.run_record), inspect.getsource(C.run_record)
T("X4-2 run_record is r1's plus exactly the one stream clause", rr_src2.replace('    bad += stream_problems(sm, p, getattr(a, "stream_record", ""))      # r2: the separated perturbation stream\n', "") == rr_src1)
mutm = CSRC.replace("REACH_TOL = 0.02", "REACH_TOL = 0.05")
M("MX4-1 a loosened K-REACH tolerance (0.05) in the r2 checks: the source-identity check refuses it", inspect.getsource(load_mod(mutm, "tol").reached) != inspect.getsource(C1.reached) or load_mod(mutm, "tol").REACH_TOL != C1.REACH_TOL)
asr_sha = hashlib.sha256((R2SHIP / "gate_dagoaccorc_assert.py").read_bytes()).hexdigest()
T("X4-3 the asserter in r2's box2_ship is byte-identical to r1's AND to the installed one", asr_sha == hashlib.sha256((R1SHIP / "gate_dagoaccorc_assert.py").read_bytes()).hexdigest() == hashlib.sha256((INST / "gate_dagoaccorc_assert.py").read_bytes()).hexdigest())


def fn_body(t, name):
    m = re.search(rf"^{re.escape(name)}\(\) \{{.*?^\}}\n", t, re.M | re.S)
    assert m, name
    return m.group(0)


def arr(t, name):
    m = re.search(rf"^{name}=\(\n(.*?)\n\)\n", t, re.M | re.S)
    return m.group(1) if m else None


def line_of(t, prefix):
    return next(x for x in t.split("\n") if x.startswith(prefix))


inv = {}
inv["gate_one"] = fn_body(R1_CHAIN, "gate_one") == fn_body(DEFAULT_CHAIN, "gate_one")
inv["runone"] = fn_body(R1_CHAIN, "runone").replace("--steps $DAG_STEPS ||", '--steps $DAG_STEPS --stream-record "$STREAM_RECORD" ||') == fn_body(DEFAULT_CHAIN, "runone")
inv["pinned_b"] = fn_body(R1_CHAIN, "pinned_b") == fn_body(DEFAULT_CHAIN, "pinned_b")
inv["plan"] = fn_body(R1_CHAIN, "plan") == fn_body(DEFAULT_CHAIN, "plan")
inv["ckptsha_x"] = fn_body(R1_CHAIN, "ckptsha_x") == fn_body(DEFAULT_CHAIN, "ckptsha_x")
inv["EXPORT"] = arr(R1_CHAIN, "EXPORT") == arr(DEFAULT_CHAIN, "EXPORT")
for var in ("OACC_CKPT_SHA", "OACC_ARGV_SHA", "OACC_SUMM_SHA", "M2B_SHA", "BANK1_SHA", "WACC", "WACC_TXT_SHA", "ASSERT_SHA", "OAF_LIB_SHA", "OAF_CHECKS_SHA", "BUILD_REF", "RUNS", "DAG_STEPS", "GATE_POLLS", "FLOOR_BYTES", "NEED_BYTES", "ART", "OACC_DIR", "DAG1_DIR"):
    inv[var] = line_of(R1_CHAIN, var + "=") == line_of(DEFAULT_CHAIN, var + "=")
k1 = [x for x in R1_CHAIN.split("\n") if "reached --summary" in x]
inv["K-REACH call"] = k1 == [x for x in DEFAULT_CHAIN.split("\n") if "reached --summary" in x] and len(k1) == 1
lev1, lev2 = arr(R1_CHAIN, "LEVER") or "", arr(DEFAULT_CHAIN, "LEVER") or ""
inv["LEVER"] = lev1 == lev2 and lev1 != ""
T("X4-4 the r2 chain carries r1's pins, RUNS, K-REACH call, LEVER greps, EXPORT, gate_one / runone (+ the one stream argument) / pinned_b / plan / ckptsha_x unchanged", all(inv.values()), [k for k, v in inv.items() if not v])
mutc = DEFAULT_CHAIN.replace("RUNS=(\"dagoaccorc 0\"", "RUNS=(\"dagoaccorcrep 9\" \"dagoaccorc 0\"", 1)
M("MX4-2 a chain whose RUNS differ from r1's: the invariant check refuses it", line_of(R1_CHAIN, "RUNS=") != line_of(mutc, "RUNS="))
T("X4-5 the r2 chain's STREAM_* lines are the ONLY unset-able part: make_chain_r2.py --check regenerates the file on disk byte for byte (STREAM_* read from the file)", subprocess.run([PYV, "-I", str(HERE / "make_chain_r2.py"), "--check"], capture_output=True, cwd=str(HERE)).returncode == 0)
T("X4-6 the r2 runner .cmd starts the r2 chain with AFTER_OUT = r1's log and logs to the r2 log (CRLF)", (lambda b: b"\r\n" in b and b"AFTER_OUT=/c/Users/Admin/box2_ship/logs/chain_v8f_dagoaccorc.out bash /c/Users/Admin/box2_ship/chain_v8f_dagoaccorc_r2.sh" in b
                                                                                                       and b"> /c/Users/Admin/box2_ship/logs/chain_v8f_dagoaccorc_r2.out 2>&1" in b)((R2SHIP / "run_chain_v8f_dagoaccorc_r2.cmd").read_bytes()))
# the guard's anchored end test knows the r2 log's end markers
gl = SCR / "guard_probe.out"
gl.write_text("10-28 waiting for ^ZZV8FMINI-DONE\nZZV8FMINI-STOP something\n", encoding="utf-8")
rcg, _ = sh(f"grep -q -E '^(ZZV8FMINI|ZZBETA|ZZV8FTAC)-(DONE|STOP)' \"{bp(gl)}\"")
gl2 = SCR / "guard_probe2.out"
gl2.write_text("10-28 01:16 waiting for the r1 chain (x: end markers ^ZZV8FMINI-DONE / ^ZZV8FMINI-STOP)" + chr(10), encoding="utf-8")
rcg2, _ = sh(f"grep -q -E '^(ZZV8FMINI|ZZBETA|ZZV8FTAC)-(DONE|STOP)' \"{bp(gl2)}\"")
T("X4-7 the installed guard's anchored end test matches an r2 stop line (and the r2 chain's own 'waiting' echo does not)", rcg == 0 and rcg2 == 1 and 'grep -q -E \'^(ZZV8FMINI|ZZBETA|ZZV8FTAC)-(DONE|STOP)\'' in (INST / "v8f_guard_dagoaccorc.sh").read_text(encoding="utf-8"))

# ====================================================================== X5. the waker
print("== X5: the r2 waker")
WK = (R2SHIP / "wake_v8fdagoaccorc_r2.sh").read_text(encoding="utf-8")


def waker(tag, lines, guard=None, pre=(), wk=None):
    d = fresh(f"wk_{tag}")
    (d / "logs").mkdir()
    (d / "logs" / "chain_v8f_dagoaccorc_r2.out").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if guard:
        (d / "logs" / "v8fdagoaccorc_r2_guard.out").write_text(guard + "\n", encoding="utf-8")
    for p in pre:
        (d / "logs" / p).write_text("")
    t = (wk or WK).replace("L=/c/Users/Admin/box2_ship/logs", f"L={bp(d)}/logs")
    assert t != (wk or WK)
    put(d / "wake.sh", t)
    return sh(f'timeout 12 bash "{bp(d)}/wake.sh"', timeout=40), d


GT = ["gate token written for M2b_s%d_%s commit x argv_sha256 y" % (s, a) for s in (0, 1) for a in ARMS2]
(rc, out), d = waker("stop", ["10-28 x", "ZZV8FMINI-STOP reached-control fail M2b_s0_dagoaccorc"])
T("X5-1 the waker wakes on the r2 chain's STOP marker", rc == 0 and "CHAIN END" in out, out[-200:])
(rc, out), d = waker("gates", ["10-28 x", *GT], pre=(".dagoaccorc_gates_seen",))
T("X5-2 the waker wakes once on the four gate tokens, with r1's gates-seen marker already present (r2 has its own marker)", rc == 0 and "GATES PASSED" in out and (d / "logs" / ".dagoaccorc_r2_gates_seen").exists(), out[-200:])
(rc, out), d = waker("named", ["10-28 x", "ZZDAGOACCORC-CONTROL-FAIL M2b_s0_dagoaccorc: x"])
T("X5-3 the waker wakes on the K-REACH named stop", rc == 0 and "NAMED STOP" in out, out[-200:])
(rc, out), d = waker("guard", ["10-28 x"], guard="ZZGUARD-STOP TEMP x")
T("X5-4 the waker wakes on a guard stop (exit 4)", rc == 4 and "GUARD" in out, (rc, out[-200:]))
wk_m = WK.replace(".dagoaccorc_r2_gates_seen", ".dagoaccorc_gates_seen")
(rc, out), d = waker("gates_m", ["10-28 x", *GT], pre=(".dagoaccorc_gates_seen",), wk=wk_m)
M("MX5-1 a waker sharing r1's gates-seen marker stays asleep when r1's marker exists", "GATES PASSED" not in out, out[-200:])
wk_m = WK.replace("chain_v8f_dagoaccorc_r2.out", "chain_v8f_dagoaccorc.out")
(rc, out), d = waker("stop_m", ["10-28 x", "ZZV8FMINI-STOP reached-control fail M2b_s0_dagoaccorc"], wk=wk_m)
M("MX5-2 a waker reading r1's log name never sees the r2 chain's stop", "CHAIN END" not in out, out[-200:])


npass, nfail = sum(1 for _, ok in RESULTS if ok), sum(1 for _, ok in RESULTS if not ok)
nkill, nsurv = sum(1 for _, ok in MUTANTS if ok), sum(1 for _, ok in MUTANTS if not ok)
print("DEBUG-X45:", npass, "tests pass,", nfail, "fail;", nkill, "mutants killed,", nsurv, "survivors")
