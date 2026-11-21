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


def sc_set(path, value):
    def f(sc):
        o = sc
        for k in path[:-1]:
            o = o[k]
        o[path[-1]] = value
    return f


def sc_del(path):
    def f(sc):
        o = sc
        for k in path[:-1]:
            o = o[k]
        o.pop(path[-1], None)
    return f


def chain_mut(*fs):
    def f(sc):
        for g in fs:
            g(sc)
    return f


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
PH_CHAIN = MC.render(MC.PH_PATTERN, MC.PH_RECORD)      # the fail-closed placeholder chain (what the generator makes while the Master Mind has not set the STREAM_* values)
REG_PATTERN, REG_RECORD = MC.DEFAULT_PATTERN, MC.DEFAULT_RECORD


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
    for f in ("gate_dagoaccorc_r2_assert.py", "v8f_dagoaccorc_r2_chain_checks.py", "oaf_dagoaccorc_r2_lib.sh"):
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



qf = bp(QUEUE)
CSRC = (R2SHIP / "v8f_dagoaccorc_r2_chain_checks.py").read_text(encoding="utf-8")
LIBSRC = (R2SHIP / "oaf_dagoaccorc_r2_lib.sh").read_text(encoding="utf-8")
RUNS_CMD = r"D:\tanitad\runs\v8f_mini_full"
ARMS2 = ("dagoaccorc", "dagoaccorcrep")
other = SCR / "other_stop.out"
def load_mod(src, tag):
    d = SCR / "mods" / tag
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    shutil.copy(R2SHIP / "gate_dagoaccorc_r2_assert.py", d / "gate_dagoaccorc_r2_assert.py")
    (d / f"chk_{tag}.py").write_text(src, encoding="utf-8", newline="\n")
    spec = importlib.util.spec_from_file_location(f"chk_{tag}", d / f"chk_{tag}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cmut(old, new):
    assert CSRC.count(old) == 1, old
    return CSRC.replace(old, new)


# ====================================================================== X2. the stream-record clause
print("== X2: the stream-record clause of the run-record check (registered spec levers.ego_perturb_stream=1;0, the typed A / REP grammar, summary.json AND selfcheck.json)")
SPEC_T = "levers.ego_perturb_stream=true"
SPEC_ANY = "levers.ego_perturb_stream"
SPEC_S = 'levers.ego_perturb_stream="domain_separated"'
SPEC_R = REG_RECORD                      # levers.ego_perturb_stream=1;0   (A: exactly 1, REP: exactly 0)


def smry(v=None, absent=False, extra=None):
    sm = {"levers": {"w_acc": 49.6969}}
    if not absent:
        sm["levers"]["ego_perturb_stream"] = v
    return sm


STREAM_ROWS = [      # name, summary, p, spec, expect no problems
    # ---- the REGISTERED spec: A exactly the int 1, REP exactly the int 0, absent fails on both, typed
    ("REG A: 1", smry(1), 0.3, SPEC_R, True),
    ("REG A: absent (the r1 trainer at e7ea20d1 recorded no such field)", smry(absent=True), 0.3, SPEC_R, False),
    ("REG A: 0 (the original, shared stream)", smry(0), 0.3, SPEC_R, False),
    ("REG A: True (== 1 in python, not the int 1)", smry(True), 0.3, SPEC_R, False),
    ("REG A: 1.0 (a float, not the int 1)", smry(1.0), 0.3, SPEC_R, False),
    ("REG A: 2", smry(2), 0.3, SPEC_R, False),
    ("REG A: null", smry(None), 0.3, SPEC_R, False),
    ("REG A: the string '1'", smry("1"), 0.3, SPEC_R, False),
    ("REG A: a nested path that does not exist", smry(1), 0.3, "levers.nope.deeper=1;0", False),
    ("REG REP: 0", smry(0), 0.0, SPEC_R, True),
    ("REG REP: absent (a REP literal is registered: absent FAILS)", smry(absent=True), 0.0, SPEC_R, False),
    ("REG REP: False (== 0 in python, not the int 0)", smry(False), 0.0, SPEC_R, False),
    ("REG REP: 0.0 (a float, not the int 0)", smry(0.0), 0.0, SPEC_R, False),
    ("REG REP: null", smry(None), 0.0, SPEC_R, False),
    ("REG REP: 1 (the replicate carries the separated stream)", smry(1), 0.0, SPEC_R, False),
    ("REG REP: the string '0'", smry("0"), 0.0, SPEC_R, False),
    # ---- the older grammars (no REP literal) are still handled
    ("A: literal true, field true", smry(True), 0.3, SPEC_T, True),
    ("A: literal true, field absent", smry(absent=True), 0.3, SPEC_T, False),
    ("A: literal true, field false", smry(False), 0.3, SPEC_T, False),
    ("A: literal true, field null", smry(None), 0.3, SPEC_T, False),
    ("A: any, field true", smry(True), 0.3, SPEC_ANY, True),
    ("A: any, field 'domain_separated'", smry("domain_separated"), 0.3, SPEC_ANY, True),
    ("A: any, field absent", smry(absent=True), 0.3, SPEC_ANY, False),
    ("A: any, field false", smry(False), 0.3, SPEC_ANY, False),
    ("A: any, field ''", smry(""), 0.3, SPEC_ANY, False),
    ("A: any, field 'shared'", smry("shared"), 0.3, SPEC_ANY, False),
    ("A: any, field 'none'", smry("none"), 0.3, SPEC_ANY, False),
    ("A: string literal, equal", smry("domain_separated"), 0.3, SPEC_S, True),
    ("A: string literal, different", smry("other"), 0.3, SPEC_S, False),
    ("A: NONE (a trainer that records no field), field absent", smry(absent=True), 0.3, "NONE", True),
    ("A: empty spec = skipped", smry(absent=True), 0.3, "", True),
    ("A: the placeholder spec", smry(1), 0.3, MC.PH_RECORD, False),
    ("REP (no REP literal): field absent", smry(absent=True), 0.0, SPEC_T, True),
    ("REP (no REP literal): field false", smry(False), 0.0, SPEC_T, True),
    ("REP (no REP literal): field null", smry(None), 0.0, SPEC_ANY, True),
    ("REP (no REP literal): field true (the replicate carries the separated stream)", smry(True), 0.0, SPEC_T, False),
    ("REP (no REP literal): field 'domain_separated'", smry("domain_separated"), 0.0, SPEC_ANY, False),
    ("REP: NONE", smry(1), 0.0, "NONE", True),
    ("REP: the placeholder spec", smry(absent=True), 0.0, MC.PH_RECORD, False),
    ("REP: a '<<' spec with no literal at all (only the placeholder guard refuses it)", smry(absent=True), 0.0, "levers.ego_perturb_stream<<", False),
]


def stream_suite(mod):
    return [n for n, sm, p, spec, ok in STREAM_ROWS if (mod.stream_problems(sm, p, spec) == []) != ok]


T(f"X2-1 stream_problems gives the registered answer on all {len(STREAM_ROWS)} rows (registered spec: A exactly the int 1, REP exactly the int 0, typed, absent fails on both)", stream_suite(C) == [], stream_suite(C))
STREAM_MUTS = [
    ("A without the field passes", cmut('return [f"the run record carries no {path}: the separated perturbation stream is not recorded on A"]', "return []")),
    ("REP without the field passes (with a REP literal registered)", cmut('return [f"the run record carries no {path}: REP must record exactly {want!r}"]', "return []")),
    ("the A literal is ignored", cmut("        if lit:\n            try:", "        if False:\n            try:")),
    ("the REP literal is ignored (REP falls back to the absent / no-stream rule)", cmut("    if rep_sep:\n        try:", "    if False:\n        try:")),
    ("A is compared against the REP literal", cmut("                want = json.loads(lit)\n            except ValueError:\n                return [f\"the stream-record literal", "                want = json.loads(lit_rep or lit)\n            except ValueError:\n                return [f\"the stream-record literal")),
    ("the A compare is loose (1 == True == 1.0)", cmut("            if json.dumps(cur, sort_keys=True) != json.dumps(want, sort_keys=True):\n                return [f\"the run record's {path} = {cur!r} on A,", "            if cur != want:\n                return [f\"the run record's {path} = {cur!r} on A,")),
    ("the REP compare is loose (0 == False == 0.0)", cmut("        if json.dumps(cur, sort_keys=True) != json.dumps(want, sort_keys=True):\n            return [f\"the run record's {path} = {cur!r} on REP,", "        if cur != want:\n            return [f\"the run record's {path} = {cur!r} on REP,")),
    ("REP carrying the stream passes (no REP literal)", cmut("    if found and not _no_stream(cur):", "    if False:")),
    ("the placeholder spec passes", cmut('    if "<<" in spec:', "    if False:")),
    ("NONE is not skipped", cmut('    if spec in ("", "NONE"):', '    if spec == "":')),
    ("an empty field value counts as a stream (A, no literal)", cmut("        if _no_stream(cur):\n            return [f\"the run record's {path} = {cur!r} on A: that is not a separated stream\"]", "        if False:\n            return []")),
    ("'shared' counts as a separated stream", cmut('NO_STREAM_STR = ("", "none", "off", "shared")', 'NO_STREAM_STR = ("", "none", "off")')),
]
for i, (name, src) in enumerate(STREAM_MUTS):
    M(f"MX2-{i + 1:02d} stream_problems {name}: the stream table catches it", len(stream_suite(load_mod(src, f"str{i}"))) >= 1)

# at the run-record level (the clause is CALLED from run_record, for summary.json AND selfcheck.json, with the argument the chain passes)


def run_dir(root, arm, seed, mut_summary=None, mut_argv=None, mut_sc=None, train_mut=None, stream="auto"):
    """A run directory built from DAG-OACC's REAL banked records.  stream='auto': the field the FIXED trainer records, in summary.json AND selfcheck.json (1 on A, 0 on REP); None: no field in either."""
    root.mkdir(parents=True, exist_ok=True)
    base = W.OACC / f"M2b_s{seed}_dagoacc"
    train = W.arm_argv(arm, seed)
    if train_mut:
        train = train_mut(list(train))
    sha16 = hashlib.sha256(" ".join(train).encode()).hexdigest()[:16]
    sm = json.load(open(base / "summary.json", encoding="utf-8"))
    sc = json.load(open(base / "selfcheck.json", encoding="utf-8"))
    p = G.ARMS[arm]
    tseed = 10 + seed if arm == "dagoaccorcrep" else seed
    sm.update(argv_sha=sha16, seed=tseed)
    sc.update(argv_sha=sha16)
    if p > 0.0:
        sm["levers"]["ego_perturb"] = {**sm["levers"]["ego_perturb"], "ego_perturb_p": p, **G.LAW_VALUES, "ego_perturb_seed": seed}
        sm["levers"]["ego_perturb_with_dagger"] = True
        sm.update(ego_perturb_n=24000, ego_perturb_eligible=80000, ego_dagger_overlap_n=0)
    if stream is not None:
        v = (1 if p > 0.0 else 0) if stream == "auto" else stream
        sm["levers"]["ego_perturb_stream"] = v
        sc["levers"]["ego_perturb_stream"] = v
    if mut_summary:
        mut_summary(sm)
    if mut_sc:
        mut_sc(sc)
    aj = {"argv": train, "sha": sha16}
    if mut_argv:
        aj = mut_argv(aj)
    for n, o in (("summary.json", sm), ("selfcheck.json", sc), ("argv.json", aj)):
        (root / n).write_text(json.dumps(o), encoding="utf-8", newline="\n")
    for n in ("train_line.txt", "pin_line.txt"):
        (root / n).write_text(W.line(train), encoding="utf-8", newline="\n")
    return root


class NS:
    def __init__(self, **k):
        self.__dict__.update(k)


def rr(root, arm, seed, mod=None, **kw):
    d = W.build(fresh("rr_w"), arm, seed)
    return quiet((mod or C).run_record, NS(run_dir=str(root), train_line=str(root / "train_line.txt"), pin_line=str(root / "pin_line.txt"), arm=arm, seed=str(seed), banked_init_sha=d["banked_init_sha"],
                                           bank_sha=d["bank_sha"], w_acc=W.W_ACC, steps="3000", expect="", **kw))


def put_both(v):
    """mutator for run_dir: the field changed in summary.json AND (via mut_sc) selfcheck.json; v == '<absent>' deletes it."""
    def f(o):
        if v == "<absent>":
            o["levers"].pop("ego_perturb_stream", None)
        else:
            o["levers"]["ego_perturb_stream"] = v
    return f


def rr_suite(mod):
    fails = []
    # name, arm, seed, run_dir kwargs, spec, expect no problems
    for nm, arm, s, kw, spec, ok in (
            ("A with the registered stream (1 / 1)", "dagoaccorc", 0, {}, SPEC_R, True),
            ("A, the field absent from BOTH files", "dagoaccorc", 1, dict(stream=None), SPEC_R, False),
            ("A, summary.json has it, selfcheck.json LACKS it", "dagoaccorc", 0, dict(mut_sc=put_both("<absent>")), SPEC_R, False),
            ("A, selfcheck.json has it, summary.json LACKS it", "dagoaccorc", 0, dict(mut_summary=put_both("<absent>")), SPEC_R, False),
            ("A, summary.json says 0 (the shared stream)", "dagoaccorc", 0, dict(mut_summary=put_both(0)), SPEC_R, False),
            ("A, selfcheck.json says 0 (the shared stream)", "dagoaccorc", 0, dict(mut_sc=put_both(0)), SPEC_R, False),
            ("A, summary.json says True", "dagoaccorc", 1, dict(mut_summary=put_both(True)), SPEC_R, False),
            ("REP with the registered stream (0 / 0)", "dagoaccorcrep", 0, {}, SPEC_R, True),
            ("REP, the field absent from BOTH files", "dagoaccorcrep", 1, dict(stream=None), SPEC_R, False),
            ("REP, summary.json says 1 (carries the separated stream)", "dagoaccorcrep", 1, dict(mut_summary=put_both(1)), SPEC_R, False),
            ("REP, selfcheck.json says 1 (carries the separated stream)", "dagoaccorcrep", 1, dict(mut_sc=put_both(1)), SPEC_R, False),
            ("A, NONE skips (no field in either file)", "dagoaccorc", 0, dict(stream=None), "NONE", True)):
        got = rr(run_dir(fresh("rrs"), arm, s, **kw), arm, s, mod=mod, stream_record=spec)
        if (got == []) != ok:
            fails.append(nm)
    return fails


T("X2-2 run_record with the registered spec, on DAG-OACC's real records + the stream field: A (1 / 1) and REP (0 / 0) pass; absent from either file, the shared stream (0) on A, the separated stream (1) on REP, True on A FAIL; NONE skips", rr_suite(C) == [], rr_suite(C))
T("X2-3 run_record WITHOUT the argument (the K section) is r1's: passes on the real records (the clause is skipped, nothing else changed)", rr(run_dir(fresh("rr0"), "dagoaccorc", 0, stream=None), "dagoaccorc", 0) == [])
CALL_SM = '    bad += stream_problems(sm, p, getattr(a, "stream_record", ""))      # r2: the separated perturbation stream (summary.json) ...\n'
CALL_SC = '    bad += ["selfcheck.json: " + x for x in stream_problems(sc, p, getattr(a, "stream_record", ""))]      # ... and the run\'s own selfcheck.json\n'
M("MX2-13 run_record never CALLS the stream clause on summary.json: the run-record table catches it", len(rr_suite(load_mod(cmut(CALL_SM, ""), "rrcall"))) >= 1)
M("MX2-14 run_record never CALLS the stream clause on selfcheck.json: the run-record table catches it (summary-only would miss a selfcheck that lacks the field)", len(rr_suite(load_mod(cmut(CALL_SC, ""), "rrcall2"))) >= 1)
M("MX2-15 run_record passes the arm's p of the OTHER arm to the clause (p fixed at 0.3): the REP rows catch it", len(rr_suite(load_mod(cmut(CALL_SM, CALL_SM.replace("stream_problems(sm, p,", "stream_problems(sm, 0.3,")), "rrp"))) >= 1)
mut = DEFAULT_CHAIN.replace(' --stream-record "$STREAM_RECORD"', "", 1)
M("MX2-16 the chain does not pass --stream-record to run-record: the text check refuses", mut != DEFAULT_CHAIN and ' --stream-record "$STREAM_RECORD"' not in mut and ' --stream-record "$STREAM_RECORD"' in DEFAULT_CHAIN)

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
T("X4-2 run_record is r1's plus exactly the two stream clauses (summary.json and the run's selfcheck.json)", rr_src2.replace(CALL_SM, "").replace(CALL_SC, "") == rr_src1)
mutm = CSRC.replace("REACH_TOL = 0.02", "REACH_TOL = 0.05")
M("MX4-1 a loosened K-REACH tolerance (0.05) in the r2 checks: the source-identity check refuses it", inspect.getsource(load_mod(mutm, "tol").reached) != inspect.getsource(C1.reached) or load_mod(mutm, "tol").REACH_TOL != C1.REACH_TOL)
import build_r2_asserter as BA                          # noqa: E402
asr_sha = hashlib.sha256((R2SHIP / "gate_dagoaccorc_r2_assert.py").read_bytes()).hexdigest()
r1_asr_txt = (R1SHIP / "gate_dagoaccorc_assert.py").read_text(encoding="utf-8")
r1_asr_sha = hashlib.sha256(r1_asr_txt.encode("utf-8")).hexdigest()
T("X4-3 the r2 asserter == build_r2_asserter.derive(r1's asserter) byte for byte (exact-count insertions only), r1's asserter is UNTOUCHED (the installed one has the same sha256) and differs from r2's",
  BA.derive(r1_asr_txt).encode("utf-8") == (R2SHIP / "gate_dagoaccorc_r2_assert.py").read_bytes() and r1_asr_sha == hashlib.sha256((INST / "gate_dagoaccorc_assert.py").read_bytes()).hexdigest() and r1_asr_sha != asr_sha)
a2 = (R2SHIP / "gate_dagoaccorc_r2_assert.py").read_text(encoding="utf-8")
r1l, r2l = r1_asr_txt.split("\n"), a2.split("\n")
removed = [x for x in r1l if x not in set(r2l)]
T("X4-3b r2's asserter keeps every line of r1's except the docstring's first line and the token dict line (the only two the derivation edits); the rest is added (the stream clause)",
  len(removed) == 2 and removed[0].startswith('"""PREREG_V8F_DAG_ORC on box 2') and '"ego_perturb_with_dagger": ARMS[a.arm] > 0.0' in removed[1] and len(r2l) > len(r1l), removed)

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
for var in ("OACC_CKPT_SHA", "OACC_ARGV_SHA", "OACC_SUMM_SHA", "M2B_SHA", "BANK1_SHA", "WACC", "WACC_TXT_SHA", "OAF_LIB_SHA", "OAF_CHECKS_SHA", "BUILD_REF", "RUNS", "DAG_STEPS", "GATE_POLLS", "FLOOR_BYTES", "NEED_BYTES", "ART", "OACC_DIR", "DAG1_DIR"):
    inv[var] = line_of(R1_CHAIN, var + "=") == line_of(DEFAULT_CHAIN, var + "=")
k1 = [x for x in R1_CHAIN.split("\n") if "reached --summary" in x]
inv["K-REACH call"] = k1 == [x for x in DEFAULT_CHAIN.split("\n") if "reached --summary" in x] and len(k1) == 1
lev1, lev2 = arr(R1_CHAIN, "LEVER") or "", arr(DEFAULT_CHAIN, "LEVER") or ""
inv["LEVER"] = lev1 == lev2 and lev1 != ""
inv["ASSERT_SHA is the r2 asserter's (and not r1's)"] = asr_sha in line_of(DEFAULT_CHAIN, "ASSERT_SHA=") and r1_asr_sha not in line_of(DEFAULT_CHAIN, "ASSERT_SHA=")
inv["ASSERT= names the r2 asserter"] = "ASSERT=$HERE_SHIP/gate_dagoaccorc_r2_assert.py" in DEFAULT_CHAIN and "ASSERT=$HERE_SHIP/gate_dagoaccorc_assert.py" not in DEFAULT_CHAIN
T("X4-4 the r2 chain carries r1's pins, RUNS, K-REACH call, LEVER greps, EXPORT, gate_one / runone (+ the one stream argument) / pinned_b / plan / ckptsha_x unchanged", all(inv.values()), [k for k, v in inv.items() if not v])
mutc = DEFAULT_CHAIN.replace("RUNS=(\"dagoaccorc 0\"", "RUNS=(\"dagoaccorcrep 9\" \"dagoaccorc 0\"", 1)
M("MX4-2 a chain whose RUNS differ from r1's: the invariant check refuses it", line_of(R1_CHAIN, "RUNS=") != line_of(mutc, "RUNS="))
mutc = DEFAULT_CHAIN.replace(asr_sha, r1_asr_sha, 1)
M("MX4-3 a chain whose ASSERT_SHA is r1's asserter's (the stream clause is not the pinned one): the invariant check refuses it", not (asr_sha in line_of(mutc, "ASSERT_SHA=") and r1_asr_sha not in line_of(mutc, "ASSERT_SHA=")))
rc, out = run_in(ship2(SCR / "ship_m", chain=mutc), "--dry-run", dict(V8F_QUEUE=qf, AFTER_OUT="x"))
M("MX4-4 a chain pinning r1's asserter sha with the r2 asserter file shipped: the chain STOPs (not the reviewed asserter)", rc == 1 and "ZZV8FMINI-STOP" in out and "is not the reviewed asserter" in out, out[-250:])
mutc = DEFAULT_CHAIN.replace("ASSERT=$HERE_SHIP/gate_dagoaccorc_r2_assert.py", "ASSERT=$HERE_SHIP/gate_dagoaccorc_assert.py", 1)
M("MX4-5 a chain whose ASSERT= names r1's asserter: the invariant check refuses it", not ("ASSERT=$HERE_SHIP/gate_dagoaccorc_r2_assert.py" in mutc and "ASSERT=$HERE_SHIP/gate_dagoaccorc_assert.py" not in mutc))
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

# ====================================================================== X6. the r2 asserter's BUILT-stream clause, the token, and the REAL records of the fixed trainer's composed run
print("== X6: ego_stream_problems (gate selfcheck levers.ego_perturb_stream EXACTLY the int 1 on A / the int 0 on REP), the token field, and the fixed trainer's REAL records")
ASRC = (R2SHIP / "gate_dagoaccorc_r2_assert.py").read_text(encoding="utf-8")
SL = ["levers", "ego_perturb_stream"]
XA_ROWS = [      # name, arm, seed, mutate(sc) or None, substrings one of which a problem must contain ([] = must PASS)
    ("A s0 clean (the record carries ego_perturb_stream 1)", "dagoaccorc", 0, None, []),
    ("A s1 clean", "dagoaccorc", 1, None, []),
    ("REP s0 clean (ego_perturb_stream 0)", "dagoaccorcrep", 0, None, []),
    ("REP s1 clean", "dagoaccorcrep", 1, None, []),
    ("A: the field is ABSENT (the r1 trainer's gate record)", "dagoaccorc", 0, sc_del(SL), ["built ego_perturb_stream"]),
    ("A s1: the field is ABSENT", "dagoaccorc", 1, sc_del(SL), ["built ego_perturb_stream"]),
    ("A: 0 (the shared, original stream)", "dagoaccorc", 0, sc_set(SL, 0), ["built ego_perturb_stream"]),
    ("A: True (== 1 in python)", "dagoaccorc", 0, sc_set(SL, True), ["built ego_perturb_stream"]),
    ("A: 1.0 (a float)", "dagoaccorc", 1, sc_set(SL, 1.0), ["built ego_perturb_stream"]),
    ("A: 2", "dagoaccorc", 0, sc_set(SL, 2), ["built ego_perturb_stream"]),
    ("A: null", "dagoaccorc", 0, sc_set(SL, None), ["built ego_perturb_stream"]),
    ("A: the string '1'", "dagoaccorc", 0, sc_set(SL, "1"), ["built ego_perturb_stream"]),
    ("REP: 1 (the replicate carries the separated stream)", "dagoaccorcrep", 0, sc_set(SL, 1), ["built ego_perturb_stream"]),
    ("REP s1: 1", "dagoaccorcrep", 1, sc_set(SL, 1), ["built ego_perturb_stream"]),
    ("REP: the field is ABSENT (a registered 0 is required: absent fails)", "dagoaccorcrep", 0, sc_del(SL), ["built ego_perturb_stream"]),
    ("REP: False (== 0 in python)", "dagoaccorcrep", 0, sc_set(SL, False), ["built ego_perturb_stream"]),
    ("REP: 0.0 (a float)", "dagoaccorcrep", 1, sc_set(SL, 0.0), ["built ego_perturb_stream"]),
    ("REP: null", "dagoaccorcrep", 0, sc_set(SL, None), ["built ego_perturb_stream"]),
]


def token_of(mod, arm, seed, tag):
    d = W.build(fresh(f"xtok_{tag}"), arm, seed)
    out = SCR / f"xtok_{tag}" / "token.json"
    args = ["token", "--selfcheck", d["selfcheck"], "--gate-line", d["gate_line"], "--train-line", d["train_line"], "--pin-line", d["pin_line"], "--arm", arm, "--seed", str(seed), "--banked-init-sha", d["banked_init_sha"],
            "--bank-sha", d["bank_sha"], "--bank-json", d["bank_json"], "--w-acc", W.W_ACC, "--b-summary", d["b_summary"], "--ref-argv-json", d["ref_argv_json"], "--commit", "a" * 40, "--queue-sha", "b" * 64,
            "--key-sha-json", '{"f": "c"}', "--run", f"M2b_s{seed}_{arm}", "--out", str(out)]
    rc = quiet(mod.main, args)
    return json.load(open(out, encoding="utf-8")) if rc == 0 and out.exists() else None


def asr_suite(mod, tokens=True):
    """The names of the rows an asserter module gets wrong: the 18 gate worlds (via world.run_check), and the token's ego_perturb_stream field (A 1, REP 0)."""
    old, fails = W.G, []
    W.G = mod
    try:
        for nm, arm, s, mut, subs in XA_ROWS:
            bad = quiet(W.run_check, W.build(fresh("xa"), arm, s, mutate=mut))
            ok = bad == [] if not subs else any(sub in x for x in bad for sub in subs)
            if not ok:
                fails.append(nm)
        if tokens:
            for arm, want in (("dagoaccorc", 1), ("dagoaccorcrep", 0)):
                tk = token_of(mod, arm, 0, arm)
                if not (tk is not None and type(tk.get("ego_perturb_stream")) is int and tk["ego_perturb_stream"] == want):
                    fails.append(f"token {arm}: ego_perturb_stream {None if tk is None else tk.get('ego_perturb_stream', '<absent>')!r}, want {want}")
    finally:
        W.G = old
    return fails


bad = asr_suite(G)
T(f"X6-1 the r2 asserter gives the registered answer on all {len(XA_ROWS)} gate worlds (the four clean worlds PASS; the field absent / 0 / True / 1.0 / 2 / null / '1' on A and 1 / absent / False / 0.0 / null on REP are REFUSED by the clause built for it) and the token records the field", bad == [], bad)
tk = token_of(G, "dagoaccorc", 0, "tka")
tr = token_of(G, "dagoaccorcrep", 1, "tkr")
T("X6-2 the launch-gate token records ego_perturb_stream (A exactly the int 1, REP exactly the int 0)", tk is not None and tr is not None and type(tk["ego_perturb_stream"]) is int and tk["ego_perturb_stream"] == 1 and type(tr["ego_perturb_stream"]) is int and tr["ego_perturb_stream"] == 0, (tk, tr))


def load_asr(src, tag):
    d = SCR / "asr_mods" / tag
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    (d / f"asr_{tag}.py").write_text(src, encoding="utf-8", newline="\n")
    spec = importlib.util.spec_from_file_location(f"asr_{tag}", d / f"asr_{tag}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def amut(old, new):
    assert ASRC.count(old) == 1, old
    return ASRC.replace(old, new)


ASR_MUTS = [
    ("never CALLS ego_stream_problems (the clause is defined but unreachable)", amut("    bad += ego_stream_problems(lv, p)      # r2: the separated perturbation stream was BUILT\n", "")),
    ("is called on A only", amut("    bad += ego_stream_problems(lv, p)      # r2: the separated perturbation stream was BUILT\n", "    if p > 0.0:\n        bad += ego_stream_problems(lv, p)\n")),
    ("drops the type check (True == 1, 1.0 == 1, False == 0, 0.0 == 0 pass)", amut("    if type(got) is not int or got != want:", "    if got != want:")),
    ("lets an ABSENT field pass (the default is the wanted value)", amut('    got = lv.get("ego_perturb_stream", "<absent>")', "    got = lv.get(\"ego_perturb_stream\", want)")),
    ("swaps the arms (A wants 0, REP wants 1)", amut('EGO_STREAM = {"dagoaccorc": 1, "dagoaccorcrep": 0}', 'EGO_STREAM = {"dagoaccorc": 0, "dagoaccorcrep": 1}')),
    ("wants the separated stream on REP too", amut('EGO_STREAM = {"dagoaccorc": 1, "dagoaccorcrep": 0}', 'EGO_STREAM = {"dagoaccorc": 1, "dagoaccorcrep": 1}')),
    ("picks the A literal for every arm", amut('    want = EGO_STREAM["dagoaccorc" if p > 0.0 else "dagoaccorcrep"]', '    want = EGO_STREAM["dagoaccorc"]')),
    ("the token does not record the field", amut(' "ego_perturb_stream": EGO_STREAM[a.arm],', "")),
]
for i, (name, src) in enumerate(ASR_MUTS):
    M(f"MX6-{i + 1:02d} the r2 asserter {name}: the gate-world / token table catches it", len(asr_suite(load_asr(src, f"m{i}"))) >= 1)

# ---- the REAL records of the fixed trainer's composed run (fixtures/compose_fix_real, copied from F:/TanitAD-scratch/compose_fix; PROVENANCE.txt)
FR = R2 / "fixtures" / "compose_fix_real"


def jl(n):
    return json.load(open(FR / n, encoding="utf-8"))


fA_sm, fA_sc, fR_sm, fR_sc, f1_sm, f1_sc = (jl(n) for n in ("A_summary.json", "A_selfcheck.json", "REPlike_summary.json", "REPlike_selfcheck.json", "r1_A_s0_summary.json", "r1_A_s0_selfcheck.json"))


def real_rows(mod_c, mod_g):
    """The names of the real-record rows a (checks, asserter) pair gets wrong."""
    bad = []
    for nm, sm, sc, p, ok in (("the fixed trainer's REAL composed A run (stream 1, p_ego 0.3)", fA_sm, fA_sc, 0.3, True), ("the fixed trainer's NON-composed run (stream 0) read as the REP arm", fR_sm, fR_sc, 0.0, True),
                              ("the fixed trainer's NON-composed run (stream 0) read as the A arm", fR_sm, fR_sc, 0.3, False), ("r1's REFUSED A s0 record (no field, ego share 0.2492) read as the A arm", f1_sm, f1_sc, 0.3, False),
                              ("the fixed trainer's REAL composed A run read as the REP arm", fA_sm, fA_sc, 0.0, False)):
        got = (mod_c.stream_problems(sm, p, SPEC_R) == [] and mod_c.stream_problems(sc, p, SPEC_R) == [] and mod_g.ego_stream_problems(sc["levers"], p) == [])
        if got != ok:
            bad.append(nm)
    return bad


bad = real_rows(C, G)
T("X6-3 the REAL records: the fixed trainer's composed A run (summary.json AND selfcheck.json: levers.ego_perturb_stream 1) PASSES the registered spec and the asserter; its non-composed run (0) passes as REP and is REFUSED as A; r1's refused A s0 record (no field) is REFUSED as A; the composed run is REFUSED as REP", bad == [], bad)
T("X6-3b the real A / REP-like records carry exactly the int 1 / 0 (so the registered values are what the fixed trainer writes, not what I assumed)", type(fA_sm["levers"]["ego_perturb_stream"]) is int and fA_sm["levers"]["ego_perturb_stream"] == 1 and fA_sc["levers"]["ego_perturb_stream"] == 1
  and type(fR_sm["levers"]["ego_perturb_stream"]) is int and fR_sm["levers"]["ego_perturb_stream"] == 0 and "ego_perturb_stream" not in f1_sm["levers"] and "ego_perturb_stream" not in f1_sc["levers"])
rA = quiet(C.reached, NS(summary=str(FR / "A_summary.json"), arm="dagoaccorc"))
r1 = quiet(C.reached, NS(summary=str(FR / "r1_A_s0_summary.json"), arm="dagoaccorc"))
T("X6-4 K-REACH (the r1 bands, unchanged) on the REAL composed A summary: PASS (the separated stream reaches ego 0.2931 of the non-swapped eligible, swap 0.0976, overlap 0); on r1's refused A s0 summary: FAIL (0.2492)", rA == 0 and r1 == 1, (rA, r1))


def rec_with(rec, v):
    """a deep copy of a real run record with levers.ego_perturb_stream set to v ('<absent>' deletes it)"""
    o = json.loads(json.dumps(rec))
    if v == "<absent>":
        o["levers"].pop("ego_perturb_stream", None)
    else:
        o["levers"]["ego_perturb_stream"] = v
    return o


for nm, v in (("stream 0 (the shared stream)", 0), ("field absent", "<absent>"), ("stream True", True), ("stream 1.0", 1.0), ("stream 2", 2)):
    M(f"MX6-R the real composed A summary with ONE thing wrong ({nm}): the registered spec refuses it", C.stream_problems(rec_with(fA_sm, v), 0.3, SPEC_R) != [])
    M(f"MX6-R the real A SELFCHECK with ONE thing wrong ({nm}): the asserter's clause refuses it", G.ego_stream_problems(rec_with(fA_sc, v)["levers"], 0.3) != [])
for nm, v in (("stream 1 (the separated stream on the replicate)", 1), ("field absent", "<absent>"), ("stream False", False), ("stream 0.0", 0.0)):
    M(f"MX6-R the real non-composed REP-like summary with ONE thing wrong ({nm}): the registered spec refuses it", C.stream_problems(rec_with(fR_sm, v), 0.0, SPEC_R) != [])
    M(f"MX6-R the real non-composed REP-like SELFCHECK with ONE thing wrong ({nm}): the asserter's clause refuses it", G.ego_stream_problems(rec_with(fR_sc, v)["levers"], 0.0) != [])


npass, nfail = sum(1 for _, ok in RESULTS if ok), sum(1 for _, ok in RESULTS if not ok)
nkill, nsurv = sum(1 for _, ok in MUTANTS if ok), sum(1 for _, ok in MUTANTS if not ok)
print("DEBUG-X26:", npass, "tests pass,", nfail, "fail;", nkill, "mutants killed,", nsurv, "survivors")
