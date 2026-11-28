#!/usr/bin/env python3
"""lit1_selftest_run.py -- the RUNNER of lit1_selftest.py (layers A python / B shell / C mutation).  Start it with `python lit1_selftest.py`."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import shutil
import sys
import threading
import time
import traceback
import types

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
import lit1_fix as X  # noqa: E402
import lit1_lib as L  # noqa: E402
import lit1_sealed as S  # noqa: E402
import lit1_tree as T  # noqa: E402
import lit1_selftest as ST  # noqa: E402

LOCK = threading.Lock()
FH = None
RES = {"pass": [], "fail": [], "killed": [], "survived": [], "bad": []}
UNMUTATED = [      # the honest list of what the mutation pass does NOT cover, and why
    ("lit1_chain.sh: the 'before' pin_check / ckpt re-hash of t1run and t0run", "defence in depth: in a closed world a file can only change DURING a step, which the 'after' checks (mutated and killed) see; the before-checks guard against an outside writer between steps"),
    ("lit1_chain.sh: the timeout stop of the wait loop (MAXWAIT_S)", "mutating it makes the scenario that times out loop forever; every scenario that ends in a timeout asserts its exact message instead"),
    ("lit1_chain.sh: the causal-REP template proof stop (proof 2)", "unreachable on its own: it can only fire when proof 1 (the same causal template, other parameters) has already passed"),
    ("lit1_lib.py: sum-mask as a GATE", "subsumed by sum-share (every n/e inside the 1/3 +- 0.02 share is inside f +- 0.02): tested with sum-share switched off in both runs, so the clause is live but never decides alone"),
    ("lit1_lib.py: ck-not-ctrl against a real control checkpoint", "tested by passing the control's sha256 to runrecord_problems; a pulled file cannot be made to hash to a control's"),
    ("lit1_tree.py: verify_manifest's 'checkpoint IS a control's' clause", "needs a file hashing to a control's sha256"),
    ("lit1_chain.sh: the '-s done_<tag>.txt' presence terms of rollout_done and of t0run's skip test", "equivalent mutants (measured: both SURVIVED): the five grep terms on the same file already fail when it is missing or empty"),
]


def emit(s):
    with LOCK:
        print(s, flush=True)
        if FH:
            print(s, file=FH, flush=True)


def safe(fn, *a):
    try:
        return fn(*a)
    except SystemExit as ex:
        return [f"SystemExit {ex}"]
    except Exception as ex:      # noqa: BLE001
        return [f"EXCEPTION {type(ex).__name__}: {ex} | {traceback.format_exc().strip().splitlines()[-3:]}"]


BASE = {}      # "A:<test>" / "B:<scenario>" -> did the UNMUTATED run pass?  A mutant is only meaningful if its unmutated scenario is green.
SC_BASE = {"s1_happy_nb": "s1_happy", "s3_tamper_r1": "s3_tamper", "s3_tamper_r2": "s3_tamper"}


def gate(mid, prefix, name):
    if prefix == "B":
        name = SC_BASE.get(name, "s14d_pins" if name.startswith("s14d_pin_") else name)
    if BASE.get(prefix + ":" + name) is False:
        RES["bad"].append(mid)
        emit(f"BASELINE-RED {mid}: the UNMUTATED {name} did not pass, so a failure under the mutant would prove nothing")
        return False
    return True


def report(name, fails, extra=""):
    BASE[name] = not fails
    if fails:
        RES["fail"].append(name)
        emit(f"FAIL   {name}: {len(fails)} failure(s)")
        for f in fails[:6]:
            emit(f"         - {f}"[:600])
    else:
        RES["pass"].append(name)
        emit(f"PASS   {name} {extra}".rstrip())


def apply_edit(path, old, new):
    raw = open(path, encoding="utf-8", newline="").read()
    n = raw.count(old)
    if n != 1:
        return f"the mutant's 'old' text occurs {n} times in {os.path.basename(path)} (must be exactly once)"
    open(path, "w", encoding="utf-8", newline="").write(raw.replace(old, new))
    return None


def mutant_result(mid, where, fails):
    if fails:
        RES["killed"].append(mid)
        emit(f"KILLED   {mid:<22} by {where}: {fails[0][:170]}")
    else:
        RES["survived"].append(mid)
        emit(f"SURVIVED {mid:<22} {where}: THE SCENARIO STILL PASSED WITH THE MUTANT APPLIED")


def run_py_mutants(tests, only=()):
    d0 = f"{L.SCRATCH}/mut_py"
    for mid, ids, tname in ST.clause_mutants():
        if only and mid not in only and tname not in only:
            continue
        if not gate(mid, "A", tname):
            continue
        with ST.dis(L, *ids):
            fails = safe(tests[tname], L, T)
        mutant_result(mid, tname, fails)
    for mid, f, old, new, tname in ST.TEXT_MUTANTS_PY:
        if only and mid not in only and tname not in only:
            continue
        if not gate(mid, "A", tname):
            continue
        d = f"{d0}/{mid}"
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        for g in ("lit1_lib.py", "lit1_tree.py"):
            shutil.copyfile(f"{HERE}/{g}", f"{d}/{g}")
        shutil.copytree(f"{HERE}/argv", f"{d}/argv")
        err = apply_edit(f"{d}/{f}", old, new)
        if err:
            RES["bad"].append(mid)
            emit(f"BAD-MUTANT {mid}: {err}")
            continue
        try:
            Lm, Tm = ST.load_pair(d)
            fails = safe(tests[tname], Lm, Tm)
        except Exception as ex:      # noqa: BLE001
            fails = [f"the mutated module does not even import: {ex}"]
        mutant_result(mid, tname, fails)


def run_sc(SH, name, mutant=None):
    wname = name if mutant is None else "m_" + mutant[0]
    w = X.World(wname)
    if mutant:
        err = apply_edit(f"{w.H}/{mutant[1]}", mutant[2], mutant[3])
        if err:
            return wname, None, [err]
    return wname, w, safe(SH.SC[name], w)


def main(argv=None):
    global FH
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--no-mutation", action="store_true")
    ap.add_argument("--layers", default="ABC")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=f"{HERE}/selftest/SELFTEST.out")
    ns = ap.parse_args(argv)
    os.makedirs(os.path.dirname(ns.out), exist_ok=True)
    FH = open(ns.out, "w", encoding="utf-8", newline="")
    os.makedirs(ST.PYW, exist_ok=True)
    S.ensure_global()
    t0 = time.time()
    emit(f"lit1_selftest {time.strftime('%Y-%m-%d %H:%M:%S')} python {sys.version.split()[0]} CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} workers {ns.workers}  (CPU only; no rollout, no ssh, no git write; every shell step sealed)")
    only = {x for x in ns.only.split(",") if x}
    import lit1_selftest_sh as SH
    tests = ST.py_tests()
    if "A" in ns.layers:
        emit("== A. PYTHON LAYER")
        for name, fn in tests.items():
            if only and name not in only:
                continue
            t1 = time.time()
            report("A:" + name, safe(fn, L, T), f"({time.time() - t1:.1f}s)")
    w1 = None
    if "B" in ns.layers:
        emit("== B. SHELL LAYER (lit1_chain.sh / wake_lit1.sh run to their end markers in sealed worlds)")
        names = [n for n in SH.SC if not n.startswith("s14d_pin_") and n not in ("s1_happy_nb", "s3_tamper_r1", "s3_tamper_r2", "s16_dry_run") and (not only or n in only)]
        with cf.ThreadPoolExecutor(max_workers=ns.workers) as ex:
            futs = {ex.submit(run_sc, SH, n): n for n in names}
            for fu in cf.as_completed(futs):
                n = futs[fu]
                wname, w, fails = fu.result()
                if n == "s1_happy":
                    w1 = w
                report("B:" + n, fails)
        if "s16_dry_run" in SH.SC and (not only or "s16_dry_run" in only):
            wname, w, fails = run_sc(SH, "s16_dry_run")
            report("B:s16_dry_run", fails)
            if w is not None and getattr(w, "dry", None):
                open(f"{HERE}/selftest/DRY_RUN_from_selftest.out", "w", encoding="utf-8", newline="").write(w.dry)
    if "C" in ns.layers and not ns.no_mutation:
        emit("== C. MUTATION PASS (every mutant is applied ALONE; the scenario that owns it must then FAIL)")
        emit("-- C1. python layer: clause switches and textual mutants of lit1_lib.py / lit1_tree.py")
        run_py_mutants(tests, only)
        emit("-- C2. shell layer: textual mutants of lit1_chain.sh / wake_lit1.sh (each runs its scenario in its own world)")
        jobs = [m for m in SH.MUT_SH if (not only or m[4] in only or m[0] in only) and gate(m[0], "B", m[4])]
        with cf.ThreadPoolExecutor(max_workers=ns.workers) as ex:
            futs = {ex.submit(run_sc, SH, m[4], m): m for m in jobs}
            for fu in cf.as_completed(futs):
                m = futs[fu]
                wname, w, fails = fu.result()
                if w is None:
                    RES["bad"].append(m[0])
                    emit(f"BAD-MUTANT {m[0]}: {fails[0]}")
                else:
                    mutant_result(m[0], m[4], fails)
        emit("-- C3. manifest verifier: textual mutants of lit1_tree.py verify_manifest against the finished happy-path world")
        if w1 is None:
            w1 = X.World("s1_happy_for_verify")
            safe(SH.SC["s1_happy_nb"], w1)
            w1.digest = T.tree_digest(f"{w1.root}/tree_stub")
        for mid, f, old, new, case in SH.MUT_VERIFY:
            if only and mid not in only and "s1_happy" not in only:
                continue
            if not gate(mid, "B", "s1_happy"):
                continue
            md = f"{w1.root}/vm_{mid}"
            shutil.rmtree(md, ignore_errors=True)
            os.makedirs(md)
            for g in ("lit1_lib.py", "lit1_tree.py"):
                shutil.copyfile(f"{HERE}/{g}", f"{md}/{g}")
            err = apply_edit(f"{md}/{f}", old, new)
            if err:
                RES["bad"].append(mid)
                emit(f"BAD-MUTANT {mid}: {err}")
                continue
            proxy = types.SimpleNamespace(H=md, out=w1.out, art=w1.art, root=w1.root, digest=w1.digest)
            fails = safe(SH.verify_battery, proxy, [case])
            mutant_result(mid, "verify:" + case, fails)
    # ---- closing checks and the summary
    gs = f"{S.GSTUB}/calls.log"
    calls = os.path.getsize(gs) if os.path.exists(gs) else 0
    home = os.listdir(S.NOHOME) if os.path.isdir(S.NOHOME) else ["?"]
    emit(f"== CLOSING CHECKS: remote-shell stub calls (bytes logged) {calls}; scratch HOME entries {len(home)}; sealed shell steps {S.STATS['sealed_runs']}, refused by the guard {S.STATS['refused']} (the harness arms)")
    if calls or home:
        RES["fail"].append("closing:remote-shell-stub-called-or-home-not-empty")
        emit("FAIL   closing check: a remote-shell stub was called or the scratch HOME is not empty")
    emit("== NOT MUTATED (documented):")
    for a, b in UNMUTATED:
        emit(f"   - {a}: {b}")
    ok = not RES["fail"] and not RES["survived"] and not RES["bad"]
    emit(f"== SUMMARY: {len(RES['pass'])} tests passed, {len(RES['fail'])} failed; mutants {len(RES['killed'])} killed, {len(RES['survived'])} survived, {len(RES['bad'])} invalid; {time.time() - t0:.0f} s")
    if RES["fail"]:
        emit("   FAILED: " + ", ".join(RES["fail"]))
    if RES["survived"]:
        emit("   SURVIVORS: " + ", ".join(RES["survived"]))
    if RES["bad"]:
        emit("   INVALID MUTANTS: " + ", ".join(RES["bad"]))
    emit("SELFTEST " + ("PASS" if ok else "FAIL"))
    FH.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
