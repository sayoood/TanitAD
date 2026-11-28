#!/usr/bin/env python3
"""Self-test of the DAG-ORC box-2 build (queue stages, gate asserter, chain checks, chain). CPU, no GPU, no box-2 contact, no git write. Every check is a MEASURED behaviour of the REAL files; every "deliberate regression"
is a doctored copy that the real check must REFUSE (a regression that is accepted is a MISSED line and turns the run RED).

Sections (select with --sections S,A,B,C,D,E):
  S  every pre-existing queue stage prints byte-identical output on the tip queue and the staged queue (a sweep of every dispatched stage x 4 argument shapes + the DAG-family arms x seeds), and the diff is INSERTIONS ONLY
  A  the three new arms render exactly DAG2-0's banked argv + the registered change (one lever), the refusals, the names, and no bank / DAGSHA is ever touched
  B  the queue run NOT dry on a fake box (stub trainer / eval): pin, pin mismatch, the pinned line is what runs, isolation (no bank directory, no queue tool call)
  C  the gate asserter (real gate record of DAG2-0, ORC's real perturbation probe, the real tip trainer's synthetic gate) + the run-record / reached / one-lever checks, each with doctored negatives
  D  the chain offline: regeneration, pins vs the banked artifacts, dry-run, lever-in-tip greps, predecessor rule, early stops with ssh/scp shims that prove no box-2 call was made
  E  the chain's real run against a fake box 2 (ssh/scp shims over a local tree, a fake git mirror): gates -> trainings -> pulls -> records; the REACHED control and the run record are ENFORCED

    CUDA_VISIBLE_DEVICES=-1 python selftest_dagorc.py [--sections S,A,B,C,D,E]
Environment: DAGORC_QUEUE (queue under test), DAGORC_SHIP (dir with the chain files under test; default box2_ship), DAGORC_SCRATCH (default F:/TanitAD-scratch/dagorc_work/st).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dagorc_lib as L  # noqa: E402

SHIP = Path(os.environ.get("DAGORC_SHIP") or HERE / "box2_ship")
sys.path.insert(0, str(SHIP))
import importlib.util  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


G = _load("gate_dagorc_assert", SHIP / "gate_dagorc_assert.py")
K = _load("v8f_dagorc_chain_checks", SHIP / "v8f_dagorc_chain_checks.py")
SCR = Path(os.environ.get("DAGORC_SCRATCH") or r"F:\TanitAD-scratch\dagorc_work\st")
SCR.mkdir(parents=True, exist_ok=True)
BASH = r"C:\Program Files\Git\bin\bash.exe"
INSTALLED = Path(r"C:\Users\Admin\box2_ship")
TREE = Path(os.environ.get("DAGORC_TREE") or r"F:\TanitAD-scratch\dagorc_tree")
FIX = HERE / "fixtures"
CKPT_SHA = ["d186fd0bcfa7ee9fbfe5c9de543fe85edda7c3e6b9c56c898958f027ae2b8cb1", "1d12ec2eb825990d57dcc45dda50b5178d1d733961b47ab0e3d50e6155fa39ef"]
ARMS = ("dagorc", "dagorcrep", "dagorc01")
RES = {"checks": [], "regressions": []}


def ok(name, cond, detail=""):
    RES["checks"].append({"name": name, "ok": bool(cond)})
    print(("PASS " if cond else "FAIL ") + name + (f"  [{str(detail)[:260]}]" if (detail and not cond) else ""), flush=True)
    return bool(cond)


def refused(name, refusal, detail=""):
    """a deliberate regression: ``refusal`` is truthy iff the real check REFUSED the doctored input"""
    RES["regressions"].append({"name": name, "caught": bool(refusal)})
    print(("CAUGHT " if refusal else "MISSED ") + name + (f"  [{str(detail)[:260]}]" if (detail and not refusal) else ""), flush=True)
    return bool(refusal)


def sh(cmd, *, env=None, cwd=None, timeout=600, inp=None):
    e = {**os.environ, **(env or {})}
    p = subprocess.run(cmd, capture_output=True, env=e, cwd=cwd, timeout=timeout, input=inp)
    return p.returncode, (p.stdout or b"").decode("utf-8", "replace").replace("\r", ""), (p.stderr or b"").decode("utf-8", "replace").replace("\r", "")


def bash(script, *, env=None, cwd=None, timeout=600):
    return sh([BASH, "-c", script], env=env, cwd=cwd, timeout=timeout)


def sha16(text_tokens) -> str:
    return hashlib.sha256(" ".join(text_tokens).encode()).hexdigest()[:16]


def lines_of(arm, s):
    """(gate line, train line, eval line) as the REAL queue under test prints them."""
    g = L.trainer_line(L.dry(f"gatedag {arm} {s}"))
    t = L.trainer_line(L.dry(f"traindag {arm} {s}"))
    e = [x for x in L.dry(f"evaldag {arm} {s}") if "v8f_mini_eval.py" in x]
    assert len(e) == 1
    return g, t, e[0]


# ============================================================================================================================ S
def section_S():
    print("== S: queue byte-identity and insertions-only")
    import build_queue as BQ
    tip_b = L.TIP_QUEUE.read_bytes()
    new_b = L.QUEUE.read_bytes()
    ok("S0 tip queue is the pinned 3b7b404e file (sha256 6d26a613...)", hashlib.sha256(tip_b).hexdigest() == BQ.TIP_SHA256)
    tl = tip_b.decode("utf-8")[:-2].split("\r\n")
    nl = new_b.decode("utf-8")[:-2].split("\r\n")
    ok("S1 the staged queue is CRLF throughout and ends with CRLF", new_b.count(b"\r\n") == new_b.count(b"\n") and new_b.endswith(b"\r\n"))
    # tip lines are a subsequence of the staged lines, in order => nothing removed, nothing changed, nothing reordered
    i = 0
    for x in nl:
        if i < len(tl) and x == tl[i]:
            i += 1
    ok("S2 every tip line occurs in the staged queue in the same order (0 lines removed or changed)", i == len(tl), f"matched {i} of {len(tl)}")
    ok("S3 the staged queue adds exactly 20 lines (1 doc, 3 :dagname, 3 :dagarm, 13 :dagorcarm)", len(nl) - len(tl) == 20, len(nl) - len(tl))
    added, k = [], 0
    for x in nl:
        if k < len(tl) and x == tl[k]:
            k += 1
        else:
            added.append(x)
    cons = BQ.DOC + BQ.DAGNAME_ARMS + BQ.DAGARM_HAND + BQ.STAGES.strip(chr(10)).split(chr(10))
    ok("S4 the added lines (those the in-order match of the tip's lines does not consume) are exactly the builder's constants", len(added) == 20 and sorted(added) == sorted(cons), (len(added), sorted(set(added) ^ set(cons))[:4]))
    ok("S5 the staged queue equals a fresh run of build_queue.build() (byte for byte)" if os.environ.get("DAGORC_QUEUE") is None else "S5 skipped (mutated queue)", os.environ.get("DAGORC_QUEUE") is not None or BQ.build() == new_b)
    # the sweep: every dispatched stage x 4 argument shapes, plus the DAG-family arms x seeds
    names = re.findall(r'(?m)^if /i "%~1"=="([A-Za-z0-9_]+)" goto :', tip_b.decode("utf-8"))
    shapes = ["", "M2b 0", "dag 0 0.1", "flatdag0 1"]
    jobs = [f"{n} {s}".strip() for n in names for s in shapes]
    for st in ("gatedag", "traindag", "evaldag"):
        for arm, p in (("dag", "0.1"), ("dag0", ""), ("dag2", "0.1"), ("dag20", ""), ("dagoacc", "0.1"), ("flatdag", "0.1"), ("flatdag0", "")):
            for sd in (0, 1):
                jobs.append(f"{st} {arm} {sd} {p}".strip())
    for extra_job in ("bankdag 0", "bankdag2 1", "bankflat 0", "waccdag", "gatedag dag 0", "gatedag dag 0 0.3", "gatedag dag2 0 0.3", "gatedag dagoacc 0 0.2", "traindag dag20 0 0.1", "evaldag nosuch 0"):
        jobs.append(extra_job)
    jobs = sorted(set(jobs))

    def run(j):
        return j, L.cmd(j, dryrun=True, queue=L.TIP_QUEUE), L.cmd(j, dryrun=True, queue=L.QUEUE)

    with ThreadPoolExecutor(4) as ex:
        results = list(ex.map(run, jobs))
    diff = [j for j, a, b in results if a != b]
    ok(f"S6 {len(jobs)} stage invocations ({len(names)} dispatched stage names x 4 shapes + the DAG-family arms) print byte-identical output on the tip queue and the staged queue", not diff, diff[:6])
    # a REAL (non-dry) refusal path of an existing arm is identical too: the refusals are cmd-level, run without V8F_DRYRUN
    for j in ("gatedag dag 0", "traindag dag2 0 0.1", "evaldag dagoacc 1"):
        a, b = L.cmd(j, dryrun=False, queue=L.TIP_QUEUE), L.cmd(j, dryrun=False, queue=L.QUEUE)
        ok(f"S7 the real (non-dry) run of `{j}` behaves identically (rc, output)", a == b, (a, b))


# ============================================================================================================================ A
EXPECT_P = {"dagorc": "0.3", "dagorc01": "0.1", "dagorcrep": "0"}


def section_A():
    print("== A: the three arms render exactly DAG2-0's argv + the registered change")
    law = {"--ego-perturb-vscale": ["0.5", "1.0"], "--ego-perturb-vmax": ["8.0"], "--ego-perturb-adrop": ["1.5"], "--ego-perturb-yawrate-std": ["2.0"]}
    for s in (0, 1):
        ref = L.banked_argv("dag20", s)
        for arm in ARMS:
            g, t, e = lines_of(arm, s)
            ta, ga = L.train_argv(t), L.train_argv(g)
            want = G.expected_from_ref(ref, arm, s)
            ok(f"A1 {arm} s{s}: the TRAIN line == DAG2-0's banked argv.json + the registered change, only --out differs (K-PAIR)", L.drop(ta) == want)
            ok(f"A2 {arm} s{s}: the GATE line == the train line with --out and --selfcheck-only changed", L.drop([x for x in ga if x != "--selfcheck-only"]) == L.drop(ta) and ga.count("--selfcheck-only") == 1 and "--selfcheck-only" not in ta)
            fl = L.flags(ta)
            ok(f"A3 {arm} s{s}: --ego-perturb-p NAMED once with {EXPECT_P[arm]}; --dagger-p NAMED once with 0; --init-ckpt is DAG_s{s}'s (not M2b's)",
               fl.get("--ego-perturb-p") == [[EXPECT_P[arm]]] and fl.get("--dagger-p") == [["0"]] and fl.get("--init-ckpt") == [[rf"D:\tanitad\runs\v8f_mini_full\M2b_s{s}_dag\ckpt_last.pt"]])
            if arm == "dagorcrep":
                ok(f"A4 {arm} s{s}: no law flag, no DAgger bank flag; --seed {10 + s}", not any(k in fl for k in G.EGO_LAW + G.DAG_ONLY) and fl.get("--seed") == [[str(10 + s)]])
            else:
                ok(f"A4 {arm} s{s}: the five law flags once each at their registered values (seed = {s}); no DAgger bank flag; --seed {s}",
                   all(fl.get(k) == [v] for k, v in law.items()) and fl.get("--ego-perturb-seed") == [[str(s)]] and not any(k in fl for k in G.DAG_ONLY) and fl.get("--seed") == [[str(s)]])
            ok(f"A5 {arm} s{s}: run / gate / tag / pin names are indexed by the init seed",
               fl.get("--out") == [[rf"D:\tanitad\runs\v8f_mini_full\M2b_s{s}_{arm}"]] and L.flags(ga).get("--out") == [[rf"D:\tanitad\runs\v8f_mini_full\gate_M2b-{arm}_s{s}"]]
               and f"--tag M2b-{arm}_s{s} " in e and rf"M2b_s{s}_{arm}\ckpt_last.pt" in e and "-tacorc" not in e)
    # B == A with p 0.1 ; REP == DAG2-0 with the seed
    for s in (0, 1):
        ta, tb = L.train_argv(lines_of("dagorc", s)[1]), L.train_argv(lines_of("dagorc01", s)[1])
        d = [(x, y) for x, y in zip(L.drop(ta), L.drop(tb)) if x != y]
        ok(f"A6 dagorc01 s{s} differs from dagorc s{s} in --ego-perturb-p only (0.3 -> 0.1) and --out", d == [("0.3", "0.1")] and len(L.drop(ta)) == len(L.drop(tb)), d)
        tr = L.train_argv(lines_of("dagorcrep", s)[1])
        d = [(x, y) for x, y in zip(L.drop(tr), L.drop(L.banked_argv("dag20", s))) if x != y]
        ok(f"A7 dagorcrep s{s} differs from DAG2-0's banked argv in --seed only ({10 + s} vs {s}) and --out", d == [(str(10 + s), str(s))], d)
    # refusals (rc 2, a REFUSED line), in the real (non-dry) mode too: they never reach a file
    for st in ("gatedag", "traindag"):
        for arm in ARMS:
            for extra in ("0.1", "0.3", "x", "0"):
                rc, out = L.cmd(f"{st} {arm} 0 {extra}")
                refused(f"A8 {st} {arm} 0 {extra}: any extra argument (a P) is REFUSED", rc == 2 and any("REFUSED" in x and "takes no P" in x for x in out), (rc, out))
        for bad_seed in ("2", "-1", "10", "a"):
            rc, out = L.cmd(f"{st} dagorc {bad_seed}")
            refused(f"A9 {st} dagorc {bad_seed}: a seed other than 0 / 1 is REFUSED", rc == 2 and any("SEED must be 0 or 1" in x for x in out), (rc, out))
        rc, out = L.cmd(f"{st} dagorc")
        refused(f"A10 {st} dagorc (no seed) is REFUSED", rc == 2 and any("a SEED is needed" in x for x in out), (rc, out))
        for bad_arm in ("dagorcx", "dagorc2", "dagorcrep0", "orc"):
            rc, out = L.cmd(f"{st} {bad_arm} 0")
            refused(f"A11 {st} {bad_arm} 0: an unknown arm is REFUSED", rc == 2 and any("ARM must be" in x for x in out), (rc, out))
    # an inherited DAGSHA_DRY / WACCDAG_DRY / DAGF0 from the environment cannot change what the arms print (the arms are bank-free)
    base = {arm: lines_of(arm, 0) for arm in ARMS}
    for arm in ARMS:
        g = L.trainer_line(L.dry(f"gatedag {arm} 0", env={"DAGSHA_DRY": "ab" * 32, "WACCDAG_DRY": "9.99", "DAGF0": "--evil", "DAGF": "--evil2", "DAGSHA": "cd" * 32, "DAGP": "0.9", "DAGORCP": "0.9", "DAGORCL": " --evil3"}))
        ok(f"A12 {arm}: an inherited DAGSHA_DRY / WACCDAG_DRY / DAGF / DAGF0 / DAGSHA / DAGP / DAGORCP / DAGORCL in the environment changes nothing", g == base[arm][0], g[-200:])
    # existing arms still print what they printed (spot check; S6 is the sweep)
    for arm, p in (("dag2", "0.1"), ("dag20", ""), ("dagoacc", "0.1")):
        ok(f"A13 the existing arm {arm} still prints its DAGGER-family lines (no dagorc flag leaks in)", "--ego-perturb-p 0 " in L.trainer_line(L.dry(f"traindag {arm} 0 {p}".strip(), env={"DAGSHA_DRY": "ab" * 32, "WACCDAG_DRY": "49.6969"})) + " ")
    # the usage doc line is in the header block next to the other later usage lines
    q = L.QUEUE.read_bytes().decode("utf-8").split("\r\n")
    k = [i for i, x in enumerate(q) if x.startswith("rem   gatedag / traindag / evaldag also take ARM dagorc ")]
    ok("A14 exactly one usage line documents the three arms, inside the header block (before the first label)", len(k) == 1 and k[0] < next(i for i, x in enumerate(q) if x.startswith(":")), k)
    ok("A15 :dagorcarm is the last label of the file and is reached only from :dagarm (no dispatch line names it)", [x for x in q if re.match(r"^:[A-Za-z0-9_]+$", x)][-1] == ":dagorcarm"
       and sum(1 for x in q if "goto :dagorcarm" in x) == 3 and not any(x.startswith('if /i "%~1"==') and "goto :" in x and "dagorc" in x for x in q))


# ============================================================================================================================ B
def fake_box(root: Path) -> dict:
    """A fake box 2 on the local disk: the repo path holds a STUB trainer / eval that record their argv, the cache / prior files :prep wants, and a stub queue tool that logs any call (it must never be called)."""
    if root.exists():
        shutil.rmtree(root)
    repo, runsm, cache, v9 = root / "repo", root / "runs", root / "cache_full", root / "v9"
    (repo / "stack" / "scripts").mkdir(parents=True)
    hj = repo / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-refav1-hj-mini" / "raw" / "p10_yaw4512"
    hj.mkdir(parents=True)
    (hj / "p10_prior.pt").write_bytes(b"prior")
    (hj / "p10_eval_pred6.npy").write_bytes(b"pred")
    pk = repo / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini" / "raw"
    pk.mkdir(parents=True)
    (pk / "eval_grid_windows.json").write_text("{}")
    for d in ("train", "eval"):
        (cache / d).mkdir(parents=True)
        (cache / d / "manifest.jsonl").write_text("{}\n")
    (cache / "native_tracks.pt").write_bytes(b"t")
    runsm.mkdir(parents=True)
    stub = (
        "import json, os, sys\n"
        "a = sys.argv[1:]\n"
        "out = a[a.index('--out') + 1]\n"
        "os.makedirs(out, exist_ok=True)\n"
        "open(os.path.join(out, 'stub_argv.json'), 'w').write(json.dumps(a))\n"
        "if '--selfcheck-only' not in a:\n"
        "    open(os.path.join(out, 'summary.json'), 'w').write('{}')\n"
        "    open(os.path.join(out, 'ckpt_last.pt'), 'w').write('x')\n"
    )
    (repo / "stack" / "scripts" / "v8f_mini_train.py").write_text(stub)
    (repo / "stack" / "scripts" / "v8f_mini_eval.py").write_text(
        "import json, os, sys\na = sys.argv[1:]\nout = a[a.index('--out-dir') + 1]\nos.makedirs(out, exist_ok=True)\ntag = a[a.index('--tag') + 1]\nopen(os.path.join(out, 'eval_' + tag + '.npz'), 'w').write('x')\n")
    (repo / "stack" / "scripts" / "v8f_dagger_queue_tool.py").write_text("import sys\nopen(%r, 'a').write(' '.join(sys.argv[1:]) + '\\n')\nprint('0' * 64)\n" % str(root / "tool_calls.log"))
    return {"REPO": str(repo), "PY": sys.executable, "CACHEFULL": str(cache), "V9": str(v9), "RUNSM": str(runsm)}


def section_B():
    print("== B: the queue run NOT dry on a fake box")
    env = fake_box(SCR / "fakebox_B")
    runs = Path(env["RUNSM"])
    root = Path(env["REPO"]).parent
    rc, out = L.cmd("gatedag dagorc 0", env=env, dryrun=False)
    ok("B1 gatedag dagorc 0 (real run): rc 0, '[ok] gate M2b-dagorc_s0'", rc == 0 and any("[ok] gate M2b-dagorc_s0" in x for x in out), (rc, out))
    pin = runs / "dagpin" / "M2b_s0_dagorc.train.argv.txt"
    ok("B2 the training argv is PINNED to dagpin\\M2b_s0_dagorc.train.argv.txt, with the lever and without --selfcheck-only", pin.exists() and "--ego-perturb-p 0.3" in pin.read_text() and "--selfcheck-only" not in pin.read_text() and "--ego-perturb-seed 0" in pin.read_text())
    stub = json.loads((runs / "gate_M2b-dagorc_s0" / "stub_argv.json").read_text())
    ok("B3 the gate ran the trainer with --selfcheck-only and the train argv (stub saw it)", "--selfcheck-only" in stub and stub[stub.index("--ego-perturb-p") + 1] == "0.3" and stub[stub.index("--dagger-p") + 1] == "0" and stub[stub.index("--seed") + 1] == "0")
    rc, out = L.cmd("gatedag dagorc 0", env=env, dryrun=False)
    ok("B4 a second gate with an unchanged argv is accepted (idempotent pin)", rc == 0, out)
    # tamper with the pin: refused
    good = pin.read_bytes()
    pin.write_bytes(good.replace(b"--ego-perturb-p 0.3", b"--ego-perturb-p 0.1"))
    rc, out = L.cmd("gatedag dagorc 0", env=env, dryrun=False)
    refused("B5 a re-gate against a pin that names another argv is REFUSED", rc == 2 and any("pins another training argv" in x for x in out), (rc, out))
    rc, out = L.cmd("traindag dagorc 0", env=env, dryrun=False)
    refused("B6 traindag against a tampered pin is REFUSED (the line derived now differs from the one the gate tokenised)", rc == 2 and any("differs from the one the gate tokenised" in x for x in out), (rc, out))
    pin.write_bytes(good)
    rc, out = L.cmd("traindag dagorc 1", env=env, dryrun=False)
    refused("B7 traindag before its gate is REFUSED (no pinned training argv)", rc == 2 and any("no pinned training argv" in x for x in out), (rc, out))
    rc, out = L.cmd("evaldag dagorc 0", env=env, dryrun=False)
    refused("B8 evaldag before the training finished is REFUSED", rc == 2 and any("has not finished training" in x for x in out), (rc, out))
    rc, out = L.cmd("traindag dagorc 0", env=env, dryrun=False)
    ok("B9 traindag dagorc 0 runs the PINNED line: rc 0 and the stub's argv is the pin's", rc == 0 and (runs / "M2b_s0_dagorc" / "summary.json").exists()
       and json.loads((runs / "M2b_s0_dagorc" / "stub_argv.json").read_text())[json.loads((runs / "M2b_s0_dagorc" / "stub_argv.json").read_text()).index("--ego-perturb-p") + 1] == "0.3", (rc, out))
    rc, out = L.cmd("traindag dagorc 0", env=env, dryrun=False)
    ok("B10 a finished run is skipped, not re-trained", rc == 0 and any("[skip] M2b_s0_dagorc" in x for x in out), out)
    rc, out = L.cmd("evaldag dagorc 0", env=env, dryrun=False)
    ok("B11 evaldag after training: rc 0, '[ok] eval M2b-dagorc_s0', the npz exists", rc == 0 and (runs / "eval" / "eval_M2b-dagorc_s0.npz").exists(), (rc, out))
    # REP: init DAG_s, trainer seed 10, indexed by the init seed
    rc, out = L.cmd("gatedag dagorcrep 1", env=env, dryrun=False)
    stub = json.loads((runs / "gate_M2b-dagorcrep_s1" / "stub_argv.json").read_text())
    ok("B12 gatedag dagorcrep 1 (real run): rc 0, the trainer gets --seed 11 and --ego-perturb-p 0 and no law flag", rc == 0 and stub[stub.index("--seed") + 1] == "11" and stub[stub.index("--ego-perturb-p") + 1] == "0" and "--ego-perturb-seed" not in stub, (rc, out))
    rc, out = L.cmd("traindag dagorcrep 1", env=env, dryrun=False)
    ok("B13 traindag dagorcrep 1 (real run): lands in M2b_s1_dagorcrep (not M2b_s11_...), pin M2b_s1_dagorcrep.train.argv.txt", rc == 0 and (runs / "M2b_s1_dagorcrep" / "summary.json").exists() and not (runs / "M2b_s11_dagorcrep").exists()
       and (runs / "dagpin" / "M2b_s1_dagorcrep.train.argv.txt").exists(), (rc, out))
    # isolation: no bank directory, no bank_sha file, the queue tool was never called, nothing of the existing arms was created
    created = sorted(p.name for p in runs.iterdir())
    ok("B14 isolation: no dagger_bank* directory, no bank_sha / w_acc file, no iteration-1 run directory was created", not any(n.startswith("dagger_bank") for n in created) and not list((runs / "dagpin").glob("bank_sha*"))
       and not any(n in created for n in ("M2b_s0_dag", "M2b_s0_dag20", "dagoacc")), created)
    ok("B15 the DAgger queue tool was never called by a DAG-ORC stage", not (root / "tool_calls.log").exists())
    # the existing DAG arm still wants its bank on the same box (the dagorc edit did not make it bank-free)
    rc, out = L.cmd("gatedag dag 0 0.1", env=env, dryrun=False)
    refused("B16 control: the existing `gatedag dag 0 0.1` on the same box still REFUSES without its bank", rc == 2 and any("no merged bank" in x for x in out), (rc, out))


# ============================================================================================================================ C
PK_REL = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
import contextlib  # noqa: E402
import io  # noqa: E402


class Case:
    """One gate input set for the asserter: the selfcheck (a dict), the three argv lines (token lists), the references. ``write`` puts them on disk; ``check`` runs the REAL asserter's main() and returns (rc, output)."""

    def __init__(self, arm, s, root):
        self.arm, self.s, self.root = arm, s, Path(root)
        g, t, _ = lines_of(arm, s)
        self.gl, self.tl = g, t
        self.gtok, self.ttok = L.train_argv(g), L.train_argv(t)
        self.ptok = list(self.ttok)
        self.sc = self.real_sc()
        self.fix_sha()
        self.init_sha = CKPT_SHA[s]
        self.bsum = L.DAG2_DIR / f"M2b_s{s}_dag20" / "summary.json"
        self.ref = L.DAG2_DIR / f"M2b_s{s}_dag20" / "argv.json"
        self.expect = None
        self.extra = []
        self.use_ref = True
        self.use_bsum = True

    def real_sc(self):
        sc = json.loads((L.DAG2_DIR / f"M2b_s{self.s}_dag20" / "selfcheck.json").read_text(encoding="utf-8"))
        if self.arm != "dagorcrep":
            orc = json.loads((TREE / PK_REL / "raw" / "2026-10-26-orecover-run" / "runs" / f"M2b_s{self.s}_orc" / "selfcheck.json").read_text(encoding="utf-8"))
            sc["ego_perturb_probe"] = copy.deepcopy(orc["ego_perturb_probe"])
            sc["levers"]["ego_perturb"] = copy.deepcopy(orc["levers"]["ego_perturb"])
            if self.arm == "dagorc01":      # B: the same record at the other dose (a doctored positive: no real B gate exists yet)
                sc["ego_perturb_probe"]["p_declared"] = 0.1
                sc["levers"]["ego_perturb"]["ego_perturb_p"] = 0.1
        return sc

    def fix_sha(self):
        self.sc["argv_sha"] = sha16(self.gtok)

    def write(self, d):
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        q = lambda toks: '"python" "v8f_mini_train.py" ' + " ".join(f'"{x}"' if (" " in x or "\\" in x) else x for x in toks)      # noqa: E731
        (d / "sc.json").write_text(json.dumps(self.sc), encoding="utf-8")
        (d / "g.txt").write_text(q(self.gtok), encoding="utf-8")
        (d / "t.txt").write_text(q(self.ttok), encoding="utf-8")
        (d / "p.txt").write_text(q(self.ptok), encoding="utf-8")
        return d

    def args(self, d, mode="check"):
        a = [mode, "--selfcheck", str(d / "sc.json"), "--gate-line", str(d / "g.txt"), "--train-line", str(d / "t.txt"), "--pin-line", str(d / "p.txt"), "--arm", self.arm, "--seed", str(self.s), "--banked-init-sha", self.init_sha]
        if self.use_bsum:
            a += ["--b-summary", str(self.bsum)]
        if self.use_ref:
            a += ["--ref-argv-json", str(self.ref)]
        if self.expect:
            a += ["--expect", str(self.expect)]
        return a + self.extra

    def check(self, tag="c"):
        d = self.write(SCR / "C" / f"{tag}_{self.arm}_{self.s}")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                rc = G.main(self.args(d))
            except SystemExit as e:
                rc = e.code
        return rc, buf.getvalue()


def edit_lines(case, fn, which=("g", "t", "p")):
    """apply fn(tokens) -> tokens to the chosen lines and refresh the gate record's argv_sha (so only the targeted assertion fires)"""
    if "g" in which:
        case.gtok = fn(case.gtok)
    if "t" in which:
        case.ttok = fn(case.ttok)
    if "p" in which:
        case.ptok = fn(case.ptok)
    case.fix_sha()


def setv(toks, flag, val):
    t = list(toks)
    t[t.index(flag) + 1] = val
    return t


def delflag(toks, flag, n=1):
    t = list(toks)
    k = t.index(flag)
    del t[k:k + 1 + n]
    return t


def swap_law(toks):
    """--ego-perturb-vmax 8.0 and --ego-perturb-adrop 1.5 exchange places: the same flags and values, another order"""
    t = list(toks)
    a, b = t.index("--ego-perturb-vmax"), t.index("--ego-perturb-adrop")
    t[a:a + 2], t[b:b + 2] = t[b:b + 2], t[a:a + 2]
    return t


def section_C():
    print("== C: the gate asserter, the run record, K-REACH and the one-lever check")
    # ---- C1: positives on the REAL gate record of DAG2-0 (REP) and its splice with ORC's real ego probe (A, B)
    for arm in ARMS:
        for s in (0, 1):
            c = Case(arm, s, SCR / "C")
            rc, out = c.check("pos")
            what = "DAG2-0's REAL gate record (seed relabelled)" if arm == "dagorcrep" else ("DAG2-0's real gate + ORC's real ego probe/levers (spliced" + (", p 0.1 doctored)" if arm == "dagorc01" else ")"))
            ok(f"C1 {arm} s{s}: the asserter ACCEPTS {what} with the queue's lines", rc == 0, out[-300:])
    # ---- C2: positives on what the REAL tip trainer wrote (synthetic world), fixtures banked by reach_synthetic.py
    fixtures = sorted(FIX.glob("*_gate")) if FIX.exists() else []
    ok("C2 the fixtures the REAL tip trainer wrote exist (reach_synthetic.py): 5 gates", len(fixtures) >= 5, [d.name for d in fixtures])
    for d in fixtures:
        arm, s = d.name.split("_")[0], int(d.name.split("_")[1][1:])
        a = ["check", "--selfcheck", str(d / "selfcheck.json"), "--gate-line", str(d / "gate_line.txt"), "--train-line", str(d / "train_line.txt"), "--pin-line", str(d / "pin_line.txt"), "--arm", arm, "--seed", str(s),
             "--banked-init-sha", (d / "init_sha.txt").read_text().strip(), "--expect", str(d / "expect.json")]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = G.main(a)
        ok(f"C2 {arm} s{s}: the asserter ACCEPTS the gate record the REAL tip trainer wrote for the queue's argv (synthetic world, expectations overridden)", rc == 0, buf.getvalue()[-300:])
        if arm == "dagorc" and s == 0:
            sc = json.loads((d / "selfcheck.json").read_text(encoding="utf-8"))
            for nm, fn in (("built p 0.1 instead of 0.3", lambda x: x["levers"]["ego_perturb"].__setitem__("ego_perturb_p", 0.1)),
                           ("probe grad_diff_all 0", lambda x: x["ego_perturb_probe"].__setitem__("grad_diff_all", 0.0)),
                           ("probe control_prior_grad_diff 1e-9", lambda x: x["ego_perturb_probe"].__setitem__("control_prior_grad_diff", 1e-9)),
                           ("init sha another checkpoint", lambda x: (x["levers"]["init_ckpt"].__setitem__("sha256", "0" * 64), x["info"].__setitem__("init_ckpt", x["levers"]["init_ckpt"])))):
                y = copy.deepcopy(sc)
                fn(y)
                (SCR / "C").mkdir(exist_ok=True)
                (SCR / "C" / "fx_sc.json").write_text(json.dumps(y), encoding="utf-8")
                b = list(a)
                b[2] = str(SCR / "C" / "fx_sc.json")
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = G.main(b)
                refused(f"C2n real-trainer record doctored: {nm}", rc != 0, buf.getvalue()[-200:])

    # ---- C3: deliberate regressions: ONE defect each, in a copy of a positive case; the asserter must REFUSE
    def neg(name, arm, s, mut, expect_msg=None):
        c = Case(arm, s, SCR / "C")
        mut(c)
        rc, out = c.check("neg")
        good = rc != 0 and (expect_msg is None or expect_msg in out)
        refused(f"C3 {arm} s{s}: {name}", good, out[-260:] if not good else "")
        return good

    A_ = "dagorc"
    neg("init sha256 is another checkpoint (M2b's, not DAG_s0's)", A_, 0, lambda c: (c.sc["levers"]["init_ckpt"].__setitem__("sha256", "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8"), c.sc["info"].__setitem__("init_ckpt", c.sc["levers"]["init_ckpt"])), "does not start from DAG_s0")
    neg("init_ckpt not applied", A_, 0, lambda c: (c.sc["levers"]["init_ckpt"].__setitem__("applied", False), c.sc["info"].__setitem__("init_ckpt", c.sc["levers"]["init_ckpt"])), "not applied")
    neg("init_ckpt read-back mismatches > 0", A_, 0, lambda c: (c.sc["levers"]["init_ckpt"].__setitem__("n_mismatch_after_load", 3), c.sc["info"].__setitem__("init_ckpt", c.sc["levers"]["init_ckpt"])), "mismatches")
    neg("init_ckpt path is M2b's directory", A_, 0, lambda c: (c.sc["levers"]["init_ckpt"].__setitem__("path", r"D:\tanitad\runs\v8f_mini_full\M2b_s0\ckpt_last.pt"), c.sc["info"].__setitem__("init_ckpt", c.sc["levers"]["init_ckpt"])), "init_ckpt path")
    neg("info.init_ckpt disagrees with levers.init_ckpt", A_, 0, lambda c: c.sc["info"]["init_ckpt"].__setitem__("applied", False), "disagree")
    neg("steps_total 2999", A_, 0, lambda c: c.sc.__setitem__("steps_total", 2999), "steps")
    neg("built lr 2e-4", A_, 0, lambda c: c.sc["levers"]["lr_schedule"].__setitem__("lr", 2e-4), "lr schedule")
    neg("built p_gt 0.2 -> 0.0", A_, 0, lambda c: c.sc["levers"].__setitem__("p_gt_end", 0.0), "p_gt")
    neg("declared_vs_built FAIL", A_, 0, lambda c: c.sc.__setitem__("declared_vs_built", "FAIL"), "declared_vs_built")
    neg("G-LIVE prior control 1e-9", A_, 0, lambda c: c.sc["live"].__setitem__("control_prior_grad_sum", 1e-9), "prior control")
    neg("G-LIVE dead param", A_, 0, lambda c: c.sc["live"].__setitem__("dead_params", ["head_o.2.weight"]), "dead set")
    neg("lru absent", A_, 0, lambda c: c.sc["info"].pop("lru"), "lru")
    neg("excluded set is not the RA-1 holdout", A_, 0, lambda c: c.sc["info"].__setitem__("excluded_sha256", "1" * 64), "holdout")
    neg("train list is not DAG2-0's", A_, 0, lambda c: c.sc["info"].__setitem__("train_sha256", "2" * 64), "TRAIN list")
    neg("a parameter group count differs from DAG2-0's", A_, 0, lambda c: c.sc["levers"]["params"].__setitem__("O", c.sc["levers"]["params"]["O"] + 1), "parameter group")
    neg("w_acc built 49.7 (a second lever)", A_, 0, lambda c: c.sc["levers"].__setitem__("w_acc", 49.7), "w_acc")
    neg("o_interface built cut (a second lever)", A_, 0, lambda c: c.sc["levers"].__setitem__("o_interface", "cut"), "o_interface")
    # the lever: probe + built perturber
    neg("ego probe reads 'off' on A", A_, 0, lambda c: c.sc.__setitem__("ego_perturb_probe", {"ego_perturb": "off", "pass": True}), "ego probe ran")
    neg("ego probe p_declared 0.1 on A", A_, 0, lambda c: c.sc["ego_perturb_probe"].__setitem__("p_declared", 0.1), "ego probe ran")
    neg("ego probe saw no eligible window", A_, 0, lambda c: c.sc["ego_perturb_probe"].__setitem__("n_selected_at_p1", 0), "no eligible")
    neg("ego probe: the perturbed batch does not move the gradient", A_, 0, lambda c: c.sc["ego_perturb_probe"].__setitem__("grad_diff_all", 0.0), "does not reach the loss")
    neg("ego probe: the perturbed batch does not move the loss", A_, 0, lambda c: c.sc["ego_perturb_probe"].__setitem__("loss_diff", 0.0), "does not reach the loss")
    for k in G.PROBE_CONTROLS:
        neg(f"ego probe control {k} = 1e-9", A_, 0, lambda c, k=k: c.sc["ego_perturb_probe"].__setitem__(k, 1e-9), "must be exactly 0.0")
    neg("ego probe: p = 0 is not the identity", A_, 0, lambda c: c.sc["ego_perturb_probe"].__setitem__("control_p0_same_object", False), "identity")
    neg("ego probe: the world-model gradient moved", A_, 0, lambda c: c.sc["ego_perturb_probe"].__setitem__("control_wm_grad_diff", 0.5), "world-model")
    neg("ego probe: G-LIVE on the perturbed batch failed", A_, 0, lambda c: c.sc["ego_perturb_probe"]["live_perturbed"].__setitem__("pass", False), "PERTURBED")
    neg("ego probe: pass False", A_, 0, lambda c: c.sc["ego_perturb_probe"].__setitem__("pass", False), "ego_perturb_probe")
    neg("built p 0.1 on A (declared 0.3)", A_, 0, lambda c: c.sc["levers"]["ego_perturb"].__setitem__("ego_perturb_p", 0.1), "built ego_perturb ego_perturb_p")
    neg("built p 0.0 on A", A_, 0, lambda c: c.sc["levers"]["ego_perturb"].__setitem__("ego_perturb_p", 0.0), "built ego_perturb ego_perturb_p")
    for k, v in (("ego_perturb_vscale", [0.4, 1.0]), ("ego_perturb_vmax", 6.0), ("ego_perturb_adrop", 1.0), ("ego_perturb_yawrate_std", 1.0), ("ego_perturb_seed", 1), ("kin_mode", "steer")):
        neg(f"built {k} {v!r} (the law differs)", A_, 0, lambda c, k=k, v=v: c.sc["levers"]["ego_perturb"].__setitem__(k, v), f"built ego_perturb {k}")
    neg("A s1 built seed 0 (the law seed must be S)", A_, 1, lambda c: c.sc["levers"]["ego_perturb"].__setitem__("ego_perturb_seed", 0), "ego_perturb_seed")
    neg("B built p 0.3", "dagorc01", 0, lambda c: c.sc["levers"]["ego_perturb"].__setitem__("ego_perturb_p", 0.3), "built ego_perturb ego_perturb_p")
    neg("B probe p_declared 0.3", "dagorc01", 0, lambda c: c.sc["ego_perturb_probe"].__setitem__("p_declared", 0.3), "ego probe ran")
    neg("REP: the ego probe is not 'off'", "dagorcrep", 0, lambda c: c.sc.__setitem__("ego_perturb_probe", {"ego_perturb": "on", "pass": True, "p_declared": 0.3}), "should read 'off'")
    neg("REP: built p 0.3", "dagorcrep", 0, lambda c: c.sc["levers"]["ego_perturb"].__setitem__("ego_perturb_p", 0.3), "built ego_perturb ego_perturb_p")
    neg("REP: built yawrate_std 2.0 (a law value built at p 0)", "dagorcrep", 0, lambda c: c.sc["levers"]["ego_perturb"].__setitem__("ego_perturb_yawrate_std", 2.0), "yawrate")
    # the swap is off on all three
    neg("the DAgger probe is not 'off'", A_, 0, lambda c: c.sc.__setitem__("dagger_probe", {"dagger": "on", "pass": True}), "dagger_probe")
    neg("dagger_probe pass False", A_, 0, lambda c: c.sc["dagger_probe"].__setitem__("pass", False), "dagger_probe")
    neg("dagger_probe absent", A_, 0, lambda c: c.sc.pop("dagger_probe"), "dagger_probe")
    neg("built dagger_p 0.1", A_, 0, lambda c: c.sc["levers"].__setitem__("dagger_p", 0.1), "dagger_p")
    neg("a bank hash was built", A_, 0, lambda c: c.sc["levers"].__setitem__("dagger_bank_sha256", "a" * 64), "bank")
    neg("a bank hash was declared", "dagorcrep", 0, lambda c: c.sc["levers"].__setitem__("dagger_bank_sha256_declared", "a" * 64), "bank")
    neg("info carries a DAgger control record", A_, 0, lambda c: c.sc["info"].__setitem__("dagger", {"bank_sha256": "a" * 64}), "DAgger control record")
    # the queue lines (train, gate, pin; edits applied identically so only the targeted rule fires)
    neg("A's lines carry p 0.1", A_, 0, lambda c: edit_lines(c, lambda t: setv(t, "--ego-perturb-p", "0.1")), "--ego-perturb-p")
    neg("A's lines drop one law flag (--ego-perturb-adrop)", A_, 0, lambda c: edit_lines(c, lambda t: delflag(t, "--ego-perturb-adrop")), "--ego-perturb-adrop")
    neg("A's lines change the law seed", A_, 0, lambda c: edit_lines(c, lambda t: setv(t, "--ego-perturb-seed", "5")), "--ego-perturb-seed")
    neg("A's lines name a law flag twice", A_, 0, lambda c: edit_lines(c, lambda t: t + ["--ego-perturb-vmax", "8.0"]), "exactly once")
    neg("A's lines drop --dagger-p 0 (the lever must be NAMED)", A_, 0, lambda c: edit_lines(c, lambda t: delflag(t, "--dagger-p")), "--dagger-p")
    neg("A's lines name --dagger-p 0.1", A_, 0, lambda c: edit_lines(c, lambda t: setv(t, "--dagger-p", "0.1")), "--dagger-p")
    neg("A's lines name a bank", A_, 0, lambda c: edit_lines(c, lambda t: t + ["--dagger-bank", r"D:\bank"]), "--dagger-bank")
    neg("A's lines name --dagger-seed", A_, 0, lambda c: edit_lines(c, lambda t: t + ["--dagger-seed", "0"]), "--dagger-seed")
    neg("A's lines carry a second lever (--w-acc 49.7)", A_, 0, lambda c: edit_lines(c, lambda t: t + ["--w-acc", "49.7"]), "second lever")
    neg("A's lines carry --o-interface cut", A_, 0, lambda c: edit_lines(c, lambda t: t + ["--o-interface", "cut"]), "second lever")
    neg("A's lines carry --no-resume", A_, 0, lambda c: edit_lines(c, lambda t: t + ["--no-resume"]), "second lever")
    neg("A's lines start from M2b_s0 (the old init)", A_, 0, lambda c: edit_lines(c, lambda t: setv(t, "--init-ckpt", r"D:\tanitad\runs\v8f_mini_full\M2b_s0\ckpt_last.pt")), "--init-ckpt")
    neg("A's lines: --seed 10 (REP's seed)", A_, 0, lambda c: edit_lines(c, lambda t: setv(t, "--seed", "10")), "--seed")
    neg("A's lines: --steps 3001", A_, 0, lambda c: edit_lines(c, lambda t: setv(t, "--steps", "3001")), "--steps")
    neg("A's lines: --lr 2e-4", A_, 0, lambda c: edit_lines(c, lambda t: setv(t, "--lr", "2e-4")), "--lr")
    neg("A's lines: --warmup 50", A_, 0, lambda c: edit_lines(c, lambda t: setv(t, "--warmup", "50")), "--warmup")
    neg("ONE-LEVER only: --threads 3 appended identically to train, gate and pin (no other rule fires)", A_, 0, lambda c: edit_lines(c, lambda t: t + ["--threads", "3"]), "ONE-LEVER")
    neg("ONE-LEVER only: --batch 32 on train, gate and pin", A_, 0, lambda c: edit_lines(c, lambda t: setv(t, "--batch", "32")), "ONE-LEVER")
    neg("ONE-LEVER only: --exclude 1:2 on train, gate and pin", A_, 0, lambda c: edit_lines(c, lambda t: setv(t, "--exclude", "1:2")), "ONE-LEVER")
    neg("ONE-LEVER only: the same law flags and values in another order", A_, 0, lambda c: edit_lines(c, swap_law), "ONE-LEVER")
    neg("the gate line differs from the train line (another batch size, train unchanged)", A_, 0, lambda c: edit_lines(c, lambda t: setv(t, "--batch", "32"), which=("g",)), "gate argv is not")
    neg("the pin differs from the planned train line (another lr)", A_, 0, lambda c: edit_lines(c, lambda t: setv(t, "--lr", "2e-4"), which=("p",)), "PIN IS NOT")
    neg("the gate line carries no --selfcheck-only", A_, 0, lambda c: (setattr(c, "gtok", [x for x in c.gtok if x != "--selfcheck-only"]), c.fix_sha()), "selfcheck-only")
    neg("the selfcheck's argv_sha is another argv's", A_, 0, lambda c: c.sc.__setitem__("argv_sha", "0" * 16), "argv_sha")
    neg("REP's lines name a law flag (--ego-perturb-seed 0: equal to the trainer's default, so only this rule can see it)", "dagorcrep", 0, lambda c: edit_lines(c, lambda t: t + ["--ego-perturb-seed", "0"]), "REP")
    neg("REP's lines name --ego-perturb-p 0.3", "dagorcrep", 0, lambda c: edit_lines(c, lambda t: setv(t, "--ego-perturb-p", "0.3")), "--ego-perturb-p")
    neg("REP's lines carry --seed 0 (DAG2-0's seed: not a replicate)", "dagorcrep", 0, lambda c: edit_lines(c, lambda t: setv(t, "--seed", "0")), "--seed")
    neg("REP s1's lines carry --seed 10 (s0's)", "dagorcrep", 1, lambda c: edit_lines(c, lambda t: setv(t, "--seed", "10")), "--seed")
    neg("REP's lines carry --seed 12", "dagorcrep", 0, lambda c: edit_lines(c, lambda t: setv(t, "--seed", "12")), "--seed")
    neg("B's lines carry p 0.3 (A's dose)", "dagorc01", 0, lambda c: edit_lines(c, lambda t: setv(t, "--ego-perturb-p", "0.3")), "--ego-perturb-p")
    neg("B's lines carry p 0.2", "dagorc01", 0, lambda c: edit_lines(c, lambda t: setv(t, "--ego-perturb-p", "0.2")), "--ego-perturb-p")
    # references
    neg("the one-lever reference argv is seed 1's (a mispaired reference)", A_, 0, lambda c: setattr(c, "ref", L.DAG2_DIR / "M2b_s1_dag20" / "argv.json"), "ONE-LEVER")
    neg("the comparator B is the other seed's summary (its TRAIN list / window set are the same, its parameter groups too: only an UNRELATED summary may differ -> a doctored copy)", A_, 0,
        lambda c: (setattr(c, "bsum", SCR / "C" / "bsum_bad.json"), (SCR / "C" / "bsum_bad.json").write_text(json.dumps({**json.loads(L.DAG2_DIR.joinpath("M2b_s0_dag20", "summary.json").read_text(encoding="utf-8")), "train_sha256": "3" * 64}), encoding="utf-8")), "TRAIN list")
    # an invalid seed
    c = Case(A_, 0, SCR / "C")
    d = c.write(SCR / "C" / "badseed")
    a = c.args(d)
    a[a.index("--seed") + 1] = "2"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = G.main(a)
    refused("C3 --seed 2 is REFUSED (rc 2: DAG_s exists for seeds 0 and 1 only)", rc == 2, buf.getvalue())
    # the control that proves the doctoring harness: a no-op mutation passes
    c0 = Case(A_, 0, SCR / "C")
    c0.sc["info"]["lru"] = c0.sc["info"]["lru"]
    rc, out = c0.check("noop")
    ok("C3c control: a no-op edit of the same record is still accepted (the doctoring harness does not break positives)", rc == 0, out[-200:])
    # ---- C4: the token
    c = Case("dagorc", 0, SCR / "C")
    d = c.write(SCR / "C" / "tok")
    tok = d / "token.json"
    base = c.args(d, "token") + ["--commit", "a" * 40, "--queue-sha", "b" * 64, "--key-sha-json", '{"x": "y"}', "--run", "M2b_s0_dagorc", "--out", str(tok)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = G.main(base)
    tj = json.loads(tok.read_text(encoding="utf-8")) if tok.exists() else {}
    ok("C4 the token is written after a passing check and binds the commit, the PINNED training argv's sha256 and the data sha256; names the arm and the dose", rc == 0 and tj.get("verdict") == "PASS" and tj["binding"]["commit"] == "a" * 40
       and tj["binding"]["argv_sha256"] == G.argv_sha256(c.ptok) and tj["binding"]["data_sha256"] == c.sc["info"]["train_sha256"] and tj.get("ego_perturb_p") == 0.3 and tj.get("trainer_seed") == 0 and tj.get("prereg") == "PREREG_V8F_DAG_ORC", buf.getvalue()[-200:])
    tok.unlink(missing_ok=True)      # (a mutated asserter that refuses the positive writes no token: that is a RED check above, not a crash here)
    args = list(base)
    args[args.index("--commit") + 1] = "abc"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = G.main(args)
    refused("C4n a commit that is not 40 hex: no token", rc != 0 and not tok.exists(), buf.getvalue()[-160:])
    c = Case("dagorc", 0, SCR / "C")
    c.sc["levers"]["ego_perturb"]["ego_perturb_p"] = 0.1
    d = c.write(SCR / "C" / "tok2")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = G.main(c.args(d, "token") + ["--commit", "a" * 40, "--queue-sha", "b" * 64, "--run", "x", "--out", str(d / "token.json")])
    refused("C4n a failing check writes NO token", rc != 0 and not (d / "token.json").exists())
    c = Case("dagorcrep", 1, SCR / "C")
    d = c.write(SCR / "C" / "tok3")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = G.main(c.args(d, "token") + ["--commit", "c" * 40, "--queue-sha", "b" * 64, "--run", "M2b_s1_dagorcrep", "--out", str(d / "token.json")])
    tj = json.loads((d / "token.json").read_text(encoding="utf-8")) if (d / "token.json").exists() else {}
    ok("C4 REP s1's token records trainer_seed 11 and ego_perturb_p 0.0", rc == 0 and tj.get("trainer_seed") == 11 and tj.get("ego_perturb_p") == 0.0, tj)
    section_C_records()


def run_dir_for(arm, s, root, mut=None):
    """A trained-run directory (argv.json, summary.json, selfcheck.json) for (arm, s), derived from DAG2-0's REAL summary.json and the queue's lines; ``mut`` edits the dicts / tokens before writing."""
    g, t, _ = lines_of(arm, s)
    ttok = L.train_argv(t)
    p = G.ARMS[arm]
    tseed = 10 + s if arm == "dagorcrep" else s
    d = Path(root)
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    sm = json.loads((L.DAG2_DIR / f"M2b_s{s}_dag20" / "summary.json").read_text(encoding="utf-8"))
    s16 = sha16(ttok)
    sm["seed"], sm["argv_sha"], sm["steps"] = tseed, s16, 3000
    sm["levers"]["ego_perturb"] = {"ego_perturb_p": p, **({**G.LAW_VALUES, "ego_perturb_seed": s} if p > 0 else G.LAW_DEFAULT), "kin_mode": "yaw"}
    sm["ego_perturb_n"], sm["ego_perturb_eligible"] = (int(round(p * 82457)), 82457)
    state = {"argv": {"argv": ttok, "sha": s16}, "summary": sm, "selfcheck": {"argv_sha": s16, "declared_vs_built": "PASS"}, "pin_tokens": list(ttok)}
    if mut:
        mut(state)
    (d / "argv.json").write_text(json.dumps(state["argv"]), encoding="utf-8")
    (d / "summary.json").write_text(json.dumps(state["summary"]), encoding="utf-8")
    (d / "selfcheck.json").write_text(json.dumps(state["selfcheck"]), encoding="utf-8")
    q = lambda toks: '"python" "v8f_mini_train.py" ' + " ".join(f'"{x}"' if (" " in x or "\\" in x) else x for x in toks)      # noqa: E731
    (d / "t.txt").write_text(q(ttok), encoding="utf-8")
    (d / "p.txt").write_text(q(state["pin_tokens"]), encoding="utf-8")
    return d


def record_rc(arm, s, d):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = K.main(["run-record", "--run-dir", str(d), "--train-line", str(d / "t.txt"), "--pin-line", str(d / "p.txt"), "--arm", arm, "--seed", str(s), "--banked-init-sha", CKPT_SHA[s]])
    return rc, buf.getvalue()


def section_C_records():
    print("-- C5: the run record, K-REACH and the one-lever subcommand")
    for arm in ARMS:
        for s in (0, 1):
            d = run_dir_for(arm, s, SCR / "C" / f"rec_{arm}_{s}")
            rc, out = record_rc(arm, s, d)
            ok(f"C5 {arm} s{s}: the run record of a correct run PASSES", rc == 0, out[-300:])

    def rneg(name, mut, msg, arm="dagorc", s=0):
        d = run_dir_for(arm, s, SCR / "C" / "recneg", mut)
        rc, out = record_rc(arm, s, d)
        refused(f"C5 {arm} s{s}: run record: {name}", rc != 0 and msg in out, out[-200:])

    rneg("argv.json is not the planned argv (a lr changed)", lambda x: x["argv"].__setitem__("argv", setv(x["argv"]["argv"], "--lr", "2e-4")), "argv.json is not the planned")
    rneg("the pin is not the planned argv", lambda x: x.__setitem__("pin_tokens", setv(x["pin_tokens"], "--lr", "2e-4")), "pinned training argv")
    rneg("argv.json sha is another", lambda x: x["argv"].__setitem__("sha", "0" * 16), "argv.json sha")
    rneg("summary.json argv_sha is another", lambda x: x["summary"].__setitem__("argv_sha", "0" * 16), "summary.json argv_sha")
    rneg("selfcheck.json argv_sha is another", lambda x: x["selfcheck"].__setitem__("argv_sha", "0" * 16), "selfcheck.json argv_sha")
    rneg("summary.json done false", lambda x: x["summary"].__setitem__("done", False), "done")
    rneg("summary.json steps 2999", lambda x: x["summary"].__setitem__("steps", 2999), "steps")
    rneg("summary.json seed 7", lambda x: x["summary"].__setitem__("seed", 7), "arm / seed")
    rneg("summary.json arm M1", lambda x: x["summary"].__setitem__("arm", "M1"), "arm / seed")
    rneg("the run's own selfcheck is not a PASS", lambda x: x["selfcheck"].__setitem__("declared_vs_built", "FAIL"), "selfcheck.json is not a PASS")
    rneg("init sha256 is M2b's (K-INIT)", lambda x: x["summary"]["levers"]["init_ckpt"].__setitem__("sha256", "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8"), "K-INIT")
    rneg("init not applied", lambda x: x["summary"]["levers"]["init_ckpt"].__setitem__("applied", False), "not applied")
    rneg("init path is M2b's", lambda x: x["summary"]["levers"]["init_ckpt"].__setitem__("path", r"D:\tanitad\runs\v8f_mini_full\M2b_s0\ckpt_last.pt"), "init_ckpt path")
    rneg("built p 0.1 on A", lambda x: x["summary"]["levers"]["ego_perturb"].__setitem__("ego_perturb_p", 0.1), "built ego_perturb ego_perturb_p")
    rneg("built law seed 1 on A s0", lambda x: x["summary"]["levers"]["ego_perturb"].__setitem__("ego_perturb_seed", 1), "ego_perturb_seed")
    rneg("built vmax 6.0", lambda x: x["summary"]["levers"]["ego_perturb"].__setitem__("ego_perturb_vmax", 6.0), "ego_perturb_vmax")
    rneg("built dagger_p 0.1", lambda x: x["summary"]["levers"].__setitem__("dagger_p", 0.1), "dagger_p")
    rneg("dagger_n 5 (a row was swapped)", lambda x: x["summary"].__setitem__("dagger_n", 5), "dagger_n")
    rneg("REP: built p 0.3", lambda x: x["summary"]["levers"]["ego_perturb"].__setitem__("ego_perturb_p", 0.3), "built ego_perturb ego_perturb_p", arm="dagorcrep")
    rneg("REP s1: summary seed 1 (the DAG seed, not the replicate's 11)", lambda x: x["summary"].__setitem__("seed", 1), "arm / seed", arm="dagorcrep", s=1)
    rneg("B: built p 0.3", lambda x: x["summary"]["levers"]["ego_perturb"].__setitem__("ego_perturb_p", 0.3), "built ego_perturb ego_perturb_p", arm="dagorc01")

    # K-REACH
    def reach(arm, n, e):
        f = SCR / "C" / "reach_summary.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"ego_perturb_n": n, "ego_perturb_eligible": e}), encoding="utf-8")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = K.main(["reached", "--summary", str(f), "--arm", arm])
        return rc

    E = 82457
    ok("C6 A: n/eligible = 0.3 exactly passes", reach("dagorc", round(0.3 * E), E) == 0)
    ok("C6 A: n/eligible = 0.32 (the upper edge, inclusive) passes", reach("dagorc", 32, 100) == 0)
    ok("C6 A: n/eligible = 0.28 (the lower edge, inclusive) passes", reach("dagorc", 28, 100) == 0)
    ok("C6 B: n/eligible = 0.1 passes", reach("dagorc01", round(0.1 * E), E) == 0)
    ok("C6 REP: n exactly 0 with eligible > 0 passes", reach("dagorcrep", 0, E) == 0)
    refused("C6 A: 0.321 is outside the band", reach("dagorc", 321, 1000) == 1)
    refused("C6 A: 0.279 is outside the band", reach("dagorc", 279, 1000) == 1)
    refused("C6 A: n = 0 (the loop line deleted) fails", reach("dagorc", 0, E) == 1)
    refused("C6 A: eligible = 0 fails", reach("dagorc", 0, 0) == 1)
    refused("C6 A: n > eligible fails", reach("dagorc", E + 1, E) == 1)
    refused("C6 B at A's rate 0.3 fails", reach("dagorc01", round(0.3 * E), E) == 1)
    refused("C6 A at B's rate 0.1 fails", reach("dagorc", round(0.1 * E), E) == 1)
    refused("C6 REP: n = 1 fails (a perturbed row on the control)", reach("dagorcrep", 1, E) == 1)
    refused("C6 REP: eligible = 0 fails (a zero that proves nothing)", reach("dagorcrep", 0, 0) == 1)
    f = SCR / "C" / "reach_summary.json"
    f.write_text("{}", encoding="utf-8")
    refused("C6 A: the keys are absent", K.reached(type("A", (), {"summary": str(f), "arm": "dagorc"})()) == 1)
    refused("C6 A: n is a bool / a float (not a count)", reach("dagorc", True, E) == 1 and reach("dagorc", 24737.0, E) == 1)
    f = FIX / "dagorcrep_s0_run_summary.json"
    if f.exists():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = K.reached(type("A", (), {"summary": str(f), "arm": "dagorcrep"})())
        ok("C6 the REAL trainer's synthetic REP run summary (ego_perturb_n exactly 0, eligible > 0) satisfies the control", rc == 0, buf.getvalue())
    f = FIX / "dagorc_s0_run_summary_LOOP_LINE_DELETED.json"
    if f.exists():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = K.reached(type("A", (), {"summary": str(f), "arm": "dagorc"})())
        refused("C6 the REAL trainer run with the loop's `ego.apply` line DELETED (ego_perturb_n 0) is refused by K-REACH", rc == 1)

    # the one-lever subcommand on the queue's lines and on doctored lines
    def onelever(arm, s, toks):
        f = SCR / "C" / "ol.txt"
        f.write_text('"python" "v8f_mini_train.py" ' + " ".join(f'"{x}"' if (" " in x or "\\" in x) else x for x in toks), encoding="utf-8")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = K.main(["one-lever", "--ref-argv-json", str(L.DAG2_DIR / f"M2b_s{s}_dag20" / "argv.json"), "--line", str(f), "--arm", arm, "--seed", str(s)])
        return rc, buf.getvalue().strip()

    for arm in ARMS:
        for s in (0, 1):
            rc, out = onelever(arm, s, L.train_argv(lines_of(arm, s)[1]))
            ok(f"C7 one-lever {arm} s{s}: the queue's line prints ONE-LEVER", rc == 0 and out == "ONE-LEVER", out)
    toks = L.train_argv(lines_of("dagorc", 0)[1])
    for nm, t in (("an extra flag", toks + ["--threads", "3"]), ("lr changed", setv(toks, "--lr", "2e-4")), ("a law flag dropped", delflag(toks, "--ego-perturb-vmax")), ("p 0.1 on A", setv(toks, "--ego-perturb-p", "0.1")),
                  ("--dagger-p 0.1", setv(toks, "--dagger-p", "0.1")), ("the init is M2b's", setv(toks, "--init-ckpt", r"D:\tanitad\runs\v8f_mini_full\M2b_s0\ckpt_last.pt"))):
        rc, out = onelever("dagorc", 0, t)
        refused(f"C7 one-lever dagorc s0: {nm} prints NOT-ONE-LEVER", rc != 0 and out == "NOT-ONE-LEVER", out)
    rc, out = onelever("dagorcrep", 0, setv(L.train_argv(lines_of("dagorcrep", 0)[1]), "--seed", "0"))
    refused("C7 one-lever dagorcrep s0: --seed 0 prints NOT-ONE-LEVER", rc != 0 and out == "NOT-ONE-LEVER", out)
    rc, out = onelever("dagorc", 0, L.train_argv(lines_of("dagorc", 1)[1]))
    refused("C7 one-lever dagorc s0 on seed 1's line prints NOT-ONE-LEVER", rc != 0 and out == "NOT-ONE-LEVER", out)


# ============================================================================================================================ D
FAKEBOX = HERE / "fakebox.py"
PYV = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
REAL_MIRROR = "C:/Users/Admin/tanitad-push"
TIP_SHORT = "3b7b404e"
GITENV = {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "safe.directory", "GIT_CONFIG_VALUE_0": "*", "GIT_AUTHOR_NAME": "selftest", "GIT_AUTHOR_EMAIL": "selftest@invalid",
          "GIT_COMMITTER_NAME": "selftest", "GIT_COMMITTER_EMAIL": "selftest@invalid"}
PRED = {"done": "10-27 11:00 the predecessor chain ran\nZZV8FMINI-DONE 10-27 12:00 tip x: the predecessor's runs\n",
        "stop": "10-27 11:00 the predecessor chain ran\nZZV8FMINI-STOP some stop of its own\n",
        "guard": "10-27 11:00 the predecessor chain ran\nZZGUARD-STOP TEMP 10-27 11:30 USB HDD hot\nZZV8FMINI-STOP guard stop\n",
        "none": "10-27 11:00 the predecessor chain is still running\n",
        "echo": "10-27 11:00 waiting for ^ZZV8FMINI-DONE or ^ZZV8FMINI-STOP\n"}


class HarnessError(RuntimeError):
    """the test rig itself could not be built (an anchor of the doctoring is gone, a git command failed): never read as a kill by the mutation runner"""


def rmtree(p):
    """shutil.rmtree that also removes git's read-only object files"""
    import stat

    def h(fn, path, *_):
        os.chmod(path, stat.S_IWRITE)
        fn(path)
    for k in range(6):
        try:
            try:
                shutil.rmtree(p, onexc=h)
            except TypeError:
                shutil.rmtree(p, onerror=h)
            return
        except PermissionError:      # a process of an earlier test still holds a handle: give it a second
            if k == 5:
                raise
            time.sleep(1.5)


def msys(p):
    s = str(p).replace("\\", "/")
    return ("/" + s[0].lower() + s[2:]) if re.match(r"^[A-Za-z]:", s) else s


def fwd(p):
    return str(p).replace("\\", "/")


def git(*a, env=None, cwd=None):
    return sh(["git", *a], env={**os.environ, **GITENV, **(env or {})}, cwd=cwd)


def real_tip():
    rc, o, e = git("-C", REAL_MIRROR, "rev-parse", f"{TIP_SHORT}^{{commit}}")
    if rc != 0 or not re.fullmatch(r"[0-9a-f]{40}", o.strip()):
        raise HarnessError(f"cannot resolve {TIP_SHORT} in the mirror: {o} {e}")
    return o.strip()


def build_origin():
    """origin.git: a bare repo that BORROWS the real mirror's objects (read-only alternates; nothing is written to the mirror). dagorc-new = the tip 3b7b404e + the staged queue (the queue under test) + the staged tests; dagorc-old = the tip itself."""
    base = SCR / "E"
    base.mkdir(parents=True, exist_ok=True)
    o, tip = base / "origin.git", real_tip()
    files = {f.relative_to(HERE / "stage").as_posix(): f for f in (HERE / "stage").rglob("*") if f.is_file()}
    files[f"{L.PK}/code/v8f_mini_box2_queue.cmd"] = L.QUEUE
    tr_rel = "stack/scripts/v8f_mini_train.py"
    tr_bytes = (TREE / tr_rel).read_bytes()
    tr_mut, n_sub = re.subn(rb"(?m)^( +)b = ego\.apply\(b, step\)", rb"\1b = b", tr_bytes)      # the CODE line in the loop (the comment that mentions it is not anchored to leading spaces + `b =`)
    if n_sub != 1:
        raise HarnessError(f"the composed tree's trainer has {n_sub} `b = ego.apply(b, step)` loop lines (need exactly 1)")
    stamp = hashlib.sha256(("|".join(f"{k}={hashlib.sha256(v.read_bytes()).hexdigest()}" for k, v in sorted(files.items())) + tip + hashlib.sha256(tr_mut).hexdigest()).encode()).hexdigest()
    if o.exists() and (base / "origin.stamp").exists() and (base / "origin.stamp").read_text() == stamp:
        return o, tip
    if o.exists():
        rmtree(o)
    rc, _, e = git("init", "-q", "--bare", fwd(o))
    if rc != 0:
        raise HarnessError("git init --bare: " + e)
    (o / "objects" / "info").mkdir(parents=True, exist_ok=True)
    (o / "objects" / "info" / "alternates").write_bytes((REAL_MIRROR + "/.git/objects\n").encode())      # bytes: a CR in this file is part of the path
    idx = base / "origin.idx"
    idx.unlink(missing_ok=True)
    env = {"GIT_DIR": fwd(o), "GIT_INDEX_FILE": fwd(idx)}
    rc, _, e = git("read-tree", tip, env=env)
    if rc != 0:
        raise HarnessError("read-tree: " + e)
    for rel, f in sorted(files.items()):
        rc, b, e = git("hash-object", "-w", "--no-filters", fwd(f), env=env)
        if rc != 0:
            raise HarnessError("hash-object: " + e)
        rc, _, e = git("update-index", "--add", "--cacheinfo", f"100644,{b.strip()},{rel}", env=env)
        if rc != 0:
            raise HarnessError("update-index: " + e)
    rc, tree, e = git("write-tree", env=env)
    rc2, c, e2 = git("commit-tree", tree.strip(), "-p", tip, "-m", "SELF-TEST ONLY: the tip + the staged DAG-ORC queue stages and tests", env=env)
    if rc or rc2:
        raise HarnessError(f"write-tree / commit-tree: {e} {e2}")
    git("update-ref", "refs/heads/dagorc-new", c.strip(), env=env)
    # dagorc-noapply: the same tip but the trainer's loop line `b = ego.apply(b, step)` is gone (the lever is declared, gated and NEVER applied): the chain must refuse it at the refresh
    mut = base / "trainer_noapply.py"
    mut.write_bytes(tr_mut)
    rc, b, e = git("hash-object", "-w", "--no-filters", fwd(mut), env=env)
    rc2, _, e2 = git("update-index", "--add", "--cacheinfo", f"100644,{b.strip()},{tr_rel}", env=env)
    rc3, tree2, e3 = git("write-tree", env=env)
    rc4, c2, e4 = git("commit-tree", tree2.strip(), "-p", c.strip(), "-m", "SELF-TEST ONLY: the trainer's ego.apply line deleted", env=env)
    if rc or rc2 or rc3 or rc4:
        raise HarnessError(f"noapply branch: {e} {e2} {e3} {e4}")
    git("update-ref", "refs/heads/dagorc-noapply", c2.strip(), env=env)
    git("update-ref", "refs/heads/dagorc-old", tip, env=env)
    (base / "origin.stamp").write_text(stamp)
    return o, tip


class World:
    pass


def _sub1(text, old, new):
    if text.count(old) != 1:
        raise HarnessError(f"anchor {old!r} occurs {text.count(old)} times (need exactly 1)")
    return text.replace(old, new)


def make_world(name, *, mirror=True, br="dagorc-new", faults="", pred="done", skip_b=True, env=None, drop=(), edit=None):
    """A ship dir (the chain + its helpers + the installed proven libs, a DERIVED mk_v8f_job.py / waitjob.sh / guard that point at THIS dir instead of C:/Users/Admin/box2_ship), ssh / scp shims over a fake box 2 tree, a fake git
    mirror (when ``mirror``), the predecessor's log. Nothing here touches C:/Users/Admin/box2_ship except READING the two proven libs and the real mk_v8f_job.py / waitjob.sh."""
    w = World()
    w.name, w.base = name, SCR / "E" / name
    if w.base.exists():
        rmtree(w.base)
    w.ship, w.bin, w.fb, w.art, w.tmp = (w.base / x for x in ("ship", "bin", "fb", "art", "tmp"))
    for d in (w.ship / "logs", w.bin, w.fb / "tanitad" / "jobs", w.fb / "tanitad" / "logs", w.fb / "tanitad" / "runs" / "v8f_mini_full", w.tmp):
        d.mkdir(parents=True)
    for f in ("chain_v8f_dagorc.sh", "gate_dagorc_assert.py", "v8f_dagorc_chain_checks.py", "wake_v8fdagorc.sh", "run_chain_v8f_dagorc.cmd"):
        if f not in drop:
            shutil.copyfile(SHIP / f, w.ship / f)
    for f in ("oaf_lib.sh", "v8f_oaf_chain_checks.py"):
        shutil.copyfile(INSTALLED / f, w.ship / f)
    g = (SHIP / "v8f_guard_dagorc.sh").read_bytes().decode("utf-8")
    g = _sub1(_sub1(g, "B=/c/Users/Admin/box2_ship;", f"B={msys(w.ship)};"), "sleep 60", "sleep 1")
    (w.ship / "v8f_guard_dagorc.sh").write_bytes(g.encode("utf-8"))
    mk = (INSTALLED / "mk_v8f_job.py").read_bytes().decode("utf-8")
    mk = _sub1(mk, '"C:/Users/Admin/box2_ship/"', '__import__("os").path.dirname(__import__("os").path.abspath(__file__)).replace(chr(92), "/") + "/"')
    (w.ship / "mk_v8f_job.py").write_bytes(mk.encode("utf-8"))
    wj = (INSTALLED / "waitjob.sh").read_bytes().decode("utf-8")
    wj = _sub1(_sub1(wj, "L=/c/Users/Admin/box2_ship/logs;", 'L=$(cd "$(dirname "$0")" && pwd)/logs;'), "sleep 30", "sleep 0.2")
    (w.ship / "waitjob.sh").write_bytes(wj.encode("utf-8"))
    for nm in ("ssh", "scp"):
        (w.bin / nm).write_bytes((f'#!/usr/bin/env bash\nexport MSYS2_ARG_CONV_EXCL=\'*\'\nexec {PYV} {fwd(FAKEBOX)} {nm} "$@"\n').encode())
    (w.base / "pred.out").write_text(PRED[pred], encoding="utf-8")
    w.mirror, w.tip_new = w.base / "mirror", None
    if mirror:
        o, tip = build_origin()
        m = w.mirror
        git("init", "-q", fwd(m))
        (m / ".git" / "objects" / "info").mkdir(parents=True, exist_ok=True)
        (m / ".git" / "objects" / "info" / "alternates").write_bytes(f"{REAL_MIRROR}/.git/objects\n{fwd(o)}/objects\n".encode())
        git("-C", fwd(m), "remote", "add", "origin", fwd(o))
        git("-C", fwd(m), "update-ref", "refs/remotes/origin/dagorc-old", tip)
        w.tip_new = git("--git-dir", fwd(o), "rev-parse", "refs/heads/dagorc-new")[1].strip()
    w.env = {"OAF_SHIP_DIR": msys(w.ship), "DAGORC_ART": msys(w.art), "OAF_MIRROR": msys(w.mirror) if mirror else "/nonexistent-mirror", "OAF_BR": br, "AFTER_OUT": msys(w.base / "pred.out"),
             "OAF_POLL_S": "1", "OAF_GRACE_S": "1", "OAF_TELEMETRY_S": "0", "OAF_TELEMETRY_S2": "0", "TMPDIR": msys(w.tmp), "FAKEBOX_ROOT": fwd(w.fb), "FAKEBOX_QUEUE": fwd(L.QUEUE), "DAGORC_TREE": fwd(TREE),
             "FAKE_FAULT": faults, "DAGORC_SKIP_B": "1" if skip_b else "0", **GITENV, **(env or {})}
    if edit:
        edit(w)
    return w


def run_chain(w, args="", timeout=1500):
    """-> (rc, chain output, the fake box's call log). The chain runs from the world's ship dir with the shims first on PATH."""
    e = {**os.environ, **w.env}
    for k in [k for k, v in w.env.items() if v is None]:
        e.pop(k, None)
    (w.ship / "logs" / "chain.rc").unlink(missing_ok=True)
    script = f'export PATH="{msys(w.bin)}:$PATH"; cd "{msys(w.ship)}" && timeout {int(timeout)} bash chain_v8f_dagorc.sh {args} > logs/chain_v8f_dagorc.out 2>&1 < /dev/null; echo $? > logs/chain.rc'
    try:      # no inherited pipe, and `timeout` inside: a chain that waits forever (a mutant with no AFTER_OUT rule) must not hang the rig
        subprocess.run([BASH, "-c", script], env=e, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout + 120)
    except subprocess.TimeoutExpired:
        pass
    outf, rcf = w.ship / "logs" / "chain_v8f_dagorc.out", w.ship / "logs" / "chain.rc"
    out = outf.read_text(encoding="utf-8", errors="replace").replace("\r", "") if outf.exists() else ""
    rc = int(rcf.read_text().strip()) if rcf.exists() else 124
    if rc == 124:      # killed by the timeout: reap whatever it left (a guard loop, sleeps) so nothing outlives the test
        subprocess.run(["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*" + msys(w.base) + "*' -and $_.Name -match 'bash|sleep|python' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"],
                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    calls = (w.fb / "calls.log").read_text(encoding="utf-8", errors="replace") if (w.fb / "calls.log").exists() else ""
    return rc, out, calls


def stop_lines(out):
    return [x for x in out.split("\n") if x.startswith("ZZV8FMINI-STOP")]


def sums_of(path):
    d = {}
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        h, n = ln.split(" ", 1)
        d[n.strip().lstrip("*")] = h
    return d


def pins_of(text):
    """the chain's pinned values by name"""
    got = {}
    for var in ("CKPT_SHA", "ARGV_SHA", "SUMM_SHA", "D20_ARGV_SHA", "D20_SUMM_SHA"):
        m = re.search(rf"^{var}=\(([0-9a-f]{{64}}) ([0-9a-f]{{64}})\)", text, re.M)
        got[var] = m.groups() if m else None
    for var in ("OAF_LIB_SHA", "OAF_CHECKS_SHA"):
        m = re.search(rf"^{var}=([0-9a-f]{{64}})", text, re.M)
        got[var] = m.group(1) if m else None
    m = re.search(r"^BUILD_REF=([0-9a-f]{40})", text, re.M)
    got["BUILD_REF"] = m.group(1) if m else None
    return got


def pins_wrong(text):
    """-> the list of pins that differ from the banked artifacts (empty = all pinned values are the banked ones)"""
    p = pins_of(text)
    bad = []
    exp = {"CKPT_SHA": [], "ARGV_SHA": [], "SUMM_SHA": [], "D20_ARGV_SHA": [], "D20_SUMM_SHA": []}
    for s in (0, 1):
        d, d2 = sums_of(L.DAG_DIR / f"M2b_s{s}_dag" / "SHA256SUMS.txt"), sums_of(L.DAG2_DIR / f"M2b_s{s}_dag20" / "SHA256SUMS.txt")
        exp["CKPT_SHA"].append(d["ckpt_last.pt"])
        exp["ARGV_SHA"].append(d["argv.json"])
        exp["SUMM_SHA"].append(d["summary.json"])
        exp["D20_ARGV_SHA"].append(d2["argv.json"])
        exp["D20_SUMM_SHA"].append(d2["summary.json"])
    for k, v in exp.items():
        if p[k] != tuple(v):
            bad.append(k)
    if p["OAF_LIB_SHA"] != hashlib.sha256((INSTALLED / "oaf_lib.sh").read_bytes()).hexdigest():
        bad.append("OAF_LIB_SHA")
    if p["OAF_CHECKS_SHA"] != hashlib.sha256((INSTALLED / "v8f_oaf_chain_checks.py").read_bytes()).hexdigest():
        bad.append("OAF_CHECKS_SHA")
    if p["BUILD_REF"] != git("-C", REAL_MIRROR, "rev-parse", "1d4f4fbb^{commit}")[1].strip():
        bad.append("BUILD_REF")
    oacc = (INSTALLED / "chain_v8f_dagoacc_r2.sh").read_text(encoding="utf-8")
    po = pins_of(oacc)
    for k in ("CKPT_SHA", "ARGV_SHA", "SUMM_SHA"):
        if p[k] != po[k]:
            bad.append(k + " (vs the DAG-OACC r2 chain's own pins)")
    return bad


def section_D():
    print("== D: the chain offline (syntax, regeneration, pins, dry-run, tip-carries-the-lever, predecessor rule, early stops with zero box-2 calls, wiring)")
    chain_p, guard_p, waker_p, launch_p = (SHIP / "chain_v8f_dagorc.sh", SHIP / "v8f_guard_dagorc.sh", SHIP / "wake_v8fdagorc.sh", SHIP / "run_chain_v8f_dagorc.cmd")
    chain = chain_p.read_bytes().decode("utf-8")
    # ---- D1 syntax / line endings
    for p in (chain_p, guard_p, waker_p):
        rc, o, e = bash(f'bash -n "{msys(p)}"')
        ok(f"D1 bash -n {p.name}", rc == 0, e)
    bad_copy = SCR / "D" / "broken.sh"
    bad_copy.parent.mkdir(parents=True, exist_ok=True)
    bad_copy.write_bytes(chain[: len(chain) // 2].encode() + b"\nif true; then\n")
    refused("D1 a truncated chain fails bash -n (the check is real)", bash(f'bash -n "{msys(bad_copy)}"')[0] != 0)
    for p in (chain_p, guard_p, waker_p, SHIP / "chain_v8f_dagorc.sh.in"):
        ok(f"D2 {p.name}: LF only (a CR in a bash script is a syntax error on box-side bash)", b"\r" not in p.read_bytes())
    lb = launch_p.read_bytes()
    ok("D2 run_chain_v8f_dagorc.cmd: CRLF throughout", lb.count(b"\r\n") == lb.count(b"\n") and lb.endswith(b"\r\n") and b"\x08" not in lb)
    # ---- D3 regeneration
    import make_chain as MC
    fresh = MC.render()
    if os.environ.get("DAGORC_SHIP"):
        ok("D3 skipped (a mutated ship dir: the mutation runner measures the behavioural checks, not the regeneration guard)", True)
    else:
        ok("D3 the chain is byte-for-byte a fresh regeneration from the template + the banked artifacts (make_chain --check)", fresh.encode("utf-8") == chain_p.read_bytes())
    refused("D3 a chain with one byte changed is not the regeneration", (fresh.replace("DAG_STEPS=3000", "DAG_STEPS=3001")).encode() != fresh.encode() and fresh.replace("DAG_STEPS=3000", "DAG_STEPS=3001").encode() != chain_p.read_bytes())
    # ---- D4 pins
    ok("D4 every pinned sha256 / BUILD_REF is the banked artifact's (DAG_s ckpt / argv / summary; DAG2-0 argv / summary; the installed libs; 1d4f4fbb; DAG-OACC r2's own copy of the DAG pins)", not pins_wrong(chain), pins_wrong(chain))
    p0 = pins_of(chain)
    for k in ("CKPT_SHA", "D20_ARGV_SHA", "OAF_LIB_SHA", "OAF_CHECKS_SHA", "BUILD_REF"):
        v = p0[k][0] if isinstance(p0[k], tuple) else p0[k]
        refused(f"D4 a doctored {k} is caught", k in " ".join(pins_wrong(chain.replace(v, ("0" if v[0] != "0" else "1") + v[1:], 1))))
    # ---- D5 guard / launcher / waker
    oacc_g = (INSTALLED / "v8f_guard_dagoacc.sh").read_bytes().decode("utf-8").split("\n")
    mine_g = guard_p.read_bytes().decode("utf-8").split("\n")
    ok("D5 the guard is DAG-OACC's guard line for line except the one comment line", len(oacc_g) == len(mine_g) and [k for k, (a, b) in enumerate(zip(oacc_g, mine_g)) if a != b] == [1], [k for k, (a, b) in enumerate(zip(oacc_g, mine_g)) if a != b][:5])
    ok("D5 the guard's anchored end test knows the markers the chain prints (so it exits when the chain ends)", "'^(ZZV8FMINI|ZZBETA|ZZV8FTAC)-(DONE|STOP)'" in guard_p.read_text())
    ol = (INSTALLED / "run_chain_v8f_dagoacc_r2.cmd").read_bytes().decode("utf-8").split("\r\n")
    nl = lb.decode("utf-8").split("\r\n")
    exp3 = ol[2].replace("chain_v8f_flatdag.out", "chain_v8f_dagoacc_r2.out").replace("chain_v8f_dagoacc_r2.sh", "chain_v8f_dagorc.sh").replace("logs/chain_v8f_dagoacc_r2.out 2>&1", "logs/chain_v8f_dagorc.out 2>&1")
    ok("D5 the launcher is the DAG-OACC r2 launcher with the names changed: AFTER_OUT = the DAG-OACC r2 chain's log, own .sh, own .out, stdin from NUL", nl[2] == exp3 and nl[0] == "@echo off" and nl[2].startswith('"C:\\Program Files\\Git\\bin\\bash.exe" -lc "AFTER_OUT=/c/Users/Admin/box2_ship/logs/chain_v8f_dagoacc_r2.out '), nl[2])
    wk = waker_p.read_text()
    outname = "chain_v8f_dagorc.out"
    ok("D5 the waker reads the chain's own .out and the guard's .out the chain writes", f"O=$L/{outname}" in wk and "v8fdagorc_guard.out" in wk and "GUARD_OUT=$L/v8fdagorc_guard.out" in chain and f"logs/{outname} 2>&1" in nl[2])
    ok("D5 the waker's markers are the ones the chain / asserter print (ZZDAGORC-CONTROL-FAIL, ZZDISK-FLOOR-STOP, `gate token written for` x4, ` pulled to `)",
       all(x in wk for x in ("'^ZZDAGORC-CONTROL-FAIL\\|^ZZDISK-FLOOR-STOP'", "ZZDISK-FLOOR-STOP", "^gate token written for", "-ge 4", " pulled to ")) and 'print("gate token written for"' in (SHIP / "gate_dagorc_assert.py").read_text() and "ZZDAGORC-CONTROL-FAIL" in chain and " pulled to $ART/$g" in chain)
    # ---- D6 wiring: order and presence of the decisions in the chain's text
    def _d6():
        iA, iB = chain.index('for run in "${AB_RUNS[@]}"; do set -- $run; gate_one'), chain.index('for run in "${AB_RUNS[@]}"; do set -- $run; runone')
        iS, iBg, iBr = chain.index('if [ "${DAGORC_SKIP_B:-0}" != 1 ]; then'), chain.index('for run in "${B_RUNS[@]}"; do set -- $run; gate_one'), chain.index('for run in "${B_RUNS[@]}"; do set -- $run; runone')
        ok("D6 order: the four A / REP gates, then their four runs, then (unless DAGORC_SKIP_B=1) B's gates and runs", iA < iB < iS < iBg < iBr < chain.index('pinned_b "after the last run"') < chain.index("finish "))
        ok("D6 the run order is PREREG section 10's: A s0, REP s0, A s1, REP s1, then B s0, B s1", 'AB_RUNS=("dagorc 0" "dagorcrep 0" "dagorc 1" "dagorcrep 1")' in chain and 'B_RUNS=("dagorc01 0" "dagorc01 1")' in chain)
        rt = chain[chain.index("runone() {"):chain.index("for run in \"${AB_RUNS[@]}\"; do set -- $run; gate_one")]
        ok("D6 runone: train -> ckpt hash -> eval -> hash again -> pull -> run record -> K-REACH, each with a hard stop",
           [rt.index(x) for x in ("queue \"t_$g\"", "ckptsha_x $arm $s trained", "queue \"e_$g\"", "ckptsha_x $arm $s after_evals", "pull \"$g/$f\"", "run-record", "reached")] == sorted(rt.index(x) for x in ("queue \"t_$g\"", "ckptsha_x $arm $s trained", "queue \"e_$g\"", "ckptsha_x $arm $s after_evals", "pull \"$g/$f\"", "run-record", "reached"))
           and "|| stop \"run DAG-ORC record $g\"" in rt and "ZZDAGORC-CONTROL-FAIL" in rt and "|| { echo \"ZZDAGORC-CONTROL-FAIL" in rt)
        gt = chain[chain.index("gate_one() {"):chain.index("ckptsha_x() {")]
        ok("D6 gate_one: the queue's gate -> pull -> the asserter's `check` (hard stop) -> only then the token", gt.index('queue "gate_$g"') < gt.index('$ASSERT)" check') < gt.index('$ASSERT)" token') and '|| stop "gate DAG-ORC assertions $g"' in gt and '|| stop "write the gate token $g"' in gt and "--ref-argv-json" in gt and "--b-summary" in gt)
        ok("D6 disk floors: the START floor precedes the wait, the re-check follows it, and pulls go through the floor", chain.index('disk_floor "$ART" "$NEED_BYTES" "the six DAG-ORC runs"') < chain.index("\nwait_predecessor\n") < chain.index("re-checked after the wait"))
        ok("D6 BOX2_BUSY is taken atomically (noclobber) AFTER the wait, and nothing is created on box 2's behalf before it (mkdir of $E only after the busy marker)", "( set -C; echo" in chain and chain.index("\nwait_predecessor\n") < chain.index("( set -C;") < chain.index("mkdir -p $E/runs $E/lines"))
        ok("D6 the refresh refuses a tip without the stages / levers BEFORE anything is shipped to box 2", chain.index("lever_check git") < chain.index("git -C $MIRROR -c core.autocrlf=false archive") < chain.index("$SCP $B/repo_v8fdagorc_tip.tar") and chain.index("for st in \"${STAGES[@]}\"") < chain.index("lever_check git"))
        ok("D6 the lever list names the loop's ego.apply, the probe call, the summary counter and the queue lines", all(x in chain for x in ("^ +b = ego\\.apply\\(b, step\\)", "egop = ego_perturb_probe", "\"ego_perturb_n\": ego_tot", "^:dagorcarm", "goto :dagorcarm", "--ego-perturb-p %DAGORCP%%DAGORCL% --dagger-p 0", "DAGSEED=1%~2")))
        ok("D6 nothing in the chain hard-codes C:/Users/Admin/box2_ship (every path comes from OAF_SHIP_DIR / the chain's own dir) except the default", chain.count("C:/Users/Admin/box2_ship") == 0 and chain.count("/c/Users/Admin/box2_ship") == 1)
    try:
        _d6()
    except (ValueError, TypeError) as e:      # an anchor the wiring pins rely on is gone from the chain text: that IS the finding
        ok("D6 wiring: every anchor the order / presence pins rely on is present in the chain text", False, repr(e))

    # ---- D7 dry-run
    qf = fwd(L.QUEUE)
    w = make_world("D_dry", mirror=False, skip_b=False)
    w.env["V8F_QUEUE"] = qf      # the dry-run reads the queue from V8F_QUEUE (without it the chain would ask the REMOTE for its tip: never in a test)
    rc, out, calls = run_chain(w, "--dry-run")
    ok("D7 --dry-run on the staged queue: rc 0, every run's planned line is ONE-LEVER and gate==train: 6 runs in PREREG order", rc == 0 and out.count("ONE-LEVER") == 6 + 0 and out.count(": SAME") == 6 and "== runs (order): dagorc 0 dagorcrep 0 dagorc 1 dagorcrep 1 dagorc01 0 dagorc01 1" in out and not stop_lines(out), out[-600:])
    ok("D7 --dry-run touched nothing: no box-2 call, no BOX2_BUSY.txt", calls == "" and not (w.ship / "BOX2_BUSY.txt").exists())
    for arm in ARMS:
        for s in (0, 1):
            tl = lines_of(arm, s)[1]
            ok(f"D7 --dry-run prints the queue's own training line for {arm} s{s}", tl in out)
    w.env["DAGORC_SKIP_B"] = "1"
    rc, out4, _ = run_chain(w, "--dry-run")
    ok("D7 DAGORC_SKIP_B=1 drops B: 4 runs, no dagorc01", rc == 0 and out4.count("ONE-LEVER") == 4 and "dagorc01" not in out4.split("== runs (order):")[1].split("\n")[0])
    # negatives: a queue that is NOT the registered one
    qb = L.QUEUE.read_bytes()

    def dryq(name, data, exp_in):
        f = SCR / "D" / f"{name}.cmd"
        f.write_bytes(data)
        w.env["V8F_QUEUE"] = fwd(f)
        r, o, c = run_chain(w, "--dry-run")
        return r != 0 and bool(stop_lines(o)) and exp_in in o, o[-300:]

    for nm, old, new, why in (("law_vmax", b"--ego-perturb-vmax 8.0", b"--ego-perturb-vmax 7.0", "the planned lines are not the one-lever"),
                              ("p_wrong", b'set "DAGORCP=0.3"', b'set "DAGORCP=0.4"', "the planned lines are not the one-lever"),
                              ("rep_seed", b'set "DAGSEED=1%~2"', b'set "DAGSEED=%~2"', "the planned lines are not the one-lever"),
                              ("dagger_p", b"--ego-perturb-p %DAGORCP%%DAGORCL% --dagger-p 0", b"--ego-perturb-p %DAGORCP%%DAGORCL% --dagger-p 0.1", "the planned lines are not the one-lever"),
                              ("steps", b"--steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p", b"--steps 3001 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p", "the planned lines are not the one-lever")):
        if qb.count(old) < 1:
            raise HarnessError(f"D7 anchor {nm}")
        r, d = dryq(nm, qb.replace(old, new), why)
        refused(f"D7 --dry-run refuses a queue where {nm} is changed", r, d)
    r, d = dryq("tipq", L.TIP_QUEUE.read_bytes(), "prints no gate / train / eval line")
    refused("D7 --dry-run refuses the tip queue (no DAG-ORC stages)", r, d)

    # ---- D8 --check-tree
    mt = SCR / "D" / "tree"
    if mt.exists():
        shutil.rmtree(mt)
    files = {"stack/scripts/v8f_mini_train.py": TREE / "stack/scripts/v8f_mini_train.py", "stack/tanitad/data/v8f_mini_data.py": TREE / "stack/tanitad/data/v8f_mini_data.py", f"{L.PK}/code/v8f_mini_box2_queue.cmd": L.QUEUE}
    for rel, src in files.items():
        (mt / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, mt / rel)

    def ctree(root):
        e = {**os.environ, **w.env, "V8F_QUEUE": ""}
        script = f'cd "{msys(w.ship)}" && bash chain_v8f_dagorc.sh --check-tree "{msys(root)}"'
        return sh([BASH, "-c", script], env=e)

    rc, o, _ = ctree(mt)
    ok("D8 --check-tree on the composed tip + staged queue: every lever pattern is present (rc 0)", rc == 0 and "every one of" in o, o[-300:])

    def mtree(name, rel, old, new):
        d = SCR / "D" / name
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(mt, d)
        p = d / rel
        t = p.read_bytes()
        if t.count(old) < 1:
            raise HarnessError(f"D8 anchor {name}: {old!r}")
        p.write_bytes(t.replace(old, new))
        return ctree(d)

    tr = "stack/scripts/v8f_mini_train.py"
    for nm, rel, old, new, pat in (("noapply", tr, b"b = ego.apply(b, step)", b"b = b", "ego\\.apply"), ("noprobe", tr, b"egop = ego_perturb_probe(model, probe, cfg, ego)", b"egop = None", "ego_perturb_probe"),
                                   ("noseedflag", tr, b'add_argument("--ego-perturb-seed"', b'add_argument("--ego-perturb-sead"', "ego-perturb-seed"), ("nocount", tr, b'"ego_perturb_n": ego_tot["n"]', b'"ego_perturb_n": 0', "ego_perturb_n"),
                                   ("noinit", tr, b'add_argument("--init-ckpt"', b'add_argument("--init-ckp"', "init-ckpt"), ("noclass", "stack/tanitad/data/v8f_mini_data.py", b"\nclass EgoPerturb", b"\nclass XEgoPerturb", "EgoPerturb"),
                                   ("noorcarm", f"{L.PK}/code/v8f_mini_box2_queue.cmd", b"\r\n:dagorcarm\r\nrem PREREG", b"\r\n:dagorcarX\r\nrem PREREG", "dagorcarm")):
        rc, o, _ = mtree(nm, rel, old, new)
        refused(f"D8 --check-tree refuses a tree where {nm}", rc == 1 and "MISSING" in o and pat in o, o[-300:])
    rc, o, _ = ctree(SCR / "D" / "no_such_tree")
    refused("D8 --check-tree on a directory that does not exist is a usage error", rc != 0)
    # the tip itself (3b7b404e) has the trainer levers but not the queue stages
    tipq = SCR / "D" / "tree_tip"
    if tipq.exists():
        shutil.rmtree(tipq)
    shutil.copytree(mt, tipq)
    shutil.copyfile(L.TIP_QUEUE, tipq / L.PK / "code" / "v8f_mini_box2_queue.cmd")
    rc, o, _ = ctree(tipq)
    refused("D8 --check-tree refuses the repo tip as it is TODAY (queue without the stages): the chain would not start until they land", rc == 1 and o.count("MISSING") >= 6, o[-200:])

    # ---- D9 the predecessor rule through --wait-only
    _wid = [0]

    def wait(pred, busy=False, env=None, timeout_s=None, unset=False):
        _wid[0] += 1
        w2 = make_world(f"D_wait{_wid[0]}", mirror=False, pred=pred, env=env)
        if busy:
            (w2.ship / "BOX2_BUSY.txt").write_text("held by someone\n")
        if unset:
            w2.env["AFTER_OUT"] = None
        if pred == "missing":
            (w2.base / "pred.out").unlink()
        if timeout_s:
            w2.env["OAF_POLL_S"] = "1"
        script = f'export PATH="{msys(w2.bin)}:$PATH"; cd "{msys(w2.ship)}" && ' + (f"timeout {timeout_s} " if timeout_s else "") + "bash chain_v8f_dagorc.sh --wait-only"
        e = {**os.environ, **{k: v for k, v in w2.env.items() if v is not None}}
        for k in [k for k, v in w2.env.items() if v is None]:
            e.pop(k, None)
        r, o, _ = sh([BASH, "-c", script], env=e, timeout=120)
        return r, o.replace("\r", ""), w2

    PRED["missing"] = ""
    r, o, _ = wait("done")
    ok("D9 predecessor ZZV8FMINI-DONE, BOX2_BUSY released -> GO", r == 0 and "ZZOAF-WAIT-GO" in o, o)
    r, o, _ = wait("stop")
    ok("D9 predecessor STOP of its own, BOX2_BUSY released -> GO (a stopped predecessor must not idle box 2)", r == 0 and "ZZOAF-WAIT-GO" in o and "STOPPED on its own" in o, o)
    r, o, _ = wait("guard")
    refused("D9 predecessor STOP with a ZZGUARD-STOP line (the box may be hot) -> HALT, no GO", r == 1 and "ZZOAF-WAIT-GO" not in o and "GUARD" in o, o)
    r, o, _ = wait("done", busy=True)
    refused("D9 predecessor DONE but BOX2_BUSY still held -> HALT", r == 1 and "ZZOAF-WAIT-GO" not in o and "still there" in o, o)
    r, o, _ = wait("stop", busy=True)
    refused("D9 predecessor STOP with BOX2_BUSY held -> BUSY three times, then HALT (never GO)", r == 1 and "ZZOAF-WAIT-GO" not in o and "still held" in o, o)
    for nm, pr in (("no end marker yet", "none"), ("an echoed `waiting for ^ZZV8FMINI-DONE` line (the marker is ANCHORED)", "echo"), ("the predecessor's log does not exist", "missing")):
        r, o, _ = wait(pr, timeout_s=6)
        refused(f"D9 {nm} -> WAIT (never GO)", r == 124 and "ZZOAF-WAIT-GO" not in o, f"rc {r} {o}")
    r, o, _ = wait("done", unset=True)
    refused("D9 AFTER_OUT unset -> STOP (no default predecessor log: a default can already be satisfied by an OLD log)", r == 1 and "AFTER_OUT is not set" in o, o)

    # ---- D10 early stops: a named STOP and ZERO box-2 calls, no busy marker, nothing written to the artifacts
    def early(name, exp, **kw):
        ed = kw.pop("edit", None)
        keep_busy = kw.pop("keep_busy", False)
        w3 = make_world(name, mirror=False, **kw, edit=ed)
        r, o, c = run_chain(w3, timeout=90)
        stops = stop_lines(o)
        good = r == 1 and any(exp in x for x in stops + [o]) and c == "" and (keep_busy or not (w3.ship / "BOX2_BUSY.txt").exists()) and not (w3.art / "K-BUILD.txt").exists()
        return good, f"rc {r} stops {stops} calls {c[:200]!r}"

    g, d = early("D_noafter", "AFTER_OUT is not set", env={"AFTER_OUT": None})
    refused("D10 AFTER_OUT unset: STOP before anything is touched", g, d)
    g, d = early("D_nofiles", "are not installed beside the chain", drop=("gate_dagorc_assert.py",))
    refused("D10 the asserter is not installed beside the chain: STOP before anything is touched", g, d)
    g, d = early("D_nochecks", "are not installed beside the chain", drop=("v8f_dagorc_chain_checks.py",))
    refused("D10 the run-record / K-REACH helper is not installed: STOP before anything is touched", g, d)

    def corrupt(fn):
        return lambda w: (w.ship / fn).write_bytes((w.ship / fn).read_bytes() + b"\n# changed\n")

    g, d = early("D_badlib", "oaf_lib.sh is not the proven file", edit=corrupt("oaf_lib.sh"))
    refused("D10 oaf_lib.sh differs from the proven file: STOP (a changed lib is a changed chain)", g, d)
    g, d = early("D_badchecks", "is not the proven file", edit=corrupt("v8f_oaf_chain_checks.py"))
    refused("D10 v8f_oaf_chain_checks.py differs from the proven file: STOP", g, d)
    g, d = early("D_floor", "ZZDISK-FLOOR-STOP", env={"OAF_TEST_FREE_BYTES": "1000"})
    refused("D10 the disk floor: E: could not take the six runs above 1 GiB -> ZZDISK-FLOOR-STOP before the wait", g, d)
    g, d = early("D_noart", "have nowhere to go", env={"DAGORC_ART": "/f/TanitAD-scratch/dagorc_work/no_such_parent/art"})
    refused("D10 the artifact directory's parent does not exist -> STOP", g, d)
    g, d = early("D_guardhalt", "predecessor verdict: HALT", pred="guard")
    refused("D10 the predecessor was stopped by the guard -> HALT before the busy marker is taken", g, d)
    g, d = early("D_busydone", "predecessor verdict: HALT", pred="done", keep_busy=True, edit=lambda w: (w.ship / "BOX2_BUSY.txt").write_text("someone else\n"))
    w3 = SCR / "E" / "D_busydone"
    ok("D10 a BOX2_BUSY.txt that is not ours is never removed by a chain that did not take it", (w3 / "ship" / "BOX2_BUSY.txt").exists() and (w3 / "ship" / "BOX2_BUSY.txt").read_text() == "someone else\n")
    ok("D10 the installed box2_ship has no DAG-ORC file (nothing here installs anything)", not [p for p in INSTALLED.glob("*dagorc*")], [p.name for p in INSTALLED.glob("*dagorc*")])


# ============================================================================================================================ E
def nstage(calls, st):
    return len([x for x in calls.split("\n") if x.startswith(f"stage {st} ")])


def stage_seq(calls):
    return [x[len("stage "):] for x in calls.split("\n") if x.startswith("stage ")]


def verify_sums(d):
    """every line of d/SHA256SUMS.txt names a file in d whose sha256 is the one listed"""
    sf = Path(d) / "SHA256SUMS.txt"
    if not sf.exists():
        return False
    rows = [x.split(None, 1) for x in sf.read_text().splitlines() if x.strip()]
    return bool(rows) and all((Path(d) / n.strip().lstrip("*")).exists() and hashlib.sha256((Path(d) / n.strip().lstrip("*")).read_bytes()).hexdigest() == h for h, n in rows)


def section_E():
    print("== E: the chain's REAL run against a fake box 2 (ssh/scp shims over a local tree, a fake git mirror = the real tip + the staged queue); the REAL queue's lines, the REAL gate asserter / run record / K-REACH")
    inst_before = sorted(p.name for p in INSTALLED.glob("*dagorc*"))
    specs = []

    def add(name, check, **kw):
        specs.append((name, check, kw))

    # ---- the happy path: all six runs
    def happy(w, rc, out, calls):
        seq = stage_seq(calls)
        order = [f"gatedag {a} {s}" for a, s in (("dagorc", 0), ("dagorcrep", 0), ("dagorc", 1), ("dagorcrep", 1))]
        ok("E1 the chain ends rc 0 with ZZV8FMINI-DONE and releases BOX2_BUSY", rc == 0 and any(x.startswith("ZZV8FMINI-DONE") for x in out.split("\n")) and not (w.ship / "BOX2_BUSY.txt").exists(), out[-500:])
        ok("E1 the four A / REP gates run FIRST, in order, before any training", seq[:4] == order, seq[:6])
        run_order = []
        for a, s in (("dagorc", 0), ("dagorcrep", 0), ("dagorc", 1), ("dagorcrep", 1)):
            run_order += [f"traindag {a} {s}", f"evaldag {a} {s}"]
        ok("E1 then train -> eval per run in PREREG order A s0, REP s0, A s1, REP s1", seq[4:12] == run_order, seq[4:12])
        ok("E1 B's two gates and two runs come AFTER the four A / REP runs", seq[12:14] == ["gatedag dagorc01 0", "gatedag dagorc01 1"] and seq[14:18] == ["traindag dagorc01 0", "evaldag dagorc01 0", "traindag dagorc01 1", "evaldag dagorc01 1"] and len(seq) == 18, seq[12:])
        ok("E1 the stages each ran exactly once (nothing retrained, nothing re-gated)", len(seq) == len(set(seq)) == 18)
        runs = [f"M2b_s{s}_{a}" for a, s in (("dagorc", 0), ("dagorcrep", 0), ("dagorc", 1), ("dagorcrep", 1), ("dagorc01", 0), ("dagorc01", 1))]
        ok("E1 every run is pulled to the artifact dir with a SHA256SUMS.txt whose every line verifies", all(verify_sums(w.art / r) for r in runs), [r for r in runs if not verify_sums(w.art / r)])
        ok("E1 the chain prints one `pulled to` line, one PASS run record and one PASS K-REACH per run", len([x for x in out.split(chr(10)) if " pulled to " in x and not x.startswith("ZZ")]) == 6 and out.count("DAG-ORC run record: PASS") == 6 and out.count("DAG-ORC perturbation-reached: PASS") == 6 and out.count("gate token written for") == 6 and out.count("DAG-ORC gate: PASS") >= 6, (len([x for x in out.split(chr(10)) if " pulled to " in x and not x.startswith("ZZ")]), out.count("DAG-ORC run record: PASS"), out.count("perturbation-reached: PASS"), out.count("gate token written for")))
        toks = {}
        for r in runs:
            f = w.art / r / "launch_gate_token.json"
            toks[r] = json.loads(f.read_text()) if f.exists() else {}
        ok("E1 every run has a launch-gate token bound to the remote tip, the gate verdict PASS, and the training argv's sha256", all(t.get("verdict") == "PASS" and t.get("binding", {}).get("commit") == w.tip_new and len(t.get("binding", {}).get("argv_sha256", "")) == 64 for t in toks.values()), [k for k, t in toks.items() if t.get("binding", {}).get("commit") != w.tip_new])
        ok("E1 a token exists only AFTER its gate passed: the gates' stage calls all precede every training stage and each token file is newer than its gate selfcheck", all((w.art / r / "launch_gate_token.json").exists() and (w.art / r / "gate" / "selfcheck.json").exists() and (w.art / r / "launch_gate_token.json").stat().st_mtime >= (w.art / r / "gate" / "selfcheck.json").stat().st_mtime for r in runs))
        kb = (w.art / "K-BUILD.txt").read_text(encoding="utf-8", errors="replace")
        ok("E1 K-BUILD is recorded against 1d4f4fbb: data and model blobs IDENTICAL, the trainer CHANGED (the F1 hunk) with its diff", kb.count(" IDENTICAL") == 2 and kb.count(" CHANGED") == 1 and "v8f_mini_train.py" in kb and "diff --git" in kb, kb[:600])
        ok("E1 K-REACH numbers printed: A / B at their p, REP exactly 0", "| dagorcrep | ego_perturb_n 0 (must be exactly 0)" in out, out[-300:])
        ok("E1 the tip the chain exported is the remote's dagorc-new (the staged stages), not the local mirror's old ref", f"tip {w.tip_new}" in out or w.tip_new in out, out[-300:])
        ok("E1 the guard ran against the fake box and exited when the chain ended (its log says so)", "chain ended; guard off" in (w.ship / "logs" / "v8fdagorc_guard.out").read_text(errors="replace") if (w.ship / "logs" / "v8fdagorc_guard.out").exists() else False)
        ok("E1 the checkpoints were hashed on box 2 after training and after the evals (equal), and the pulled ones equal them", all(k in (w.ship / "logs" / "v8f_dagorc" / "ckpt_sha256.json").read_text() for k in ("M2b-dagorc_s0_trained", "M2b-dagorc01_s1_after_evals")))

    add("E_happy", happy, skip_b=False)

    def skipb(w, rc, out, calls):
        seq = stage_seq(calls)
        ok("E1b DAGORC_SKIP_B=1: a complete chain of exactly the four A / REP runs (4 gates + 4 x train/eval = 12 stages), no dagorc01 anywhere, ZZV8FMINI-DONE",
           rc == 0 and len(seq) == 12 and not any("dagorc01" in x for x in seq) and any(x.startswith("ZZV8FMINI-DONE") for x in out.split(chr(10))) and not (w.art / "M2b_s0_dagorc01").exists(), (rc, seq))

    add("E_skipb", skipb, skip_b=True)

    # ---- faults: each must STOP, name the stop, and leave the later runs UNTRAINED
    def stopped(name, expect, **cnt):
        def chk(w, rc, out, calls):
            st = stop_lines(out)
            hit = any(expect in x for x in st) or any(expect in x for x in out.split("\n"))
            good = rc == 1 and bool(st) and hit and not (w.ship / "BOX2_BUSY.txt").exists() and not any(x.startswith("ZZV8FMINI-DONE") for x in out.split("\n"))
            for k, v in cnt.items():
                good = good and (nstage(calls, k) == v)
            refused(f"E2 {name}", good, f"rc {rc} stops {st} counts { {k: nstage(calls, k) for k in cnt} } want {cnt}")
        return chk

    add("E_gate_init", stopped("a gate whose init checkpoint is NOT DAG_s0's: STOP at that gate; no training started, no token for it", "gate DAG-ORC assertions M2b_s0_dagorc", gatedag=1, traindag=0), faults="gate_init:dagorc:0")
    add("E_gate_probe", stopped("a gate whose ego probe moves no gradient (A s1): the first two gates pass, the third STOPs; no training started", "gate DAG-ORC assertions M2b_s1_dagorc", gatedag=3, traindag=0), faults="gate_probe:dagorc:1")
    add("E_rep_applied", stopped("REP s0's run APPLIED the perturbation (ego_perturb_n 123): ZZDAGORC-CONTROL-FAIL after REP s0, the s1 pair never trains", "reached-control fail M2b_s0_dagorcrep", traindag=2), faults="rep_applied:dagorcrep:0")
    add("E_reach0", stopped("A s0's loop never applied the perturbation (ego_perturb_n 0 although the gate passed): ZZDAGORC-CONTROL-FAIL, nothing after it trains", "reached-control fail M2b_s0_dagorc", traindag=1), faults="reach0:dagorc:0")
    add("E_argv", stopped("A s0's argv.json is not the planned training argv: the run record STOPs", "run DAG-ORC record M2b_s0_dagorc", traindag=1), faults="argv_mismatch:dagorc:0")
    add("E_jobrc", stopped("A s0's training job exits rc 1: STOP, no eval, nothing after it", "rc=1", traindag=1, evaldag=0), faults="job_rc1:dagorc:0")
    add("E_ckpt1", stopped("box 2's DAG_s1 checkpoint is not the banked one: STOP before any gate", "before the gates", gatedag=0, traindag=0), faults="bad_sha:M2b_s1_dag/ckpt_last.pt:0")
    add("E_md5", stopped("a v9 release whose md5 is not the pinned one: STOP before any gate", "v9_labels_train.npz md5", gatedag=0), faults="bad_sha:v9_labels_train.npz:0")
    add("E_keysha", stopped("box 2's trainer is not the tip's: STOP before any gate", "box 2 stack/scripts/v8f_mini_train.py != tip", gatedag=0), faults="bad_sha:stack/scripts/v8f_mini_train.py:0")

    add("E_pullcorrupt", stopped("a pull that lands A s0's checkpoint with the right size and one byte wrong: STOP at the pulled-checkpoint hash, nothing after it trains", "the pulled checkpoint's sha256", traindag=1, evaldag=1), faults="pull_corrupt:M2b_s0_dagorc/ckpt_last.pt:0")
    add("E_evaltouch", stopped("an eval that rewrites the checkpoint it evaluated (hashed after training and again after the evals): STOP", "checkpoint changed during the evals", traindag=1, evaldag=1), faults="eval_touch:dagorc:0")

    def nojobs(name, expect):
        def chk(w, rc, out, calls):
            st = stop_lines(out)
            good = rc == 1 and any(expect in x for x in st) and not (w.ship / "BOX2_BUSY.txt").exists() and "launch.ps1" not in calls and "\njob " not in "\n" + calls
            refused(f"E3 {name}", good, f"rc {rc} stops {st} calls {calls[:300]!r}")
        return chk

    add("E_oldtip", nojobs("the remote tip WITHOUT the queue stages (dagorc-old = 3b7b404e): the chain refuses at the refresh, ships nothing, launches no job, releases BOX2_BUSY", "the tip's queue has no dagorcarm stage"), br="dagorc-old")
    add("E_noapply", nojobs("a remote tip with the stages but WITHOUT the trainer's `b = ego.apply(b, step)` loop line: the chain refuses at the refresh (the lever is declared and gated and never applied)", "does not carry the DAG-ORC queue stages / the perturbation levers"), br="dagorc-noapply")

    only = [x for x in os.environ.get("DAGORC_E_ONLY", "").split(",") if x]
    if only:
        specs = [s for s in specs if s[0] in only]
        print("-- DAGORC_E_ONLY:", [s[0] for s in specs])
    worlds = [(n, c, make_world(n, **k)) for n, c, k in specs]

    def go(item):
        n, c, w = item
        t0 = time.time()
        rc, out, calls = run_chain(w, timeout=1700)
        return n, c, w, rc, out, calls, time.time() - t0

    with ThreadPoolExecutor(max_workers=int(os.environ.get("DAGORC_E_PAR", "3"))) as ex:
        results = list(ex.map(go, worlds))
    for n, c, w, rc, out, calls, dt in results:
        print(f"-- {n}: rc {rc}, {dt:.0f} s, {len(calls.splitlines())} fake-box calls")
        c(w, rc, out, calls)
    ok("E9 no DAG-ORC file appeared in the installed box2_ship during the whole section (the harness wrote only under its own scratch)", sorted(p.name for p in INSTALLED.glob("*dagorc*")) == inst_before)
    return results


# ============================================================================================================================ main
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="S,A,B,C,D,E")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    t0 = time.time()
    fn = {"S": section_S, "A": section_A, "B": section_B, "C": section_C, "D": section_D, "E": section_E}
    for s in a.sections.split(","):
        try:
            fn[s.strip()]()
        except L.QueueStageError as e:      # a stage of the QUEUE UNDER TEST failed to print its line: the queue's defect, a RED check
            ok(f"{s.strip()} the queue under test prints every stage line the section needs", False, str(e)[:300])
        except HarnessError as e:      # the rig could not be built: neither green nor red, exit 2
            print("HARNESS ERROR:", e)
            return 2
    nc, nr = len(RES["checks"]), len(RES["regressions"])
    pc, cr = sum(1 for x in RES["checks"] if x["ok"]), sum(1 for x in RES["regressions"] if x["caught"])
    red = [x["name"] for x in RES["checks"] if not x["ok"]] + [x["name"] for x in RES["regressions"] if not x["caught"]]
    print(f"RESULT: {pc}/{nc} checks pass, {cr}/{nr} deliberate regressions caught, {nr - cr} missed  ({time.time() - t0:.0f} s)  sections {a.sections}")
    if red:
        print("RED:", *red, sep="\n  ")
    if a.json:
        Path(a.json).write_text(json.dumps({"sections": a.sections, "checks": RES["checks"], "regressions": RES["regressions"], "pass": not red}, indent=1), encoding="utf-8")
    return 0 if not red else 1


if __name__ == "__main__":
    sys.exit(main())
