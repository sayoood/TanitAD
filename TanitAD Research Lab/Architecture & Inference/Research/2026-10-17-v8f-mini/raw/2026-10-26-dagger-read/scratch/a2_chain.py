"""Detached closer for the A2 pass (BANK-CONFIG follows the generator's --limit-clips convention). No model turns. Run from anywhere; everything is absolute.

R77 snapshot of the deliverables -> wait for a pytest slot (machine cap 2, polled every 60 s) -> TARGETED tests (fast fail) -> full suite -> full mutation proof on COPIES
-> byte-identity re-check -> evidence appended to NOTES.md.
Markers (all under out/, which holds ONLY this pass; the previous pass is out_pre_a2/): chain_log.txt, pytest_targeted.out, pytest_full.out, mutation.out, mutation_console.out,
snap_before_SHA256.txt, snap_after_check.out, final_SHA256.txt, finish_a2.done (success path ran to the end), chain_failed_a2 (a stage failed: the reason is inside).
"""
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
import time

HERE = pathlib.Path("C:/Users/Admin/dagread")
OUT = HERE / "out"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
RUNS = "C:/Users/Admin/orcread/pytest_runs.py"
BT_ROOT = pathlib.Path("E:/Projects/TanitAD-artifacts/dagread_build/pt")
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
TARGET_K = "limit_clips or bank_config or real_generator_smoke or bank_accounting or bank_stage_names"
os.chdir(HERE)
OUT.mkdir(exist_ok=True)


def stamp(msg):
    with open(OUT / "chain_log.txt", "a", encoding="utf-8", newline="\n") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def fail(msg):
    stamp("FAILED " + msg)
    (OUT / "chain_failed_a2").write_text(msg + "\n", encoding="utf-8")
    sys.exit(1)


def n_pytest():
    r = subprocess.run([PY, RUNS], capture_output=True, text=True, env=ENV)
    try:
        return int(r.stdout.strip())
    except ValueError:
        return 99


def wait_slot():
    while n_pytest() >= 2:
        time.sleep(60)


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def deliverables():
    fx = sorted(str(p.relative_to(HERE)).replace("\\", "/") for d in ("fixtures/dagger_smoke", "fixtures/trainer_kb45", "fixtures/generator_a2") for p in (HERE / d).iterdir() if p.is_file())
    return ["dag_read.py", "test_dag_read.py", "mutate_dag.py", "bank_audit.py", "PREREG_V8F_DAGGER.md"] + fx


def pytest_run(tag, out_name, extra):
    bt = BT_ROOT / tag
    shutil.rmtree(bt, ignore_errors=True)
    with open(OUT / out_name, "w", encoding="utf-8", newline="\n") as f:
        r = subprocess.run([PY, "-m", "pytest", "-v", "-p", "no:cacheprovider", f"--basetemp={bt}", "test_dag_read.py", *extra], stdout=f, stderr=subprocess.STDOUT, env=ENV)
    shutil.rmtree(bt, ignore_errors=True)                                   # the synthetic worlds write ~0.4 GB of small files per run
    return r.returncode


for f in ("chain_failed_a2", "finish_a2.done"):
    (OUT / f).unlink(missing_ok=True)
stamp("chain start (A2)")
FILES = deliverables()
snap = {f: sha(f) for f in FILES}
(OUT / "snap_before_SHA256.txt").write_text("".join(f"{h} *{f}\n" for f, h in snap.items()), encoding="utf-8", newline="\n")      # R77: BEFORE the suite and the proof
stamp(f"R77 snapshot of {len(FILES)} files written")

wait_slot()
stamp("pytest slot free; targeted tests start")
rc = pytest_run("a2_targeted", "pytest_targeted.out", ["-k", TARGET_K])
stamp(f"targeted tests rc={rc}")
if rc != 0:
    fail(f"targeted pytest rc={rc}: read out/pytest_targeted.out")

wait_slot()
stamp("pytest slot free; full suite starts")
rc = pytest_run("a2_full", "pytest_full.out", [])
stamp(f"full suite rc={rc}")
if rc != 0:
    fail(f"pytest rc={rc}: read out/pytest_full.out")

(OUT / "mutation_started.txt").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")
with open(OUT / "mutation_console.out", "w", encoding="utf-8", newline="\n") as f:
    r = subprocess.run([PY, "mutate_dag.py", "--out", str(OUT / "mutation.out")], stdout=f, stderr=subprocess.STDOUT, env=ENV)
    f.write(f"mutation rc={r.returncode}\n")
stamp(f"mutation proof rc={r.returncode}")

after, ok = {f: sha(f) for f in deliverables()}, True
lines = []
for f in sorted(set(snap) | set(after)):
    good = snap.get(f) == after.get(f)
    ok &= good
    lines.append(f"{f}: {'OK' if good else 'FAILED'}")
(OUT / "snap_after_check.out").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
stamp(f"R77 re-check ok={ok}")

pyt = (OUT / "pytest_full.out").read_text(encoding="utf-8").strip().splitlines()[-1]
mut = (OUT / "mutation.out").read_text(encoding="utf-8").strip().splitlines()[-1] if (OUT / "mutation.out").exists() else "(no mutation.out)"
note = f"""
## 12. Evidence of the A2 pass (written by scratch/a2_chain.py after the mutation run)
* `out/pytest_targeted.out`: the targeted tests (`-k "{TARGET_K}"`) were green before the full suite started.
* `out/pytest_full.out`: {pyt}
* `out/mutation.out` (live log `out/mutation_console.out`): {mut}
* R77: {"the deliverables (dag_read.py, test_dag_read.py, mutate_dag.py, bank_audit.py, the landed PREREG copy, fixtures/dagger_smoke/*, fixtures/trainer_kb45/*, fixtures/generator_a2/*) are BYTE-IDENTICAL before the suite and after the mutation run (`out/snap_before_SHA256.txt` vs `out/snap_after_check.out`; mutants ran on copies under E:/Projects/TanitAD-artifacts/dagread_build/mut)." if ok else "MISMATCH after the mutation run, see `out/snap_after_check.out` -- do not land."}
* sha256 of every deliverable and NOTES.md after the run: `out/final_SHA256.txt`. The previous pass's evidence is untouched in `out_pre_a2/`.
"""
with open(HERE / "NOTES.md", "a", encoding="utf-8", newline="\n") as f:
    f.write(note)
final = deliverables() + ["NOTES.md"]
(OUT / "final_SHA256.txt").write_text("".join(f"{sha(f)} *{f}\n" for f in final), encoding="utf-8", newline="\n")
(OUT / "finish_a2.done").write_text(f"snap_ok={ok} mutation_rc={r.returncode}\n{pyt}\n{mut}\n", encoding="utf-8")
if r.returncode != 0 or not ok:
    (OUT / "chain_failed_a2").write_text(f"mutation rc={r.returncode} snap_ok={ok}: read out/mutation.out (SURVIVED/ERROR lines) and out/snap_after_check.out\n", encoding="utf-8")
stamp("chain done")
