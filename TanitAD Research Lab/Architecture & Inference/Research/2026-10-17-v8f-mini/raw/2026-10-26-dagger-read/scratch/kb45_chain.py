"""Detached closer for the K-B4/K-B5 refresh pass (no model turns). Run from anywhere; everything is absolute.

R77 snapshot of the deliverables -> wait for a pytest slot (machine cap 2, polled every 60 s) -> full suite -> full mutation proof on COPIES -> byte-identity re-check -> evidence appended to NOTES.md.
On any failure writes out/chain_failed (with the reason) and stops. On success writes out/finish_kb45.done.
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
BT = pathlib.Path("E:/Projects/TanitAD-artifacts/dagread_build/pt/full6")
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
os.chdir(HERE)
OUT.mkdir(exist_ok=True)


def stamp(msg):
    with open(OUT / "chain_log.txt", "a", encoding="utf-8", newline="\n") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def fail(msg):
    stamp("FAILED " + msg)
    (OUT / "chain_failed").write_text(msg + "\n", encoding="utf-8")
    sys.exit(1)


def n_pytest():
    r = subprocess.run([PY, RUNS], capture_output=True, text=True, env=ENV)
    try:
        return int(r.stdout.strip())
    except ValueError:
        return 99


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def deliverables():
    fx = sorted(str(p.relative_to(HERE)).replace("\\", "/") for d in ("fixtures/dagger_smoke", "fixtures/trainer_kb45") for p in (HERE / d).iterdir() if p.is_file())
    return ["dag_read.py", "test_dag_read.py", "mutate_dag.py", "bank_audit.py", "PREREG_V8F_DAGGER.md"] + fx


for f in ("chain_failed", "finish_kb45.done"):
    (OUT / f).unlink(missing_ok=True)
stamp("chain start")
FILES = deliverables()
snap = {f: sha(f) for f in FILES}
(OUT / "snap_before_SHA256.txt").write_text("".join(f"{h} *{f}\n" for f, h in snap.items()), encoding="utf-8", newline="\n")      # R77: BEFORE the suite and the proof
stamp(f"R77 snapshot of {len(FILES)} files written")

while n_pytest() >= 2:
    time.sleep(60)
stamp("pytest slot free; full suite starts")
shutil.rmtree(BT, ignore_errors=True)
with open(OUT / "pytest_full.out", "w", encoding="utf-8", newline="\n") as f:
    r = subprocess.run([PY, "-m", "pytest", "-v", "-p", "no:cacheprovider", f"--basetemp={BT}", "test_dag_read.py"], stdout=f, stderr=subprocess.STDOUT, env=ENV)
shutil.rmtree(BT, ignore_errors=True)
stamp(f"full suite rc={r.returncode}")
if r.returncode != 0:
    fail(f"pytest rc={r.returncode}: read out/pytest_full.out")

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
## 10. Evidence of the K-B4/K-B5 refresh pass (written by scratch/kb45_chain.py after the mutation run)
* `out/pytest_full.out`: {pyt}
* `out/mutation.out` (live log `out/mutation_console.out`): {mut}
* R77: {"the deliverables (dag_read.py, test_dag_read.py, mutate_dag.py, bank_audit.py, the landed PREREG copy, fixtures/dagger_smoke/*, fixtures/trainer_kb45/*) are BYTE-IDENTICAL before the suite and after the mutation run (`out/snap_before_SHA256.txt` vs `out/snap_after_check.out`; mutants ran on copies under E:/Projects/TanitAD-artifacts/dagread_build/mut)." if ok else "MISMATCH after the mutation run, see `out/snap_after_check.out` -- do not land."}
* sha256 of every deliverable and NOTES.md after the run: `out/final_SHA256.txt`.
"""
with open(HERE / "NOTES.md", "a", encoding="utf-8", newline="\n") as f:
    f.write(note)
final = deliverables() + ["NOTES.md"]
(OUT / "final_SHA256.txt").write_text("".join(f"{sha(f)} *{f}\n" for f in final), encoding="utf-8", newline="\n")
(OUT / "finish_kb45.done").write_text(f"snap_ok={ok} mutation_rc={r.returncode}\n", encoding="utf-8")
stamp("chain done")
