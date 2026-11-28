"""BEFORE (the three files at their pre-F3 bytes, no new test) and AFTER (the F3 patch) pytest runs, same tree, same
env: the merge's DVB-related set + the gate test (the parent's 256-baseline set) and the existing resume tests."""
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

W = Path("C:/Users/Admin/v7f_ckpt")
STACK = W / "stack"
OUT = W / "work" / "suites"
OUT.mkdir(exist_ok=True)
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
FILES = {"scripts/train_v6_staged.py": "train_v6_staged.py", "scripts/launch_gate.py": "launch_gate.py",
         "tests/test_launch_gate_v7f.py": "test_launch_gate_v7f.py"}
NEW = STACK / "tests" / "test_v6_exact_resume.py"
MERGE_SET = ["tests/test_launch_gate_v7f.py", "tests/test_declared_vs_built.py",
             "tests/test_tactical_label_reach_v6.py", "tests/test_v6_effective_weights.py",
             "tests/test_v7f_r1r4.py", "tests/test_v7f_r2_nav_fixes.py", "tests/test_v7f_r6_strategic_off.py"]
RESUME_SET = ["tests/test_ckpt_trajectory.py", "tests/test_dinov3_seed.py", "tests/test_eval_speed_ckpt.py",
              "tests/test_refa_v1_ema_resume.py", "tests/test_replay.py", "tests/test_seam_dump_import_guard.py",
              "tests/test_v6_chain.py", "tests/test_v6_ladder_edges.py", "tests/test_v6_staged.py",
              "tests/test_resume_continues_the_data_order.py"]
env = dict(os.environ, PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="4",
           PYTHONPATH=f"{STACK};{W}",
           TANITAD_R3_REF_TRAINER="C:/Users/Admin/v7f_ckpt_gate/ref/train_v6_staged_c36b6ddd.py",
           TANITAD_V6_PRE_R1R4="C:/Users/Admin/v7f_ckpt_gate/ref/v6_c36b6ddd.py")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def run(tag, files):
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rfEs", *files], cwd=STACK,
                       env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=7200)
    head = "# " + " | ".join(f"{k} {sha(STACK / k)}" for k in FILES) + "\n# env TANITAD_R3_REF_TRAINER=" + \
        env["TANITAD_R3_REF_TRAINER"] + " TANITAD_V6_PRE_R1R4=" + env["TANITAD_V6_PRE_R1R4"] + "\n"
    (OUT / f"{tag}.txt").write_text(head + r.stdout + "\n--- stderr ---\n" + r.stderr[-5000:], encoding="utf-8")
    last = [ln for ln in r.stdout.splitlines() if (" passed" in ln or " failed" in ln) and " in " in ln][-1:]
    print(tag, r.returncode, last, flush=True)


after = {k: (STACK / k).read_bytes() for k in FILES}
new_bytes = NEW.read_bytes()
which = sys.argv[1:] or ["before", "after"]
try:
    if "before" in which:
        for k, name in FILES.items():
            shutil.copyfile(W / "pristine" / name, STACK / k)
        NEW.unlink()
        run("before_merge_set", MERGE_SET)
        run("before_resume_set", RESUME_SET)
finally:
    for k, b in after.items():
        (STACK / k).write_bytes(b)
    NEW.write_bytes(new_bytes)
if "after" in which:
    run("after_merge_set_plus_new", MERGE_SET + ["tests/test_v6_exact_resume.py"])
    run("after_resume_set", RESUME_SET)
print("restored:", {k: sha(STACK / k) for k in FILES}, sha(NEW))
