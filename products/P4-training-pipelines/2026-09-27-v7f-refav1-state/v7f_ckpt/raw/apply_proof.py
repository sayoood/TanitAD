"""PROVE the anchor patch: (a) on fresh copies of the pre-F3 files (sha 8c212e5f / f1534182 / 428a647b) it reproduces
the working files BYTE-FOR-BYTE; (b) it applies cleanly onto a FRESH copy of the CURRENT merge copy
(C:/Users/Admin/v7f_merge/stack), taken now. Writes work/apply_proof.json."""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

W = Path("C:/Users/Admin/v7f_ckpt")
SCRIPT = W / "pkg/code/apply_f3_exact_resume.py"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
REL = ["scripts/train_v6_staged.py", "scripts/launch_gate.py", "tests/test_launch_gate_v7f.py"]
NEW = "tests/test_v6_exact_resume.py"
PRIS = {"scripts/train_v6_staged.py": "train_v6_staged.py", "scripts/launch_gate.py": "launch_gate.py",
        "tests/test_launch_gate_v7f.py": "test_launch_gate_v7f.py"}


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fresh(root: Path, src_of) -> None:
    if root.exists():
        shutil.rmtree(root)
    for rel in REL:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_of(rel), root / rel)


out = {"script_sha256": sha(SCRIPT), "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
# (a) reproduction from the pre-F3 bytes
ra = Path("C:/Users/Admin/v7f_ckpt_gate/proof_repro/stack")
fresh(ra, lambda rel: W / "pristine" / PRIS[rel])
inp = {rel: sha(ra / rel) for rel in REL}
r = subprocess.run([PY, str(SCRIPT), "--stack", str(ra)], capture_output=True, text=True)
rep = {}
for rel in REL + [NEW]:
    a, b = sha(ra / rel), sha(W / "stack" / rel)
    rep[rel] = {"reproduced": a, "working": b, "byte_identical": a == b}
out["a_reproduction"] = {"inputs": inp, "rc": r.returncode, "stdout": r.stdout, "files": rep,
                         "ALL_BYTE_IDENTICAL": r.returncode == 0 and all(v["byte_identical"] for v in rep.values())}
# (b) the CURRENT merge copy, fresh copy taken now
rb = Path("C:/Users/Admin/v7f_ckpt_gate/proof_merge_now/stack")
fresh(rb, lambda rel: Path("C:/Users/Admin/v7f_merge/stack") / rel)
inp_b = {rel: sha(rb / rel) for rel in REL}
r2 = subprocess.run([PY, str(SCRIPT), "--stack", str(rb)], capture_output=True, text=True)
out["b_current_merge_copy"] = {"source": "C:/Users/Admin/v7f_merge/stack", "inputs": inp_b, "rc": r2.returncode,
                               "stdout": r2.stdout, "stderr": r2.stderr[-2000:],
                               "outputs": {rel: sha(rb / rel) for rel in REL + [NEW] if (rb / rel).exists()},
                               "inputs_equal_pre_f3": {rel: inp_b[rel] == inp[rel] for rel in REL}}
# the merged result of (b) for the files the parent had NOT touched must equal my working file
out["b_current_merge_copy"]["outputs_equal_working_where_input_was_pre_f3"] = {
    rel: out["b_current_merge_copy"]["outputs"].get(rel) == sha(W / "stack" / rel)
    for rel in REL if inp_b[rel] == inp[rel]}
(W / "work/apply_proof.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps({"a": out["a_reproduction"]["ALL_BYTE_IDENTICAL"], "b_rc": r2.returncode,
                  "b_inputs_equal_pre_f3": out["b_current_merge_copy"]["inputs_equal_pre_f3"],
                  "b_equal_working": out["b_current_merge_copy"]["outputs_equal_working_where_input_was_pre_f3"]},
                 indent=1))
sys.exit(0 if out["a_reproduction"]["ALL_BYTE_IDENTICAL"] and r2.returncode == 0 else 1)
