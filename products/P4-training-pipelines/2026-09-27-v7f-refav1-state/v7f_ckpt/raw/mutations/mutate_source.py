"""SOURCE mutations of the F3 trainer ON DISK: each must turn test_v6_exact_resume.py's POSITIVE tests RED, and the
file is restored byte-for-byte afterwards (sha256 asserted). Output: mutation_record.json + one pytest log each."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

STACK = Path("C:/Users/Admin/v7f_ckpt/stack")
TR = STACK / "scripts" / "train_v6_staged.py"
OUT = Path("C:/Users/Admin/v7f_ckpt/work/mutations")
OUT.mkdir(exist_ok=True)
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
orig = TR.read_bytes()
sha0 = hashlib.sha256(orig).hexdigest()
CRLF = b"\r\n"

MUTATIONS = {
    "M1_restore_call_deleted": (
        b"_rl = apply_resume_state(resume_state, device=device, rng=rng, gen=gen,",
        b"_rl = (lambda *_a, **_k: {'t3_alpha_applied': None, 't3_prog_applied': None, 'monitors': {}, "
        b"'applied': []})(resume_state, device=device, rng=rng, gen=gen,"),
    "M2_lr_reset_deleted": (
        b'            g["lr"] = g["initial_lr"]',
        b"            pass  # MUTATION: the F3b reset deleted"),
    "M3_capture_not_saved": (
        b"                       extra=resume_payload)",
        b"                       extra=None)  # MUTATION: the F3 payload is not saved"),
}
env = dict(os.environ, PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="4",
           PYTHONPATH="C:/Users/Admin/v7f_ckpt/stack;C:/Users/Admin/v7f_ckpt")
record = {"trainer": str(TR), "sha256_before": sha0, "mutations": {}}
try:
    for name, (old, new) in MUTATIONS.items():
        assert orig.count(old) == 1, (name, orig.count(old))
        TR.write_bytes(orig.replace(old, new))
        r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_v6_exact_resume.py",
                            "-k", "POSITIVE"], cwd=STACK, env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=1800)
        TR.write_bytes(orig)
        log = r.stdout + "\n--- stderr ---\n" + r.stderr
        (OUT / f"{name}.pytest.txt").write_text(log, encoding="utf-8")
        tail = [ln for ln in r.stdout.splitlines() if " passed" in ln or " failed" in ln][-1:]
        failed = [ln.split(" - ")[0] for ln in r.stdout.splitlines() if ln.startswith("FAILED")]
        record["mutations"][name] = {"anchor": old.decode(), "replacement": new.decode(),
                                     "pytest_exit": r.returncode, "summary": tail, "failed_tests": failed,
                                     "RED": r.returncode != 0 and bool(failed)}
        print(name, record["mutations"][name]["summary"], "RED" if record["mutations"][name]["RED"] else "GREEN",
              flush=True)
finally:
    TR.write_bytes(orig)
    record["sha256_after_restore"] = hashlib.sha256(TR.read_bytes()).hexdigest()
    record["restored_byte_identical"] = record["sha256_after_restore"] == sha0
    (OUT / "mutation_record.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print("restored", record["restored_byte_identical"])
sys.exit(0 if all(m["RED"] for m in record["mutations"].values()) and record["restored_byte_identical"] else 1)
