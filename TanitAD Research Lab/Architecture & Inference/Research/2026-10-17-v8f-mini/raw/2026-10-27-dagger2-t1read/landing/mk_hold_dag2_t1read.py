"""Hold: the DAGGER2 T1-kin read chain + reader (dag2t1/), coding PREREG_V8F_DAGGER2 as amended (A1-2026-10-27), and Read A's driver
code it runs (readA_code/, which had no repo copy).  The lead-block npz is NOT copied: it is the repo blob raw/2026-10-22-dk/... (7a879baa).
python mk_hold_dag2_t1read.py <live tip>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
S = "E:/Projects/TanitAD-artifacts/dag2t1/"
H = "C:/Users/Admin/qland/hold_dag2_t1read/"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-27-dagger2-t1read/"
assert not os.path.exists(H), H
SKIP_DIRS = ("_st", "_mut", "_tmp", "__pycache__")
SKIP = {"inputs/b1_eval_lead_block_k30.npz"}


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


n = 0
for root, dirs, files in os.walk(S):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    for f in files:
        rel = os.path.relpath(os.path.join(root, f), S).replace("\\", "/")
        if rel in SKIP or os.path.basename(rel).startswith("_"):
            continue
        b = open(S + rel, "rb").read()
        if rel.endswith((".py", ".sh", ".md")):
            assert b"\r" not in b, rel
        assert len(b) > 0 and blob(PK + rel) is None and len(H + "stage/" + PK + rel) < 260, rel
        os.makedirs(os.path.dirname(H + "stage/" + PK + rel), exist_ok=True)
        open(H + "stage/" + PK + rel, "wb").write(b)
        n += 1
me = open(__file__, "rb").read().replace(b"\r\n", b"\n")
os.makedirs(H + "stage/" + PK + "landing", exist_ok=True)
open(H + "stage/" + PK + "landing/mk_hold_dag2_t1read.py", "wb").write(me)
lb = blob("TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-22-dk/b1_eval_lead_block_k30.npz")
assert lb == open(S + "inputs/b1_eval_lead_block_k30.npz", "rb").read()
open(H + "msg.txt", "wb").write((
    "DAGGER2 T1-kin read chain + reader (raw/2026-10-27-dagger2-t1read/, %d files), built and self-tested, NOT run on any DAG2 rollout"
    " (none exists). dag2_read.py codes PREREG_V8F_DAGGER2 as amended (A1-2026-10-27, pinned by raw sha256 c4239e45...): the 16 guarded"
    " cells read FIXED 1 / PARTIAL 11 / NO EFFECT 4, asserted against enumerate_cells.py; new fields PARTIAL_one_seed, adopts_DAG2,"
    " outcome_qualifier. Self-test 196/196, mutation proof 68/68 killed, control green (selftest_output.txt, mutation.json). RC-DAG:"
    " layer A reproduces DAG's banked Read A table with 0 differences (repro/). Preflight: 45 controls PASS on the real iteration-2 banks,"
    " COVERAGE 0.773 / 0.764. Blinding: the default self-test does not hash the real DAG2 eval files (--real-pulls does, before launch);"
    " dry_run_real_config.txt predates the amendment. readA_code/ = Read A's driver (rr_lib, readA, build_report), until now on one disk"
    " only. The lead-block npz is the repo blob raw/2026-10-22-dk/b1_eval_lead_block_k30.npz (byte-identical, not copied). Not built:"
    " the T0 four-family beside rows, the HACC re-fit beside row, DAG's SMOKE mode (declared NOT RUN in the output).\n\n"
    "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n" % (n + 1)).encode())
print("hold written", H, n + 1)
