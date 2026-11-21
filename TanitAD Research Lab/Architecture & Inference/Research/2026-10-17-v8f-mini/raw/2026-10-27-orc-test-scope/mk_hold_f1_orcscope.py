"""Hold: (1) F1, a FLAT-DAG launch blocker: INIT_CKPT_MATCH gains o_interface (read only when the checkpoint has it) + its test;
(2) test_v8f_orc_chain.py's two landing-moment queue tests scoped to ORC's own region (RED on the tip since the DAGGER stages landed);
(3) the composition-check package (raw/2026-10-27-oaccflat-check/) and the scope fix's mutation proof (raw/2026-10-27-orc-test-scope/).
python mk_hold_f1_orcscope.py <live tip>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
C = "E:/Projects/TanitAD-artifacts/oaccflat_check/"
DL = "E:/Projects/TanitAD-artifacts/dagland/"
X = "C:/Users/Admin/ofx/"
H = "C:/Users/Admin/qland/hold_f1_orcscope/"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
assert not os.path.exists(H), H


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b):
    assert len(b) > 0 and len(H + rel) < 260, rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


def rd(p):
    return open(p, "rb").read()


# (1) the trainer: the tip's file plus exactly one tuple key (and its comment lines)
TR = "stack/scripts/v8f_mini_train.py"
old, new = blob(TR), rd(C + "stage/" + TR)
assert b"\r" not in new
ol, nl = old.split(b"\n"), new.split(b"\n")
assert len(nl) == len(ol) + 3 and nl[:506] == ol[:506] and nl[510:] == ol[507:]      # one line replaced by four, everything else identical
assert nl[506] == ol[506].replace(b'"max_age_s")', b'"max_age_s",') and b'"o_interface")' in nl[507]
put("mods-rw/" + TR, new)
T1 = "stack/tests/test_v8f_init_ckpt_o_interface.py"
assert blob(T1) is None
put("stage/" + T1, rd(C + "stage/" + T1))

# (2) the ORC chain test, scoped (patched from the tip's bytes by patch_orc_chain_test.py)
T2 = "stack/tests/test_v8f_orc_chain.py"
assert rd(X + "test_v8f_orc_chain.tip.py") == blob(T2)
t2 = rd(X + "stack/tests/test_v8f_orc_chain.py")
assert b"\r" not in t2 and b"LATER_LABELS = {" in t2
put("mods-rw/" + T2, t2)
for fn in ("test_v8f_flat.py", "test_v8f_oacc.py"):       # widened by DAG_LABELS (patch_flat_oacc_later.py), their own ORC_LABELS convention
    T = "stack/tests/" + fn
    assert rd(X + fn.replace(".py", ".tip.py")) == blob(T)
    t = rd(X + T)
    assert b"\r" not in t and t.count(b"DAG_LABELS") == 2
    put("mods-rw/" + T, t)
for fn, tok, k in (("test_v8f_kind.py", b"DAG_LABELS", 2), ("test_v8f_mixo.py", b"DAG_LABELS", 2), ("test_v8f_lora_ar2.py", b" dag = {", 1)):
    T = "stack/tests/" + fn                                # widened by patch_kind_mixo_lora_later.py, the same convention
    assert rd(X + fn.replace(".py", ".tip.py")) == blob(T)
    t = rd(X + T)
    assert b"\r" not in t and t.count(tok) == k, fn
    put("mods-rw/" + T, t)

# (3) packages
R1 = PK + "raw/2026-10-27-oaccflat-check/"
keep = ("RESULT.md", "HANDOFF.md", "arm1_dag_oacc.py", "arm1_pair_records.py", "arm2_flat_dag.py", "cg_lib.py", "final_all.sh", "run_mutants.py",
        "run_pytest_staged.py", "verify_tree.py")
n = 0
for f in keep:
    put("stage/" + R1 + f, rd(C + f)); n += 1
for sub in ("out", "logs", "flat_runs"):
    for root, _, files in os.walk(C + sub):
        for f in files:
            p = os.path.join(root, f).replace("\\", "/")
            assert blob(R1 + p[len(C):]) is None
            if os.path.getsize(p) == 0:
                continue                                       # an empty log (final_all.out) carries nothing
            put("stage/" + R1 + p[len(C):], rd(p)); n += 1
R2 = PK + "raw/2026-10-27-orc-test-scope/"
for f in ("patch_orc_chain_test.py", "mutate_orc_scope.py", "mutate_orc_scope.out", "pytest_orcfix_f1.log", "mk_hold_f1_orcscope.py",
          "patch_flat_oacc_later.py", "mutate_flat_oacc_later.out", "pytest_queue_others.log", "pytest_queue_others2.log",
          "patch_kind_mixo_lora_later.py", "pytest_queue_all.log", "pytest_sweep_by_file.log", "sweep_by_file.sh", "sweep_resume.sh"):
    b = rd(DL + f)
    put("stage/" + R2 + f, b.replace(b"\r\n", b"\n") if f.endswith(".py") else b)
put("msg.txt", ("F1 (FLAT-DAG launch blocker, found by the composition check): INIT_CKPT_MATCH gains o_interface, read only when the checkpoint"
                " has it, so a fine-tune of a cut-trained checkpoint under a full argv (or the reverse) is REFUSED at load instead of passing the"
                " gate; every M2b checkpoint (no key) loads as before. New test RED on the old trainer (3 failed), GREEN on the fix (7 passed)."
                " test_v8f_orc_chain.py: its two landing-moment queue tests are scoped to ORC's own region with the later, insertion-only edits"
                " (DAGGER, DAGGER2) declared by label; they were RED on the tip since b43c7a9; mutation proof 8/8 killed."
                " test_v8f_flat / oacc / kind / mixo / lora_ar2: their label-set tests widened by the DAGGER / DAGGER2 labels (their own"
                " ORC_LABELS convention), RED on the tip for the same reason; an undeclared label still turns them RED. All 15 queue-reading"
                " test files + the DAgger / init-ckpt tests run on the fixed tree: pytest_sweep_by_file.log (per file; pytest_queue_all.log is the run C: running full wrecked, kept as evidence of that). Not re-run for lack of disk: mini_full, pgt, tac, wc, dagger (untouched by this change), mixo beyond its label test. Packages:"
                " raw/2026-10-27-oaccflat-check/ (DAG-OACC and FLAT-DAG compose on the real gate, synthetic data; %d files),"
                " raw/2026-10-27-orc-test-scope/\n\n"
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n" % n).encode())
print("hold written", H, n)
