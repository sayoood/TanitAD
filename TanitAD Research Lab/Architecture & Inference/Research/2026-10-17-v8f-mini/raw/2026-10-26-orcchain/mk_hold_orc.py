"""Hold: PREREG_V8F_ORECOVER Amendment 3 (before any run) + the reader brought to it + the box-2 chain (queue stages, chain, its test, the five
widened queue-pin tests). python mk_hold_orc.py <tip> [--tree]. --tree also writes the verification tree E:/Projects/TanitAD-artifacts/oc/tree
(a full archive of the tip with this hold overlaid) for the suites. The five widened tests are re-derived on the TIP's bytes by widen_pins.py."""
import os
import re
import shutil
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*", "-c", "core.autocrlf=false"]
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
OC = "E:/Projects/TanitAD-artifacts/oc/"
CH = "C:/Users/Admin/orcchain/"
RD = "C:/Users/Admin/orcread/"
SCR = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/t1kin/"
TREE_ONLY = "--tree-only" in sys.argv            # build the verification tree before the evidence exists: the hold goes to oc/hold_tmp
H = OC + "hold_tmp/" if TREE_ONLY else "C:/Users/Admin/qland/hold_orc_chain/"
if TREE_ONLY and os.path.exists(H):
    shutil.move(H, OC + f"hold_tmp_old_{os.getpid()}")
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
QUEUE = PK + "code/v8f_mini_box2_queue.cmd"
TESTS = ("test_v8f_kind.py", "test_v8f_flat.py", "test_v8f_lora_ar2.py", "test_v8f_mixo.py", "test_v8f_oacc.py")
assert not os.path.exists(H), H


def blob(p, rev=TIP):
    return subprocess.run(GIT + ["show", f"{rev}:{p}"], capture_output=True, check=True).stdout


def rd(p):
    return open(p, "rb").read()


def put(rel, b):
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


def lf(b, name):
    assert b"\r\n" not in b, name
    return b


# 1. the queue (CRLF, the committed form): the agent edited the 98713259 bytes, and the tip still carries them
assert blob(QUEUE) == blob(QUEUE, "98713259"), "the queue moved on the tip since the chain was built: rebase"
q = rd(CH + "files/" + QUEUE)
assert q.count(b"\r\n") == q.count(b"\n") and q.startswith(blob(QUEUE)[:200]), "the queue must stay CRLF"
put("mods-rw/" + QUEUE, q)
# 2. the five widened pin tests, re-derived on the tip's bytes
W = OC + "widen/"
shutil.rmtree(W, ignore_errors=True)
os.makedirs(W)
for t in TESTS:
    open(W + t, "wb").write(blob("stack/tests/" + t))
r = subprocess.run([PY, CH + "tools/widen_pins.py", W], capture_output=True, text=True)
assert r.returncode == 0, r.stdout + r.stderr
for t in TESTS:
    b = lf(rd(W + t), t)
    assert b != blob("stack/tests/" + t) and b"ORC_LABELS" in b or b"orc = " in b, t
    if blob("stack/tests/" + t) == blob("stack/tests/" + t, "98713259"):
        assert b == rd(CH + "files/stack/tests/" + t), f"{t}: re-derived widening differs from the agent's on the same base"
    put("mods-rw/stack/tests/" + t, b)
# 3. the chain and its test (new files)
put("stage/" + PK + "code/box2_ops/chain_v8f_orc.sh", lf(rd(CH + "files/" + PK + "code/box2_ops/chain_v8f_orc.sh"), "chain"))
put("stage/stack/tests/test_v8f_orc_chain.py", lf(rd(CH + "files/stack/tests/test_v8f_orc_chain.py"), "chain test"))
R = "stage/" + PK + "raw/2026-10-26-orcchain/"
for f in sorted(os.listdir(CH + "tools")):
    if os.path.isfile(CH + "tools/" + f):
        put(R + f, rd(CH + "tools/" + f))
put(R + "NOTES.md", lf(rd(CH + "NOTES.md"), "notes"))
for f in sorted(os.listdir(CH + "out")):
    if os.path.isfile(CH + "out/" + f) and os.path.getsize(CH + "out/" + f) > 0:
        put(R + "out/" + f, rd(CH + "out/" + f))
# 4. PREREG Amendment 3 (append) and the reader brought to it
pre = blob("Project Steering/PREREG_V8F_ORECOVER.md")
a3 = lf(rd(SCR + "AMEND_ORC3.md"), "A3")
assert b"## Amendment 3" not in pre and rd(RD + "PREREG_V8F_ORECOVER.A3.md") == pre + a3
put("mods/Project Steering/PREREG_V8F_ORECOVER.md", pre + a3)
RR = PK + "raw/2026-10-26-orcread/"
for f in ("orc_read.py", "test_orc_read.py", "mutate_orc.py"):
    assert blob(RR + f) == rd(RD + f + ".pre_a3"), f"{f}: the tip is not the A3 edit's base"
    put("mods-rw/" + RR + f, lf(rd(RD + f), f))
for f in ("apply_a3.py", "apply_a3b.py", "run_a3_tests.sh", "PREREG_V8F_ORECOVER.A3.md"):
    put("stage/" + RR + f, lf(rd(RD + f), f))
for f in () if TREE_ONLY else ("pytest_full_a3.out", "mutation_a3.out", "mutation_a3.log", "pytest_full_a3b.out"):
    put("stage/" + RR + "out/" + f, rd(RD + "out/" + f))
def evidence():
    s3 = rd(RD + "out/pytest_full_a3b.out").decode("utf-8", "replace")
    m = re.search(r"(\d+) passed", s3)
    assert m and " failed" not in s3 and " error" not in s3, s3[-300:]
    n_rd = int(m.group(1))
    mu = rd(RD + "out/mutation_a3.out").decode("utf-8", "replace")
    mk = re.search(r"# (\d+) of (\d+) mutants KILLED; survived: none", mu)
    assert mk and mk.group(1) == mk.group(2), mu[-400:]
    n_mu = int(mk.group(2))
    notes = blob(RR + "NOTES.md")
    put("mods/" + RR + "NOTES.md", notes + (
        "\n## Amendment 3 (A3-2026-10-26, before any run)\n"
        "* `apply_a3.py`: `g_argv` requires ORC to name the law once at its registered values plus `--ego-perturb-seed <seed>`, and ORC-0 to name none\n"
        "  of it. `g_pair` drops the law flags from the compared set (`drop = {--out, --ego-perturb-p, --ego-perturb-seed} | LAW`).\n"
        "* `apply_a3b.py`: `PREREG_ID` names A3; `PREREG_V8F_ORECOVER.A3.md` is the landed PREREG with Amendment 3 appended; one test checks it extends A2\n"
        f"  and carries the reader's law value for value.\n* `out/pytest_full_a3b.out`: {n_rd} passed. `out/mutation_a3.out`: {n_mu}/{n_mu} mutants KILLED (M24, M25 are A3's).\n").encode())
    # 5. the register
    g = blob("Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8")
    row = [ln for ln in g.split("\n") if ln.startswith("| **H-V8F-ORECOVER-1**")]
    assert len(row) == 1
    cells = row[0].split(" | ")
    assert len(cells) == 5 and cells[3] == "HYPOTHESIS", "row layout: id | claim | status | class | link |"
    cells[2] = cells[2] + (
        " **2026-10-26, before any run: Amendment 3.** ORC-0 names only `--ego-perturb-p 0`, because the trainer's gate refuses law flags at p = 0"
        " and the guard is kept. ORC names the law at its registered values. The pair rule excludes the ORC-only flags. **Box-2 chain built:**"
        " `code/box2_ops/chain_v8f_orc.sh` with queue stages `gateorc` / `trainorc` / `evalorc` (`test_v8f_orc_chain.py`; mutation 37/37 RED;"
        f" `raw/2026-10-26-orcchain/`). Reader at A3: {n_rd} passed, mutation {n_mu}/{n_mu}. Queued on box 2 directly after OACC: MIXO has no"
        " box-2 run left (s0 FAIL-O-LEARNED, s1 not trained, as committed).")
    put("mods-rw/Project Steering/GOALS_AND_CLAIMS.md", g.replace(row[0], " | ".join(cells)).encode("utf-8"))
    put("msg.txt", ("PREREG_V8F_ORECOVER Amendment 3 (before any run: ORC-0 names only --ego-perturb-p 0) + reader at A3 + the box-2 chain"
                    " (queue stages gateorc/trainorc/evalorc, chain_v8f_orc.sh, test_v8f_orc_chain.py, five queue-pin tests widened)\n\n"
                    "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n").encode())
    print("hold written", H, "reader", n_rd, "passed,", n_mu, "mutants")


if not TREE_ONLY:
    evidence()
if "--tree" in sys.argv or TREE_ONLY:
    T = OC + "tree/"
    if os.path.exists(T):
        shutil.move(T, OC + f"tree_old_{os.getpid()}")
    os.makedirs(T)
    a = subprocess.Popen(GIT + ["archive", TIP, "--", "stack", "taniteval", PK[:-1], "Project Steering/PREREG_V8F_ORECOVER.md",
                                "Project Steering/PREREG_V8F_MIXO.md", "Project Steering/PREREG_V8F_OACC.md", "Project Steering/PREREG_V8F_FLAT.md"],
                         stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", T], stdin=a.stdout, check=True)
    assert a.wait() == 0
    for top in ("mods-rw", "stage", "mods"):
        for dp, _, fs in os.walk(H + top):
            for f in fs:
                src = os.path.join(dp, f)
                rel = os.path.relpath(src, H + top).replace("\\", "/")
                if rel.startswith("Project Steering/GOALS_AND_CLAIMS") or rel.startswith("Project Steering/PI_"):
                    continue
                os.makedirs(os.path.dirname(T + rel), exist_ok=True)
                shutil.copyfile(src, T + rel)
    print("tree written", T)
