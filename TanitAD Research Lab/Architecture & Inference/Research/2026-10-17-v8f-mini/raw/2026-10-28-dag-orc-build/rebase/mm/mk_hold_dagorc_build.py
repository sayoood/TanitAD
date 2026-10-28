"""Hold: the DAG-ORC build (PREREG_V8F_DAG_ORC Amendment 1, before any DAG-ORC run) -- the trainer's composition flag `--ego-perturb-with-dagger`
(+ its tests and the flag-off golden), the queue stage `:dagoaccorcarm` and the six label tests (mods-rw), the package raw/2026-10-28-dag-orc-build/
(rebase/ = the re-based chain, gate asserter, chain checks, reader, self-tests, mutation proofs, K-RC evidence, the MM fixes; compose/ = the flag's
build record) and a register note on H-V8F-DAG-ORC-1.
python mk_hold_dagorc_build.py <live tip>"""
import hashlib
import json
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
RB = "E:/Projects/TanitAD-artifacts/dagorc_rebase/"
CP = "E:/Projects/TanitAD-artifacts/dagland/ego_dagger_compose/"
H = "C:/Users/Admin/qland/hold_dagorc_build/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-28-dag-orc-build/"
assert not os.path.exists(H), H


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b):
    assert len(b) > 0 and len(H + rel) < 260, (rel, len(H + rel))
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


rd = lambda p: open(p, "rb").read()
sha = lambda b: hashlib.sha256(b).hexdigest()

# ---- the proofs this hold rests on, read from their artifacts (never retyped)
mj = json.load(open(RB + "read/mutation.json", encoding="utf-8"))["summary"]
assert mj["control_rc"] == 0 and mj["killed"] == mj["mutants"] and not mj["not_killed"], mj
READER = sha(rd(RB + "read/dag_orc_read.py"))
assert mj["reader_sha256"] == READER == "2e5f9f1778e104615d4480f51c6246703c9e3bb8f6c0eeaa3fc0c5621bd8adcf"
for f in ("RC_OACC.json", "RC_DAG2.json"):
    r = json.load(open(RB + "read/" + f, encoding="utf-8"))
    assert r["PASS"] is True and r["differences"] == 0 and not r["controls_failed"] and r["reader_sha256"] == READER, f
st = json.load(open(RB + "chain/selftest_chain_result.json", encoding="utf-8"))
sct = rd(RB + "chain/logs/selftest_chain_try2.out").decode("utf-8")
assert "SELFTEST: 56/56 tests pass; MUTATION: 155/155 mutants killed, 0 survivors" in sct
res = rd(CP + "RESULTS.md").decode("utf-8")
assert "SUMMARY: 19/19 mutants killed; survivors: none" in res and "| failed |" in res
for row in [l for l in res.splitlines() if l.startswith("| test_")]:
    assert row.rstrip().endswith("| - |") and "failed" not in row.split("|")[2], row
comb = rd(RB + "mm/combined_pytest.out").decode("utf-8")
assert re.search(r"\d+ passed", comb) and not re.search(r"\d+ (failed|error)", comb), comb[-400:]
base = rd(RB + "chain/logs/label_tests_baseline.out").decode("utf-8")
edit = rd(RB + "chain/logs/label_tests_edited.out").decode("utf-8")
fails = lambda t: sorted(set(re.findall(r"^(?:FAILED|ERROR) (\S+)", t, re.M)))
assert set(fails(edit)) <= set(fails(base)), sorted(set(fails(edit)) - set(fails(base)))     # the label edit adds no failure to the overlay's own set
mut_n = mj["mutants"]

# ---- code: the flag (mods-rw trainer + data lib, new test + golden), the queue stage and the label tests (mods-rw)
CODE_RW = [(CP + "stage/", "stack/scripts/v8f_mini_train.py", "6085d4b5c54d2c7e0a1b69a9f8969ff31f3611475f033ea6a07f30fb9fa7b4e2"),
           (CP + "stage/", "stack/tanitad/data/v8f_mini_data.py", "f1f54c36e969d6652705746f5c98b6d2ec3e53879955491a18cad140bdb6fbcf"),
           (RB + "chain/stage/", V8 + "code/v8f_mini_box2_queue.cmd", None)]
CODE_RW += [(RB + "chain/stage/", f"stack/tests/test_v8f_{t}.py", None) for t in ("flat", "oacc", "kind", "mixo", "lora_ar2", "orc_chain")]
for base_, rel, want in CODE_RW:
    b = rd(base_ + rel)
    assert blob(rel) is not None and b != blob(rel), rel
    assert want is None or sha(b) == want, rel
    assert rel.endswith(".cmd") or b"\r" not in b, rel
    put("mods-rw/" + rel, b)
q = rd(RB + "chain/stage/" + V8 + "code/v8f_mini_box2_queue.cmd")
assert sha(q).startswith("d8aa4f83") and b'add_argument("--ego-perturb-with-dagger"' in rd(CP + "stage/stack/scripts/v8f_mini_train.py")
for rel in ("stack/tests/test_v8f_ego_dagger_compose.py", "stack/tests/ego_dagger_compose_golden.json"):
    b = rd(CP + "stage/" + rel)
    assert blob(rel) is None and b"\r" not in b, rel
    put("stage/" + rel, b)

# ---- the package
SKIP_DIR = ("__pycache__", "/stage/", "_work/")
LEAD = "read/ref/b1_eval_lead_block_k30.npz"
assert sha(rd(RB + LEAD)) == sha(blob(V8 + "raw/2026-10-22-dk/b1_eval_lead_block_k30.npz"))
n = 0
for root, label in ((RB, "rebase/"), (CP, "compose/")):
    for dp, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(dp, f).replace("\\", "/")
            rel = p[len(root):]
            if any(s in "/" + rel for s in SKIP_DIR) or rel == LEAD or rel.endswith(".pyc"):
                continue
            b = rd(p)
            if not b:
                continue
            assert len(b) < 400_000, rel
            if rel.endswith((".py", ".sh", ".md", ".in")):
                assert b"\r" not in b, rel
            assert not re.search(rb"hf_[A-Za-z0-9]{20,}", b), rel
            assert blob(PK + label + rel) is None, rel
            put("stage/" + PK + label + rel, b)
            n += 1

# ---- register note on H-V8F-DAG-ORC-1
GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
END = (b" Launch blocker: the composition flag and its reachability proofs. | MEASURED (N0); HYPOTHESIS (A) |"
       b" `Project Steering/PREREG_V8F_DAG_ORC.md` |")
assert g.count(END) == 1
add = (" Launch blocker: the composition flag and its reachability proofs. **Build (2026-10-28), before any DAG-ORC run:** the trainer's"
       " `--ego-perturb-with-dagger` (default off; flag-off bit-identical to the unmodified trainer on DAG-, ORC- and M2b-style argv; overlap"
       " exactly 0 on a composed run; 26 new tests, the seven existing v8F suites green, 19 / 19 mutants killed), the queue stage and its label"
       " tests, the re-based chain (self-test 56 / 56, 155 / 155 mutants killed) and reader (`2e5f9f17…`: self-test 331 / 331, K-RC 0"
       f" differences on both legs, {mut_n} / {mut_n} mutants killed). The launch blocker is cleared; the waiter takes box 2 when it is free"
       " (A + REP, ≈ 2.8 h). Package `raw/2026-10-28-dag-orc-build/`. | MEASURED (N0); HYPOTHESIS (A) | `Project Steering/PREREG_V8F_DAG_ORC.md` |"
       ).encode("utf-8")
for w in (b"void", b"superseded", b"retracted", b"formerly", b"no longer", b"correction", b"withdrawn"):
    assert w not in add.lower(), w
g2 = g.replace(END, add)
assert len(g2.split(b"\n")) == len(g.split(b"\n"))
put("mods-rw/" + GC, g2)
put("msg.txt", ("DAG-ORC build (PREREG_V8F_DAG_ORC A1, before any DAG-ORC run): the trainer's default-off --ego-perturb-with-dagger (swap first,"
                " perturbation only on non-swapped samples, overlap exactly 0; flag-off bit-identical to the unmodified trainer; 26 new tests,"
                " the seven existing v8F suites green, 19/19 mutants killed); the queue stage :dagoaccorcarm and its six label tests; the"
                " re-based chain (56/56, 155/155 mutants) and reader (331/331, K-RC 0 differences on both legs, %d/%d mutants). Package"
                " raw/2026-10-28-dag-orc-build/ (%d files). Register note on H-V8F-DAG-ORC-1.\n\n"
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n" % (mut_n, mut_n, n)).encode())
print("hold written", H, n, "package files; reader mutants", mut_n)
