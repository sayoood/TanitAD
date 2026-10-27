"""ONE hold for the DAgger arm: the code (di/tree), the reader (dagread), the box-2 queue + chain (dagq), the dev-box T1 chain (dagt1), the integration
evidence (proofs, red-suite triage, E7, the K-B5 real-magnitude probe), the stale lora spy fix, and the register row.  python mk_hold_dagger.py <live tip>
Every byte is pinned: di/tree by its SNAPSHOT_AFTER, dagread by its final_SHA256, dagq by its MANIFEST.txt sha256 (R77)."""
import hashlib
import json
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
BASE = "2a78f059cd4dcd29c279467ccca84581123d73b5"
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
A = "E:/Projects/TanitAD-artifacts/"
H = "C:/Users/Admin/qland/hold_dagger/"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
RAW = PK + "raw/"
INTEG = RAW + "2026-10-26-dagger-integration/"
assert not os.path.exists(H), H


def blob(c, p):
    r = subprocess.run(GIT + ["show", f"{c}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def rd(p):
    return open(p, "rb").read()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def put(rel, b):
    assert len(H + rel) < 260 and len(rel) <= 220, rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


staged = []


def stage(rel, b):
    assert blob(TIP, rel) is None, f"{rel} exists on the tip"
    assert rel not in staged, rel
    staged.append(rel)
    put("stage/" + rel, b)


def rewrite(rel, b):
    base, tip = blob(BASE, rel), blob(TIP, rel)
    assert base is not None and tip == base, f"{rel}: changed on the tip since {BASE[:8]} (would revert another session)"
    put("mods-rw/" + rel, b)


# 1. the code + its proofs, from di/tree, pinned by the landing manifest computed after the proofs were banked
T = A + "di/tree/"
m = json.load(open(A + "di/out/MANIFEST_LANDING.json"))
assert m["tip"] == BASE and set(m["CHANGED"]) == {"stack/scripts/v8f_mini_train.py"}
snap = json.load(open(A + "di/out/SNAPSHOT_AFTER.json"))
for f, h in m["NEW"].items():
    b = rd(T + f)
    assert sha(b) == h, f
    if f in snap:
        assert sha(b) == snap[f], f"{f}: differs from the proof's SNAPSHOT_AFTER"
    stage(f, b)
b = rd(T + "stack/scripts/v8f_mini_train.py")
assert sha(b) == m["CHANGED"]["stack/scripts/v8f_mini_train.py"]
rewrite("stack/scripts/v8f_mini_train.py", b)

# 2. the stale lora spy (pre-existing red on the tip: declared_vs_built gained ego/sched in O-RECOVER and dag here); 39/40 after, the 40th is pre-existing
lo = rd(A + "redtriage/dagger/stack/tests/test_v8f_lora.py")
assert b"*rest, **kw" in lo and b"\r\n" not in lo
rewrite("stack/tests/test_v8f_lora.py", lo)

# 3. the reader, pinned by its final_SHA256 (taken after the proof)
R = "C:/Users/Admin/dagread/"
fin = {ln.split(None, 1)[1].lstrip("*").strip(): ln.split()[0] for ln in rd(R + "out/final_SHA256.txt").decode().splitlines() if ln.strip()}
for f, h in fin.items():
    assert sha(rd(R + f)) == h, f"dagread/{f} moved since its proof"
RD = RAW + "2026-10-26-dagger-read/"
keep_out = ("pytest_full.out", "mutation.out", "snap_after_check.out", "final_SHA256.txt", "smoke_audit_kb45.out", "chain_log.txt", "snap_before_SHA256.txt")
for root, dirs, files in os.walk(R):
    dirs[:] = [d for d in dirs if d not in ("__pycache__", "out_pre_a1", "out_a1_run3_N89_survived", "out_pre_kb45", "pt")]
    for fn in files:
        p = os.path.join(root, fn).replace("\\", "/")
        rel = p[len(R):]
        if rel.startswith("out/") and rel[4:] not in keep_out:
            continue
        if rel.endswith((".pyc",)) or rel == "PREREG_V8F_DAGGER.md":
            continue
        stage(RD + rel, rd(p))

# 4. the box-2 queue + chain, pinned by dagq/MANIFEST.txt; the queue .cmd replaces the tip's (insertions only), the tool is NEW
Q = A + "dagq/"
lines = rd(Q + "MANIFEST.txt").decode().splitlines()
QD = RAW + "2026-10-26-dagger-queue/"
for i, ln in enumerate(lines):
    mm = re.match(r"^([0-9a-f]{64})  (\S.*)$", ln)
    if not mm:
        continue
    h, src = mm.group(1), mm.group(2).strip()
    b = rd(Q + src)
    assert sha(b) == h, f"dagq/{src} moved since its MANIFEST"
    dest = next((x for x in lines[i + 1:i + 4] if x.strip().startswith("->")), "")
    if "stage/" + PK + "code/v8f_mini_box2_queue.cmd" == src:
        rewrite(PK + "code/v8f_mini_box2_queue.cmd", b)
    elif src == "stage/stack/scripts/v8f_dagger_queue_tool.py":
        stage("stack/scripts/v8f_dagger_queue_tool.py", b)
    else:                                                                  # everything else (incl. the DEV BOX files) is banked for provenance
        stage(QD + src, b)
for f in ("NOTES.md", "MANIFEST.txt", "selftest_output.txt", "mutation_output.txt", "selftest_dag_queue.py", "mutate_dag_queue.py", "real_gate_proof.py",
          "kb_shape_probe.py", "kb_shape_probe.txt", "bankpub.diff"):
    if os.path.exists(Q + f) and QD + f not in staged:
        stage(QD + f, rd(Q + f))

# 5. the dev-box T1 chain (text and scripts; the loader test's npz dumps stay local)
D1 = A + "dagt1/"
TD = RAW + "2026-10-26-dagger-t1/"
for root, dirs, files in os.walk(D1):
    dirs[:] = [d for d in dirs if d not in ("bt", "__pycache__")]
    for fn in files:
        p = os.path.join(root, fn).replace("\\", "/")
        rel = p[len(D1):]
        if fn.endswith((".npz", ".pt", ".pyc")) or rel.startswith("PREREG_"):
            continue
        b = rd(p)
        if fn.endswith((".py", ".sh", ".md")):
            assert b"\r\n" not in b or fn.endswith(".cmd"), rel
        stage(TD + rel, b)

# 6. integration evidence: E7, the K-B5 real-magnitude probe, the red-suite triage, the kscore full-tip run
L = A + "dagland/"
stage(INTEG + "E7_DECISION.md", rd(L + "E7_DECISION.md"))
for f in ("kb5_realmag_probe.py", "kb5_realmag_out.txt"):
    stage(INTEG + "kb5_realmag/" + f, rd(L + "kb5_realmag/" + f))
RT = A + "redtriage/"
for fn in sorted(os.listdir(RT + "out")):
    if fn.endswith((".out", ".json")) or fn in ("COMPARE.txt", "runner.log"):
        stage(INTEG + "redtriage/" + fn, rd(RT + "out/" + fn))
for fn in sorted(os.listdir(RT)):
    if fn.endswith((".py", ".sh", ".cmd", ".md")):
        stage(INTEG + "redtriage/" + fn, rd(RT + fn))
stage(INTEG + "redtriage/kscore_full_tip_plus_dagger.out", rd(A + "kscore_chk/kscore_full_tip_plus_dagger.out"))
stage(INTEG + "redtriage/REDTRIAGE.md", rd(L + "REDTRIAGE.md"))
stage(INTEG + "mk_hold_dagger.py", rd(L + "mk_hold_dagger.py"))

# 7. the register row, edited on the LIVE tip's text
g = blob(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8")
row = [ln for ln in g.split("\n") if ln.startswith("| **H-V8F-DAGGER-1**")]
assert len(row) == 1, len(row)
cells = row[0].split(" | ")
cells[2] = cells[2] + (
    " **2026-10-27: BUILT, PROVED AND LANDED; box-2 go given.** The PI, 2026-10-26: *\"once the DAgger code lands and the queue self-test passes,"
    " run the bank (both seeds) and DAG / DAG-0 at both seeds on box 2, through the launch gate, under PREREG_V8F_DAGGER.\"* Proofs on the final"
    " bytes: data suite 57 passed, 86/86 mutants; trainer suite 97 passed, 105/105 mutants; reader 115 passed, 153/153 mutants; box-2 queue"
    " self-test 127/127 with 153/153 regressions, 51/51 mutants; dev-box T1 chain 219/219. K-B4 / K-B5 are written by the generator and gate the"
    " chain; K-B5 on the 141 eval clips' real magnitudes reads 5.91e-05 m against 1e-4. Six other v8f suites were red on a partial tree; each is"
    " green or identical on the pristine tip (`raw/2026-10-26-dagger-integration/redtriage/`). Nothing has run on box 2 yet.")
put("mods-rw/Project Steering/GOALS_AND_CLAIMS.md", g.replace(row[0], " | ".join(cells)).encode("utf-8"))
put("msg.txt", ("DAgger arm (PREREG_V8F_DAGGER + A1): the own-state bank library + generator (K-B0..K-B5 written and gated), the trainer's four flags"
                " (--dagger-p/-bank/-bank-sha256/-seed, default off, bit-identical when off; train_window_list_sha256 on every run), the reader"
                " (115 passed, 153/153 mutants), the box-2 queue stages + chain (bank publication, COVERAGE stop, chain-pinned gate argv; self-test"
                " 127/127, 51/51 mutants), the dev-box T1 chain (219/219). Data 57 passed 86/86 mutants, trainer 97 passed 105/105. E7: the bank"
                " is bound through argv -> bank.json -> shards, no launch_gate.py entry. Stale lora spy fixed (pre-existing red). Red-suite triage:"
                " none DAgger-caused. Register H-V8F-DAGGER-1 with the PI's box-2 go\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n").encode())
print("hold written:", len(staged), "staged + 4 rewrites")
