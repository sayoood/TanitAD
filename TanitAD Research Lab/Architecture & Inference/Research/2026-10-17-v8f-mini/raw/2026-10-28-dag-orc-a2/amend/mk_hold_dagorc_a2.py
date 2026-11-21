"""Hold: PREREG_V8F_DAG_ORC Amendment 2 (A2-2026-10-28, before any DAG-ORC row is read) -- the composed perturbation's own stream (trainer + data lib
+ tests, mods-rw), the r2 chain and the T1-kin chain (banked; installed outside the repo), the refused r1 run's record, a TRAPS entry and a register note.
python mk_hold_dagorc_a2.py <live tip>"""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
A = "E:/Projects/TanitAD-artifacts/"
FX, R2, T1, HERE = A + "dagland/compose_fix/", A + "dagorc_r2/", A + "dagorc_t1/", A + "dagland/dagorc_amend2/"
R1 = A + "v8f-dag-orc-oacc/M2b_s0_dagoaccorc/"
H = "C:/Users/Admin/qland/hold_dagorc_a2/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-28-dag-orc-a2/"
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

# ---- the fix: the sweep has FINISHED (R77: never build from a tree a mutation run is editing) and the staged bytes are the pre-sweep ones
assert os.path.isfile(FX + "FINISHED.txt")
res = rd(FX + "RESULTS.md").decode("utf-8")
m = re.search(r"SUMMARY: (\d+)/(\d+) mutants killed; survivors: none", res)
assert m and m.group(1) == m.group(2), "mutation sweep not all killed"
for row in [l for l in res.splitlines() if l.startswith("| test_")]:
    assert row.rstrip().endswith("| - |"), row
FIX = {"stack/scripts/v8f_mini_train.py": "1c888280", "stack/tanitad/data/v8f_mini_data.py": "c117e1cf",
       "stack/tests/test_v8f_ego_dagger_compose.py": "faad3be2"}
for rel, pre in FIX.items():
    b = rd(FX + "stage/" + rel)
    assert sha(b).startswith(pre) and b"\r" not in b and blob(rel) is not None and b != blob(rel), rel
    put("mods-rw/" + rel, b)
assert re.search(rb"^EGO_PERTURB_COMPOSED_STREAM = 1\s*$", rd(FX + "stage/stack/tanitad/data/v8f_mini_data.py"), re.M)

# ---- Amendment 2 (append) and the TRAPS entry (append)
am = rd(HERE + "AMEND2.md")
assert b"\r" not in am and am.startswith(b"\n## Amendment 2 (A2-2026-10-28)") and b"(Filled in at landing" not in am
PR = "Project Steering/PREREG_V8F_DAG_ORC.md"
p = blob(PR)
assert p.endswith(b"\n") and b"## Amendment 1" in p and b"## Amendment 2" not in p
put("mods/" + PR, p + am)
tr = rd(HERE + "TRAPS_ENTRY.md")
TP = "Project Steering/TRAPS.md"
t = blob(TP)
assert t.endswith(b"\n") and b"\r" not in tr and b"ONE GENERATOR" not in t
put("mods/" + TP, t + tr)

# ---- the package
n = 0
lfn = []
SKIP = (".pt", ".npz", ".npy", ".pyc")
for root, label, skipdirs in ((FX, "fix/", ("/stage/",)), (R2, "r2/", ()), (T1, "t1/", ("/out/", "/smoke_out/")), (HERE, "amend/", ())):
    for dp, _, fs in os.walk(root):
        for f in fs:
            pth = os.path.join(dp, f).replace("\\", "/")
            rel = pth[len(root):]
            if f.endswith(SKIP) or "__pycache__" in rel or any(s in "/" + rel for s in skipdirs):
                continue
            b = rd(pth)
            if not b or len(b) >= 400_000:
                print("skip", label + rel, len(b)); continue
            if rel.endswith((".py", ".sh", ".md")) and b"\r" in b:
                b = b.replace(b"\r\n", b"\n")              # the repo's .py / .sh / .md are LF: normalised on banking, listed in LF_NORMALISED.txt
                assert b"\r" not in b, rel
                lfn.append(label + rel)
            assert not re.search(rb"hf_[A-Za-z0-9]{20,}", b), rel
            assert blob(PK + label + rel) is None, rel
            put("stage/" + PK + label + rel, b)
            n += 1
for f in ("summary.json", "selfcheck.json", "argv.json", "launch_gate_token.json", "SHA256SUMS.txt", "train_log.jsonl"):
    put("stage/" + PK + "r1_refused/M2b_s0_dagoaccorc/" + f, rd(R1 + f)); n += 1
if lfn:
    put("stage/" + PK + "LF_NORMALISED.txt", ("CRLF -> LF on banking (content otherwise unchanged):\n" + "\n".join(sorted(lfn)) + "\n").encode()); n += 1
put("stage/" + PK + "r1_refused/chain_v8f_dagoaccorc.out", rd("C:/Users/Admin/box2_ship/logs/chain_v8f_dagoaccorc.out")); n += 1

# ---- register note
GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
END = b" Package `raw/2026-10-28-dag-orc-build/`. | MEASURED (N0); HYPOTHESIS (A) | `Project Steering/PREREG_V8F_DAG_ORC.md` |"
assert g.count(END) == 1
add = (" Package `raw/2026-10-28-dag-orc-build/`. **Amendment 2 (A2-2026-10-28), before any DAG-ORC row is read:** r1's four gates passed, A s0"
       " trained and K-REACH refused it (perturbation share 19,161 / 76,898 = 0.2492, outside 0.3 ± 0.02): the swap and the perturbation drew"
       " from one stream (same seed formula, same seed, same first draw), so the perturbation fired only where the swap did not. Fixed: on the"
       " composed path the perturbation takes its own stream (`levers.ego_perturb_stream` = 1; every other path bit-identical); all four runs"
       " re-run from their gates (r2); r1's artifacts kept aside as `_r1_kreach_stop`, never read. Package `raw/2026-10-28-dag-orc-a2/`."
       " | MEASURED (N0); HYPOTHESIS (A) | `Project Steering/PREREG_V8F_DAG_ORC.md` |").encode("utf-8")
for w in (b"void", b"superseded", b"retracted", b"formerly", b"no longer", b"correction", b"withdrawn"):
    assert w not in add.lower() and w not in am.lower() and w not in tr.lower(), w
g2 = g.replace(END, add)
assert len(g2.split(b"\n")) == len(g.split(b"\n"))
put("mods-rw/" + GC, g2)
put("msg.txt", ("PREREG_V8F_DAG_ORC Amendment 2 (A2-2026-10-28), before any DAG-ORC row is read: K-REACH refused r1's A s0 (perturbation share"
                " 0.2492, outside 0.3 +- 0.02) because the swap and the perturbation drew from one RNG stream. The composed path now gives the"
                " perturbation its own stream (levers.ego_perturb_stream = 1; every other path bit-identical; new tests RED on e7ea20d1, %s/%s"
                " mutants killed); the r2 chain re-runs all four runs from their gates; the T1-kin chain is banked. TRAPS entry; register note."
                " Package raw/2026-10-28-dag-orc-a2/ (%d files).\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
                % (m.group(1), m.group(2), n)).encode())
print("hold written", H, n, "package files")
