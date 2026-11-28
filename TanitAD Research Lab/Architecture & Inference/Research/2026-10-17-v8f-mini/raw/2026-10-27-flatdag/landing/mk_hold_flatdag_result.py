"""Hold: RESULT_V8F_FLATDAG (NOT ESTABLISHED) + the package raw/2026-10-27-flatdag/ (box-2 run metadata, the dev-box T1-kin read chain
and reader, the read) + register row H-V8F-FLATDAG-1.
python mk_hold_flatdag_result.py <live tip>"""
import hashlib
import os
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
A = "E:/Projects/TanitAD-artifacts/"
B2 = A + "v8f-flatdag/"                      # box-2 pulls
T1 = A + "v8f-flatdag-t1/"                   # dev-box T1-kin outputs
RC = A + "flatdag_t1/"                       # the read chain + reader
LG = "C:/Users/Admin/box2_ship/logs/"
HERE = A + "dagland/flatdag_result/"
H = "C:/Users/Admin/qland/hold_flatdag_result/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-27-flatdag/"
assert not os.path.exists(H), H


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b):
    assert len(b) > 0 and len(H + rel) < 260, rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


rd = lambda p: open(p, "rb").read()
sha = lambda b: hashlib.sha256(b).hexdigest()
TEXT = (".py", ".sh", ".md")                 # LF-normalised in the repo; the originals' sha256 are listed in LF_NORMALISED.txt

res = rd(HERE + "RESULT_V8F_FLATDAG.md")
assert b"\r" not in res and blob(V8 + "RESULT_V8F_FLATDAG.md") is None
put("stage/" + V8 + "RESULT_V8F_FLATDAG.md", res)

files = {}
for root, _, fs in os.walk(B2):              # run and bank metadata; checkpoints, bank shards and eval arrays stay on E:
    for f in fs:
        p = os.path.join(root, f).replace("\\", "/")
        if f.endswith((".pt", ".npz")):
            continue
        assert os.path.getsize(p) < 100_000, p
        files["box2/" + p[len(B2):]] = p
for f in os.listdir(LG):
    if f.startswith("v8fflatdag") and os.path.isfile(LG + f) or f == "chain_v8f_flatdag.out":
        files["box2/logs/" + f] = LG + f
LEAD = "inputs/b1_eval_lead_block_k30.npz"   # the repo blob raw/2026-10-22-dk/b1_eval_lead_block_k30.npz, byte-identical
assert sha(rd(RC + LEAD)) == sha(blob(V8 + "raw/2026-10-22-dk/b1_eval_lead_block_k30.npz"))
for root, _, fs in os.walk(RC):
    for f in fs:
        p = os.path.join(root, f).replace("\\", "/")
        rel = p[len(RC):]
        if rel == LEAD or rel.startswith("readA_code/"):   # readA_code/ = raw/2026-10-27-dag-orc-read/ref/readA_code/ (checked below)
            continue
        files["t1read/" + rel] = p
for f in ("build_report.py", "readA.py", "rr_lib.py"):
    assert rd(RC + "readA_code/" + f).replace(b"\r\n", b"\n") == blob(V8 + "raw/2026-10-27-dag-orc-read/ref/readA_code/" + f), f
files["t1read/reader_src/t1kf_read_a1_arm.py"] = A + "dagreread/reader_src/t1kf_read_a1_arm.py"   # the only reader dependency not in the repo
for f in os.listdir(T1):
    if f in ("PINS.txt", "runs.json") or f.startswith("done_") or f.startswith("t1v8f_") and f.endswith(".json"):
        files["t1/" + f] = T1 + f
for sub in ("argv", "ids"):
    for f in os.listdir(T1 + sub):
        files[f"t1/{sub}/{f}"] = T1 + sub + "/" + f
files["read/FLATDAG_READ.json"] = T1 + "read/FLATDAG_READ.json"
files["read/FLATDAG_compact.txt"] = T1 + "read/FLATDAG_compact.txt"
files["landing/mk_hold_flatdag_result.py"] = HERE + "mk_hold_flatdag_result.py"
assert sha(rd(files["read/FLATDAG_READ.json"])) == "7274b53209944305b998b40df3707d9d433cfb2d105d12e58342bff681e1c841"
assert sha(rd(RC + "flatdag_read.py")) == "f978c4e354486b0530ec95c50fea68e7b6a44f0430865336c2ab4c06447d59c8"

lf = ["original sha256 (the file that ran, CRLF)  ->  repo sha256 (LF)  path"]
n = 0
for rel, src in sorted(files.items()):
    b = rd(src)
    if not b:
        continue                                          # an empty log carries nothing
    assert blob(PK + rel) is None, rel
    if rel.endswith(TEXT) and b"\r\n" in b:
        b2 = b.replace(b"\r\n", b"\n")
        lf.append(f"{sha(b)}  ->  {sha(b2)}  {rel}")
        b = b2
    put("stage/" + PK + rel, b)
    n += 1
if len(lf) > 1:
    put("stage/" + PK + "t1read/LF_NORMALISED.txt", ("\n".join(lf) + "\n").encode())
    n += 1

GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
old = b"| **PRE-REGISTERED (2026-10-27), NOT RUN.** Promoted from the draft banked at 5eaa0232"
assert g.count(old) == 1 and g.count(b"**H-V8F-FLATDAG-1**") == 1
new = ("| **NOT ESTABLISHED (MEASURED 2026-10-27, read by the pre-registered reader; all 91 controls PASS).** PRIMARY ADE6 DAG − FLAT-DAG"
       " −0.3451 [−0.4421, −0.2506] m at s0 (meets −2F) and −0.0544 [−0.1605, +0.0547] at s1 (not separated); family rows separated-worse:"
       " s0 DK-TG +0.0734, DK-VIOL +0.0370; s1 accel MAE +0.8252. STRATEGIC paired rows not computed (no paired instrument; per-arm S-POS vs CTRV"
       " separated-better for both arms). The cut survives the fine-tune (oracle − own exactly 0 [0, 0] on 2,399 windows at both seeds). Beside:"
       " DAgger helps flat too, FLAT-DAG − FLAT-DAG-0 ADE6 −0.1522 / −0.1905 (separated, R-T0 held), accel MAE +0.38 / +0.28 (separated-worse);"
       " the hierarchy keeps 58 % / 15 % of B-2's margin and leads at both seeds on heading, cross-track and M1 crawl share with T's head F1"
       " not different. T1-kin self-action open loop (R-T0 at T0); paired episode-cluster bootstrap, B = 2,000, 141 episodes; one training"
       " run per seed. Committed next: MIXO's T-side arm, which the register resolves to perception (SPEEDCAL → PREREG_V8F_LORA, A40 by its"
       " venue rule, not allocated); a box-2-feasible form is drafted under its own PREREG. `RESULT_V8F_FLATDAG.md`,"
       " `raw/2026-10-27-flatdag/read/FLATDAG_READ.json`. Was: **PRE-REGISTERED (2026-10-27), NOT RUN.** Promoted from the draft banked at"
       " 5eaa0232").encode("utf-8")
g2 = g.replace(old, new)
i = g2.index(b"**H-V8F-FLATDAG-1**")
j = g2.index(b"\n", i)
row = g2[i:j]
END = b"| HYPOTHESIS | `Project Steering/PREREG_V8F_FLATDAG.md` |"
assert row.endswith(END), row[-120:]
g2 = g2[:i] + row[: -len(END)] + b"| MEASURED | `Project Steering/PREREG_V8F_FLATDAG.md` |" + g2[j:]
assert len(g2.split(b"\n")) == len(g.split(b"\n"))
for w in (b"void", b"superseded", b"retracted", b"formerly", b"no longer", b"correction", b"withdrawn"):
    assert w not in new.lower() and w not in res.lower(), w
put("mods-rw/" + GC, g2)
put("msg.txt", ("RESULT_V8F_FLATDAG: NOT ESTABLISHED (H-V8F-FLATDAG-1). Read by the pre-registered reader on 4/4 new T1-kin rollouts (dev-box CPU) +"
                " DAG's banked ones, all 91 controls PASS. PRIMARY DAG - FLAT-DAG ADE6 -0.3451 m at s0 (meets -2F), -0.0544 at s1 (n.s.); family"
                " rows separated-worse at both seeds (s0 DK-TG, DK-VIOL; s1 accel MAE). Beside: DAgger helps flat too (-0.15 / -0.19 m) and costs"
                " accel on both architectures; the hierarchy keeps 58 %% / 15 %% of B-2's margin. Committed next resolved: MIXO's T-side line ends at"
                " perception (SPEEDCAL -> LORA, A40 by its venue rule, not allocated). Package raw/2026-10-27-flatdag/ (%d files: box-2 run and bank"
                " metadata, the read chain and reader, the read; checkpoints, bank shards, dumps and ticks stay on E:). Register row updated.\n\n"
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n" % n).encode())
print("hold written", H, n)
