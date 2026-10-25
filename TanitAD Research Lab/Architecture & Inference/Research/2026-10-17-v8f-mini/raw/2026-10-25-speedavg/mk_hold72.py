"""Build qland/hold_v8f_speedavg_instr: PREREG_V8F_SPEEDAVG's instruments (19 NEW, from S/speedavg, the test guarded for the stack-only import probe; the smoke's stand-in evals renamed so no
file can be read as the arm) + PREREG Amendment 1 (the PARTIAL reading, before any SPEEDAVG data) + register note. python mk_hold72.py <tip>"""
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
T = SC + "speedavg/"
A2 = SC + "grid_a2/"
H = "C:/Users/Admin/qland/hold_v8f_speedavg_instr"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RS = f"{PK}/raw/2026-10-25-speedavg"
TOKEN = re.compile(rb"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")


def git(*a):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", *a], capture_output=True)
    return r.stdout if r.returncode == 0 else None


BS = chr(92)


def lp(p):   # Windows long-path form: the scratch prefix + the repo path exceed MAX_PATH (260)
    return BS + BS + "?" + BS + os.path.abspath(p).replace("/", BS)


def rd(p):
    return open(lp(p), "rb").read()


def put(kind, rel, data):
    assert not TOKEN.search(data), f"token-shaped string in {rel}"
    p = os.path.join(H, kind, rel)
    os.makedirs(lp(os.path.dirname(p)), exist_ok=True)
    open(lp(p), "wb").write(data)


for n in ("hold_v8f_speedavg_instr", "landed_hold_v8f_speedavg_instr"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
rows = [x.split("\t")[0] for x in open(T + "FILES.txt", encoding="utf-8").read().splitlines() if x.strip() and not x.startswith("#")]
assert len(rows) == 19, rows
for p in rows:
    b = rd(T + p).replace(b"\r\n", b"\n")
    dst = p
    if f"{RS}/smoke/eval/" in p:   # stand-in B checkpoints under the arm's tags: rename the folder so no file reads as X
        dst = p.replace(f"{RS}/smoke/eval/", f"{RS}/smoke/eval_STANDIN_B_checkpoints_NOT_X/")
    assert git("rev-parse", f"{TIP}:{dst}") is None, dst
    put("stage", dst, b)
lab = f"{RS}/smoke/SMOKE_LABEL.txt"
put("stage", lab, rd(T + lab).replace(b"\r\n", b"\n").rstrip(b"\n") + (
    "\n\nMM, 2026-10-25, at banking: the per-mode eval JSONs are under `eval_STANDIN_B_checkpoints_NOT_X/` (renamed from `eval/`). Their file names"
    " and tags say `M2b-mixo_s0/s1` because the driver names outputs after the arm it will run; the checkpoints inside are B's (each JSON's"
    " `ckpt` field names `.../M2b_s0|s1/ckpt_last.pt`). X_s1 did not exist when this smoke ran.\n").encode())
put("stage", f"{RS}/HANDOFF.md", rd(T + "HANDOFF.md").replace(b"\r\n", b"\n"))
put("stage", f"{RS}/mk_hold72.py", open(A2 + "mk_hold72.py", "rb").read().replace(b"\r\n", b"\n"))

pr = "Project Steering/PREREG_V8F_SPEEDAVG.md"
old = git("show", f"{TIP}:{pr}")
assert old is not None and b"Amendment 1" not in old
put("mods-rw", pr, old.replace(b"\r\n", b"\n").rstrip(b"\n") + (
    "\n\n## Amendment 1 (2026-10-25, MM): written BEFORE any SPEEDAVG data (no X_s1 exists; only a stand-in plumbing smoke ran)\n\n"
    "The instrument build found one case the PARTIAL row does not settle: PRIMARY negative and separated at both seeds, ≤ −F at one seed and"
    " above −F at the other (for example −0.30 / −0.05 m). It is fixed here, before data:\n"
    "* **That case is PARTIAL** (the both-seed bar is missed at one seed, which is what \"but above −F\" excludes from VARIANCE PAYS)."
    " `v8f_speedavg_read.py` applies this reading and sets `partial_reading_dependent: true` whenever the other literal reading (above −F at"
    " BOTH seeds) would have said MIXED; the result reports that flag.\n"
    "* Both seeds ≤ −F with a separated-worse SECONDARY stays **MIXED** (unchanged).\n\n"
    "Instrument: `stack/scripts/v8f_mini_eval_speedavg.py` (modes `avg` / `self` / `gt` / `own` = the arm and controls 1–3),"
    " `code/v8f_speedavg_read.py`, `code/devbox_ops/run_speedavg.sh` (dev-box CPU; pulls the two X checkpoints read-only from box 2 and"
    " sha-checks them; STOPs unless KINDX's seed-1 read made SPEED a carrier). Launch with `SPEEDAVG_SF`, `SPEEDAVG_OUT` and `SPEEDAVG_PY` set:"
    " the defaults point at the builder's scratch. 53 tests; 38 of 38 mutants RED.\n").encode("utf-8"))

g = "Project Steering/GOALS_AND_CLAIMS.md"
lines = git("show", f"{TIP}:{g}").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-SPEEDAVG-1** (2026-10-25) |")]
assert len(ix) == 1
lines[ix[0]] = lines[ix[0]].rstrip().rstrip("|").rstrip() + (
    " **Instruments landed 2026-10-25** (`v8f_mini_eval_speedavg.py`, `v8f_speedavg_read.py`, `run_speedavg.sh`; 53 tests, 38 / 38 mutants RED;"
    " a stand-in plumbing smoke on B's checkpoints passes controls 1–3, no bar read). Amendment 1 (before data): a seed pair with one seed ≤ −F"
    " and the other above −F reads PARTIAL. |")
put("mods-rw", g, "\n".join(lines).encode("utf-8"))
msg = ("SPEEDAVG instruments (conditional zero-training arm) + PREREG_V8F_SPEEDAVG Amendment 1 (the PARTIAL reading, before data)\n\n"
       "Driver (avg / self / gt / own), reader, dev-box runner; 53 tests, 38/38 mutants RED; stand-in smoke passes controls 1-3.\n"
       "Runs only if KINDX's seed-1 read makes SPEED a carrier. Register: H-V8F-SPEEDAVG-1.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok")
