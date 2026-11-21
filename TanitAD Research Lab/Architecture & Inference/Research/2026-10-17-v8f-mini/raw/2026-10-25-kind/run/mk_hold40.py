"""Build qland/hold_v8fkind_run: PREREG_V8F_KIND's run banked (RESULT_V8F_KIND.md + raw/2026-10-25-kind/run/), register rows H-V8F-KIND-1 (verdict)
   and H-V8F-MIXO-1 (its condition met).   python mk_hold40.py <remote tip sha>"""
import glob
import os
import subprocess
import sys

TIP = sys.argv[1]
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/grid_a2/"
H = "C:/Users/Admin/qland/hold_v8fkind_run"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RUN = f"{PK}/raw/2026-10-25-kind/run/"
LG = "C:/Users/Admin/box2_ship/logs/"
E = LG + "v8f_kind/"
B2 = "C:/Users/Admin/box2_ship/"


def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    assert len(rel) <= 220, rel
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


lf = lambda b: b.replace(b"\r\n", b"\n")


def stage(src, rel, binary=False):
    assert show(TIP, rel) is None, f"{rel} already exists on the tip"
    b = open(src, "rb").read()
    assert b, src
    put("stage", rel, b if binary else lf(b))


for n in ("hold_v8fkind_run", "landed_hold_v8fkind_run"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
assert b"ZZV8FMINI-DONE" in open(LG + "chain_v8f_kind.out", "rb").read()
stage(S + "RESULT_V8F_KIND.md", f"{PK}/RESULT_V8F_KIND.md")
stage(S + "raw_readme_kind_run.md", RUN + "README.md")
for f in ("mk_hold40.py", "kind_facts.py", "kind_facts.out"):
    stage(S + f, RUN + f)
for f in ("kind_read.json", "check_inputs.json", "ckpt_sha256.json", "telemetry_v8fkind.csv"):
    stage(E + f, RUN + f)
ev = sorted(glob.glob(E + "eval/eval_M2b_s[01]-kind-*"))
assert len(ev) == 90, len(ev)
for p in ev:
    stage(p, RUN + "eval/" + os.path.basename(p), binary=p.endswith(".npz"))
for f in ("chain_v8f_kind.out", "chain_v8f_kind.wait1_stopped.out", "chain_v8f_kind_dryrun.out", "v8fkind_guard.out",
          "v8fkind10250717_kindrun.log", "v8fkind10250717_refresh.log"):
    stage(LG + f, RUN + "box2_logs/" + f)
for f in ("run_chain_v8f_kind.cmd", "wake_v8fkind.sh"):
    stage(B2 + f, RUN + "box2_logs/" + f)
lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-KIND-1** (2026-10-25) |")]
assert len(ix) == 1
pre = "| **PRE-REGISTERED, NOT RUN (box 2, eval only, under 1 h ESTIMATED; awaits instruments and the PI's go).** "
assert lines[ix[0]].count(pre) == 1
new = ("| **INTERACTIVE (as committed), 2026-10-25. MEASURED, T0, paired episode-cluster bootstrap, 141 episodes; `…/2026-10-17-v8f-mini/raw/2026-10-25-kind/run/kind_read.json`.**"
       " gain_k (m, s0 / s1): speed +0.270 / +0.185 (2F at s0 only, so no carrier by Amendment 2), angle +0.075 / +0.043, time-to-event −0.237 / −0.315,"
       " geometry −0.434 / −0.421, time gap and lead gap ≈ 0; the union of all 26 cells +1.076 / +0.923; A −0.30 / −0.55. GT in one kind alone makes O"
       " WORSE (tte, geom), and leaving only speed out of the full oracle costs 1.8–2.0 m: O reads the cells jointly and cannot use a partly right code."
       " Identity controls exactly +0 at both seeds. Next (committed): `PREREG_V8F_MIXO`. `RESULT_V8F_KIND.md`.")
lines[ix[0]] = lines[ix[0]].replace(pre, new + " ~~" + pre[2:].strip() + "~~ ")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-MIXO-1** (2026-10-25) |")]
assert len(ix) == 1 and lines[ix[0]].endswith(" |")
lines[ix[0]] = lines[ix[0]][:-2] + (" **Condition met 2026-10-25:** `H-V8F-KIND-1` read INTERACTIVE. Instruments are being built (no GPU); the box-2"
                                    " training run needs the PI's go. |")
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
msg = ("PREREG_V8F_KIND banked: INTERACTIVE as committed -> PREREG_V8F_MIXO (O-side) next\n\n"
       "MEASURED, T0, eval only on the banked M2b s0/s1 checkpoints, paired episode-cluster bootstrap (141 episodes), identity controls\n"
       "exactly +0 at both seeds. gain_k (m, s0/s1): speed +0.270/+0.185 (2F at s0 only), angle +0.075/+0.043, tte -0.237/-0.315,\n"
       "geom -0.434/-0.421, timegap/leadgap ~0; union +1.076/+0.923; A -0.30/-0.55 (bar 0.8). GT in one kind alone makes O worse, and\n"
       "dropping only speed from the full oracle costs 1.8-2.0 m: O cannot use a partly right code. Strata (Amendment 1): GT angle is worth\n"
       "+0.77/+0.55 m inside the 140 bend windows. Register: H-V8F-KIND-1 verdict, H-V8F-MIXO-1 condition met.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok", len(ev), "evals")
