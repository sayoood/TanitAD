"""Build qland/hold_v8flora_ar2_pass: AR-2 run 3 banked (raw/2026-10-25-lora-ar2/run3/), the register note on H-V8F-LORA-1 (AR-2 PASS; Amendment 3's
   diagnosis MEASURED).   python mk_hold41.py <remote tip sha>"""
import glob
import os
import subprocess
import sys

TIP = sys.argv[1]
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/grid_a2/"
H = "C:/Users/Admin/qland/hold_v8flora_ar2_pass"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RUN = f"{PK}/raw/2026-10-25-lora-ar2/run3/"
LG = "C:/Users/Admin/box2_ship/logs/"
E = LG + "v8f_lora_ar2/"


def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    assert len(rel) <= 220, rel
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


lf = lambda b: b.replace(b"\r\n", b"\n")


def stage(src, rel):
    assert show(TIP, rel) is None, f"{rel} already exists on the tip"
    b = open(src, "rb").read()
    assert b, src
    put("stage", rel, lf(b))


for n in ("hold_v8flora_ar2_pass", "landed_hold_v8flora_ar2_pass"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
chain = open(LG + "chain_v8f_lora_ar2.out", "rb").read()
assert b"ZZV8FMINI-DONE" in chain and b"smoke-path identity probe rel-L2 0.00e+00" in chain
stage(S + "raw_readme_ar2_run3.md", RUN + "README.md")
stage(S + "mk_hold41.py", RUN + "mk_hold41.py")
for f in ("identity_read.json", "ar2_read.json", "ar2_read.out", "frames_vs_banked.json", "preflight_manifest_only.json", "frames_sha256.txt",
          "smoke/smoke_report.json", "smoke/smoke_steps.jsonl", "smoke/smoke.log", "smoke/run/selfcheck.json", "smoke/run/summary.json",
          "smoke/run/argv.json", "smoke/run/train_log.jsonl"):
    stage(E + f, RUN + f)
for p in sorted(glob.glob(E + "identity/*")):
    stage(p, RUN + "identity/" + os.path.basename(p))
for f in ("chain_v8f_lora_ar2.out", "v8floraar2_guard.out", "v8f_lora_ar2_HANDOFF.md"):
    stage(LG + f, RUN + "box2_logs/" + f)
logs = sorted(glob.glob(LG + "v8flora10250735_*.log"))
assert len(logs) == 3, logs
for p in logs:
    stage(p, RUN + "box2_logs/" + os.path.basename(p))
lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-LORA-1** (2026-10-25) |")]
assert len(ix) == 1 and lines[ix[0]].endswith(" |")
lines[ix[0]] = lines[ix[0]][:-2] + (" **AR-2 PASSED, run 3 (07:35–07:49, tip `c0769eaa`):** with builder-shaped encoder calls (Amendment 3) the training"
                                    " path's identity probe reads rel L2 **exactly 0.0** (was 2.41e-2), so the batch-shape diagnosis is MEASURED. The"
                                    " same-device identity control passed bit for bit again (the mutant was caught at 0.028). Smoke: 1.96 s/step, 1.44"
                                    " GiB; 470 h per seed projected on box 2, so the full run needs the A40 and the PI's go"
                                    " (`raw/2026-10-25-lora-ar2/run3/`). After `H-V8F-KIND-1` read INTERACTIVE, the MM recommends MIXO before it. |")
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
msg = ("PREREG_V8F_LORA AR-2 PASSED (run 3): the training path is bit for bit the cache with builder-shaped calls\n\n"
       "Box 2, tip c0769eaa, PI go: same-device identity PASS (rel-L2 0.0; mutant caught at 0.028), and the training path's identity\n"
       "probe now reads rel-L2 exactly 0.0 (2.41e-2 in run 2): Amendment 3's batch-shape diagnosis is MEASURED. Smoke 200 steps:\n"
       "1.96 s/step, peak 1.44 GiB; 470 h per seed projected on box 2 (ESTIMATED) => the A40 by the venue rule, which needs the\n"
       "PI's go. Register: H-V8F-LORA-1.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok")
