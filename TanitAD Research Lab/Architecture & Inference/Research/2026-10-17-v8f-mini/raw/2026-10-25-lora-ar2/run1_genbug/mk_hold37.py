"""Build qland/hold_v8flora_genfix: the LoRA live check's generator on the model's device (AR-2 run 1 crashed on it), its pinning test,
   run 1's evidence (raw/2026-10-25-lora-ar2/run1_genbug/) and the register note on H-V8F-LORA-1.   python mk_hold37.py <remote tip sha>"""
import glob
import os
import subprocess
import sys

TIP = sys.argv[1]
BASE = "1cfa5163"
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
X, A2 = S + "fix_gen/", S + "grid_a2/"
H = "C:/Users/Admin/qland/hold_v8flora_genfix"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RUN = f"{PK}/raw/2026-10-25-lora-ar2/run1_genbug/"
R1 = "C:/Users/Admin/box2_ship/logs/v8f_lora_ar2.run1_genbug/"
LG = "C:/Users/Admin/box2_ship/logs/"


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


for n in ("hold_v8flora_genfix", "landed_hold_v8flora_genfix"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
tr = "stack/scripts/v8f_mini_train.py"
old = show(TIP, tr)
assert old is not None and old == show(BASE, tr), f"{tr}: the tip moved since {BASE}"
new = lf(open(X + tr, "rb").read())
assert new != old and new.count(b"torch.Generator(device=dev).manual_seed(s_)") == 1
assert new.replace(b"torch.Generator(device=dev).manual_seed(s_))   # the generator lives on the model's device: a CPU one crashed AR-2's cuda smoke",
                   b"torch.Generator().manual_seed(s_))") == old, "the fix must be exactly one line"
put("mods-rw", tr, new)
stage(X + "stack/tests/test_v8f_lora_device.py", "stack/tests/test_v8f_lora_device.py")
stage(A2 + "raw_readme_ar2_run1.md", RUN + "README.md")
stage(A2 + "mk_hold37.py", RUN + "mk_hold37.py")
stage(A2 + "fixgen_suites.out", RUN + "fixgen_suites.out")
for f in ("identity_read.json", "frames_vs_banked.json", "preflight_manifest_only.json", "frames_sha256.txt", "smoke/smoke.log"):
    stage(R1 + f, RUN + f)
for p in sorted(glob.glob(R1 + "identity/*")):
    stage(p, RUN + "identity/" + os.path.basename(p))
stage(LG + "chain_v8f_lora_ar2.run1_genbug.out", RUN + "box2_logs/chain_v8f_lora_ar2.run1_genbug.out")
stage(LG + "v8floraar2_guard.run1_genbug.out", RUN + "box2_logs/v8floraar2_guard.run1_genbug.out")
stage(LG + "chain_v8f_lora_ar2_dryrun.out", RUN + "box2_logs/chain_v8f_lora_ar2_dryrun.out")
for p in sorted(glob.glob(LG + "v8flora10250614_*.log")):
    stage(p, RUN + "box2_logs/" + os.path.basename(p))
lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-LORA-1** (2026-10-25) |")]
assert len(ix) == 1 and lines[ix[0]].endswith(" |")
lines[ix[0]] = lines[ix[0]][:-2] + (" **AR-2 run 1 (box 2, 2026-10-25 06:14–06:25, PI go):** the same-device identity control PASSED, bit for bit:"
                                    " rel L2 exactly 0.0 and 100 % bitwise-equal tokens on all 6 clips (RTX 4060). The pixel-scale mutant read 0.028 and"
                                    " was caught (`raw/2026-10-25-lora-ar2/run1_genbug/identity_read.json`). The smoke then crashed in the LoRA live"
                                    " check on a CPU `torch.Generator` against the cuda model. Fixed in `stack/scripts/v8f_mini_train.py` and pinned by"
                                    " `stack/tests/test_v8f_lora_device.py` (control + mutant); rerun. |")
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
msg = ("LoRA live check: the generator on the model's device (AR-2 run 1's smoke crashed on a CPU one); run 1's identity PASS banked\n\n"
       "AR-2 run 1 on box 2 (PI go): the same-device identity control PASSED bit for bit (rel L2 0.0, 100 % bitwise-equal tokens, 6 clips,\n"
       "RTX 4060; pixel-scale mutant 0.028 caught). The 200-step smoke then raised 'Expected a cuda device type for generator but found\n"
       "cpu' in live_check_lora, which built torch.Generator() on the CPU for a cuda model. CPU-only tests could not see it.\n"
       "Fix: torch.Generator(device=dev), one line. test_v8f_lora_device.py pins it by AST, with a control (the CPU live check's\n"
       "device-less generator must be found) and a mutant (the old line turns it RED). v8F suites: run1_genbug/fixgen_suites.out.\n"
       "Register: H-V8F-LORA-1.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok")
