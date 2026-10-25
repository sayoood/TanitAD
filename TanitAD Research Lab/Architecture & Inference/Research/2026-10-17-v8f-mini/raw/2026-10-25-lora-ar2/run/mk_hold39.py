"""Build qland/hold_v8flora_shape: PREREG_V8F_LORA Amendment 3 (builder-shaped encoder calls of exactly 8 frames on the training path), its test,
   AR-2 run 2's evidence (raw/2026-10-25-lora-ar2/run/) and the register note on H-V8F-LORA-1.   python mk_hold39.py <remote tip sha>"""
import glob
import os
import subprocess
import sys

TIP = sys.argv[1]
BASE = "1cfa5163"
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
X, A2 = S + "fix_gen2/", S + "grid_a2/"
H = "C:/Users/Admin/qland/hold_v8flora_shape"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RUN = f"{PK}/raw/2026-10-25-lora-ar2/run/"
E = "C:/Users/Admin/box2_ship/logs/v8f_lora_ar2/"
LG = "C:/Users/Admin/box2_ship/logs/"
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


def stage(src, rel):
    assert show(TIP, rel) is None, f"{rel} already exists on the tip"
    b = open(src, "rb").read()
    assert b, src
    put("stage", rel, lf(b))


for n in ("hold_v8flora_shape", "landed_hold_v8flora_shape"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
enc = "stack/tanitad/models/v8f_lora_encoder.py"
old = show(TIP, enc)
assert old is not None and old == show(BASE, enc), f"{enc}: the tip moved since {BASE}"
new = open(X + enc, "rb").read()
assert b"\r" not in new and new.count(b"pooled_builder_shaped(enc,") == 2 and new.count(b"def pooled_builder_shaped(") == 1 and b"BUILDER_BATCH = 8\n" in new
# the ONLY changes: the new constant + helper, and the two enc.pooled calls routed through it
back = new.replace(b"outs.append(pooled_builder_shaped(enc, frames_u8[s:s + chunk].to(device)))", b"outs.append(enc.pooled(frames_u8[s:s + chunk].to(device)))")
back = back.replace(b"p = pooled_builder_shaped(enc, frames_u8[s:s + chunk].to(device))", b"p = enc.pooled(frames_u8[s:s + chunk].to(device))")
a = back.index(b"#: PREREG_V8F_LORA Amendment 3")
z = back.index(b"def encode_chunked(")
back = back[:a] + back[z:]
back = back.replace(b'(no autograd unless ``grad``). Inside a\n    chunk the encoder runs builder-shaped (``pooled_builder_shaped``)."""',
                    b'(no autograd unless ``grad``)."""')
assert back == old, "the encoder diff is wider than Amendment 3"
put("mods-rw", enc, new)
stage(X + "stack/tests/test_v8f_lora_shape.py", "stack/tests/test_v8f_lora_shape.py")
pr = "Project Steering/PREREG_V8F_LORA.md"
o = show(TIP, pr)
assert o.endswith(b"\n") and b"## Amendment 2" in o and b"## Amendment 3" not in o
put("mods", pr, o + lf(open(A2 + "lora_amend3.md", "rb").read()))
stage(A2 + "raw_readme_ar2_run2.md", RUN + "README.md")
stage(A2 + "mk_hold39.py", RUN + "mk_hold39.py")
stage(A2 + "shape_suites.out", RUN + "shape_suites.out")
for f in ("identity_read.json", "ar2_read.json", "ar2_read.out", "frames_vs_banked.json", "preflight_manifest_only.json", "frames_sha256.txt",
          "smoke/smoke_report.json", "smoke/smoke_steps.jsonl", "smoke/smoke.log", "smoke/run/selfcheck.json", "smoke/run/summary.json",
          "smoke/run/argv.json", "smoke/run/train_log.jsonl", "cache/eval/manifest.jsonl"):
    stage(E + f, RUN + f)
for p in sorted(glob.glob(E + "identity/*")):
    stage(p, RUN + "identity/" + os.path.basename(p))
for f in ("chain_v8f_lora_ar2.out", "v8floraar2_guard.out", "v8f_lora_ar2_HANDOFF.md"):
    stage(LG + f, RUN + "box2_logs/" + f)
for p in sorted(glob.glob(LG + "v8flora10250655_*.log")):
    stage(p, RUN + "box2_logs/" + os.path.basename(p))
for f in ("run_chain_v8f_lora_ar2.cmd", "wake_v8floraar2.sh"):
    stage(B2 + f, RUN + "box2_logs/" + f)
lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-LORA-1** (2026-10-25) |")]
assert len(ix) == 1 and lines[ix[0]].endswith(" |")
lines[ix[0]] = lines[ix[0]][:-2] + (" **AR-2 run 2 (06:55–07:10): FAILED as committed.** The identity control PASSED bit for bit again, and the smoke ran"
                                    " (2.02 s/step, 1.65 GiB; 494 h per seed projected on box 2, so the A40 by the venue rule). But the TRAINING path's"
                                    " identity probe read rel L2 2.41e-2 > τ_device 1e-3: 75.6 % bitwise equal, max |Δ| one fp8 step, from encoding 9"
                                    " frames in one call against the builder's batches of 8 (`raw/2026-10-25-lora-ar2/run/`). **Amendment 3:** every"
                                    " training-path encoder call is builder-shaped (exactly 8 frames, pads dropped, zero pad gradient; pinned by"
                                    " `stack/tests/test_v8f_lora_shape.py`); AR-2 reruns. The full run needs the PI's A40 go and is recommended to wait"
                                    " for `H-V8F-KIND-1`. |")
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
msg = ("PREREG_V8F_LORA Amendment 3 + AR-2 run 2 banked: builder-shaped encoder calls (exactly 8 frames) on the training path\n\n"
       "AR-2 run 2 (box 2, PI go): same-device identity PASS bit for bit; the smoke ran (2.02 s/step, 1.65 GiB, 494 h/seed projected on\n"
       "box 2 => A40 by the venue rule); AR-2 FAILED as committed: the training path's identity probe read rel-L2 2.41e-2 > 1e-3, 75.6 %\n"
       "bitwise equal, max |delta| one fp8 step - fp8 rounding flips from encoding 9 frames in one call against the builder's batches of 8.\n"
       "Fix: v8f_lora_encoder.pooled_builder_shaped runs every training-path encoder call on exactly BUILDER_BATCH = 8 frames, the last\n"
       "group padded by repeating its final frame, the pad rows dropped (no output, zero gradient). test_v8f_lora_shape.py: a spy sees\n"
       "only batch-8 calls, the pad gradient is analytic, the constant is pinned to the builder's loop, the unshaped call is a mutant.\n"
       "Register: H-V8F-LORA-1.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok")
