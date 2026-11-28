"""Build qland/hold_v8f_lora_ar3: AR-3 (the LoRA arm's launch-gate dry run, dev-box CPU) banked, with the MM's reading in the register row
H-V8F-LORA-1 and on D-V8F-SPEED-PERCEPTION.  python mk_hold86.py <tip>"""
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
A = SC + "ar3/"
H = "C:/Users/Admin/qland/hold_v8f_lora_ar3"
MIR = "C:/Users/Admin/tanitad-push"
RD = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-26-ar3"
TOKEN = re.compile(rb"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")


def git(*a):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", *a], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    assert not TOKEN.search(data), f"token-shaped string in {rel}"
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


for n in ("hold_v8f_lora_ar3", "landed_hold_v8f_lora_ar3"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
n = 0
for dp, _, fs in os.walk(A):
    for f in fs:
        src = os.path.join(dp, f)
        rel = os.path.relpath(src, A).replace("\\", "/")
        dst = f"{RD}/{rel}"
        assert git("rev-parse", f"{TIP}:{dst}") is None, dst
        put("stage", dst, open(src, "rb").read().replace(b"\r\n", b"\n"))
        n += 1
put("stage", f"{RD}/mk_hold86.py", open(SC + "grid_a2/mk_hold86.py", "rb").read().replace(b"\r\n", b"\n"))
NOTE = (" **AR-3 dry run (2026-10-26, dev-box CPU, tip `c2fad328`; `raw/2026-10-26-ar3/HANDOFF.md`):** the plan's AR-3 names `launch_gate.py`"
        " `PROFILES[\"v8f\"]`, which is the refcv8-chassis profile; its own parser refuses L's argv (73 refusals, `--arm M2b` is not a choice)."
        " v8F-mini's gate is the in-process one every box-2 arm runs (declared-vs-built, G-LIVE with an exact-zero frozen-prior control,"
        " `gate_token.json`; the trainer refuses without it). On that gate L's exact argv is refused on ONE item: 3,972 train clips have no"
        " `<clip>.v2ep.pt` frames on the dev box (AR-4). G-DVB covers 66 / 66 parser flags; two entries are weak: `accum` is checked by divisibility"
        " only (`v8f_mini_train.py:228`), and `--amp` is not refused under `--encoder lora` although Amendment 2 pins it off. Both are open: the"
        " trainer is what OACC's chain launches from the tip. G-LIVE, G-EVAL / G-CKPT and G-SUITE were not completed; the MM stopped their CPU"
        " runs to protect the T1-kin read's memory (3.2 GB of commit headroom), and any token needs the then-current tip anyway. Open placeholders"
        " in L's argv: `--out`, `--frames-root`, `--accum`; no `trainlora` queue stage exists. ⇒ AR-3 is a wording fix (name v8F-mini's"
        " in-process gate) plus the two G-DVB fixes; AR-4 (the frames on the run host) is the real blocker. |")
g = "Project Steering/GOALS_AND_CLAIMS.md"
lines = git("show", f"{TIP}:{g}").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-LORA-1** (2026-10-25) |")]
assert len(ix) == 1
lines[ix[0]] = lines[ix[0]].rstrip().rstrip("|").rstrip() + NOTE
put("mods-rw", g, "\n".join(lines).encode("utf-8"))
q = "Project Steering/PI_DECISION_QUEUE.md"
qt = git("show", f"{TIP}:{q}").decode("utf-8")
old = "  - Still to do: AR-3, a launch-gate PASS for the LoRA arm, and AR-4, the data staged on the A40.\n"
assert qt.count(old) == 1
new = old + ("  - **AR-3 dry run (MM, 2026-10-26, `H-V8F-LORA-1`):** on v8F-mini's in-process gate, L's exact argv is refused on ONE item, the"
             " missing train frames (AR-4). The plan's AR-3 wording names the refcv8 profile and needs correcting. Two weak G-DVB entries (`accum`, `--amp`)"
             " are to be fixed before the run.\n")
put("mods-rw", q, qt.replace(old, new).encode("utf-8"))
msg = ("AR-3 dry run banked: L's argv passes v8F-mini's in-process gate except the missing train frames (AR-4); plan wording names refcv8 profile\n\n"
       "G-DVB 66/66 flags covered; two weak entries (accum divisibility only, --amp not refused under lora) open. CPU gate jobs stopped (memory).\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok", n, "files")
