"""Build qland/hold_v8f_kindx_s1_result: KINDX's seed-1 confirmation (CARRIER: speed) banked under raw/2026-10-25-kindx-s1/ (reads, controls,
run records, eval JSONs, npz sha256, chain + recovery logs and scripts), the addendum on RESULT_V8F_KINDX, register H-V8F-KINDX-1.
python mk_hold75.py <tip>"""
import glob
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
H = "C:/Users/Admin/qland/hold_v8f_kindx_s1_result"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RK = f"{PK}/raw/2026-10-25-kindx-s1"
B = "C:/Users/Admin/box2_ship/"
L = B + "logs/"
E = L + "v8f_kindx_s1/"
TOKEN = re.compile(rb"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")


def git(*a):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", *a], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    assert not TOKEN.search(data), f"token-shaped string in {rel}"
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


def new(src, rel):
    assert git("rev-parse", f"{TIP}:{rel}") is None, rel
    put("stage", rel, open(src, "rb").read().replace(b"\r\n", b"\n"))


for n in ("hold_v8f_kindx_s1_result", "landed_hold_v8f_kindx_s1_result"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
for f in ("READ_s1.json", "controls_s1.json", "check_inputs_s1.json", "ckpt_sha256.json", "gate_token.json", "queue_trainmixo_s0.txt", "queue_trainmixo_s1.txt"):
    new(E + f, f"{RK}/{f}")
new(E + "gate_M2b-mixo/selfcheck.json", f"{RK}/gate_M2b-mixo/selfcheck.json")
for f in ("argv.json", "selfcheck.json", "summary.json", "train.log", "train_log.jsonl"):
    new(E + f"runs/M2b_s1_mixo/{f}", f"{RK}/runs/M2b_s1_mixo/{f}")
sha = []
for p in sorted(glob.glob(E + "eval/*")):
    n = os.path.basename(p)
    if n.endswith(".npz"):
        sha.append(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  eval/{n}")
    else:
        new(p, f"{RK}/eval/{n}")
sha.append(f"{hashlib.sha256(open(E + 'runs/M2b_s1_mixo/ckpt_last.pt', 'rb').read()).hexdigest()}  runs/M2b_s1_mixo/ckpt_last.pt (not banked; box 2 + dev box)")
put("stage", f"{RK}/SHA256_NOT_BANKED.txt", ("# the npz dumps and the checkpoint are not banked (size); they live in box2_ship/logs/v8f_kindx_s1/ and on box 2\n" + "\n".join(sha) + "\n").encode())
for f in ("chain_v8f_kindx_s1.driver_died_1945.out", "chain_v8f_kindx_s1.out", "relaunch_kindx_s1.out", "kindx_s1_finish.out", "v8fkindx_guard.out",
          "BOX2_BUSY.stale_kindxs1_driver_died_1945.txt", "chain_v8f_flat.hold_kindx_pull_2140.out"):
    new(L + f, f"{RK}/ops/{f}")
for f in ("relaunch_kindx_s1.sh", "kindx_s1_finish.sh", "run_guard_v8fkindx.cmd", "run_relaunch_kindx_s1.cmd", "wake_v8fkindxs1_relaunch.sh"):
    new(B + f, f"{RK}/ops/{f}")
put("stage", f"{RK}/mk_hold75.py", open(SC + "grid_a2/mk_hold75.py", "rb").read().replace(b"\r\n", b"\n"))

rk = f"{PK}/RESULT_V8F_KINDX.md"
old = git("show", f"{TIP}:{rk}")
assert old is not None and b"## SEED-1 CONFIRMATION" not in old
put("mods-rw", rk, old.replace(b"\r\n", b"\n").rstrip(b"\n") + b"\n" + open(SC + "kindx_s1_res/ADDENDUM.md", "rb").read().replace(b"\r\n", b"\n"))

g = "Project Steering/GOALS_AND_CLAIMS.md"
lines = git("show", f"{TIP}:{g}").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-KINDX-1** (2026-10-25) |")]
assert len(ix) == 1
lines[ix[0]] = lines[ix[0]].rstrip().rstrip("|").rstrip() + (
    " **MEASURED 2026-10-25, seed 1: CARRIER (speed)** (`RESULT_V8F_KINDX.md` §SEED-1, `raw/2026-10-25-kindx-s1/`; T0 oracle, paired"
    " episode-cluster bootstrap). gain_speed(X_s1) +0.668 [+0.575, +0.761] m against +0.716 at s0; no other kind clears 2F. B_s1's"
    " gain_speed is +0.185, and B's tte / geom gains are negative (−0.315 / −0.421) where X's are positive at both seeds: MIXO's mechanism"
    " replicates. Controls PASS. Committed next (Amendment 1): PREREG_V8F_SPEEDAVG runs (launched 21:40, dev-box CPU). |")
put("mods-rw", g, "\n".join(lines).encode("utf-8"))
msg = ("KINDX seed 1: CARRIER (speed) -- gain_speed(X_s1) +0.668 m (s0 +0.716); SPEEDAVG runs (Amendment 1)\n\n"
       "T0 oracle on X = M2b-mixo_s1 (sha 68a0f335); controls PASS. MIXO's mechanism replicates: B's tte / geom gains are negative,\n"
       "X's positive at both seeds. Banked under raw/2026-10-25-kindx-s1/ with the driver-death recovery (relaunch + step-7 finisher).\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok")
