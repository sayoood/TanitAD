"""Build qland/hold_v8f_sfresh: PREREG_V8F_SFRESH's run banked (RESULT_V8F_SFRESH.md + raw/2026-10-25-sfresh/, WITHOUT the 2.8 GB eval
   cache, the fetched checkpoints and the code export) and the register: H-V8F-SFRESH-1 (STALENESS ARTEFACT), H-MINI-STRAT-1 note.
   python mk_hold50.py <remote tip sha>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
SF, A2 = S + "sfresh/", S + "grid_a2/"
H = "C:/Users/Admin/qland/hold_v8f_sfresh"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = f"{PK}/raw/2026-10-25-sfresh/"
BIN = (".npz", ".npy", ".pt")
SKIP = ("cache/", "inputs/", "code/")


def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    assert len(rel) <= 220, rel
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


def stage(src, rel):
    assert show(TIP, rel) is None, f"{rel} already exists on the tip"
    b = open(src, "rb").read()
    put("stage", rel, b if rel.endswith(BIN) else b.replace(b"\r\n", b"\n"))


for n in ("hold_v8f_sfresh", "landed_hold_v8f_sfresh"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
n, skipped = 0, []
for root, dirs, files in os.walk(SF):
    for f in files:
        rel = os.path.relpath(os.path.join(root, f), SF).replace("\\", "/")
        if rel.startswith(SKIP) or "__pycache__" in rel:
            skipped.append(rel); continue
        assert os.path.getsize(SF + rel) < 10_000_000, rel
        stage(SF + rel, RAW + rel); n += 1
stage(A2 + "mk_hold50.py", RAW + "mk_hold50.py")
stage(SF + "SFRESH_RESULT.md", f"{PK}/RESULT_V8F_SFRESH.md")
lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-SFRESH-1** (2026-10-25) |")]
assert len(ix) == 1
pre = "| **PRE-REGISTERED, NOT RUN (2026-10-25).** "
assert lines[ix[0]].count(pre) == 1
new = ("| **STALENESS ARTEFACT, 2026-10-25 (both conditions at both seeds). MEASURED, T1-class open loop, DIAGNOSTIC (fresh ages are not the"
       " deployed schedule), eval-141 (2,319 windows with a 50 m checkpoint), paired episode-cluster bootstrap;"
       " `…/2026-10-17-v8f-mini/raw/2026-10-25-sfresh/READ.json`, `RESULT_V8F_SFRESH.md`.** Q-LAYER (fresh S − nav-only baseline, 50 m) ="
       " −2.43 [−3.22, −1.74] / −2.53 [−3.33, −1.79] m; Q-STALE (fresh − cycle) = −5.72 [−6.44, −5.06] / −5.73 [−6.46, −5.07] m. Beside: 80 m"
       " Q-LAYER −5.11 / −5.10, heading at 50 m −4.72 / −4.78 deg, all separated. Controls passed: checkpoint and clip hashes, the CPU cycle"
       " re-run reproduces the banked S error to ≤ 1e-4 m and ADE6 exactly, and the 291 windows whose cycle ages are (0, 0) are bit-identical."
       " Mechanism: the cycle − fresh gap grows with the ego's travel since S's tick (0 to 13.8 m) while the fresh error stays at 2.0–2.6 m."
       " ⇒ Clause 1 of `H-MINI-STRAT-1` PASSES on fresh ages. Committed next: the deployed-schedule cost (a faster S tick or a re-anchored"
       " checkpoint) becomes the strategic lever. ~~" + pre[2:].strip() + "~~ ")
lines[ix[0]] = lines[ix[0]].replace(pre, new)
js = [k for k, x in enumerate(lines) if x.startswith("| **H-MINI-STRAT-1** |")]
assert len(js) == 1
old = "| **OPEN.** If the scene adds nothing"
assert lines[js[0]].count(old) == 1
lines[js[0]] = lines[js[0]].replace(old, (
    "| **2026-10-25, `PREREG_V8F_SFRESH` (STALENESS ARTEFACT): clause 1 PASSES on fresh ages.** MEASURED, M2b: fresh S − nav-only baseline at"
    " 50 m = −2.43 [−3.22, −1.74] / −2.53 [−3.33, −1.79] m, at 80 m −5.11 / −5.10 m, separated at both seeds"
    " (`…/raw/2026-10-25-sfresh/READ.json`). The registered cycle-age metric still reads S worse (+3.28 / +3.21 m) because it scores the"
    " distance travelled since S's tick. M1 was not re-read. **OPEN** on the remaining clauses. If the scene adds nothing"))
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
msg = ("PREREG_V8F_SFRESH: STALENESS ARTEFACT -- the strategic layer beats nav-only kinematics once scored at its own tick\n\n"
       "MEASURED, T1-class open loop, DIAGNOSTIC, eval-141, paired episode-cluster bootstrap; controls passed (cycle re-run reproduces <= 1e-4 m).\n"
       "Q-LAYER fresh S - nav-only at 50 m: -2.43 [-3.22,-1.74] / -2.53 [-3.33,-1.79] m; Q-STALE fresh - cycle: -5.72 / -5.73 m, separated.\n"
       "H-MINI-STRAT-1 clause 1 passes on fresh ages. Committed next: the deployed-schedule cost (faster S tick or re-anchored checkpoint).\n"
       "Banked without the 2.8 GB eval cache, the fetched checkpoints and the code export.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok", n, "raw files; skipped", len(skipped))
