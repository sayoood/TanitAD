"""Build qland/hold_v8f_video2: evidence video v2 (mp4 beside v1) + its provenance under raw/2026-10-25-video-v2/ + register row D-V8F-VIDEO2-1. python mk_hold61.py <tip>"""
import glob
import json
import os
import subprocess
import sys

TIP = sys.argv[1]
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
V = SC + "viz2/"
A2 = SC + "grid_a2/"
H = "C:/Users/Admin/qland/hold_v8f_video2"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = f"{PK}/raw/2026-10-25-video-v2"
TEXT = (".md", ".json", ".py", ".log", ".txt")


def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


def stage(src, rel):
    assert show(TIP, rel) is None, rel
    b = open(src, "rb").read()
    if src.endswith(TEXT):
        b = b.replace(b"\r\n", b"\n")
    put("stage", rel, b)


for n in ("hold_v8f_video2", "landed_hold_v8f_video2"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
stage(V + "v8f_evidence_v2.mp4", f"{PK}/report/2026-10-25/v8f_evidence_v2.mp4")
for f in ("SELECTION_V2.md", "CONTROLS_V2.json", "CONTROLS_V2_part1.json", "CONTROLS_V2_part2.json", "VIDEO_NOTES.md",
          "video_chapters.json", "build_video.log", "cf_evalwide.log"):
    stage(V + f, f"{RAW}/{f}")
for f in sorted(glob.glob(V + "*.py")):
    stage(f, f"{RAW}/scripts/{os.path.basename(f)}")
for f in sorted(glob.glob(V + "data/*")):
    stage(f, f"{RAW}/data/{os.path.basename(f)}")
for f in ("qa_02e_listen_navpath.png", "qa_03b_junction_all.png", "qa_03g_summary.png", "qa_04e_plans_table.png",
          "qa_05c_S_card.png", "_t_s_card.png"):
    stage(V + f, f"{RAW}/stills/{f}")
stage(A2 + "mk_hold61.py", f"{RAW}/mk_hold61.py")
put("stage", f"{RAW}/NOT_BANKED.txt", (
    "Not banked, on purpose:\n"
    "* viz2/code/ (38 MB): an export of tip 9a6551d5's stack/ and taniteval/, which is in git already.\n"
    "* 19 of the 25 qa_*.png stills: every frame is reproducible from the mp4; the 6 banked under stills/ are the ones the review read.\n"
    "Pending, not in the mp4: three source fixes in scripts/v2_scenes.py (segment 5's S-age chart, preview stills/_t_s_card.png;\n"
    "a shortened segment-5 caption; the 23-of-24 wording). The rebuild was skipped because C: was below 1.5 GB free.\n"
    "Rebuild: VIDEO_NOTES.md section 0.\n").encode("utf-8"))

cf = json.load(open(V + "data/cf_evalwide.json"))["seeds"]


def d6(s, cfg):
    x = cf[s][cfg]["D6"]
    return f"{x['mean']:.2f} [{x['lo']:.2f}, {x['hi']:.2f}]"


def fol(s):
    t = cf[s]["a_all"]["T_lat_argmax_follows_cmd"]
    return "/".join(f"{100 * t[c]:.0f}" for c in ("FOLLOW", "TURN_L", "TURN_R"))


cells = "; ".join(f"{name} {d6('M2b_s0', k)} / {d6('M2b_s1', k)}" for name, k in
                  (("all three layers", "a_all"), ("S only", "d_S"), ("T only", "c_T"), ("O only", "b_O")))
lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-LISTEN2-1** (2026-10-25) |")]
assert len(ix) == 1 and not any(x.startswith("| **D-V8F-VIDEO2-1**") for x in lines)
row = ("| **D-V8F-VIDEO2-1** (2026-10-25) | Evidence video v2 (`report/2026-10-25/v8f_evidence_v2.mp4`, 146.4 s): who listens to what;"
       " the nav counterfactual (a constant command injected at the named layer(s), the model's own S -> T -> O chain);"
       " better plans (B / H / HS / P10 by horizon); S at its own tick. Strategic family PAUSED (PI, 2026-09-19). |"
       " **MEASURED, DESCRIPTIVE (2026-10-25).** T1-class open loop, an INTERVENTION on the input, not driving."
       " Eval-wide D6 (m; distance at 6 s from the FOLLOW plan, mean of TURN_L/R; 2,399 windows; episode-cluster bootstrap, B = 2,000,"
       f" 141 episodes; s0 / s1): {cells}. T-lateral argmax equals the command (FOLLOW/TURN_L/TURN_R, all three layers):"
       f" {fol('M2b_s0')} % (s0), {fol('M2b_s1')} % (s1). ⇒ The command moves the plan through S's checkpoints and T's code; O's own nav token"
       " is nearly inert, which agrees with H-V8F-LISTEN-1 (Q-NAV). Controls ALL_PASS: own nav reproduces `forward_infer` to 0.0;"
       " drawn plans equal the banked arrays to 0.0 m; zeroed nav gives exactly 0.0 over 160 cells."
       f" `{RAW}/` (VIDEO_NOTES.md, data/cf_evalwide.json). |")
lines[ix[0] + 1:ix[0] + 1] = [row]
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
msg = ("v8F evidence video v2: who listens to what, the nav counterfactual, better plans, S at its own tick\n\n"
       "mp4 beside v1 in report/2026-10-25/; provenance (pre-declared selection, controls ALL_PASS, data, scripts, 6 stills) in\n"
       "raw/2026-10-25-video-v2/. Register: D-V8F-VIDEO2-1 (MEASURED, descriptive, T1-class open loop, intervention).\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok;", row[:200])
