"""Build qland/hold_v8fnavturn: the nav-vs-turn-label audit (raw/2026-10-25-navturn/), R71, PREREG_V8F_KIND Amendment 1 (before any KIND data),
   register rows D-V8F-T-ATTRIB-1 (corrected) and H-V8F-KIND-1 (amendment note).   python mk_hold35.py <remote tip sha>"""
import glob
import os
import subprocess
import sys

TIP = sys.argv[1]
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/grid_a2/"
NT = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/navturn/"
H = "C:/Users/Admin/qland/hold_v8fnavturn"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = f"{PK}/raw/2026-10-25-navturn/"


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


for n in ("hold_v8fnavturn", "landed_hold_v8fnavturn"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
stage(S + "raw_readme_navturn.md", RAW + "README.md")
stage(S + "mk_hold35.py", RAW + "mk_hold35.py")
stage(NT + "NAVTURN_AUDIT.md", RAW + "NAVTURN_AUDIT.md")
stage(NT + "nt_common.py", RAW + "nt_common.py")
scripts = sorted(p for p in glob.glob(NT + "*.py") if not p.endswith("nt_common.py"))
assert len(scripts) == 8, [os.path.basename(p) for p in scripts]
for p in scripts:
    stage(p, RAW + os.path.basename(p))
    stage(p[:-3] + ".out", RAW + os.path.basename(p)[:-3] + ".out")
for f in ("navturn_windows.csv", "repro_turns.json"):
    stage(NT + f, RAW + f)
for f in ("repro_eval141.npz", "navturn_cause.npy"):
    stage(NT + f, RAW + f, binary=True)

rl = "Project Steering/RETRACTION_LOG.md"
old = show(TIP, rl)
assert old.endswith(b"\n") and b"### R71 " not in old and b"### R70 " in old
put("mods", rl, old + lf(open(S + "r71.md", "rb").read()))
pr = "Project Steering/PREREG_V8F_KIND.md"
old = show(TIP, pr)
assert old.endswith(b"\n") and b"## Amendment 1" not in old
put("mods", pr, old + lf(open(S + "kind_amend1.md", "rb").read()))

lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **D-V8F-T-ATTRIB-1** (2026-10-25) |")]
assert len(ix) == 1
c = "T is at chance there, 0.95 SD, against 0.40 when the route speaks"
assert lines[ix[0]].count(c) == 1
lines[ix[0]] = lines[ix[0]].replace(c, "~~" + c + "~~ CORRECTED R71: confounded by the nav's speed gate; like-for-like 0.65 [0.50, 0.75] against"
                                    " 0.41 SD at s0, and 74 % of those windows are fast bends the nav never announces")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-KIND-1** (2026-10-25) |")]
assert len(ix) == 1 and lines[ix[0]].endswith(" |")
lines[ix[0]] = lines[ix[0]][:-2] + (" **Amendment 1 (before any data):** every gain is also read by window stratum (junction / curve / no-turn), beside,"
                                    " deciding nothing; it is the curve-only vs junction-only oracle at no extra eval (R71). |")
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
msg = ("Nav-vs-turn-label audit (EXPLORATORY, CPU) + R71 + PREREG_V8F_KIND Amendment 1 (before any data)\n\n"
       "MEASURED on eval-141, raw/2026-10-25-navturn/: the nav token announces only is_turn turns (vmin <= 8 m/s, not suppressed);\n"
       "the turn cells take any >= 20 deg segment in [now+2, now+8] s. 156 of the 210 'route-silent' turn windows (74 %) are fast bends\n"
       "the speed gate never announces; 129 nav-turn windows without a turn label are horizon / band mismatches.\n"
       "R71: T_ATTRIB's 0.95-vs-0.40 SD contrast was junction vs bend; like-for-like it is 0.65 [0.50, 0.75] vs 0.41 at s0, on 37 windows.\n"
       "KIND Amendment 1: every gain is also read by window stratum (junction / curve / no-turn), beside, deciding nothing.\n"
       "Register: D-V8F-T-ATTRIB-1 corrected, H-V8F-KIND-1 noted.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok")
