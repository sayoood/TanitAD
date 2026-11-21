"""Build qland/hold_v8fdscale_prereg: PREREG_V8F_DSCALE + the 2026-10-25 zero-GPU facts + R66 corrections + the register.
   python mk_hold25.py <remote tip sha>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
A2, EN = S + "grid_a2/", S + "ens/"
H = "C:/Users/Admin/qland/hold_v8fdscale_prereg"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = f"{PK}/raw/2026-10-25-ensemble/"


def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    assert len(rel) <= 220, rel
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


def lf(b):
    return b.replace(b"\r\n", b"\n")


def stage(src, rel):
    assert show(TIP, rel) is None, f"{rel} already exists on the tip"
    b = open(src, "rb").read()
    assert b, src
    put("stage", rel, lf(b))


def append(rel, src, guard):
    cur = show(TIP, rel)
    assert cur is not None and guard not in cur, rel
    put("mods", rel, cur.rstrip(b"\n") + b"\n" + lf(open(src, "rb").read()))


for n in ("hold_v8fdscale_prereg", "landed_hold_v8fdscale_prereg"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
stage(A2 + "PREREG_V8F_DSCALE.md", "Project Steering/PREREG_V8F_DSCALE.md")
stage(A2 + "raw_readme_ens.md", RAW + "README.md")
stage(A2 + "mk_hold25.py", RAW + "mk_hold25.py")
for b in ("ens_facts", "lon_floor", "paired_swing"):
    for e in ("py", "json", "out"):
        stage(EN + f"{b}.{e}", RAW + f"{b}.{e}")
append(f"{PK}/RESULT_V8F_HIST.md", A2 + "hist_corr6.md", b"## 6. Correction")
append(f"{PK}/RESULT_V8F_GRID.md", A2 + "grid_corr.md", b"## Correction (2026-10-25)")
append("Project Steering/RETRACTION_LOG.md", A2 + "r66.md", b"### R66 ")

reg = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8")
lines = reg.split("\n")


def row(prefix):
    ix = [k for k, x in enumerate(lines) if x.startswith(prefix)]
    assert len(ix) == 1, prefix
    assert lines[ix[0]].endswith(" |") and lines[ix[0]].count(" | ") == 2, prefix
    return ix[0]


assert not any(x.startswith("| **H-V8F-DSCALE-1**") for x in lines)
ih = row("| **H-V8F-HIST-1** (2026-10-23) |")
lines[ih] = lines[ih][:-2] + " " + open(A2 + "reg25_hist_tail.txt", "rb").read().decode("utf-8").strip() + " |"
isc = row("| **H-DATA-SCALE-1** |")
lines[isc] = lines[isc][:-2] + " v8F-mini instance pre-registered 2026-10-25: `H-V8F-DSCALE-1` (`PREREG_V8F_DSCALE.md`). |"
new = open(A2 + "reg25_dscale.txt", "rb").read().decode("utf-8").strip()
assert new.startswith("| **H-V8F-DSCALE-1**") and new.count(" | ") == 2 and new.endswith(" |")
idd = row("| **H-V8F-D1-1** (2026-10-23) |")
lines.insert(idd + 1, new)
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
open(os.path.join(H, "msg.txt"), "wb").write(lf(open(A2 + "msg25.txt", "rb").read()))
print("hold ok")
