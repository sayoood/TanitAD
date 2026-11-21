"""Build qland/hold_v8f_hacc_bake: HACC's committed next step 2, part 1 -- the scored weights baked into the deployable checkpoints (files on
E:, not in git), with the bake script, its log and report, a MODEL_REGISTRY §17.1 addendum and the H-V8F-HACC-1 register note.  python mk_hold83.py <tip>"""
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
BK = SC + "hacc_bake/"
ART = "E:/Projects/TanitAD-artifacts/v8f-mini-hacc/"
H = "C:/Users/Admin/qland/hold_v8f_hacc_bake"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RD = f"{PK}/raw/2026-10-26-hacc-bake"
TOKEN = re.compile(rb"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")


def git(*a):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", *a], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    assert not TOKEN.search(data), f"token-shaped string in {rel}"
    assert git("rev-parse", f"{TIP}:{rel}") is None or kind == "mods-rw", f"NEW path exists at the tip: {rel}"
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


for n in ("hold_v8f_hacc_bake", "landed_hold_v8f_hacc_bake"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
log = open(BK + "bake.log", "rb").read()
assert log.rstrip().endswith(b"ZZBAKE-OK"), "the bake did not finish OK"
for f, src in (("bake_hacc.py", BK + "bake_hacc.py"), ("bake.log", BK + "bake.log"), ("BAKE_REPORT.json", ART + "BAKE_REPORT.json"),
               ("mk_hold83.py", SC + "grid_a2/mk_hold83.py")):
    put("stage", f"{RD}/{f}", open(src, "rb").read().replace(b"\r\n", b"\n"))

r = "Project Steering/MODEL_REGISTRY.md"
rt = git("show", f"{TIP}:{r}").decode("utf-8")
assert "### 17.1 v8F-mini M2b + HACC" not in rt
put("mods-rw", r, (rt.rstrip("\n") + "\n" + open(BK + "REG_ADD.md", encoding="utf-8").read().replace("\r\n", "\n")).encode("utf-8"))

g = "Project Steering/GOALS_AND_CLAIMS.md"
lines = git("show", f"{TIP}:{g}").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-HACC-1** (2026-10-25) |")]
assert len(ix) == 1 and "Committed next step 1 DONE (2026-10-26)" in lines[ix[0]]
lines[ix[0]] = lines[ix[0]].rstrip().rstrip("|").rstrip() + (
    " **Next step 2, part 1 DONE (2026-10-26): the weights are baked into the deployable checkpoints** `M2b_s{0,1}_hacc.pt`"
    " (`E:\\Projects\\TanitAD-artifacts\\v8f-mini-hacc\\`, `MODEL_REGISTRY.md` §17.1). Through the eval harness's strict load the baked file equals the"
    " runtime-set weights bit for bit, every source tensor is carried bit for bit, and the plans agree with the banked HACC plans to 7.7e-6 /"
    " 6.6e-6 m (`raw/2026-10-26-hacc-bake/`). Still open: the map's provenance in the eval JSON. |")
put("mods-rw", g, "\n".join(lines).encode("utf-8"))
msg = ("HACC weights baked into deployable checkpoints M2b_s{0,1}_hacc.pt (E: artifacts); registry 17.1; bake verified bit for bit\n\n"
       "Strict harness load == runtime-set weights on every output tensor; vs banked HACC plans 7.7e-6 / 6.6e-6 m; other seed's table moves it.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok")
