"""Stage the test_rls1_queue.py fix and its evidence into out/ (idempotent): the file, its diff vs c2d4042, its base blob, the diagnosis code + raw logs, and paths.json entries.
Edits only C:/Users/Admin/p6gate/out/.  LF everywhere."""
import difflib
import json
import subprocess
from pathlib import Path

GIT = "C:/Users/Admin/tanitad-push"
R = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
OUT = Path("C:/Users/Admin/p6gate/out")
DIAG = Path("C:/Users/Admin/p6gate/rls1q_diag")
rp = f"{R}/refe/test_rls1_queue.py"
new = (DIAG / "edit/new_test_rls1_queue.py").read_bytes()
assert b"\r" not in new
tested = Path("C:/Users/Admin/p6gate/trees/fix_rls1q") / rp
assert tested.read_bytes() == new, "the TESTED copy must equal the delivered copy"
base = subprocess.run(["git", "-C", GIT, "-c", "core.autocrlf=false", "show", f"c2d4042:{rp}"], capture_output=True, check=True).stdout
blob = subprocess.run(["git", "-C", GIT, "rev-parse", f"c2d4042:{rp}"], capture_output=True, text=True, check=True).stdout.strip()
assert base == (DIAG / "edit/base_test_rls1_queue.py").read_bytes()
dst = OUT / rp
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_bytes(new)
d = difflib.unified_diff(base.decode("utf-8").splitlines(True), new.decode("utf-8").splitlines(True), fromfile=f"a/{rp}", tofile=f"b/{rp}")
(OUT / "diffs/test_rls1_queue.py.diff").write_text("".join(d), encoding="utf-8", newline="\n")
bb = json.loads((OUT / "base_blobs.json").read_text(encoding="utf-8"))
bb["base_commits"]["test_rls1_queue.py"] = "c2d4042"
bb["blobs"][rp] = {"base_commit": "c2d4042", "base_blob": blob, "c2d4042_blob": blob, "unchanged_5448a05_to_c2d4042": None,
                   "note": "added 2026-10-18 by the rls1-queue flake diagnosis (RESULT.md section 6); NOT in make_out.py's CHANGED list"}
(OUT / "base_blobs.json").write_text(json.dumps(bb, indent=1) + "\n", encoding="utf-8", newline="\n")
# evidence: code + raw logs into the diagnosis package
PD = OUT / R / "raw/2026-10-18-p6-gate-diagnosis"
files = {f"code/rls1q/{n}": DIAG / n for n in ("run1.py", "s1probe.py", "launch_probe.py", "collide.py", "envmon.py", "stage_rls1q.py", "mutcheck.py", "rerun_when_quiet.py")}
files["code/rls1q/apply_edit.py"] = DIAG / "edit/apply_edit.py"
for p in sorted(DIAG.glob("probe_*.log")) + sorted(DIAG.glob("fixq_*.out")) + sorted(DIAG.glob("fixq_v2r*__*.log")) + sorted(DIAG.glob("envmon.tsv")) + sorted(DIAG.glob("mutcheck_*.log")):
    files[f"raw/rls1q_flake/{p.name}"] = p
added = []
for rel, src in files.items():
    data = src.read_bytes().replace(b"\r\n", b"\n")
    t = PD / rel
    t.parent.mkdir(parents=True, exist_ok=True)
    t.write_bytes(data)
    added.append(t)
paths = json.loads((OUT / "paths.json").read_text(encoding="utf-8"))
have = {p[0] for p in paths}
for t in [dst] + added:
    r = str(t.relative_to(OUT)).replace("\\", "/")
    if r not in have:
        paths.append([r, str(t).replace("\\", "/"), "raw"])
        have.add(r)
(OUT / "paths.json").write_text(json.dumps(paths, indent=1) + "\n", encoding="utf-8", newline="\n")
print("base blob", blob, "| staged", 1 + len(added), "files |", len(paths), "paths")
