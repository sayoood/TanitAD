#!/usr/bin/env python3
"""YP7 finalize: freeze the final package bytes, mirror every changed/new file into YP7/tree/, write CHANGES.md (tip blob = the land base guard, final blob) and RESULT.txt.
Run by run_pipeline.sh AFTER the traced ship-list derive, its check and the FINAL suite-prerun.  Reads only; never touches the tree under pyp7 except to COPY out of it."""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PKW = Path("C:/Users/Admin/pyp7")
S = Path("E:/Projects/TanitAD-artifacts/yawrate_wp/YP7")
REPO = "C:/Users/Admin/tanitad-push"
TIP = "b17b549232a80f72db28c7214468ec65eb443fe1"
PKPFX = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/"
YT_BASE = "a521c0736a70b7659ce26a2254324a52cf9d5345"
YT_FILES = {"refe/train.py", "refe/gate_checks.py", "refe/test_launch_gate.py", "refe/test_p7_head_wiring.py", "refe/ego_yaw_table.py", "refe/ego_yaw.py", "refe/test_ego_yaw.py"}
EXPECTED = sorted(YT_FILES | {"SPEC_P7.md", "refe/launch_gate.py", "refe/p7_pod.sh", "refe/test_p7_chain.py", "refe/test_p7_integration.py", "refe/p7_ship_list.json"})


def ho(path):
    return subprocess.run(["git", "hash-object", "--no-filters", str(path)], capture_output=True, text=True, check=True).stdout.strip()


def tip_blob(rel):
    r = subprocess.run(["git", "-C", REPO, "rev-parse", f"{TIP}:{PKPFX}{rel}"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def tail(p, n=1):
    try:
        ls = [ln for ln in Path(p).read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
        return " | ".join(ls[-n:])
    except OSError:
        return "(absent)"


# every file in the working tree vs the tip's PK tree
ls = subprocess.run(["git", "-C", REPO, "ls-tree", "-r", "-z", TIP, PKPFX], capture_output=True).stdout.split(b"\0")
tip = {}
for e in ls:
    if e:
        meta, p = e.split(b"\t", 1)
        tip[p.decode("utf-8")[len(PKPFX):]] = meta.decode().split()[2]
loc = {}
for r, d, fs in os.walk(PKW):
    for f in fs:
        loc[os.path.relpath(os.path.join(r, f), PKW).replace(os.sep, "/")] = None
inp = "\n".join(loc) + "\n"
hs = subprocess.run(["git", "hash-object", "--no-filters", "--stdin-paths"], input=inp, capture_output=True, text=True, cwd=str(PKW)).stdout.split()
loc = dict(zip(loc, hs))
changed = sorted(f for f in loc if tip.get(f) != loc[f])
missing = sorted(f for f in tip if f not in loc)
unexpected = sorted(set(changed) - set(EXPECTED))
absent = sorted(set(EXPECTED) - set(changed))

out_tree = S / "tree"
if out_tree.exists():
    shutil.rmtree(out_tree)
for rel in changed:
    dst = out_tree / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PKW / rel, dst)
rows = []
for rel in changed:
    t = tip.get(rel)
    who = "YT (unchanged by YP7)" if rel in YT_FILES and ho(PKW / rel) == ho(Path("E:/Projects/TanitAD-artifacts/yawrate_wp/YT_tree") / rel) else ("YT + YP7" if rel in YT_FILES else "YP7")
    rows.append((rel, "NEW" if t is None else "MODIFIED", t or "(absent at the tip)", loc[rel], who))

head = (S / "CHANGES_head.md").read_text(encoding="utf-8")
tbl = ["", "## Files changed or new, with the tip blob (= the land base guard) and the final blob", "",
       f"Tip: `{TIP}` (`agent/arch-inf-20260803`, fetched into the mirror `C:/Users/Admin/tanitad-push`). Paths are relative to `{PKPFX}`. Blob = `git hash-object --no-filters`.", "",
       "| file | change | tip blob (base guard) | final blob | by |", "|---|---|---|---|---|"]
for rel, ch, tb, fb, who in rows:
    tbl.append(f"| `{rel}` | {ch} | `{tb}` | `{fb}` | {who} |")
tbl += ["", f"Mirror of every row above: `{out_tree.as_posix()}/<path>`. {len(rows)} files.",
        f"Unexpected changed files (not in the expected list): {unexpected or 'none'}. Expected but unchanged: {absent or 'none'}. Tip files missing from the work tree: {len(missing)}.",
        "`refe/planner.py` and everything under `eval/` are byte-identical to the tip (not in the table)."]

import datetime
lines = [f"YP7 RESULT (generated {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')})", ""]
lines.append("pre-run A (sanity, before the derive): " + tail(S / "suite_prerun_PRE.out", 2))
lines.append("ship-list derive:   " + tail(S / "ship" / "derive.log", 3))
lines.append("ship-list check:    " + tail(S / "ship" / "check.log", 2))
lines.append("FINAL suite-prerun: " + tail(S / "suite_prerun_FINAL.out", 2))
lines.append("full test_ego_yaw.py (local, incl. SLOW + mutation harness): " + tail(S / "logs" / "test_ego_yaw_full.log", 2))
try:
    sl = json.loads((PKW / "refe" / "p7_ship_list.json").read_text(encoding="utf-8"))
    inl = {k: {"in_scripts": f"refe/{k}" in sl["scripts"], "in_files": f"refe/{k}" in sl["files"]} for k in ("ego_yaw_table.py", "ego_yaw.py", "test_ego_yaw.py")}
    lines.append(f"ship list: files={len(sl['files'])} scripts={len(sl['scripts'])} closure={sl['closure_digest'][:12]} trace={sl['trace']}; in the list: {inl}; suite={sl['suite']}")
except Exception as e:  # noqa: BLE001
    lines.append(f"ship list unreadable: {e}")
lines.append(f"changed/new files: {len(rows)}; unexpected: {unexpected}; expected-but-unchanged: {absent}")
(S / "RESULT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
(S / "CHANGES.md").write_text(head + "\n".join(tbl) + "\n\n## Pipeline results\n\n```\n" + "\n".join(lines) + "\n```\n", encoding="utf-8")
print("\n".join(lines))
sys.exit(1 if unexpected or absent else 0)
