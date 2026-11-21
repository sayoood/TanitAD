"""Stage every deliverable file by EXACT path (never a directory, never -A), then verify each: the staged blob
(`git ls-files --stage`) must equal `git hash-object` of the working file, both 40 hex chars; anything else is
INCONCLUSIVE. The record file itself is staged + verified in a second pass (it cannot contain its own blob)."""
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(r"D:/Projects/TanitAD")
DIR = "products/P4-training-pipelines/2026-09-27-v7f-refav1-state/v7f_gate"
RECORD = f"{DIR}/raw/staging_verification.json"
SHA = re.compile(r"^[0-9a-f]{40}$")


def git(*args: str) -> str:
    p = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"git {' '.join(args[:3])} failed rc={p.returncode}: {p.stderr[-400:]}")
    return p.stdout


def verify(rel: str) -> dict:
    st = git("ls-files", "--stage", "--", rel).split()
    staged = st[1] if len(st) >= 4 else None
    work = git("hash-object", "--", rel).strip()
    ok = bool(staged and work and SHA.match(staged) and SHA.match(work) and staged == work)
    return {"path": rel, "staged_blob": staged, "hash_object": work,
            "verdict": "VERIFIED" if ok else "INCONCLUSIVE"}


files = sorted(str(p.relative_to(REPO)).replace("\\", "/") for p in (REPO / DIR).rglob("*") if p.is_file())
bad = [f for f in files if re.search(r"(PASS|FAIL|INCOMPLETE|PI-DECISION)_[0-9a-f]{12}\.json$|\.key$|Keys\.txt$", f)]
if bad:
    sys.exit(f"refusing: token/key-like files in the deliverable: {bad}")
body = [f for f in files if f != RECORD]
for i in range(0, len(body), 20):
    git("add", "--", *body[i:i + 20])
rows = [verify(f) for f in body]
rec = {"_read": "staged by exact path, never committed; staged blob == git hash-object (both 40 hex) per file",
       "when_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
       "n_files": len(rows), "n_verified": sum(r["verdict"] == "VERIFIED" for r in rows),
       "record_itself": "staged + verified in a second pass (reported by the agent, not in this file)",
       "files": rows}
(REPO / RECORD).write_text(json.dumps(rec, indent=1), encoding="utf-8")
git("add", "--", RECORD)
me = verify(RECORD)
print(json.dumps({"n_files": rec["n_files"], "n_verified": rec["n_verified"],
                  "inconclusive": [r["path"] for r in rows if r["verdict"] != "VERIFIED"],
                  "record": me}, indent=1))
