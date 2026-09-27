"""Regenerate LANDING_READY.txt for BATCH 2: every code/fix2/ file with its tip base blob (read
from the lander repo at the named tip, 40-char asserted) and its new blob
(`git hash-object --no-filters`), plus every PACKAGE file that is NEW or CHANGED relative to the
tip (blob comparison, both sides 40-char; a file whose blob cannot be read is listed, never
silently dropped). Files still being written by a live run are held back and named.

    python write_landing_ready_b2.py <tip-commit>
"""
import pathlib
import subprocess
import sys

TIP = sys.argv[1]
PK = pathlib.Path(__file__).resolve().parents[1]
REL = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-declared-vs-built"
GIT = ["git", "--git-dir=C:/Users/Admin/tanitad-push/.git"]
FIX2 = PK / "code" / "fix2"
# a live run writes here until it banks its summary; land it only when the summary exists
LIVE = {"raw/gate_r34_ab436ee": PK / "raw" / "gate_r34_ab436ee" / "r34_fix5_summary.json",
        "raw/r34_fix5_driver.log": PK / "raw" / "gate_r34_ab436ee" / "r34_fix5_summary.json"}


def blob_at(rev_path: str) -> str | None:
    r = subprocess.run(GIT + ["rev-parse", rev_path], capture_output=True, text=True)
    out = r.stdout.strip()
    return out if (r.returncode == 0 and len(out) == 40) else None


def new_blob(p: pathlib.Path) -> str:
    out = subprocess.run(["git", "hash-object", "--no-filters", str(p)], capture_output=True,
                         text=True).stdout.strip()
    assert len(out) == 40, (p, out)
    return out


def eol(p: pathlib.Path) -> str:
    d = p.read_bytes()
    return "CRLF" if d.count(b"\r\n") == d.count(b"\n") else ("LF" if b"\r\n" not in d else "MIXED")


full = subprocess.run(GIT + ["rev-parse", TIP], capture_output=True, text=True).stdout.strip()
assert len(full) == 40, full
# ⛔ REBASED onto NEW-1 (Master Mind 2026-09-26: NEW-1 lands first). Until it lands, the base of a
# file NEW-1 also rewrites is NEW-1's package blob; after it lands, re-run with its commit as TIP
# (and no N1 argument) so every base is read from the landed tip.
N1 = pathlib.Path(sys.argv[2]) / "code" / "fix" if len(sys.argv) > 2 else None
lines = [f"## 2026-09-26 batch 2: declared-vs-built -- D3 (the per-head gradient-reach rows reach "
         f"metrics.jsonl: one fill rule, a config.json declaration, a first-row refusal, "
         f"check_logged_rows for the launch smoke) + --nav-compliance-tau-file (SPEC_REFCV7 §7: "
         f"verified against the float, sha256 stamped in config.json; G-HYG fields, G-DVB entry, "
         f"registry 204) + the tau script's evidence class derived from the host -- full files at "
         f"their REPO paths under code/fix2/, REBASED onto NEW-1 (3-way merge against {TIP[:7]})",
         f"# batch 1 LANDED as {TIP[:7]} (68/68 blobs verified by the lander). Bases: NEW-1's package "
         f"blobs where NEW-1 rewrites the file, else {full} (40-char asserted). ⛔ NEW-1 lands "
         f"FIRST; then re-run `code/write_landing_ready_b2.py <NEW-1 commit>` and superset-check.",
         "# land these at their repo paths. Full files; line endings as each tip blob (EOL column).",
         "# <package path>  ->  <repo path>     (comment above each: base blob [whose] | new blob = "
         "git hash-object --no-filters | EOL)"]
for p in sorted(x for x in FIX2.rglob("*") if x.is_file() and "__pycache__" not in x.parts):
    rp = p.relative_to(FIX2).as_posix()
    if N1 is not None and (N1 / rp).is_file():
        base = f"{new_blob(N1 / rp)} [NEW-1 package]"
    else:
        base = f"{blob_at(f'{TIP}:{rp}') or 'NEW'} [tip {TIP[:7]}]"
    lines.append(f"# {rp}: base {base} | new {new_blob(p)} | {eol(p)}")
    lines.append(f"{REL}/code/fix2/{rp}  ->  {rp}")

pkg, held = [], []
for x in sorted(PK.rglob("*")):
    if not x.is_file() or "__pycache__" in x.parts:
        continue
    rp = x.relative_to(PK).as_posix()
    # code/fix/ and code/fix2/ are REPO files staged in the package (mapped above / landed with
    # batch 1 at their repo paths), never package files of their own
    if rp.startswith(("code/fix/", "code/fix2/")) or rp == "LANDING_READY.txt":
        continue
    live = next((k for k in LIVE if rp == k or rp.startswith(k + "/")), None)
    if live and not LIVE[live].exists():
        held.append(rp)
        continue
    # the queued neighbour runs append to gate_b2_dev/driver.log and write *NB* files until the
    # TIP2 pass reports its exit (or the queue gives up)
    if rp == "raw/gate_b2_dev/driver.log" or (rp.startswith("raw/gate_b2_dev/") and "NB" in rp):
        drv = (PK / "raw" / "gate_b2_dev" / "driver.log").read_text(encoding="utf-8",
                                                                      errors="replace")
        if not any(k in drv for k in ("TIP2 neighbours exit=", "NOT RUN", "CANCELLED")):
            held.append(rp)
            continue
    b = blob_at(f"{TIP}:{REL}/{rp}")
    if b is None:
        pkg.append((rp, "NEW"))
    elif b != new_blob(x):
        pkg.append((rp, "CHANGED"))
lines.append("# package files NEW or CHANGED since the tip (blob comparison); LANDING_READY.txt itself:")
lines.append(f"{REL}/LANDING_READY.txt")
for rp, why in pkg:
    lines.append(f"# {why}")
    lines.append(f"{REL}/{rp}")
if held:
    lines.append("# HELD BACK -- still being written by a live run (lands with its summary, next batch):")
    lines += [f"#   {REL}/{h}" for h in held]
(PK / "LANDING_READY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print(f"{len(lines)} lines; repo files {sum(1 for l in lines if '->' in l and not l.startswith('#'))}; "
      f"package files {len(pkg) + 1}; held {len(held)}")
print(lines[0])
