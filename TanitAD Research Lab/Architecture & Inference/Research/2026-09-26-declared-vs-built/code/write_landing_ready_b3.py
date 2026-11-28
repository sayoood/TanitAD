"""Regenerate LANDING_READY.txt for BATCH 3 (the refcv7 launch blockers). Master Mind's format,
exactly: a `# <repo path>: base <40-char> | new <40-char> | <EOL>` line above each
`<package path>  ->  <repo path>` line; provenance notes go on SEPARATE comment lines, so the gate
builder's parser reads nothing but those two shapes.

    python write_landing_ready_b3.py <tip-commit> [--b2-pending]

* a file present at <tip> -> its base is the tip blob (40-char asserted);
* `--b2-pending`: batch 2 has NOT landed yet, so a file batch 2 rewrites takes the batch-2 blob
  as its base (code/fix2, hashed here) -- after batch 2 lands, run WITHOUT the flag against the
  landing commit and every base is read from the tip;
* a file absent from both -> `NEW`.
Package files NEW or CHANGED relative to the tip (blob comparison) are listed after the map.
"""
import pathlib
import subprocess
import sys

TIP = sys.argv[1]
B2_PENDING = "--b2-pending" in sys.argv
PK = pathlib.Path(__file__).resolve().parents[1]
REL = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-declared-vs-built"
GIT = ["git", "--git-dir=C:/Users/Admin/tanitad-push/.git"]
FIX3 = PK / "code" / "fix3"
FIX2 = PK / "code" / "fix2"


def blob_at(rev_path: str) -> str | None:
    r = subprocess.run(GIT + ["rev-parse", "--verify", "--quiet", rev_path], capture_output=True,
                       text=True)
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
b2_files = {p.relative_to(FIX2).as_posix() for p in FIX2.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts}
lines = [f"## 2026-09-27 batch 3: declared-vs-built -- the refcv7 launch blockers: (a) the three "
         f"zero-gradient modules (core.decoder.control_head, core.decoder.offset_head, "
         f"scorer.goal_point) are bypassed BY CONSTRUCTION, so FROZEN and DECLARED "
         f"(_gradreach, kept built) and held against argv by G-DVB (check/probe_grad_unreachable); "
         f"(b) G-HYG strict_fields on the six open config classes; (c) SPEC section 10 (A5): "
         f"check_refcv7_required refuses any residual prior but ha0_ext_pose (argv AND built) and "
         f"a missing --ego-history -- full files at their REPO paths under code/fix3/",
         f"# base commit {full}" + (" + batch 2 (NOT YET LANDED: its 4 shared files' bases below "
                                    "are the batch-2 blobs; re-run without --b2-pending against "
                                    "the batch-2 landing commit)" if B2_PENDING else ""),
         "# land these at their repo paths. Full files; line endings as each tip blob (EOL column).",
         "# <package path>  ->  <repo path>     (comment above each: base blob | new blob = "
         "git hash-object --no-filters | EOL)"]
for p in sorted(x for x in FIX3.rglob("*") if x.is_file() and "__pycache__" not in x.parts):
    rp = p.relative_to(FIX3).as_posix()
    if B2_PENDING and rp in b2_files:
        base = new_blob(FIX2 / rp)
        lines.append(f"# (base of {rp} = the batch-2 blob; batch 2 is on the Thor gate)")
    else:
        base = blob_at(f"{full}:{rp}") or "NEW"
    lines.append(f"# {rp}: base {base} | new {new_blob(p)} | {eol(p)}")
    lines.append(f"{REL}/code/fix3/{rp}  ->  {rp}")

pkg, held = [], []
LIVE_DIRS = ("raw/gate_b3_dev",)
for x in sorted(PK.rglob("*")):
    if not x.is_file() or "__pycache__" in x.parts:
        continue
    rp = x.relative_to(PK).as_posix()
    if rp.startswith(("code/fix/", "code/fix2/", "code/fix3/", "code/fix3_r")) \
            or rp == "LANDING_READY.txt":
        continue
    if rp.startswith(LIVE_DIRS):
        drv = PK / "raw" / "gate_b3_dev" / "driver.log"
        if not (drv.exists() and "DONE" in drv.read_text(encoding="utf-8", errors="replace")):
            held.append(rp)
            continue
    b = blob_at(f"{full}:{REL}/{rp}")
    if b is None:
        pkg.append((rp, "NEW"))
    elif b != new_blob(x):
        pkg.append((rp, "CHANGED"))
lines.append("# package files NEW or CHANGED since the base commit (blob comparison); "
             "LANDING_READY.txt itself:")
lines.append(f"{REL}/LANDING_READY.txt")
for rp, why in pkg:
    lines.append(f"# {why}")
    lines.append(f"{REL}/{rp}")
if held:
    lines.append("# HELD BACK -- still being written by a live run (lands with its result):")
    lines += [f"#   {REL}/{h}" for h in held]
(PK / "LANDING_READY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print(f"{len(lines)} lines; repo files {sum(1 for l in lines if '->' in l and not l.startswith('#'))}"
      f"; package files {len(pkg) + 1}; held {len(held)}")
