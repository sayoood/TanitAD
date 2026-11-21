"""Regenerate LANDING_READY.txt from the files on disk: every code/fix/ file with its tip base
blob (read from the lander repo at the named tip, 40-char asserted) and its new blob
(`git hash-object --no-filters`), plus every package file that exists. Run after any change.

    python write_landing_ready.py <tip-commit>
"""
import pathlib
import subprocess
import sys

TIP = sys.argv[1]
PK = pathlib.Path(__file__).resolve().parents[1]
REL = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-declared-vs-built"
GIT = ["git", "--git-dir=C:/Users/Admin/tanitad-push/.git"]


def blob_at(rev_path: str) -> str:
    r = subprocess.run(GIT + ["rev-parse", rev_path], capture_output=True, text=True)
    out = r.stdout.strip()
    return out if (r.returncode == 0 and len(out) == 40) else "NEW"


def new_blob(p: pathlib.Path) -> str:
    out = subprocess.run(["git", "hash-object", "--no-filters", str(p)], capture_output=True,
                         text=True).stdout.strip()
    assert len(out) == 40, (p, out)
    return out


def eol(p: pathlib.Path) -> str:
    d = p.read_bytes()
    return "CRLF" if d.count(b"\r\n") == d.count(b"\n") else ("LF" if b"\r\n" not in d else "MIXED")


fix = PK / "code" / "fix"
lines = [f"## 2026-09-26 batch 1: declared-vs-built -- FIX-3 (C26 field), G-HYG, G-DVB, FIX-4 (selection "
         f"flags + a declaration read off the decoder), FIX-5 (per-stage ImageNet fingerprint, sidecar "
         f"meta required), G3 (label-time guard + E2(b) per-family eval mode), PI E1 ruling (speed "
         f"ceiling INFERENCE-ONLY + REFCV7_REQUIRED_ON), legacy as-trained rebuild -- full files "
         f"at their REPO paths under code/fix/, based on tip {TIP[:7]} blobs",
         "# land these at their repo paths. Full files; line endings as each tip blob (EOL column).",
         f"# Base blobs read at {TIP} (40-char asserted). The six shared files are byte-identical at "
         f"59f0d46, 5de9363 and {TIP[:7]}. If the tip moves, superset-check against the NEW tip.",
         "# <package path>  ->  <repo path>     (comment above each: tip base blob | new blob = "
         "git hash-object --no-filters | EOL)"]
for p in sorted(x for x in fix.rglob("*") if x.is_file()):
    rp = p.relative_to(fix).as_posix()
    lines.append(f"# {rp}: base {blob_at(f'{TIP}:{rp}')} | new {new_blob(p)} | {eol(p)}")
    lines.append(f"{REL}/code/fix/{rp}  ->  {rp}")
lines.append("# package files:")
pkg = ["RESULT.md", "LANDING_READY.txt"]
pkg += sorted(x.relative_to(PK).as_posix() for x in (PK / "code").rglob("*")
              if x.is_file() and "fix" not in x.relative_to(PK / "code").parts[:1]
              and "__pycache__" not in x.parts)
pkg += sorted(x.relative_to(PK).as_posix() for x in (PK / "raw").rglob("*")
              if x.is_file() and "__pycache__" not in x.parts)
for f in pkg:
    lines.append(f"{REL}/{f}")
(PK / "LANDING_READY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print(f"{len(lines)} lines; {sum(1 for l in lines if '->' in l and not l.startswith('#'))} repo files; "
      f"{len(pkg)} package files")
