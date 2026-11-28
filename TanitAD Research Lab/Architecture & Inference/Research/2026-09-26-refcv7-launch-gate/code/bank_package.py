"""Bank the gate's work into its D: package and (re)write LANDING_READY.txt.

    python bank_package.py <tip sha>

Repo files -> code/fix/<repo path>; package tools -> code/ (never __pycache__). Every copy is
verified by sha256 against its source; the landing lines carry the tip base blob (or NEW), the new
blob (`git hash-object --no-filters`) and the file's line endings, counted in BYTES.
"""
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

TIP = sys.argv[1]
GITDIR = "C:/Users/Admin/tanitad-push/.git"
W = Path("C:/Users/Admin/lg0926/work")
PKG_REL = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv7-launch-gate"
PKG = Path("D:/Projects/TanitAD") / PKG_REL
REPO_FILES = {  # repo path -> source
    "stack/scripts/launch_gate.py": W / "stack/scripts/launch_gate.py",
    "stack/scripts/launch_gate_refcv7.py": W / "stack/scripts/launch_gate_refcv7.py",
    "stack/scripts/launch_gate_arms.py": W / "stack/scripts/launch_gate_arms.py",
    "stack/scripts/closure_run.py": W / "stack/scripts/closure_run.py",
    "stack/tests/test_launch_gate.py": W / "stack/tests/test_launch_gate.py",
    "stack/ops/sup_refcv7.sh": W / "stack/ops/sup_refcv7.sh",
    "stack/ops/launch_gate_thor_env_failures.json": W / "stack/ops/launch_gate_thor_env_failures.json",
    "stack/ops/runs.d/refcv7-r101-s0.argv.json": W / "stack/ops/runs.d/refcv7-r101-s0.argv.json",
    ".gitignore": W / "pkgfix/.gitignore",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*a: str) -> str:
    return subprocess.run(["git", f"--git-dir={GITDIR}", *a], capture_output=True, text=True).stdout.strip()


def eol(p: Path) -> str:
    b = p.read_bytes()
    crlf, lf = b.count(b"\r\n"), b.count(b"\n")
    if crlf == 0:
        return "LF"
    return "CRLF" if crlf == lf else f"MIXED({crlf}/{lf})"


def main() -> int:
    tip = git("rev-parse", TIP)
    assert len(tip) == 40, f"tip {TIP!r} does not resolve"
    lines = [f"## 2026-09-27 the refcv7 LAUNCH GATE (SPEC_REFCV7 sec. 2 + A1..A11 as landed) -- full files "
             f"at their REPO paths under code/fix/, based on tip {tip}",
             "# land these at their repo paths. Full files; the EOL column is the file's own (bytes counted).",
             f"# Base blobs read at {tip} (40-char asserted). If the tip moves, superset-check against the NEW tip.",
             "# <package path>  ->  <repo path>     (comment above each: tip base blob | new blob = git hash-object --no-filters | EOL)"]
    bad = []
    for rp, src in REPO_FILES.items():
        dst = PKG / "code" / "fix" / rp
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        if sha(dst) != sha(src):
            bad.append(rp)
        base = git("rev-parse", f"{tip}:{rp}")
        if len(base) != 40:
            # "NEW" only when a path that EXISTS at the tip resolves in the same breath: an empty
            # answer from a failed query is not an absence
            ctl = git("rev-parse", f"{tip}:.gitignore")
            assert len(ctl) == 40, f"control rev-parse failed ({ctl!r}): {rp} is INCONCLUSIVE"
            assert subprocess.run(["git", f"--git-dir={GITDIR}", "cat-file", "-e", f"{tip}:{rp}"],
                                  capture_output=True).returncode != 0, f"{rp}: rev-parse empty, cat-file finds it"
            base = "NEW"
        new = git("hash-object", "--no-filters", str(dst))
        assert len(new) == 40, f"hash-object failed for {dst}"
        lines.append(f"# {rp}: base {base} | new {new} | {eol(dst)}")
        lines.append(f"{PKG_REL}/code/fix/{rp}  ->  {rp}")
    lines.append("# ⛔ stack/ops/sup_refcv7.sh must be staged BYTE-EXACT with LF endings (bash); "
                 "verify with `git ls-files --stage` vs the new blob above.")
    lines.append("# package files (not repo files):")
    tools = []
    for src in sorted((W / "pkg" / "code").rglob("*")):
        if src.is_dir() or "__pycache__" in src.parts:
            continue
        rel = src.relative_to(W / "pkg" / "code")
        dst = PKG / "code" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        if sha(dst) != sha(src):
            bad.append(str(rel))
        tools.append(f"{PKG_REL}/code/{rel.as_posix()}")
    extra = [p for p in sorted(PKG.rglob("*")) if p.is_file() and "code" not in p.relative_to(PKG).parts[:1]]
    lines += tools
    lines += [f"{PKG_REL}/{p.relative_to(PKG).as_posix()}" for p in extra
              if p.name != "LANDING_READY.txt"]
    lines.append(f"{PKG_REL}/LANDING_READY.txt")
    (PKG / "LANDING_READY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))
    if bad:
        print("COPY MISMATCH:", bad)
        return 1
    print(f"BANKED {len(REPO_FILES)} repo files + {len(tools)} tools, all sha256-verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
