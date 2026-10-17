#!/usr/bin/env python3
"""apply_all.py -- re-apply the SFT-4R readiness change set onto a BRANCH-TIP tree, byte-exact.

usage:  python apply_all.py <repo_root> [--patchers-only] [--dry-run]

  <repo_root>       a tree with the LF (git-archive / mirror) layout:  <repo_root>/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/...
                    (a CRLF autocrlf worktree is REFUSED: the patchers assert LF so that they can never half-apply)
  --dry-run         apply to a temp copy of refe/ and verify, touch nothing in <repo_root>
  --patchers-only   skip the pre-image sha check (use when the tip moved on a file this change set edits: the patchers still assert their anchors and the post-image sha is still verified;
                    a post-image mismatch then means "tip moved under the patch" and the diffs in diff/ are the reference)

What it does, in order (every step asserts; any failure aborts before anything is written to <repo_root> unless it is the last, verified copy):
  1. pre-image sha256 of every EXISTING target == the sha at tip 6c3d469c (SHA256.json "pre"),
  2. launch_gate.py   <- patch/patch_launch_gate.py            (adds the refe_sft4r profile)
     gate_checks.py   <- patch/patch_gate_checks.py            (adds scorer_expect_rows / data_expect)
  3. patch/patch_suite_nesting.py   (gate_checks.check_suite refuses to nest; adds test_suite_nesting.py to SFT4R_SUITE)
  4. sft4r_chain.sh, sft4r_pod.sh, test_sft4r_gate.py  <- post_p0/ (the previous agent's v2 files; no patcher existed for them)
     patch/patch_test_sft4r_gate.py, patch/patch_sft4r_chain_precheck.py
  5. test_suite_nesting.py  <- post/ (new file)
  6. sha256 of every result == SHA256.json "post"; then (not --dry-run) the six files are copied into <repo_root>.
Nothing is staged, committed or pushed.
"""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKR = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
FILES = ["launch_gate.py", "gate_checks.py", "sft4r_chain.sh", "sft4r_pod.sh", "test_sft4r_gate.py", "test_suite_nesting.py"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    flags = {x for x in sys.argv[1:] if x.startswith("--")}
    if len(a) != 1:
        print(__doc__)
        return 2
    root = Path(a[0])
    refe = root / PKR / "refe"
    if not refe.is_dir():
        print(f"REFUSED: {refe} is not a directory")
        return 2
    table = json.loads((HERE / "SHA256.json").read_text(encoding="utf-8"))
    for f in ("launch_gate.py", "gate_checks.py", "sft4r_chain.sh", "sft4r_pod.sh"):
        if b"\r\n" in (refe / f).read_bytes():
            print(f"REFUSED: {f} has CRLF line endings: apply on an LF tree (git archive of the tip / the mirror), not an autocrlf worktree")
            return 2
    work = Path(tempfile.mkdtemp(prefix="sft4r_apply_"))
    w = work / "refe"
    w.mkdir()
    for f in ("launch_gate.py", "gate_checks.py", "sft4r_chain.sh", "sft4r_pod.sh"):
        shutil.copyfile(refe / f, w / f)
    if "--patchers-only" not in flags:
        bad = [f for f in ("launch_gate.py", "gate_checks.py", "sft4r_chain.sh", "sft4r_pod.sh") if sha(w / f) != table["pre"][f]]
        if bad:
            print(f"REFUSED: the tip moved under this change set: pre-image sha256 differs for {bad} (expected the tip {table['tip_commit'][:12]}). Re-run with --patchers-only and review diff/.")
            return 3
    py = sys.executable

    def run(script, target=None):
        r = subprocess.run([py, "-X", "utf8", str(HERE / "patch" / script), str(target if target is not None else w)], capture_output=True, text=True)
        print(f"  {script}: rc {r.returncode} {r.stdout.strip()[-120:]} {r.stderr.strip()[-300:]}")
        if r.returncode != 0:
            print("ABORTED: nothing was written to the repo tree")
            sys.exit(4)
    run("patch_launch_gate.py", w / "launch_gate.py")          # the previous agent's two patchers take the FILE, mine take the refe dir
    run("patch_gate_checks.py", w / "gate_checks.py")
    run("patch_suite_nesting.py")
    for f in ("sft4r_chain.sh", "sft4r_pod.sh", "test_sft4r_gate.py"):
        shutil.copyfile(HERE / "post_p0" / "refe" / f, w / f)
    run("patch_test_sft4r_gate.py")
    run("patch_sft4r_chain_precheck.py")
    shutil.copyfile(HERE / "post" / "refe" / "test_suite_nesting.py", w / "test_suite_nesting.py")
    bad = [f for f in FILES if sha(w / f) != table["post"][f]]
    if bad:
        print(f"ABORTED: post-image sha256 differs for {bad}: the result is NOT the verified change set (the tip moved; use diff/ as the reference)")
        return 5
    print("all six post-image sha256 equal the verified set")
    if "--dry-run" in flags:
        print(f"dry run: {w} holds the result; the repo tree is untouched")
        return 0
    for f in FILES:
        shutil.copyfile(w / f, refe / f)
        print("  wrote", refe / f)
    shutil.rmtree(work, ignore_errors=True)
    print("APPLIED. Nothing staged or committed.  Next: python refe/test_suite_nesting.py && python refe/test_sft4r_gate.py (each prints ZZTEST_PASS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
