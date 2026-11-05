"""Build / refresh the qland hold `hold_v8f_level_motion` for lever (a) `--level-head` / `--w-level` (builder output; the MM lands it, nothing here commits, pushes or lands).

  python -I mk_level_hold.py create      first build: mods-rw (the two edited tracked files), stage (new tests + the raw package), SHA_CHANGES.txt, MANIFEST.txt
  python -I mk_level_hold.py logs        refresh ONLY the package's logs/ (suite logs, COMPARE.txt, targeted and mutation logs) and MANIFEST.txt, after the background suite has finished

The tree sources are the patched export F:/TanitAD-scratch/motion_c/build/new (stack/ + taniteval/ of tip b07982c plus this patch); the tip export is F:/.../build/tip. LF only for .py/.md/.sh (checked in python)."""
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

TIP = "b07982c0dc7ebce6815badde6ca02d9aff5f09c0"
BUILD = Path("E:/Projects/TanitAD-artifacts/dagland/motion_c/build")
NEW, OLD = Path("F:/TanitAD-scratch/motion_c/build/new"), Path("F:/TanitAD-scratch/motion_c/build/tip")
FLOGS = Path("F:/TanitAD-scratch/motion_c/build/logs")
HOLD = Path("C:/Users/Admin/qland/hold_v8f_level_motion")
GIT = "C:/Users/Admin/tanitad-push"
REL = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-11-05-level-head-code"
MODS = ["stack/tanitad/models/v8f_mini.py", "stack/scripts/v8f_mini_train.py", "stack/tests/test_v8f_launch_intent.py"]
NEWF = ["stack/tests/test_v8f_level_head.py", "stack/tests/v8f_mini_tip_snapshot_level.txt"]
PKG_FILES = ["HANDOFF.md", "patches/pa_model.py", "patches/pa_train.py", "mutations_level.py", "export_tree.py", "mk_level_hold.py", "runner/run_files.py", "runner/compare_suites.py",
             "runner/chain_tip_and_compare.py", "runner/run_suite.cmd", "smoke_a.py"]
LOG_COPY = {FLOGS / "mutation_level.log": "logs/mutation_level.log", FLOGS / "targeted_regression.log": "logs/targeted_regression.log", FLOGS / "level_head_tests.log": "logs/level_head_tests.log",
            FLOGS / "real_c0_forward.log": "logs/real_c0_forward.log", FLOGS / "launch_intent_after_edit.log": "logs/launch_intent_after_edit.log", FLOGS / "three_on_tip.log": "logs/targeted_three_failures_on_tip.log",
            FLOGS / "three_on_new.log": "logs/targeted_three_failures_on_new_before_golden_edit.log", BUILD / "logs/files_NEW.log": "logs/files_NEW.log", BUILD / "logs/files_TIP.log": "logs/files_TIP.log",
            BUILD / "logs/COMPARE.txt": "logs/COMPARE.txt", BUILD / "logs/suite_run.out": "logs/suite_run.out", BUILD / "logs/README_SUITE.md": "logs/README_SUITE.md"}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def put(src_bytes, dst):
    if dst.suffix in (".py", ".md", ".sh", ".txt"):
        assert b"\r\n" not in src_bytes or dst.name.endswith(".log"), f"CRLF in {dst}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src_bytes)


def blob_id(path):
    return subprocess.run(["git", "-C", GIT, "rev-parse", f"{TIP}:{path}"], capture_output=True, text=True).stdout.strip() or "(absent at the tip)"


def manifest():
    base = HOLD / "stage" / REL
    rows = []
    for p in sorted(base.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.txt":
            b = p.read_bytes()
            rows.append(f"./{p.relative_to(base).as_posix()} | {len(b)} | {sha(b)}\n")
    (base / "MANIFEST.txt").write_bytes("".join(rows).encode())
    return len(rows)


mode = sys.argv[1]
if mode == "create":
    assert not HOLD.exists(), f"{HOLD} exists"
    lines = [f"tip = {TIP}; sha256 of the file content (LF)\n"]
    for rel in MODS:
        old, new = (OLD / rel).read_bytes(), (NEW / rel).read_bytes()
        assert b"\r\n" not in new and b"\r\n" not in old
        put(new, HOLD / "mods-rw" / rel)
        lines.append(f"\n{rel}   [tracked, edited -> hold/mods-rw]\n  base blob {blob_id(rel)}\n  tip {sha(old)}\n  new {sha(new)}\n")
    for rel in NEWF:
        new = (NEW / rel).read_bytes()
        put(new, HOLD / "stage" / rel)
        lines.append(f"\n{rel}   [new file -> hold/stage]\n  tip (new file: absent at the tip)\n  new {sha(new)}\n")
    put("".join(lines).encode(), HOLD / "stage" / REL / "SHA_CHANGES.txt")
    put("".join(lines).encode(), BUILD / "SHA_CHANGES.txt")


def copy_pkg():
    base = HOLD / "stage" / REL
    for rel in PKG_FILES:
        src = BUILD / rel
        if rel == "smoke_a.py":
            src = Path("C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/smoke_a.py")
        if not src.exists():
            print("MISSING (skipped)", rel)
            continue
        put(src.read_bytes(), base / rel)


copy_pkg()                                 # both modes: the package's own files (HANDOFF.md, patches, runner) follow the build dir
for src, rel in LOG_COPY.items():
    if src.exists():
        put(src.read_bytes(), HOLD / "stage" / REL / rel)
    else:
        print("log not present yet:", src)
print("hold", HOLD, "files in the package manifest:", manifest())
