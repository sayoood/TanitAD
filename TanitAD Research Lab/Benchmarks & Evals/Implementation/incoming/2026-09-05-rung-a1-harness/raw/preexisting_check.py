"""Are the 21 failures + 8 errors PRE-EXISTING? Demonstrate, do not assert.

Swaps the mirror's THREE modified files back to their pre-Rung-A1 copies and
moves the TWO new test files aside, re-runs exactly the failing selection, then
restores everything and verifies the restore by md5.

⛔ Run with nothing else touching the mirror.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

MIRROR = Path(r"C:\Users\Admin\tanitad-wt")
SP = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
          r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
          r"\8e7cfa33-c625-47cf-88aa-711db80ac113\scratchpad")
PY = r"C:\Users\Admin\venvs\tanitad\Scripts\python.exe"

MODIFIED = {                       # live path -> pristine copy
    "taniteval/tools/refav1_arm.py": SP / "refav1_arm.UNPATCHED.py",
    "taniteval/tools/refcv3_arm.py": SP / "refcv3_arm.UNPATCHED.py",
    "stack/tests/test_refcv3_arm.py": SP / "test_refcv3_arm.UNPATCHED.py",
}
NEW = ["stack/tests/test_refcv3_ha0_ext_shared.py",
       "stack/tests/test_refcv3_ablations.py"]

SELECTION = [
    "stack/tests/test_bev_consumer_fov.py",
    "stack/tests/test_build_parity_guard.py",
    "stack/tests/test_decision_check.py",
    "stack/tests/test_eval_contamination.py",
    "stack/tests/test_launch_closure_audit.py",
    "stack/tests/test_refav1_kin_contract.py",
    "stack/tests/test_runbook_commands.py",
    "stack/tests/test_secret_scan.py",
    "stack/tests/test_text_encoding_is_explicit.py",
    "stack/tests/test_v6_chain.py",
    "stack/tests/test_v6_st_launch_fixes.py",
    "stack/tests/test_closedloop_floor.py",
]


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def run(tag: str) -> str:
    r = subprocess.run(
        [PY, "-m", "pytest", *SELECTION, "-q", "-p", "no:cacheprovider",
         "--continue-on-collection-errors"],
        cwd=str(MIRROR), capture_output=True, text=True, encoding="utf-8",
        errors="replace",
        env={**__import__("os").environ, "PYTHONPATH": str(MIRROR / "stack")})
    out = r.stdout + r.stderr
    (SP / f"preexisting_{tag}.txt").write_text(out, encoding="utf-8")
    lines = [ln for ln in out.splitlines()
             if ln.startswith(("FAILED", "ERROR")) or " passed" in ln
             or " failed" in ln]
    return "\n".join(lines)


def main():
    live_md5 = {p: md5(MIRROR / p) for p in MODIFIED}
    print("=== A) WITH Rung A1 applied ===", flush=True)
    after = run("patched")
    print(after, flush=True)

    print("\n=== B) with the pre-Rung-A1 files restored ===", flush=True)
    for rel, pristine in MODIFIED.items():
        shutil.copyfile(pristine, MIRROR / rel)
    for rel in NEW:
        (MIRROR / rel).rename(MIRROR / (rel + ".aside"))
    try:
        before = run("unpatched")
        print(before, flush=True)
    finally:
        for rel in NEW:
            (MIRROR / (rel + ".aside")).rename(MIRROR / rel)
        for rel in MODIFIED:
            shutil.copyfile(SP / (Path(rel).name.replace(".py", ".PATCHED.py")),
                            MIRROR / rel)
        bad = [p for p in MODIFIED if md5(MIRROR / p) != live_md5[p]]
        print("\nRESTORE:", "OK" if not bad else f"⛔ MISMATCH {bad}", flush=True)
        if bad:
            return 1

    a = {ln.split(" ")[1] for ln in after.splitlines()
         if ln.startswith(("FAILED", "ERROR")) and len(ln.split(" ")) > 1}
    b = {ln.split(" ")[1] for ln in before.splitlines()
         if ln.startswith(("FAILED", "ERROR")) and len(ln.split(" ")) > 1}
    print("\n=== VERDICT ===")
    print(f"failing in BOTH (PRE-EXISTING): {len(a & b)}")
    only_after = sorted(a - b)
    only_before = sorted(b - a)
    print(f"failing ONLY WITH Rung A1 (REGRESSIONS): {len(only_after)}")
    for x in only_after:
        print("   ⛔", x)
    print(f"failing ONLY WITHOUT Rung A1 (FIXED by it): {len(only_before)}")
    for x in only_before:
        print("   ✅", x)
    return 0


if __name__ == "__main__":
    sys.exit(main())
