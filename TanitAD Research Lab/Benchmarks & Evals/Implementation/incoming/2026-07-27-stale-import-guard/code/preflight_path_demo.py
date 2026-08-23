#!/usr/bin/env python3
"""RED -> GREEN for `PREFLIGHT: OK` printed over a path that does not exist.

⭐ THE RED HALF RUNS THE **SHIPPED** CODE, not a simulation of it: the pre-fix
`scripts/train_flagship_v4.py` is extracted from a git revision into a temp dir
and executed with the SAME argv as the fixed one. Two interpreters, one argv,
opposite verdicts.

    python3 preflight_path_demo.py --repo <repo> --base-rev 40aa6ff \
        --out raw/preflight_path_demo.json

⛔ Launches nothing: every invocation carries `--print-launch`, which the trainer
documents as "print the staged pod launch command and the preflight gates, then
exit".
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _argv_for(tmp: Path, anchors: str) -> list[str]:
    """The v5 launch shape, reduced to the flags the path layer sees. Every
    other gate is satisfied so the ONLY difference between RED and GREEN is
    which `--anchors-dense` is passed."""
    (tmp / "traincache").mkdir(exist_ok=True)
    (tmp / "valcache").mkdir(exist_ok=True)
    return ["--print-launch", "--from-scratch",
            "--train-cache", str(tmp / "traincache"),
            "--val-cache", str(tmp / "valcache"),
            "--anchors-dense", anchors,
            "--out", str(tmp / "run"),
            "--steps", "30000", "--gate-step", "10000",
            "--batch", "8", "--accum", "8",
            "--phase-a-steps", "2000", "--phase-b-steps", "8000",
            "--heldout-every", "2000", "--heldout-episodes", "8",
            "--heldout-patience", "2"]


def _run(script: Path, repo: Path, argv: list[str]) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo / "stack"), str(repo / "stack" / "scripts")])
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run([sys.executable, "-u", str(script), *argv],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env,
                       cwd=str(repo / "stack"), timeout=600)
    tail = [ln for ln in r.stdout.splitlines()
            if ln.startswith("PREFLIGHT:") or ln.strip().startswith("- [")]
    return {"exit": r.returncode, "verdict": tail, "stderr_tail":
            r.stderr.strip().splitlines()[-3:] if r.returncode not in (0, 2) else []}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--base-rev", default="HEAD",
                    help="the revision whose train_flagship_v4.py is the RED one")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    repo = Path(a.repo).resolve()

    tmp = Path(tempfile.mkdtemp(prefix="preflight_demo_"))
    try:
        real = tmp / "flagship_v4_anchors_dense.pt"
        real.write_bytes(b"\0" * 8)
        # The exact string in BOTH published v5 launch commands (V5_EVALUABLE
        # §7.1 and §7.3), re-rooted here so the demo needs no pod.
        (tmp / "experiments" / "anchors").mkdir(parents=True)   # empty, as on pod2
        bogus = tmp / "experiments" / "anchors" / "anchors_dense_1to20.pt"

        old_src = subprocess.run(
            ["git", "show", f"{a.base_rev}:stack/scripts/train_flagship_v4.py"],
            capture_output=True, cwd=str(repo), timeout=120)
        red_script = tmp / "train_flagship_v4_RED.py"
        red_script.write_bytes(old_src.stdout)

        green_script = repo / "stack" / "scripts" / "train_flagship_v4.py"

        res = {
            "base_rev": subprocess.run(["git", "rev-parse", "--short",
                                        a.base_rev], capture_output=True,
                                       text=True, cwd=str(repo)).stdout.strip(),
            "bogus_path_exists": bogus.exists(),
            "bogus_parent_is_empty_dir": (bogus.parent.is_dir()
                                          and not any(bogus.parent.iterdir())),
            "RED  (shipped code, missing anchors)":
                _run(red_script, repo, _argv_for(tmp, str(bogus))),
            "GREEN(fixed code,   missing anchors)":
                _run(green_script, repo, _argv_for(tmp, str(bogus))),
            "GREEN(fixed code,   REAL anchors)":
                _run(green_script, repo, _argv_for(tmp, str(real))),
        }
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=2))
        red_ok = res["RED  (shipped code, missing anchors)"]["exit"] == 0
        green_blocked = res["GREEN(fixed code,   missing anchors)"]["exit"] == 2
        print(f"\nRED printed PREFLIGHT: OK for a missing path : {red_ok}")
        print(f"GREEN blocks the same command                : {green_blocked}")
        return 0 if (red_ok and green_blocked) else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
