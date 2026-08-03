"""Test-suite wall-clock profiler + cold-I/O regression guard (Tools&DevEnv, G-E cost).

Why this exists
---------------
Every scheduled agent runs `pytest` to satisfy gate G-E, and pays that cost on a
COLD process against a repo that lives on Google Drive File Stream. Measured on the
dev machine 2026-07-09 (venv is off-Drive, repo/tests/fixtures are on-Drive):

    cold first run of the day : 40.6 s wall   (181 passed, 1 skipped)
    warm subsequent runs      : 10.7 s wall   (same suite)
    pytest-reported test time :  9.2 s        (compute only)
    pytest --collect-only     :  4.9 s        (imports torch once, all 28 modules)
    import torch              :  1.9 s
    full read of stack/ src   :  0.13 s       (87 files, 0.44 MB, WARM)

The ~30 s cold penalty is NOT byte volume (the source tree is 0.44 MB) and NOT
compute (9.2 s) — it is Google-Drive *hydration latency*: per-file metadata +
on-demand fetch round-trips the first time each file is touched in a fresh
session. The lever is therefore I/O locality, not test speed.

This module gives Tools&DevEnv two things with no third-party deps:
  1. `profile` — decompose a suite run into collection / execution / overhead and
     rank the slowest tests, so the cost story stays measured over time.
  2. `check`   — a CI/agent regression guard: fail if warm overhead or any single
     test exceeds a budget, so a newly-added slow fixture is caught at commit.

Parsers work on captured pytest text so they are unit-testable without spawning
pytest (see tests/). The subprocess runner is a thin wrapper around them.

Usage
-----
    python profile_testsuite.py profile --stack-dir stack --out report.json
    python profile_testsuite.py check   --stack-dir stack --max-warm-overhead 4 \
                                        --max-test 6.0
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --- parsers (pure, unit-tested) -------------------------------------------------

# e.g. "3.02s call     tests/test_smoke_train.py::test_smoke_training"
_DURATION_RE = re.compile(
    r"^\s*([0-9]+\.[0-9]+)s\s+(call|setup|teardown)\s+(\S+)\s*$"
)
# e.g. "181 passed, 1 skipped in 9.23s"  /  "5 passed, 2 failed in 1.10s"
_SUMMARY_RE = re.compile(
    r"(?:(\d+) passed)?(?:, )?(?:(\d+) failed)?(?:, )?(?:(\d+) skipped)?"
    r".*?\bin\s+([0-9]+\.[0-9]+)s"
)


@dataclass
class Duration:
    seconds: float
    phase: str  # call | setup | teardown
    nodeid: str


@dataclass
class Summary:
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    reported_sec: float = 0.0


def parse_durations(text: str) -> list[Duration]:
    """Extract the `--durations` table rows from captured pytest output."""
    out: list[Duration] = []
    for line in text.splitlines():
        m = _DURATION_RE.match(line)
        if m:
            out.append(Duration(float(m.group(1)), m.group(2), m.group(3)))
    return out


def parse_summary(text: str) -> Summary:
    """Extract the final summary line ('N passed, M skipped in T.TTs')."""
    best: Summary | None = None
    for line in text.splitlines():
        m = _SUMMARY_RE.search(line)
        if m and (m.group(1) or m.group(2) or m.group(3)):
            best = Summary(
                passed=int(m.group(1) or 0),
                failed=int(m.group(2) or 0),
                skipped=int(m.group(3) or 0),
                reported_sec=float(m.group(4)),
            )
    if best is None:
        raise ValueError("no pytest summary line found in output")
    return best


# --- report model ----------------------------------------------------------------

@dataclass
class RunReport:
    label: str
    wall_sec: float
    summary: Summary
    slowest: list[Duration] = field(default_factory=list)

    @property
    def overhead_sec(self) -> float:
        """Wall time not accounted for by pytest's own reported test time."""
        return round(self.wall_sec - self.summary.reported_sec, 3)


# --- subprocess runner (thin) ----------------------------------------------------

def _run_pytest(stack_dir: Path, durations: int) -> tuple[str, float]:
    cmd = [
        sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
        f"--durations={durations}",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, cwd=str(stack_dir), capture_output=True, text=True
    )
    wall = time.perf_counter() - t0
    return proc.stdout + proc.stderr, wall


def measure(stack_dir: Path, label: str, durations: int = 25) -> RunReport:
    text, wall = _run_pytest(stack_dir, durations)
    summary = parse_summary(text)
    slowest = sorted(parse_durations(text), key=lambda d: -d.seconds)[:durations]
    return RunReport(label=label, wall_sec=round(wall, 3), summary=summary,
                     slowest=slowest)


def prime_tree(stack_dir: Path, patterns: tuple[str, ...] = ("*.py",)) -> int:
    """Force-read the source tree once to hydrate the Drive cache. Returns file count."""
    n = 0
    for pat in patterns:
        for p in stack_dir.rglob(pat):
            if p.is_file():
                p.read_bytes()
                n += 1
    return n


# --- commands --------------------------------------------------------------------

def cmd_profile(args: argparse.Namespace) -> int:
    stack = Path(args.stack_dir)
    # cold run first (whatever the Drive cache currently is), then a warm re-run.
    cold = measure(stack, "cold", args.durations)
    warm = measure(stack, "warm", args.durations)
    report = {
        "cold": asdict(cold), "warm": asdict(warm),
        "cold_wall_sec": cold.wall_sec, "warm_wall_sec": warm.wall_sec,
        "cold_io_tax_sec": round(cold.wall_sec - warm.wall_sec, 3),
        "warm_overhead_sec": warm.overhead_sec,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if cold.summary.failed == 0 else 1


def cmd_check(args: argparse.Namespace) -> int:
    stack = Path(args.stack_dir)
    warm = measure(stack, "warm", args.durations)
    problems: list[str] = []
    if warm.summary.failed:
        problems.append(f"{warm.summary.failed} test(s) failed")
    if warm.overhead_sec > args.max_warm_overhead:
        problems.append(
            f"warm overhead {warm.overhead_sec}s > budget {args.max_warm_overhead}s "
            "(import/collection creep)"
        )
    over = [d for d in warm.slowest if d.phase == "call" and d.seconds > args.max_test]
    for d in over:
        problems.append(f"slow test {d.nodeid} {d.seconds}s > {args.max_test}s "
                        "(mark @pytest.mark.slow or speed up)")
    if problems:
        print("TESTSUITE CHECK FAILED:")
        for p in problems:
            print("  - " + p)
        return 1
    print(f"TESTSUITE CHECK OK: {warm.summary.passed} passed, "
          f"overhead {warm.overhead_sec}s, wall {warm.wall_sec}s")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("profile", help="cold/warm decomposition + slowest tests")
    p.add_argument("--stack-dir", default="stack")
    p.add_argument("--durations", type=int, default=25)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_profile)

    c = sub.add_parser("check", help="regression guard (nonzero exit on budget breach)")
    c.add_argument("--stack-dir", default="stack")
    c.add_argument("--durations", type=int, default=25)
    c.add_argument("--max-warm-overhead", type=float, default=4.0)
    c.add_argument("--max-test", type=float, default=6.0)
    c.set_defaults(func=cmd_check)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
