"""TanitAD one-command commit gate — the logic behind ``scripts/ci.ps1``.

Why this exists (Tools&DevEnv backlog P0 #1): every agent and the loop commit
many times a day; we want a single command that (a) proves the safety
tripwires are green, (b) refuses to let the suite silently rot into a slow
mess, and (c) reports a warm wall-clock so the ">15 s and climbing" moment is
visible instead of discovered months later.

Three gates, most-load-bearing first:

  1. **I2 tripwire** — the encoder batch-1 consistency check
     (``tests/test_instruments.py::test_i2_batch_consistency_of_encoder``).
     I2 is the instrument that catches the entire BatchNorm-in-inference class
     of bugs (D-004); it runs first and alone so a red I2 fails CI in <2 s
     before the full suite even starts.
  2. **Per-test latency budget** — no single test's ``call`` phase may exceed
     ``--slow-test-s`` (default 6 s). A newly-added slow fixture makes CI exit
     nonzero. This is the falsifier the backlog registered for this item.
  3. **Suite green + optional warm-wall budget** — the (quick or full) suite
     must pass; if ``--warm-budget-s`` > 0, the warm wall-clock must be under it.

The pure helpers (``parse_durations``, ``max_call_duration``, ``evaluate``,
``QUICK_SUITE``) are unit-tested in ``tests/test_ci.py`` without spawning pytest;
``main`` is the thin subprocess orchestrator that ``ci.ps1`` calls.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass

# The I2 collapse/batch-statistic tripwire — run first, alone, fail-fast.
I2_NODE = "tests/test_instruments.py::test_i2_batch_consistency_of_encoder"

# The curated pre-commit safety subset: instrument doctrine (I1-I4), the gate
# runner (BLOCKED != FAIL + I2-missing gating), SigReg anti-collapse, the
# end-to-end smoke train, and the data-contract integrity tests that guard the
# silent-wrong-data class of bugs. Fast (all node call-times < ~2.2 s) and it
# covers "did this commit break a safety invariant" without the full 350-test
# breadth. Paths are relative to ``stack/``.
QUICK_SUITE = (
    "tests/test_instruments.py",
    "tests/test_gates.py",
    "tests/test_sigreg.py",
    "tests/test_smoke_train.py",
    "tests/test_mixing.py",
    "tests/test_epcache_key.py",
    "tests/test_comma2k19_contract.py",
)

_DURATION_RE = re.compile(r"^\s*([0-9.]+)s\s+(call|setup|teardown)\s+(\S+)", re.M)
_SUMMARY_RE = re.compile(r"(\d+)\s+passed")
_FAIL_RE = re.compile(r"(\d+)\s+(failed|error|errors)")
_INTIME_RE = re.compile(r"in\s+([0-9.]+)s")


@dataclass(frozen=True)
class Duration:
    seconds: float
    phase: str
    nodeid: str


def parse_durations(pytest_output: str) -> list[Duration]:
    """Extract the ``--durations`` table from pytest's stdout.

    pytest prints lines like ``2.15s call     tests/test_x.py::test_y``.
    Only well-formed rows are returned; order is preserved.
    """
    out: list[Duration] = []
    for m in _DURATION_RE.finditer(pytest_output):
        out.append(Duration(float(m.group(1)), m.group(2), m.group(3)))
    return out


def max_call_duration(durations: list[Duration]) -> Duration | None:
    """Slowest ``call``-phase test (the one the latency budget guards)."""
    calls = [d for d in durations if d.phase == "call"]
    if not calls:
        return None
    return max(calls, key=lambda d: d.seconds)


def suite_passed(pytest_output: str) -> bool:
    """True iff pytest reported passes and no failures/errors."""
    if _FAIL_RE.search(pytest_output):
        return False
    return _SUMMARY_RE.search(pytest_output) is not None


def reported_wall(pytest_output: str) -> float | None:
    """pytest's own ``... in Xs`` timing (a warm-wall proxy), if present."""
    m = None
    for m in _INTIME_RE.finditer(pytest_output):
        pass  # keep the last match (the summary line)
    return float(m.group(1)) if m else None


@dataclass
class Verdict:
    ok: bool
    reasons: list[str]


def evaluate(
    *,
    i2_passed: bool,
    suite_ok: bool,
    slowest: Duration | None,
    slow_budget_s: float,
    warm_wall_s: float | None,
    warm_budget_s: float,
) -> Verdict:
    """Combine the three gates into a single go/no-go verdict.

    Pure and total — no I/O — so the CI decision logic is unit-tested directly.
    """
    reasons: list[str] = []
    if not i2_passed:
        reasons.append(f"I2 tripwire FAILED ({I2_NODE}) - batch-statistic leak in the inference path")
    if not suite_ok:
        reasons.append("test suite did not pass (failures/errors or no tests collected)")
    if slowest is not None and slowest.seconds > slow_budget_s:
        reasons.append(
            f"slow test: {slowest.nodeid} took {slowest.seconds:.2f}s call "
            f"(budget {slow_budget_s:.1f}s) - split it or mark it 'slow'"
        )
    if warm_budget_s > 0 and warm_wall_s is not None and warm_wall_s > warm_budget_s:
        reasons.append(
            f"warm wall {warm_wall_s:.1f}s over budget {warm_budget_s:.1f}s "
            f"- the suite is getting slow; curate or parallelize"
        )
    return Verdict(ok=not reasons, reasons=reasons)


def _run(python: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [python, "-m", "pytest", *args],
        capture_output=True,
        text=True,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TanitAD one-command commit gate")
    ap.add_argument("--quick", action="store_true",
                    help="run only the curated safety subset (pre-commit); default is the full suite")
    ap.add_argument("--slow-test-s", type=float, default=6.0,
                    help="per-test call-duration budget in seconds (default 6.0)")
    ap.add_argument("--warm-budget-s", type=float, default=0.0,
                    help="fail if warm wall exceeds this (0 = no wall budget)")
    ap.add_argument("--python", default=sys.executable,
                    help="python interpreter to run pytest with (default: current)")
    args = ap.parse_args(argv)

    py = args.python
    print(f"[ci] I2 tripwire: {I2_NODE}")
    t0 = time.perf_counter()
    i2 = _run(py, ["-q", I2_NODE])
    i2_ok = i2.returncode == 0
    print(i2.stdout.strip().splitlines()[-1] if i2.stdout.strip() else "(no output)")
    if not i2_ok:
        # Fail fast: don't spend 20 s on the full suite if the core tripwire is red.
        sys.stdout.write(i2.stdout)
        sys.stderr.write(i2.stderr)
        print("\n[ci] BLOCKED - I2 tripwire failed (fail-fast, suite not run)")
        return 2

    suite_args = ["-q", "--durations=0", "-p", "no:cacheprovider"]
    suite_args += list(QUICK_SUITE) if args.quick else []
    mode = "quick" if args.quick else "full"
    print(f"[ci] running {mode} suite ...")
    suite = _run(py, suite_args)
    warm = time.perf_counter() - t0

    durations = parse_durations(suite.stdout)
    slowest = max_call_duration(durations)
    ok = suite_passed(suite.stdout)
    rep_wall = reported_wall(suite.stdout)

    verdict = evaluate(
        i2_passed=i2_ok,
        suite_ok=ok,
        slowest=slowest,
        slow_budget_s=args.slow_test_s,
        warm_wall_s=warm,
        warm_budget_s=args.warm_budget_s,
    )

    # Always surface the suite's own summary line.
    summary = [ln for ln in suite.stdout.splitlines() if " passed" in ln or " failed" in ln or " error" in ln]
    print(summary[-1].strip() if summary else "(pytest produced no summary line)")
    if slowest is not None:
        print(f"[ci] slowest test: {slowest.nodeid} {slowest.seconds:.2f}s call")
    print(f"[ci] warm wall (I2 + {mode} suite): {warm:.1f}s"
          + (f" (pytest reported {rep_wall:.1f}s)" if rep_wall is not None else ""))

    if verdict.ok:
        print(f"[ci] OK - safe to commit ({mode} gate)")
        return 0
    if not ok:
        sys.stdout.write(suite.stdout)  # show the failing tests
    print("[ci] BLOCKED:")
    for r in verdict.reasons:
        print(f"  - {r}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
