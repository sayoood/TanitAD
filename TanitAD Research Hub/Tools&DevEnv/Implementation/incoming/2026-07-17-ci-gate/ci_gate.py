"""ci_gate — one-command pre-commit / pre-push test gate for the TanitAD stack.

Runs the pytest suite once and turns its result into a HARD, self-explaining
gate. It fails (non-zero exit) on any of:

  * a pytest failure OR **collection error** (the class of breakage that silently
    left the whole suite un-runnable for every agent on 2026-07-17 — an untracked
    TDD test importing a symbol its implementation never shipped);
  * any single test whose call phase exceeds ``--max-test-seconds`` (default 15 s)
    — the "newly-added slow fixture" falsifier from the Tools&DevEnv backlog;
  * a total wall-clock over ``--max-wall-seconds`` (default 90 s);
  * a **required tripwire node** (default the I2 batch-consistency encoder test)
    being absent, skipped, or failing — so nobody can quietly delete/skip the
    instrument doctrine and still get a green gate.

It is stdlib-only and OS-agnostic: the Windows one-liner ``ci.ps1`` and the pod
(``python scripts/ci_gate.py``) share this exact logic. pytest's own JUnit XML
(`--junitxml`) is the machine-readable source of per-test timing and outcomes;
pytest's exit code is the primary pass/fail signal, the XML adds the detail.

Usage::

    python ci_gate.py                       # gate the ./tests suite (cwd)
    python ci_gate.py --max-test-seconds 4  # tighter per-test budget
    python ci_gate.py -- -k comma2k19       # pass extra args through to pytest

Exit codes: 0 = gate PASS, 1 = gate FAIL, 3 = could not run pytest at all.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_REQUIRE = "test_i2_batch_consistency_of_encoder"  # the I2 CI tripwire


@dataclass
class Case:
    name: str
    classname: str
    time: float
    status: str  # "passed" | "failed" | "error" | "skipped"

    @property
    def nodeid(self) -> str:
        return f"{self.classname}::{self.name}" if self.classname else self.name


@dataclass
class Report:
    exit_code: int
    wall_s: float
    cases: list[Case] = field(default_factory=list)
    parse_error: str | None = None

    def by_status(self, status: str) -> list[Case]:
        return [c for c in self.cases if c.status == status]

    @property
    def n(self) -> int:
        return len(self.cases)


def _classify(case_el: ET.Element) -> str:
    """Map a JUnit <testcase> child element to a coarse status."""
    if case_el.find("error") is not None:
        return "error"
    if case_el.find("failure") is not None:
        return "failed"
    if case_el.find("skipped") is not None:
        return "skipped"
    return "passed"


def parse_junit(xml_path: Path) -> list[Case]:
    """Parse a pytest JUnit XML into Case rows. Raises on unreadable XML."""
    root = ET.parse(xml_path).getroot()
    cases: list[Case] = []
    # <testsuites><testsuite><testcase>… — handle both a bare testsuite root and
    # the testsuites wrapper pytest emits.
    suites = root.iter("testsuite")
    for suite in suites:
        for el in suite.findall("testcase"):
            try:
                t = float(el.get("time", "0") or 0)
            except ValueError:
                t = 0.0
            cases.append(Case(name=el.get("name", "?"),
                              classname=el.get("classname", ""),
                              time=t,
                              status=_classify(el)))
    return cases


def run_pytest(pytest_args: list[str], rootdir: Path) -> Report:
    """Run pytest once under ``rootdir`` and collect a Report."""
    with tempfile.TemporaryDirectory() as td:
        xml = Path(td) / "junit.xml"
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
               f"--junitxml={xml}", *pytest_args]
        t0 = time.monotonic()
        try:
            proc = subprocess.run(cmd, cwd=str(rootdir))
        except OSError as exc:  # pragma: no cover - environment failure
            return Report(exit_code=3, wall_s=0.0,
                          parse_error=f"could not launch pytest: {exc}")
        wall = time.monotonic() - t0
        cases: list[Case] = []
        parse_error = None
        if xml.is_file():
            try:
                cases = parse_junit(xml)
            except ET.ParseError as exc:
                parse_error = f"unreadable junit xml: {exc}"
        else:
            parse_error = "pytest wrote no junit xml (hard crash before collection)"
        return Report(exit_code=proc.returncode, wall_s=wall,
                      cases=cases, parse_error=parse_error)


def evaluate(report: Report, max_test_seconds: float, max_wall_seconds: float,
             require: list[str]) -> list[str]:
    """Return a list of gate-failure reasons (empty list == PASS)."""
    reasons: list[str] = []

    # 1) pytest's own verdict is authoritative for pass/fail/collection error.
    #    exit 0 = all green; 5 = no tests collected; anything else = trouble.
    if report.exit_code == 5:
        reasons.append("pytest collected NO tests (exit 5) — wrong rootdir or all deselected")
    elif report.exit_code not in (0,):
        errs = report.by_status("error")
        fails = report.by_status("failed")
        detail = ""
        if errs:
            detail = f" — collection/setup ERROR in: {', '.join(c.nodeid for c in errs[:5])}"
        elif fails:
            detail = f" — {len(fails)} failing: {', '.join(c.nodeid for c in fails[:5])}"
        reasons.append(f"pytest exit {report.exit_code}{detail}")

    # 2) per-test slowness budget (the backlog falsifier).
    slow = sorted((c for c in report.cases if c.time > max_test_seconds),
                  key=lambda c: c.time, reverse=True)
    for c in slow:
        reasons.append(f"slow test {c.nodeid}: {c.time:.2f}s > {max_test_seconds:.2f}s budget")

    # 3) total wall budget.
    if report.wall_s > max_wall_seconds:
        reasons.append(f"suite wall {report.wall_s:.1f}s > {max_wall_seconds:.1f}s budget")

    # 4) required tripwire nodes must exist AND be green.
    for needed in require:
        if not needed:
            continue
        hits = [c for c in report.cases if c.name == needed or c.nodeid.endswith(needed)]
        if not hits:
            reasons.append(f"required tripwire '{needed}' was not collected (deleted or renamed?)")
        elif any(c.status != "passed" for c in hits):
            bad = ", ".join(f"{c.nodeid}={c.status}" for c in hits if c.status != "passed")
            reasons.append(f"required tripwire not green: {bad}")

    if report.parse_error and report.exit_code == 0:
        # green exit but we could not read detail — surface it, don't hide it.
        reasons.append(f"warning: {report.parse_error}")
    return reasons


def _summary(report: Report, top: int = 5) -> str:
    counts = {s: len(report.by_status(s))
              for s in ("passed", "failed", "error", "skipped")}
    line = (f"{counts['passed']} passed, {counts['failed']} failed, "
            f"{counts['error']} error, {counts['skipped']} skipped "
            f"in {report.wall_s:.1f}s ({report.n} collected)")
    slowest = sorted(report.cases, key=lambda c: c.time, reverse=True)[:top]
    if slowest:
        rows = "\n".join(f"    {c.time:6.2f}s  {c.nodeid}" for c in slowest)
        line += f"\n  slowest {min(top, len(slowest))}:\n{rows}"
    return line


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TanitAD one-command test gate")
    ap.add_argument("--rootdir", default=".",
                    help="dir to run pytest in (default: cwd)")
    # 15 s clears today's tall pole (test_replay ~10.9 s, 2026-07-17) with
    # headroom while still catching a runaway fixture; tighten as tests speed up.
    ap.add_argument("--max-test-seconds", type=float, default=15.0)
    ap.add_argument("--max-wall-seconds", type=float, default=90.0)
    ap.add_argument("--require", action="append", default=None,
                    help="test node that MUST exist and pass (repeatable). "
                         f"default: {DEFAULT_REQUIRE}. pass --require '' to disable.")
    ap.add_argument("pytest_args", nargs="*",
                    help="extra args forwarded to pytest (after --)")
    args = ap.parse_args(argv)

    require = args.require if args.require is not None else [DEFAULT_REQUIRE]
    rootdir = Path(args.rootdir).resolve()

    print(f"[ci_gate] pytest in {rootdir}  "
          f"(per-test<={args.max_test_seconds:g}s, wall<={args.max_wall_seconds:g}s, "
          f"require={[r for r in require if r] or 'none'})", flush=True)

    report = run_pytest(args.pytest_args, rootdir)
    if report.exit_code == 3:
        print(f"[ci_gate] FAIL — {report.parse_error}", flush=True)
        return 3

    reasons = evaluate(report, args.max_test_seconds, args.max_wall_seconds, require)
    print(f"[ci_gate] {_summary(report)}", flush=True)
    if reasons:
        print(f"[ci_gate] GATE FAIL ({len(reasons)}):", flush=True)
        for r in reasons:
            print(f"    - {r}", flush=True)
        return 1
    print("[ci_gate] GATE PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
