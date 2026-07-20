"""ci_gate — one-command pre-commit / pre-push test gate for the TanitAD stack.

Runs the pytest suite once and turns its result into a HARD, self-explaining
gate. It fails (non-zero exit) on any of:

  * a pytest failure OR **collection error** (the class of breakage that silently
    left the whole suite un-runnable for every agent on 2026-07-17 — an untracked
    TDD test importing a symbol its implementation never shipped);
  * any single test whose call phase exceeds ``--max-test-seconds`` (default 15 s)
    — the "newly-added slow fixture" falsifier from the Tools&DevEnv backlog;
  * a total wall-clock over ``--max-wall-seconds`` (default 150 s);
  * a **required tripwire node** (default the I2 batch-consistency encoder test)
    being absent, skipped, or failing — so nobody can quietly delete/skip the
    instrument doctrine and still get a green gate;
  * v2 — a **required SUITE** collecting fewer tests than its manifest floor, or
    holding a non-green test. Single-node tripwires only guard nodes somebody
    thought to name; the suite manifest guards whole modules against silent
    deletion/rename, which is how coverage actually disappears when six agents
    edit one tree. ``--min-total`` is the same idea at whole-suite granularity;
  * v2 — the **CUDA device-parity tripwire** (``--gpu-smoke require``). The
    `stack/tests` suite is 100 % CPU-only (measured 2026-07-20: ``grep -rl cuda
    stack/tests`` is empty) while every trainer, eval and deploy tick runs on a
    GPU, so device/dtype/NaN regressions were structurally invisible to this
    gate. See ``tools/gpu_tripwire.py``.

It is stdlib-only and OS-agnostic: the Windows one-liner ``ci.ps1`` and the pod
(``python tools/ci_gate.py``) share this exact logic. pytest's own JUnit XML
(`--junitxml`) is the machine-readable source of per-test timing and outcomes;
pytest's exit code is the primary pass/fail signal, the XML adds the detail.

Usage::

    python tools/ci_gate.py --rootdir stack            # gate the stack suite
    python tools/ci_gate.py --rootdir stack --gpu-smoke require
    python tools/ci_gate.py --max-test-seconds 4       # tighter per-test budget
    python tools/ci_gate.py --json gate.json           # machine-readable report
    python tools/ci_gate.py -- -k comma2k19            # extra args -> pytest

Exit codes: 0 = gate PASS, 1 = gate FAIL, 3 = could not run pytest at all.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_REQUIRE = "test_i2_batch_consistency_of_encoder"  # the I2 CI tripwire

# Load-bearing suites and their collected-test floors, measured 2026-07-20 on
# `agent/tools-devenv-20260718` (396 collected total). These are the modules whose
# silent disappearance would cost the most: the instrument doctrine, the data
# contracts, the calibration trio, the three reference arms, and the eval/metric
# surfaces the gate protocol reads. A floor is a FLOOR — adding tests is always
# fine; removing them is a deliberate act that must edit this line.
#
# Deliberately NOT every module: a 46-entry hand-maintained manifest rots. The
# targeted list plus --min-total covers both the "one module vanished" and the
# "half the suite vanished" failures.
SUITE_MANIFEST: dict[str, int] = {
    "tests/test_instruments.py": 4,          # I2/I3/I4/I7 doctrine
    "tests/test_calib.py": 12,               # calib "trio" part 1
    "tests/test_physicalai_rig.py": 5,       # calib "trio" part 2 (two-rig, D-016)
    "tests/test_lake.py": 9,
    "tests/test_metrics.py": 22,
    "tests/test_eval_behavior.py": 13,
    "tests/test_refa_flagship_parity.py": 6,
    "tests/test_refa.py": 5,
    "tests/test_refb.py": 15,
    "tests/test_refc.py": 15,
    "tests/test_flagship4b.py": 11,
    "tests/test_gates.py": 16,
    "tests/test_driving_diagnostic.py": 9,
    "tests/test_comma2k19.py": 7,
    "tests/test_scena.py": 22,
    "tests/test_resim.py": 30,
}
# Whole-suite floor: catches wholesale loss (a broken conftest deselecting half
# the tree) that a per-module manifest of 16 entries would miss.
DEFAULT_MIN_TOTAL = 390


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
    gpu: dict | None = None          # tools.gpu_tripwire report, when run

    def by_status(self, status: str) -> list[Case]:
        return [c for c in self.cases if c.status == status]

    @property
    def n(self) -> int:
        return len(self.cases)

    def to_dict(self) -> dict:
        return {
            "exit_code": self.exit_code,
            "wall_s": round(self.wall_s, 2),
            "collected": self.n,
            "counts": {s: len(self.by_status(s))
                       for s in ("passed", "failed", "error", "skipped")},
            "slowest": [{"nodeid": c.nodeid, "time": round(c.time, 3)}
                        for c in sorted(self.cases, key=lambda c: c.time,
                                        reverse=True)[:5]],
            "parse_error": self.parse_error,
            "gpu": self.gpu,
        }


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


def suite_key(spec: str) -> str:
    """Normalize a suite spec to the JUnit ``classname`` form.

    ``tests/test_lake.py`` / ``tests\\test_lake.py`` / ``tests.test_lake`` all map
    to ``tests.test_lake`` — pytest's JUnit writer emits the dotted module path,
    so the manifest can be written in the natural path form."""
    s = spec.replace("\\", "/").strip()
    if s.endswith(".py"):
        s = s[:-3]
    return s.strip("/").replace("/", ".")


def parse_suite_spec(spec: str) -> tuple[str, int]:
    """``"tests/test_lake.py>=9"`` -> ``("tests.test_lake", 9)``. A spec with no
    ``>=`` floor means "must exist and be green", i.e. floor 1."""
    if ">=" in spec:
        path, _, n = spec.partition(">=")
        try:
            floor = int(n.strip())
        except ValueError as exc:
            raise ValueError(f"bad suite floor in {spec!r}: {exc}") from exc
    else:
        path, floor = spec, 1
    return suite_key(path), floor


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
             require: list[str], suites: dict[str, int] | None = None,
             min_total: int = 0) -> list[str]:
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

    # 5) v2 — required SUITES must meet their collected floor and be green.
    for key, floor in (suites or {}).items():
        members = [c for c in report.cases if c.classname == key]
        if len(members) < floor:
            reasons.append(
                f"suite '{key}' collected {len(members)} test(s) < floor {floor} "
                f"(module deleted, renamed, or tests removed)")
            continue
        # A skip is legitimate (optional model download, sim-only path), so it is
        # not a failure -- but a suite that is ENTIRELY skipped is coverage loss
        # wearing a green hat, and that is exactly what this manifest exists to
        # catch. Measured case: tests.test_scena skips one MiniLM search test.
        bad = [c for c in members if c.status in ("failed", "error")]
        if bad:
            reasons.append(
                f"suite '{key}' not green: "
                + ", ".join(f"{c.name}={c.status}" for c in bad[:5]))
        elif all(c.status == "skipped" for c in members):
            reasons.append(f"suite '{key}' is entirely SKIPPED "
                           f"({len(members)} tests) — no coverage")

    # 6) v2 — whole-suite floor.
    if min_total and report.n < min_total:
        reasons.append(f"only {report.n} tests collected < --min-total {min_total} "
                       f"(wholesale coverage loss?)")

    # 7) v2 — CUDA device-parity tripwire, when one was run.
    if report.gpu is not None:
        g = report.gpu
        if not g.get("available"):
            reasons.append(f"gpu tripwire required but no CUDA device: {g.get('error')}")
        elif g.get("error"):
            reasons.append(f"gpu tripwire could not run: {g['error']}")
        else:
            for p in g.get("probes", []):
                if not p.get("ok"):
                    reasons.append(f"gpu tripwire {p['name']}: {p['detail']}")

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


def _run_gpu_tripwire(rootdir: Path) -> dict:
    """Run the CUDA parity probes in-process, with ``rootdir`` importable so the
    stack resolves the same way pytest just did."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    added = str(rootdir)
    if added not in sys.path:
        sys.path.insert(0, added)
    import gpu_tripwire  # noqa: PLC0415 - optional, only when --gpu-smoke is on

    return gpu_tripwire.run().to_dict()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TanitAD one-command test gate")
    ap.add_argument("--rootdir", default=".",
                    help="dir to run pytest in (default: cwd)")
    # 15 s clears today's tall pole (test_replay 7.2 s off-Drive / 7.9 s on the
    # Drive tree, 2026-07-20) with headroom while still catching a runaway
    # fixture; tighten as tests speed up.
    ap.add_argument("--max-test-seconds", type=float, default=15.0)
    # 150 s: the full 531-test Drive tree measured 60.2 s on 2026-07-20, so this
    # is ~2.5x headroom and still an order of magnitude under the 5-min ceiling
    # the backlog set for "shard it instead".
    ap.add_argument("--max-wall-seconds", type=float, default=150.0)
    ap.add_argument("--require", action="append", default=None,
                    help="test node that MUST exist and pass (repeatable). "
                         f"default: {DEFAULT_REQUIRE}. pass --require '' to disable.")
    ap.add_argument("--require-suite", action="append", default=[],
                    metavar="PATH[>=N]",
                    help="test module that MUST collect >=N tests, all green "
                         "(repeatable). Adds to the built-in manifest.")
    ap.add_argument("--no-default-suites", action="store_true",
                    help="drop the built-in SUITE_MANIFEST (use with "
                         "--require-suite for a custom set, or on a subset run)")
    ap.add_argument("--min-total", type=int, default=DEFAULT_MIN_TOTAL,
                    help=f"minimum total collected tests (default "
                         f"{DEFAULT_MIN_TOTAL}; 0 disables)")
    ap.add_argument("--gpu-smoke", choices=("off", "warn", "require"),
                    default="off",
                    help="run the CUDA device-parity tripwire: 'require' makes a "
                         "missing/failing GPU a gate failure, 'warn' only prints")
    ap.add_argument("--json", default=None, help="write the report as JSON")
    ap.add_argument("pytest_args", nargs="*",
                    help="extra args forwarded to pytest (after --)")
    args = ap.parse_args(argv)

    require = args.require if args.require is not None else [DEFAULT_REQUIRE]
    rootdir = Path(args.rootdir).resolve()

    suites: dict[str, int] = {} if args.no_default_suites else dict(SUITE_MANIFEST)
    suites = {suite_key(k): v for k, v in suites.items()}
    for spec in args.require_suite:
        key, floor = parse_suite_spec(spec)
        suites[key] = floor

    print(f"[ci_gate] pytest in {rootdir}  "
          f"(per-test<={args.max_test_seconds:g}s, wall<={args.max_wall_seconds:g}s, "
          f"require={[r for r in require if r] or 'none'}, "
          f"suites={len(suites)}, min-total={args.min_total}, "
          f"gpu-smoke={args.gpu_smoke})", flush=True)

    report = run_pytest(args.pytest_args, rootdir)
    if report.exit_code == 3:
        print(f"[ci_gate] FAIL — {report.parse_error}", flush=True)
        return 3

    if args.gpu_smoke != "off":
        try:
            gpu = _run_gpu_tripwire(rootdir)
        except Exception as exc:                  # noqa: BLE001 - never mask the suite result
            gpu = {"available": False, "probes": [],
                   "error": f"gpu_tripwire raised: {exc!r}"}
        import gpu_tripwire  # noqa: PLC0415 - already imported above on success

        print(gpu_tripwire.format_report(gpu_tripwire.GpuReport(
            available=gpu.get("available", False),
            device_name=gpu.get("device_name", ""),
            torch_version=gpu.get("torch_version", ""),
            probes=[gpu_tripwire.Probe(**p) for p in gpu.get("probes", [])],
            encode_batch1_ms=gpu.get("encode_batch1_ms"),
            wall_s=gpu.get("wall_s", 0.0),
            error=gpu.get("error"))), flush=True)
        # 'warn' surfaces the report but never contributes a gate reason.
        report.gpu = gpu if args.gpu_smoke == "require" else None

    reasons = evaluate(report, args.max_test_seconds, args.max_wall_seconds,
                       require, suites=suites, min_total=args.min_total)
    print(f"[ci_gate] {_summary(report)}", flush=True)
    if args.json:
        payload = report.to_dict()
        payload["reasons"] = reasons
        payload["pass"] = not reasons
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if reasons:
        print(f"[ci_gate] GATE FAIL ({len(reasons)}):", flush=True)
        for r in reasons:
            print(f"    - {r}", flush=True)
        return 1
    print("[ci_gate] GATE PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
