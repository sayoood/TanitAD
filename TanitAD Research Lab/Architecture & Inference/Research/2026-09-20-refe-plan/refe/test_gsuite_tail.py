#!/usr/bin/env python3
"""G-SUITE FAILURE TAIL (2026-11-13): a failing suite script's last stdout/stderr lines ride in the G-SUITE evidence.

WHY.  VDAC arm HC-s0 failed G-SUITE on the pod with `test_aux_drivable_head.py: rc 1, 'ZZTEST_PASS' NOT seen` and check_suite had thrown the script's output away, so the cause (a missing shipped file) had to
be re-derived on the dev box; each pod retry costs ~45 min.  Now a FAILING script's dict carries `stdout_tail` (last 40 lines) and `stderr_tail` (last 20), each line cut to 300 chars, any line with an HF token
pattern or `HF_TOKEN=` replaced by `<redacted>`.  EVIDENCE ONLY: verdict, reasons, token and rc are unchanged; a PASSING script's dict gets no new key.

CHECKS (CPU only, a few s; prints `ZZTEST_PASS <n>`):  A tails exact + limits + cut; B redaction; C passing entry has no new keys; D verdict/reasons identical to the tail-less engine (the mutant);
E the MUTANT (temp copy of gate_checks.py with the tail lines removed) goes RED on A, B (and the mutant is what D compares against).
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_checks as C  # noqa: E402

FAKE_TOKEN = "hf_" + "AbCdEf0123456789XyZ"
OLD_KEYS = {"script", "rc", "token", "token_seen", "skips", "seconds"}


def raw(code: str, token: str = "ZZTEST_PASS") -> dict:
    return {"raw": [sys.executable, "-c", code], "token": token}


def run(mod, entries: list) -> dict:
    saved = os.environ.pop(mod.SUITE_NEST_ENV, None)           # this test may itself run inside a suite: start from an empty nesting stack
    try:
        return mod.check_suite({"suite": entries})
    finally:
        if saved is not None:
            os.environ[mod.SUITE_NEST_ENV] = saved


def scripts(res: dict) -> list:
    return res["details"]["scripts"] if "details" in res else res["evidence"]["scripts"]  # verdict() carries the dict under "details"


FAIL_CODE = ("import sys\n"
             "for i in range(60): print('out line', i)\n"
             "for i in range(30): print('err line', i, file=sys.stderr)\n"
             "print('x' * 500)\n"
             "print('leak ' + " + repr(FAKE_TOKEN) + ")\n"
             "print('HF_TOKEN=whatever')\n"
             "sys.exit(1)\n")


def judge(mod) -> list:
    """Every check against engine `mod`; returns the list of failures (empty = all hold)."""
    bad = []
    res = run(mod, [raw(FAIL_CODE), raw("print('ZZTEST_PASS')")])
    sc = scripts(res)
    f, p = sc[0], sc[1]
    so, se = f.get("stdout_tail"), f.get("stderr_tail")
    if so is None or se is None:
        bad.append("failing entry has no stdout_tail/stderr_tail")
        so, se = so or [], se or []
    exp_out = [f"out line {i}" for i in range(60)] + ["x" * 500, "leak " + FAKE_TOKEN, "HF_TOKEN=whatever"]
    if len(so) != 40 or [ln for ln in so if ln != "<redacted>" and len(ln) > 300]:
        bad.append(f"stdout_tail limit: len {len(so)} (want 40) or a line over 300 chars")
    if so and so[-3:] != ["x" * 300, "<redacted>", "<redacted>"]:
        bad.append(f"stdout_tail last 3 wrong (cut / redaction): {[s[:20] for s in so[-3:]]}")
    if so and so[:-3] != exp_out[-40:-3]:
        bad.append("stdout_tail body is not exactly the last lines printed")
    if se != [f"err line {i}" for i in range(10, 30)]:
        bad.append(f"stderr_tail not exactly the last 20 lines: {se[:2]}..{se[-1:]}")
    if FAKE_TOKEN in json.dumps(res, default=str) or "HF_TOKEN=" in json.dumps(res, default=str):
        bad.append("a token line survived into the evidence")
    if set(p) != OLD_KEYS:
        bad.append(f"passing entry keys changed: {sorted(set(p) ^ OLD_KEYS)}")
    small = scripts(run(mod, [raw("import sys; print('a'); print('b', file=sys.stderr); sys.exit(2)")]))[0]
    if small.get("stdout_tail") != ["a"] or small.get("stderr_tail") != ["b"]:
        bad.append(f"short output tails wrong: {small.get('stdout_tail')} {small.get('stderr_tail')}")
    return bad


def make_mutant(tmp: Path):
    src = (HERE / "gate_checks.py").read_text(encoding="utf-8").splitlines(keepends=True)
    keep = [ln for ln in src if '["stdout_tail"]' not in ln and '["stderr_tail"]' not in ln]
    assert len(src) - len(keep) == 2, "mutation must remove exactly the two tail lines"
    p = tmp / "gate_checks_mut.py"
    p.write_text("".join(keep), encoding="utf-8", newline="")
    spec = importlib.util.spec_from_file_location("gate_checks_mut", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gate_checks_mut"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    n, fails = 0, []
    bad = judge(C)
    n += 1
    if bad:
        fails += ["A-C patched engine: " + b for b in bad]
    tmp = Path(tempfile.mkdtemp(prefix="gsuite_tail_"))
    try:
        mut = make_mutant(tmp)
        n += 1
        mbad = judge(mut)
        if not mbad:
            fails.append("E MUTANT (tail code removed) passed the judge: the test is vacuous")
        # D: verdict, reasons, rc, token and old keys identical between the patched and the tail-less engine
        ents = [raw(FAIL_CODE), raw("print('ZZTEST_PASS')"), raw("print('nope')"), raw("print('ZZTEST_PASS'); print('ZZTEST_SKIP why')")]
        a, b = run(C, [dict(e) for e in ents]), run(mut, [dict(e) for e in ents])
        n += 1
        strip = lambda r: {k: r[k] for k in ("check", "status", "reasons", "arm", "rehearsal")}  # noqa: E731
        if strip(a) != strip(b):
            fails.append(f"D verdict/reasons differ: {strip(a)} vs {strip(b)}")
        for x, y in zip(scripts(a), scripts(b)):
            if {k: v for k, v in x.items() if k in OLD_KEYS and k != "seconds"} != {k: v for k, v in y.items() if k != "seconds"}:
                fails.append(f"D old-key evidence differs for {x['script']}")
        pa = [x for x in scripts(a) if x["rc"] == 0 and x["token_seen"]]
        if any(set(x) != OLD_KEYS for x in pa):
            fails.append("D passing entry gained keys")
        if a["status"] != "FAIL" or len(a["reasons"]) != 3:
            fails.append("D sanity: the failing suite did not FAIL")
    finally:
        sys.modules.pop("gate_checks_mut", None)
        shutil.rmtree(tmp, ignore_errors=True)
    if fails:
        print("ZZTEST_FAIL " + " | ".join(fails))
        return 1
    print(f"ZZTEST_PASS {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
