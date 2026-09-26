#!/usr/bin/env python3
"""Did a bench run SUCCEED? — judged on the CONTENT of summary.json, never on its existence.

⛔⛔ WHY THIS FILE EXISTS (MEASURED 2026-09-20, TWICE, BOTH MINE). The suite writes ``summary.json`` for
a FAILED run too — every arm ``status: FAILED``, every headline ``UNAVAILABLE``. Two of my scripts tested
``[ -f summary.json ]`` and called that success:
  * ``finish_when_done.sh`` produced a FINISH_REPORT.md that looked like a result, UNAVAILABLE in every
    cell (caught and fixed the same evening);
  * ``retry_scoring.sh:73`` — the SAME test surviving in the sibling script — logged
    ``BENCH_STATUS=FAILED`` and then ``SUCCESS`` on the next line, wrote ``SUCCESSFUL_RUN_DIR.txt``
    pointing at an all-FAILED run (d3c2b2), and exited with 5 of 6 attempts unused.
⇒ ONE judge, used by every script, so the defect cannot survive in a copy again.

SUCCESS iff EVERY required arm reads ``status: OK`` in summary.json. The counts are printed so the gate
is auditable from a log line, and the verdict is a single literal token on the last line
(``JUDGE=SUCCESS`` / ``JUDGE=FAILED``) that a caller greps instead of reading an exit code through a
shell that can overwrite it.

    python judge_run.py <run dir> --require CV,STOP,ECHO,A1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def judge(run_dir, require) -> dict:
    p = Path(run_dir) / "summary.json"
    if not p.exists():
        return {"verdict": "FAILED", "reason": "summary.json ABSENT", "ok": [], "not_ok": list(require)}
    try:
        s = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:                                               # noqa: BLE001
        return {"verdict": "FAILED", "reason": f"summary.json unreadable: {type(e).__name__}",
                "ok": [], "not_ok": list(require)}
    arms = s.get("arms") or {}
    ok = [a for a in require if (arms.get(a) or {}).get("status") == "OK"]
    not_ok = {a: (arms.get(a) or {}).get("status", "ABSENT") for a in require if a not in ok}
    return {"verdict": "SUCCESS" if not not_ok else "FAILED",
            "n_required": len(require), "n_ok": len(ok), "ok": ok, "not_ok": not_ok,
            "reason": ("every required arm is status OK" if not not_ok else
                       f"{len(not_ok)} of {len(require)} required arms are not OK: {not_ok}")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--require", default="CV,STOP,ECHO,A1")
    a = ap.parse_args(argv)
    req = [x.strip() for x in a.require.split(",") if x.strip()]
    r = judge(a.run_dir, req)
    print(json.dumps(r))
    print(f"JUDGE={r['verdict']}")
    return 0 if r["verdict"] == "SUCCESS" else 1


if __name__ == "__main__":
    sys.exit(main())
