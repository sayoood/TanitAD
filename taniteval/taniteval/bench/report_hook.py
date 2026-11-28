"""End-of-run report hook: ``python -m taniteval.benchreport <run_dir>`` (W5's package).

Orchestrator ruling 2026-09-19: W5's package is ``taniteval/taniteval/benchreport/`` (renamed so it
never shadows the legacy ``taniteval/taniteval/report.py``). The suite calls it in a SUBPROCESS —
a report failure is recorded, never allowed to corrupt a scored run — and only if the package is
importable; otherwise the run records ``REPORT_PENDING`` with the reason.
⛔ The legacy ``taniteval.report`` module is a DIFFERENT tool and is never invoked on a run dir.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPORT_MODULE = "taniteval.benchreport"


def _last_error_line(log_path) -> str:
    """The most informative line of a failed render's log: the final traceback line if there is one,
    else the last non-empty line. ⚠️ Returns a STATED reason on failure rather than an empty string --
    an empty `reason` reads as "no reason", which is the same lie as no field at all."""
    try:
        lines = [ln.rstrip() for ln in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
                 if ln.strip()]
    except Exception as e:                                              # noqa: BLE001
        return f"UNAVAILABLE: could not read the render log ({type(e).__name__}: {e})"
    if not lines:
        return "UNAVAILABLE: the render log is EMPTY (the renderer produced no output at all)"
    for ln in reversed(lines):
        if ln and not ln.startswith((" ", "	")) and ":" in ln and "Traceback" not in ln:
            return ln[:400]
    return lines[-1][:400]


def report_available() -> tuple:
    try:
        spec = importlib.util.find_spec(REPORT_MODULE)
    except Exception as e:                                              # noqa: BLE001
        return False, f"find_spec raised {type(e).__name__}: {e}"
    if spec is None:
        return False, f"{REPORT_MODULE} is not importable (W5 has not landed it)"
    if not spec.submodule_search_locations:
        return False, f"{REPORT_MODULE} resolves to a MODULE, not W5's package — refusing to guess its CLI"
    main_py = Path(list(spec.submodule_search_locations)[0]) / "__main__.py"
    if not main_py.exists():
        return False, f"{REPORT_MODULE} has no __main__.py yet (W5 in progress)"
    return True, str(main_py)


def render(run_dir: Path, timeout_s: float = 1800.0) -> dict:
    ok, why = report_available()
    if not ok:
        return {"status": "REPORT_PENDING", "reason": why,
                "how_to_render_later": f"python -m {REPORT_MODULE} {str(run_dir).replace(os.sep, '/')}"}
    t0 = time.time()
    log = Path(run_dir) / "raw" / "benchreport.log"
    try:
        with open(log, "w", encoding="utf-8") as fh:
            r = subprocess.run([sys.executable, "-m", REPORT_MODULE, str(run_dir)], stdout=fh,
                               stderr=subprocess.STDOUT, timeout=timeout_s, env=dict(os.environ))
        index = Path(run_dir) / "report" / "index.html"
        # ⛔ ASSERT ON THE ARTIFACT, NOT THE STATUS. MEASURED 2026-09-20: W5's CLI exits 3 when the
        # report renders but its failure GALLERY is unavailable (it needs the NavSim venv + frames);
        # the page itself was complete and its own verifier read "PASS (90 numbers, 0 errors)".
        # A non-zero code is carried as a WARNING beside the rendered page, never as a failure.
        rendered = index.exists() and index.stat().st_size > 0
        out = {"status": "RENDERED" if rendered else "REPORT_FAILED", "rc": r.returncode,
               "index": "report/index.html" if rendered else None, "index_exists": index.exists(),
               "log": "raw/benchreport.log", "wall_s": round(time.time() - t0, 1)}
        if not rendered:
            # ⛔ A FAILURE MUST CARRY ITS REASON IN THE RECORD, not only a log path. MEASURED
            # 2026-09-20 on the navhard run: bench_run.json said REPORT_FAILED / rc 1 / index_exists
            # false and NOTHING ELSE, so the cause (Path(None) in the gallery) was invisible to every
            # consumer and to the leaderboard -- a reader had to know a log file existed and open it.
            out["reason"] = _last_error_line(log)
            out["how_to_retry"] = f"python -m {REPORT_MODULE} <run_dir>  (safe to re-run; it rewrites report/)"
        if rendered and r.returncode != 0:
            out["warning"] = (f"taniteval.benchreport exited {r.returncode} but wrote a non-empty page — "
                              "read raw/benchreport.log and report/verify.json for what it could not do")
        vp = Path(run_dir) / "report" / "verify.json"
        if vp.exists():
            try:
                out["verify"] = json.loads(vp.read_text(encoding="utf-8"))
            except Exception as e:                                       # noqa: BLE001
                out["verify_parse_error"] = f"{type(e).__name__}: {e}"
        return out
    except Exception as e:                                              # noqa: BLE001
        return {"status": "REPORT_FAILED", "reason": f"{type(e).__name__}: {e}"[:400], "log": "raw/benchreport.log"}
