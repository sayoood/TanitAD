#!/usr/bin/env python3
"""p_check_arm.py — the per-arm gate for a P arm, written as an ARTIFACT.

⛔ WHY THIS EXISTS AND WHY IT IS NOT OPTIONAL. `p_runner.py` treats an arm directory with no
`p_arm_check.json` as **PARTIAL** and moves it aside before re-running. Without a checker, a
FINISHED 11-hour arm reads as partial and the next invocation destroys it. The check is therefore
the thing that makes the runner resumable, not a nicety.

⛔ AND IT ASSERTS ON THE ARTIFACT, NEVER ON AN EXIT CODE. The trainer's return status travels
through a shell that can overwrite it (`$?` after a pipeline is the LAST element's status); the
admissible evidence that an arm ran is its `metrics.jsonl` and its `config.json`.

What must be true for ``VALID``:
  1. ``run/config.json`` exists and its recorded ``argv`` carries the EXPECTED seed and steps —
     so an arm that silently ran a different configuration cannot pass.
  2. ``run/metrics.jsonl`` carries an eval row at the expected final step.
  3. The pre-registration's own label-density identity holds on that row:
     ``eval_box3d_n_matched == eval_box3d_n_target`` (a `min(targets, queries)` identity while
     32 ≤ 100). A matcher that pairs fewer than it should breaks it, and the arm is then not
     comparable to its base.
  4. The headline ``eval_box3d_centre`` is finite.

The four metric families are carried through to the check so no downstream read has to re-open
the metrics file to find them.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HEAD = "eval_box3d_centre"
FAMILY_KEYS = ("eval_box3d", "eval_box3d_centre", "eval_box3d_size", "eval_box3d_yaw",
               "eval_box3d_presence", "eval_box3d_cls", "eval_box3d_rates", "eval_box3d_occ",
               "eval_box3d_z", "eval_box3d_h", "eval_box3d_n_matched", "eval_box3d_n_target",
               "eval_box3d_n_dropped", "eval_box3d_n_windows", "eval_box3d_n_labelled")


def flag_value(argv, flag, n=1):
    """The value(s) after ``flag`` in a recorded argv, or None."""
    for i, t in enumerate(argv):
        if t == flag:
            return argv[i + 1] if n == 1 else argv[i + 1:i + 1 + n]
    return None


def last_eval_row(metrics_path: Path) -> tuple[dict | None, int]:
    """The LAST row carrying the headline, and the COUNT OF UNPARSEABLE LINES.

    ⛔ Not the last line: train rows have no eval keys.
    ⛔ AND THE SKIP IS COUNTED, NEVER SILENT. A bare `except: continue` here cannot distinguish
    "this line is a different kind of thing" from "this file is damaged". MEASURED in the pinned tree: every metrics write is a COMPLETE line followed immediately by `log.flush()` (`refc_v3_train.py` :7180-7181, and four sibling sites 7093/7098, 7261/7263, 7273/7274, 7324/7328; the file is opened once in append mode at :6958).
    ⇒ A hard kill therefore loses the final ROW, not half of one: killed between rows, or between `write()` and `flush()`, the file PARSES PERFECTLY and is simply SHORT. The only partial-line window is inside the flush syscall itself. ⛔ SO FOR A SIGKILL OR A cgroup-OOM — this box's documented trainer death — `n_unparseable_metrics_lines` is STRUCTURALLY BLIND, and the STEP-COUNT check is what fires. The unparseable counter's real triggers are power loss, a filesystem fault, or a different writer that does not flush per row.
    Swallowing it would let the checker read an EARLIER row as the final one and pass an arm
    whose artifact never fully parsed. The count is returned so the caller can refuse.

    ⚠️ A7-IN-s0, the arm paused on 2026-09-20, is the CLEAN counter-example and was VERIFIED rather than assumed: 107 lines, **0 unparseable**, file ends with a newline, last row step 1,070 = 107 x 10. ⚠️ I had written that this arm's kill DID truncate it; that was FALSE — a plausible mechanism asserted about a specific file without opening it. Why it stayed clean is NOT established: the ordered kill (trainer last, by explicit PID) and a per-row flush are both candidates and I have separated neither. *(Class named with the Master Mind 2026-09-20: the swallow
    living in the CHECKER's own error handling rather than in the code under test.)*
    """
    best, n_bad = None, 0
    with open(metrics_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:                               # noqa: BLE001 — COUNTED, not dropped
                n_bad += 1
                continue
            if HEAD in d:
                best = d
    return best, n_bad


def check(arm_dir: Path, *, seed: int, steps: int) -> dict:
    run = arm_dir / "run"
    bad, rep = [], {"arm_dir": str(arm_dir), "expected_seed": seed, "expected_steps": steps}
    cfg_p, met_p = run / "config.json", run / "metrics.jsonl"
    if not cfg_p.exists():
        bad.append("run/config.json MISSING — the arm did not get far enough to stamp itself")
    else:
        try:
            argv = list(json.loads(cfg_p.read_text(encoding="utf-8")).get("argv") or [])
        except Exception as exc:                            # noqa: BLE001
            argv = []
            bad.append(f"run/config.json UNREADABLE ({type(exc).__name__}) — INCONCLUSIVE")
        got_seed, got_steps = flag_value(argv, "--seed"), flag_value(argv, "--steps")
        rep["recorded_seed"], rep["recorded_steps"] = got_seed, got_steps
        if argv and str(got_seed) != str(seed):
            bad.append(f"seed mismatch: config says {got_seed}, expected {seed}")
        if argv and str(got_steps) != str(steps):
            bad.append(f"steps mismatch: config says {got_steps}, expected {steps}")
    if not met_p.exists():
        bad.append("run/metrics.jsonl MISSING")
        row = None
    else:
        row, n_bad = last_eval_row(met_p)
        rep["n_unparseable_metrics_lines"] = n_bad
        if n_bad:
            bad.append(f"⛔ {n_bad} unparseable line(s) in metrics.jsonl — the artifact did not "
                       f"fully parse, so the 'last' eval row may not be the last one written")
        if row is None:
            bad.append(f"no eval row carrying {HEAD} — the arm never completed an eval")
    if row is not None:
        rep["final_step"] = row.get("step")
        if str(row.get("step")) != str(steps):
            bad.append(f"final eval at step {row.get('step')}, expected {steps}")
        rep["metrics"] = {k: row[k] for k in FAMILY_KEYS if k in row}
        nm, nt = row.get("eval_box3d_n_matched"), row.get("eval_box3d_n_target")
        rep["identity_matched_equals_target"] = (nm is not None and nt is not None
                                                 and abs(float(nm) - float(nt)) < 1e-6)
        if not rep["identity_matched_equals_target"]:
            bad.append(f"⛔ label-density identity BROKEN: n_matched {nm} != n_target {nt}")
        h = row.get(HEAD)
        if h is None or not math.isfinite(float(h)):
            bad.append(f"{HEAD} is {h} — not a finite number")
        else:
            rep["headline_" + HEAD] = float(h)
    rep["reasons"] = bad
    rep["status"] = "VALID" if not bad else "INVALID"
    return rep


# ⛔ A CHECKER MUST NOT DIE ON ITS OWN OUTPUT. MEASURED 2026-09-20: three separate
# readouts crashed with a cp1252 `UnicodeEncodeError` on this box mid-print -- one of
# them after reporting "lines lost = 1" but BEFORE naming the line, i.e. it had verified
# nothing while looking like it had. Relying on the caller to export PYTHONIOENCODING is
# a habit; this is a guard. `errors="replace"` means the print degrades instead of
# raising even if the stream cannot take utf-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001 -- a stream that cannot be reconfigured is not fatal
    pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_dir")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    a = ap.parse_args(argv)
    d = Path(a.arm_dir)
    rep = check(d, seed=a.seed, steps=a.steps)
    d.mkdir(parents=True, exist_ok=True)
    (d / "p_arm_check.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    print("ZZP-CHECK-%s ZZ" % rep["status"], flush=True)
    return 0 if rep["status"] == "VALID" else 5


if __name__ == "__main__":
    raise SystemExit(main())
