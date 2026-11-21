"""PREREG_V8F_STOPPERC arm A: fix ``--w-stop-aux`` from the gate smoke, never tune it.

The card (section 5, arm A): *W is fixed at the gate smoke so that the term equals T's goal-BCE term at step 0.* W is therefore ``goal / stop_aux`` where both are the UNWEIGHTED terms of the same step-0 forward
(``V8FMini.loss`` ``T["goal"]`` and ``T["stop_aux"] = stop_cell + stop_pres``), so that ``W x stop_aux == goal``.

Two sources (the probe's is the one written into the argv; the log's is a cross-check that warns when it is more than 10 % away):

  selfcheck.json  ``stop_aux_probe.w_stop_aux_for_goal_equal`` (the G-LIVE probe's step-0 forward on a COPY of the model: ``--stop-aux on --w-stop-aux <any W > 0> --selfcheck-only``);
  train_log.jsonl the first row's ``goal`` and ``stop_aux`` (a ``--log-every 1`` smoke; row 1 is the first forward, before any optimiser step).

  python stop_aux_w_from_smoke.py --selfcheck <out>/selfcheck.json [--train-log <out>/train_log.jsonl] [--digits 4]

Prints ONE number (W, rounded to ``--digits`` decimals; the argv carries the rounded value) and exits 0; exits non-zero with a message when the inputs are missing, non-finite or not positive.
"""
from __future__ import annotations

import argparse
import json
import math
import sys


def _ratio(goal, term, where):
    if not (isinstance(goal, (int, float)) and isinstance(term, (int, float)) and math.isfinite(goal) and math.isfinite(term) and goal > 0.0 and term > 0.0):
        raise SystemExit(f"stop_aux_w_from_smoke: {where}: goal {goal!r} / stop_aux {term!r} must be finite and > 0")
    return goal / term


def w_from_selfcheck(sc: dict) -> float:
    p = sc.get("stop_aux_probe")
    if not p or p.get("stop_aux") == "off":
        raise SystemExit("stop_aux_w_from_smoke: selfcheck.json has no stop_aux_probe (run the smoke with --stop-aux on --w-stop-aux <W> --stop-aux-extr <table> --selfcheck-only)")
    w = _ratio(p.get("step0_goal_term"), p.get("step0_stop_aux_term"), "selfcheck stop_aux_probe")
    if p.get("w_stop_aux_for_goal_equal") is not None and abs(p["w_stop_aux_for_goal_equal"] - w) > 1e-6 * max(1.0, w):
        raise SystemExit(f"stop_aux_w_from_smoke: the probe's own W ({p['w_stop_aux_for_goal_equal']}) is not goal / stop_aux ({w})")
    return w


def w_from_train_log(path) -> float:
    with open(path, "r", encoding="utf-8") as fh:
        first = next((json.loads(ln) for ln in fh if ln.strip()), None)
    if first is None or "stop_aux" not in first or "goal" not in first:
        raise SystemExit(f"stop_aux_w_from_smoke: {path}: no first row carrying `goal` and `stop_aux` (a --stop-aux on smoke with --log-every 1)")
    return _ratio(first["goal"], first["stop_aux"], f"{path} row 1")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selfcheck", required=True)
    ap.add_argument("--train-log", default=None)
    ap.add_argument("--digits", type=int, default=4)
    a = ap.parse_args(argv)
    with open(a.selfcheck, "r", encoding="utf-8") as fh:
        w = w_from_selfcheck(json.load(fh))
    if a.train_log:
        w2 = w_from_train_log(a.train_log)
        if abs(w - w2) > 0.10 * max(1.0, w):                    # ADVISORY: the probe forces p_gt 0.5 and the loop starts at p_gt_start, so T's goal term differs a little; a gap this large is a different model or batch
            print(f"stop_aux_w_from_smoke: WARNING the selfcheck probe gives W {w:.6f} and the smoke log's first row gives {w2:.6f} (> 10 % apart): check that both read the same smoke", file=sys.stderr)
    print(f"{round(w, a.digits):.{a.digits}f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
