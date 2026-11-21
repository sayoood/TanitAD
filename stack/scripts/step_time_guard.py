#!/usr/bin/env python3
"""A step-time abort criterion that CAN ACTUALLY FIRE.

⛔ WHY THIS EXISTS — C112, and it is an instrument failure, not an arithmetic one
================================================================================
``train_v6_staged.py`` logs ``step_s``, whose own note says it is
``elapsed/step over the N steps THIS process ran``. That is a **CUMULATIVE MEAN
since process start**, and a ``+5 %`` abort criterion built on it is
**STRUCTURALLY UNABLE TO FIRE**:

    at N=6300 steps and elapsed=166,791.9 s, driving the mean to 28.0 needs
    k further steps at instantaneous rate r, with k*(r-28) = 9608.1:
      r = 27.7 s/step  (the intended +5 % trip point)  -> NEVER (asymptote 27.7)
      r = 30.0 s/step  (+13 %)                         -> k = 4804 ~ 40 h
      r = 40.0 s/step  (+51 %, catastrophic)           -> k =  801 ~  8.9 h

The concurrency pilot ran ~2 h. **It would have reported "safe" no matter what
happened.** ⚠️ And the reassuring reference band *"26.47–26.66 all day"* is not
day variation — it is a **converging mean, strictly non-increasing over 100
points**. *A quantity that cannot rise is not a monitor.*

Same family as ``df`` on a pod, ``free``/``tegrastats`` on Thor, and cgroup
``usage_in_bytes``: **a real number answering a narrower question than the one
asked.**

⇒ The admissible quantity is the **MARGINAL** rate over one log interval.
``train_v6_staged.py`` now logs it directly as ``step_s_interval``; for the
**banked logs that predate that field** this module recovers it exactly, because
the cumulative series is invertible::

    elapsed_i = step_s_i * n_i           (n = steps THIS process ran)
    r(i) = (step_s_i*n_i - step_s_{i-1}*n_{i-1}) / (n_i - n_{i-1})

⚠️ THE WARM-UP IS THE REASON A PERSISTENCE-ONLY GUARD IS NOT ENOUGH
===================================================================
MEASURED on the live ``v6F-SW-30k`` log (254 logged rows, 252 marginal points,
``…/incoming/2026-08-18-o2-live-and-ridge-reread/raw/v6F-SW-30k_train_log.jsonl``):

===========================  ====  ==========  ===============================
segment                         n  median s/s  worst excess
===========================  ====  ==========  ===============================
era-2 STEADY (n > 900)        110     26.3594  **+2.589 %** over its own median
era-2 WARM-UP (n <= 900)       17     27.1252  **+3.229 %** over steady median
===========================  ====  ==========  ===============================

⛔ The post-resume warm-up is **+3.2 % for 17 CONSECUTIVE rows**, so requiring
*k consecutive* over-threshold points does **not** exclude it — only knowing
where the process restarted does. That is what ``steps_this_process`` is for: it
**resets** on restart, which is how a segment boundary is detected.

⇒ **The tolerance, stated honestly in both directions** (MEASURED on the live
log; a first draft of this file overstated the case and the test caught it):

* **+5 %** is ``OK`` on the live run **with or without** the warm-up exclusion
  (2.41 pp of margin over steady, 1.77 pp over warm-up). At this tolerance the
  warm-up exclusion buys robustness; it is **not load-bearing**.
* **+3 %** is ``OK`` on steady state but ``TRIP`` on the warm-up ⇒ **without the
  exclusion it fires on EVERY RESUME** — the rejects-everything shape (C95).
  With the exclusion it survives, but on **0.41 pp** of margin.
* ⇒ **The warm-up exclusion becomes load-bearing at any tolerance ≤ +3.23 %**,
  and no tolerance at all rescues ``step_s`` itself, which is the
  passes-everything shape (C97).

**+5 % on the marginal rate is therefore the defensible setting** — ~1.9× the
measured steady worst case, not a comfortable one.

⚠️ EVIDENCE CLASS: MEASURED (ours) — every number above is computed by
``stack/tests/test_step_time_guard.py`` from the banked log named there, not
copied from prose.

Usage::

    python3 stack/scripts/step_time_guard.py <train_log.jsonl> --baseline 26.36
    python3 stack/scripts/step_time_guard.py <log> --baseline 26.36 --tol 0.05
    python3 stack/scripts/step_time_guard.py <log> --json verdict.json

Read-only. Exit 0 = OK, 1 = TRIP, 2 = INSUFFICIENT/unusable — but ⚠️ **the exit
code is not the evidence; the printed verdict is.**
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

#: ``n`` out of the prose note, for logs written before ``steps_this_process``
#: became a first-class key. ⚠️ A regex over prose is a fallback, never the
#: primary path — the note's wording is not a contract.
_N_IN_NOTE = re.compile(r"over the (\d+) steps")

#: MEASURED: the post-resume transient runs +3.229 % over steady state for the
#: first ~900 steps (17 consecutive logged rows at --log-every 50). Excluding it
#: is not cosmetic — it is the difference between a guard and a false alarm.
WARMUP_STEPS = 900

#: MEASURED: steady variation tops out at +2.589 %, warm-up at +3.229 %.
DEFAULT_TOL_FRAC = 0.05

#: Two consecutive over-threshold intervals, so a single slow window (disk
#: hiccup, checkpoint write) is not an abort. ⚠️ Persistence alone would NOT
#: have excluded the warm-up — see the module docstring.
DEFAULT_PERSIST = 2

OK = "OK"
TRIP = "TRIP"
INSUFFICIENT = "INSUFFICIENT"


# --------------------------------------------------------------- extraction
def steps_this_process(row: dict) -> int | None:
    """``n`` for a log row — first-class key first, prose only as a fallback.

    Returns ``None`` rather than a guess. ⚠️ C105 cost three rounds to a
    locator that returned something plausible instead of failing.
    """
    val = row.get("steps_this_process")
    if isinstance(val, bool):          # bool is an int subclass; not a count
        return None
    if isinstance(val, int):
        return val
    match = _N_IN_NOTE.search(str(row.get("step_s_note") or ""))
    return int(match.group(1)) if match else None


def load_rows(path: str | Path) -> list[dict]:
    """Read a ``train_log.jsonl`` and keep only rows that carry a step time.

    A v6 log interleaves ``run_start``, spectrum records and log rows; the
    spectrum records carry ``step`` but no ``step_s``, and pairing them with log
    rows would manufacture nonsense intervals.
    """
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and (
                    "step_s" in rec or "step_s_interval" in rec):
                rows.append(rec)
    return rows


def marginal_rates(rows: list[dict]) -> list[dict]:
    """Marginal s/step per logged interval, with segment boundaries marked.

    Prefers the trainer's native ``step_s_interval``; falls back to
    first-differencing the cumulative ``step_s``, which is **exact**, not an
    approximation. A row whose rate cannot be established gets
    ``s_per_step=None`` and ``source="unusable"`` — it is never dropped
    silently, because a shrinking denominator is how a monitor quietly stops
    monitoring.

    ``segment`` increments whenever ``n`` goes backwards, i.e. the trainer
    process restarted. ⛔ A rate must NEVER be differenced across that boundary:
    the elapsed clock resets, so the arithmetic yields a meaningless (often
    negative) number that reads like a speed-up.
    """
    out: list[dict] = []
    segment = 0
    prev_n: int | None = None
    prev_cum: float | None = None

    for row in rows:
        n = steps_this_process(row)
        if prev_n is not None and n is not None and n <= prev_n:
            segment += 1                      # process restart
            prev_n, prev_cum = None, None     # nothing to difference against

        native = row.get("step_s_interval")
        cum = row.get("step_s")
        cum = float(cum) if isinstance(cum, (int, float)) else None

        rate: float | None = None
        source = "unusable"
        d_n: int | None = None
        if prev_n is not None and n is not None and n > prev_n:
            d_n = n - prev_n

        if isinstance(native, (int, float)) and not isinstance(native, bool):
            rate, source = float(native), "native"
        elif prev_cum is not None and cum is not None and d_n:
            rate = (cum * n - prev_cum * prev_n) / d_n
            source = "derived"

        out.append({"step": row.get("step"), "n": n, "d_n": d_n,
                    "s_per_step": rate, "source": source, "segment": segment})
        if n is not None:
            prev_n, prev_cum = n, cum
    return out


# ------------------------------------------------------------------- verdict
def check(rates: list[dict], baseline_s: float | None = None,
          tol_frac: float = DEFAULT_TOL_FRAC,
          persist: int = DEFAULT_PERSIST,
          warmup_steps: int = WARMUP_STEPS) -> dict:
    """Does the marginal step time exceed ``baseline*(1+tol)`` persistently?

    ``INSUFFICIENT`` is a distinct verdict from ``OK`` on purpose. A run with no
    admissible points has proved nothing, and reporting that as ``OK`` is the
    dead-pod failure: *absence of a finding is not a finding of absence.*
    """
    # ⚠️ Partitioned in ONE pass by identity, not by `r not in warm`: two
    # intervals can be dict-equal (same step, n, rate) and membership testing
    # would then drop a real point. A monitor must not lose samples to its own
    # bookkeeping.
    usable, warm, adm = [], [], []
    for r in rates:
        if r["s_per_step"] is None:
            continue
        usable.append(r)
        (warm if (r["n"] is None or r["n"] <= warmup_steps) else adm).append(r)

    if baseline_s is None:
        baseline_s = st.median([r["s_per_step"] for r in adm]) if adm else None
        baseline_source = ("auto-median of admissible points — ⚠️ CANNOT "
                           "detect a uniformly slow run, since the slowdown "
                           "would be inside the baseline. Pass --baseline for "
                           "a decision-grade check.")
    else:
        baseline_source = "explicit"

    result = {
        "verdict": INSUFFICIENT,
        "baseline_s": baseline_s,
        "baseline_source": baseline_source,
        "tol_frac": tol_frac,
        "threshold_s": None if baseline_s is None else baseline_s * (1 + tol_frac),
        "persist": persist,
        "warmup_steps": warmup_steps,
        "n_rows": len(rates),
        "n_admissible": len(adm),
        "n_excluded_warmup": len(warm),
        "n_unusable": len(rates) - len(usable),
        "n_segments": 1 + max((r["segment"] for r in rates), default=0),
        "sources": sorted({r["source"] for r in rates}),
        "max_s_per_step": None,
        "max_excess_frac": None,
        "trip_run": [],
        "note": "",
        "_evidence_class": "MEASURED (ours)",
    }

    if baseline_s is None or len(adm) < persist:
        result["note"] = (
            f"only {len(adm)} admissible interval(s) against persist={persist} "
            f"({len(warm)} excluded as post-restart warm-up, "
            f"{result['n_unusable']} unusable). ⛔ This run proves NOTHING "
            f"about the step time — it is not a clean result.")
        return result

    threshold = result["threshold_s"]
    worst = max(adm, key=lambda r: r["s_per_step"])
    result["max_s_per_step"] = worst["s_per_step"]
    result["max_excess_frac"] = worst["s_per_step"] / baseline_s - 1.0

    run: list[dict] = []
    for point in adm:
        # ⛔ Never carry a run across a process restart: the two sides are
        # different processes, and the second side's first points are warm-up.
        if run and point["segment"] != run[-1]["segment"]:
            run = []
        if point["s_per_step"] > threshold:
            run.append(point)
            if len(run) >= persist:
                result["verdict"] = TRIP
                result["trip_run"] = [
                    {"step": p["step"], "n": p["n"],
                     "s_per_step": round(p["s_per_step"], 4),
                     "excess_frac": round(p["s_per_step"] / baseline_s - 1, 6)}
                    for p in run]
                result["note"] = (
                    f"{len(run)} consecutive intervals over "
                    f"{threshold:.4f} s/step (baseline {baseline_s:.4f} "
                    f"+{100 * tol_frac:.1f} %), ending at step "
                    f"{run[-1]['step']}.")
                return result
        else:
            run = []

    result["verdict"] = OK
    result["note"] = (
        f"{len(adm)} admissible intervals, worst "
        f"{worst['s_per_step']:.4f} s/step at step {worst['step']} "
        f"(+{100 * result['max_excess_frac']:.3f} %) — under the "
        f"+{100 * tol_frac:.1f} % threshold of {threshold:.4f} s/step.")
    return result


def format_verdict(result: dict) -> str:
    lines = [
        "=" * 74,
        "step-time guard — MARGINAL rate (step_s is a cumulative mean and",
        "CANNOT fire; see C112 and this module's docstring)",
        "=" * 74,
        f"  VERDICT           {result['verdict']}",
        f"  baseline          {result['baseline_s']} s/step "
        f"({result['baseline_source'].splitlines()[0]})",
        f"  threshold         {result['threshold_s']} s/step "
        f"(+{100 * result['tol_frac']:.1f} %, persist={result['persist']})",
        f"  intervals         {result['n_admissible']} admissible · "
        f"{result['n_excluded_warmup']} warm-up-excluded · "
        f"{result['n_unusable']} unusable · "
        f"{result['n_segments']} segment(s) · sources={result['sources']}",
    ]
    if result["max_s_per_step"] is not None:
        lines.append(f"  worst interval    {result['max_s_per_step']:.4f} s/step "
                     f"({result['max_excess_frac'] * 100:+.3f} %)")
    lines.append(f"  {result['note']}")
    if result["verdict"] == TRIP:
        for point in result["trip_run"]:
            lines.append(f"    TRIP  step {point['step']}  "
                         f"{point['s_per_step']} s/step  "
                         f"{point['excess_frac'] * 100:+.3f} %")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Abort criterion on the MARGINAL step time. The trainer's "
                    "step_s is a cumulative mean and cannot fire (C112).")
    ap.add_argument("log", help="train_log.jsonl (or any jsonl of log rows)")
    ap.add_argument("--baseline", type=float, default=None,
                    help="reference s/step. Omit only for exploration — an "
                         "auto baseline cannot see a uniformly slow run.")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL_FRAC,
                    help=f"fractional tolerance (default {DEFAULT_TOL_FRAC} = "
                         f"+5 %%; steady variation MEASURED at +2.589 %%)")
    ap.add_argument("--persist", type=int, default=DEFAULT_PERSIST)
    ap.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    ap.add_argument("--json", help="write the verdict here")
    args = ap.parse_args(argv)

    rows = load_rows(args.log)
    if not rows:
        print(f"[guard] ⛔ no log rows with a step time in {args.log} — this is "
              f"NOT a clean result.", file=sys.stderr)
        return 2
    result = check(marginal_rates(rows), args.baseline, args.tol,
                   args.persist, args.warmup_steps)
    print(format_verdict(result))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2),
                                   encoding="utf-8")
        print(f"[guard] wrote {args.json}")
    return {OK: 0, TRIP: 1, INSUFFICIENT: 2}[result["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
