#!/usr/bin/env python3
"""refcv3_metrics_read.py -- derive the honest read of refcv3's in-training metrics.

WHY THIS FILE EXISTS
--------------------
`refc_v3_train.py` appends BOTH per-batch train rows and `eval_*` rows (160 FIXED
held-out windows every `--eval-every` steps) to ONE `metrics.jsonl`, across every
relaunch. The run this was written for died six times and was deliberately switched
three times, so the file is NOT a time series:

  * a resume replays the 500 steps since the last checkpoint, so **steps repeat**
    and row order runs backwards at every boundary;
  * the nav source changed mid-run (refb-derived -> v7.2 token) and the eval was
    leaking the held-out label marginals into the model's tactical prior until a
    later step, so **the series spans three different measurements**;
  * `elapsed_s` is cumulative SINCE THE CURRENT LAUNCH, so a naive diff across a
    boundary is negative and a naive total is wrong.

Reading the file without those three facts produces a plausible, wrong answer --
which is why this reader exists and why it prints the reconstruction before any
metric.

TIER
----
⛔ EVERYTHING THIS TOOL EMITS IS **T0** (EVAL_DOCTRINE.md): an in-training loss
monitor on the training loss surface, over held-out windows. It is a WM/readout
diagnostic and may NEVER be quoted as driving performance, nor as the four binding
metric families -- those are a separate T1 job with paired episode-cluster CIs.

⛔ NO INTERVAL IS AVAILABLE FROM THIS FILE, BY ANY ESTIMATOR. Each `eval_*` value
is already the POOLED MEAN over the 160 windows; the file carries no per-window
values and no episode index, and `taniteval.ci.episode_cluster_bootstrap` needs
both. Every statement derived here is a DIRECTION, not a verdict. The tool prints
that refusal in its own output so a number cannot travel without it.

ESTIMATOR / FIT DISCIPLINE
--------------------------
Slopes are OLS on (step, value) within ONE era, and each carries its R^2, its n and
its exact fit window. Per CLAUDE.md a fit below **R^2 0.80** is NOT quotable as a
rate; the tool marks those `quotable_rate: false` and the reading falls back to the
half-split ratio, which is reported as direction only. `sd_step` (the scatter
implied by consecutive evals) is a DESCRIPTIVE scale, not a standard error: the
points are autocorrelated and share one 160-window sample.

Stdlib only -- no numpy, no torch, no pod paths. 0 GPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

#: Columns emitted by the eval block that are METRICS (not bookkeeping).
EVAL_METRIC_COLS = (
    "eval_loss", "eval_traj", "eval_cls", "eval_law", "eval_route",
    "eval_lat", "eval_lon", "eval_lat_tac", "eval_lon_tac",
    "eval_goal_tac", "eval_goal2s_err_m", "eval_anchor_acc", "eval_sel_v3",
    "eval_goal_gate", "eval_goal_gate_grad", "eval_goal_score_absmean",
)
#: Columns that MUST be constant if the held-out window set really is fixed.
#: A varying value here withdraws every cross-step comparison in the output.
FIXED_SET_INVARIANTS = (
    "eval_windows", "eval_batches", "eval_slot_valid_frac",
    "eval_tac_label_rows", "eval_tac_label_v7", "eval_nav_injected",
)
#: train column <-> eval column for the generalisation-gap block.
GAP_PAIRS = (
    ("loss", "eval_loss"), ("traj", "eval_traj"), ("cls", "eval_cls"),
    ("law", "eval_law"), ("route", "eval_route"),
    ("lat_tac", "eval_lat_tac"), ("lon_tac", "eval_lon_tac"),
    ("goal2s_err_m", "eval_goal2s_err_m"),
)
#: CLAUDE.md: below this R^2 there is no quotable exponent/rate at all.
R2_QUOTABLE = 0.80
#: no-information cross-entropy of the v7.0 8-class tactical heads.
LN8 = math.log(8.0)

_SUP_RELAUNCH = re.compile(
    r"^\[sup (?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\] relaunch #\d+ "
    r"from step (?P<step>\d+)")


# --------------------------------------------------------------------------- #
# loading + reconstruction                                                     #
# --------------------------------------------------------------------------- #
def load_rows(path):
    """Every JSON row, in file order."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def kind(row):
    if "eval_error" in row:
        return "eval_error"
    return "eval" if "eval_loss" in row else "train"


def split_launches(rows):
    """Tag every row with its launch index.

    A launch boundary is a DECREASE of ``elapsed_s`` on a train row (``elapsed_s``
    is ``time.time() - t0`` with ``t0`` at launch). Eval rows carry no
    ``elapsed_s`` and inherit the launch that is current when they are written.
    """
    out, launch, prev = [], 0, None
    for r in rows:
        if kind(r) == "train":
            e = r["elapsed_s"]
            if prev is not None and e < prev:
                launch += 1
            prev = e
        out.append((launch, r))
    return out


def canonical(tagged):
    """Last-writer-wins per (step, kind).

    A resume REPLAYS the steps since the last checkpoint. The surviving
    trajectory is the LAST row written for a step, so that is the canonical one;
    the earlier copies are the recomputation cost, counted separately.
    """
    ev, tr = {}, {}
    for lch, r in tagged:
        if kind(r) == "eval":
            ev[r["step"]] = (lch, r)
        elif kind(r) == "train":
            tr[r["step"]] = (lch, r)
    return ev, tr


def replay_accounting(tagged, log_every=50):
    """Steps recomputed after each resume (the real cost of a death/switch)."""
    per, maxseen = {}, -1
    for lch, r in tagged:
        if kind(r) != "train":
            continue
        per.setdefault(lch, 0)
        if r["step"] <= maxseen:
            per[lch] += 1
        else:
            maxseen = r["step"]
    return {l: {"replayed_rows": n, "replayed_steps": n * log_every}
            for l, n in sorted(per.items())}


def parse_supervisor(path):
    """The relaunch lines: (timestamp, DEATH step). The resume is step-500."""
    if not path or not Path(path).exists():
        return []
    out = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        m = _SUP_RELAUNCH.match(ln.strip())
        if m:
            out.append({"ts": m.group("ts"), "from_step": int(m.group("step"))})
    return out


# --------------------------------------------------------------------------- #
# fitting                                                                      #
# --------------------------------------------------------------------------- #
def ols(xs, ys):
    """(slope, intercept, R^2). Returns Nones when n < 3 or x is degenerate."""
    n = len(xs)
    if n < 3:
        return None, None, None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None, None, None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    sst = sum((y - my) ** 2 for y in ys)
    ssr = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = (1 - ssr / sst) if sst > 0 else None
    return b, a, r2


def sd_step(values):
    """Scatter implied by CONSECUTIVE evals: pstdev(first differences)/sqrt(2).

    ⚠️ A descriptive scale for "is this move bigger than the series' own jitter",
    NOT a standard error and NOT an interval.
    """
    if len(values) < 3:
        return None
    d = [values[i] - values[i - 1] for i in range(1, len(values))]
    return statistics.pstdev(d) / math.sqrt(2.0)


def era_block(ev, steps, cols):
    """Per-column stats + a slope that carries its R^2, n and fit window."""
    block = {"n": len(steps),
             "fit_window": [steps[0], steps[-1]] if steps else None,
             "cols": {}}
    for c in cols:
        ys = [ev[s][1][c] for s in steps]
        b, _a, r2 = ols(steps, ys)
        row = {
            "first": ys[0], "last": ys[-1], "min": min(ys), "max": max(ys),
            "mean": sum(ys) / len(ys),
            "sd": statistics.pstdev(ys) if len(ys) > 1 else 0.0,
            "sd_step": sd_step(ys),
            "slope_per_1k_steps": (b * 1000.0) if b is not None else None,
            "r2": r2, "n": len(ys),
            "quotable_rate": bool(r2 is not None and r2 >= R2_QUOTABLE),
        }
        row["ratio_last_over_first"] = (ys[-1] / ys[0]) if ys[0] else None
        block["cols"][c] = row
    return block


def half_split(ev, steps, cols):
    """Descriptive first-half vs second-half move, scaled by the series' jitter."""
    if len(steps) < 6:
        return {"status": "SKIPPED", "reason": f"n={len(steps)} < 6"}
    mid = len(steps) // 2
    h1, h2 = steps[:mid], steps[mid:]
    out = {"h1": [h1[0], h1[-1]], "h1_n": len(h1),
           "h2": [h2[0], h2[-1]], "h2_n": len(h2),
           "note": ("|delta| / sd_step is a DESCRIPTIVE signal-to-scatter ratio, "
                    "NOT a t-statistic and NOT a CI: the points are "
                    "autocorrelated and share one 160-window sample."),
           "cols": {}}
    for c in cols:
        allv = [ev[s][1][c] for s in steps]
        a = [ev[s][1][c] for s in h1]
        b = [ev[s][1][c] for s in h2]
        s = sd_step(allv)
        d = sum(b) / len(b) - sum(a) / len(a)
        out["cols"][c] = {
            "h1_mean": sum(a) / len(a), "h2_mean": sum(b) / len(b),
            "delta": d, "sd_step": s,
            "abs_delta_over_sd_step": (abs(d) / s) if s else None,
            "moves_beyond_2x_scatter": bool(s and abs(d) / s >= 2.0),
        }
    return out


# --------------------------------------------------------------------------- #
# the derived blocks                                                           #
# --------------------------------------------------------------------------- #
def invariants(ev):
    """R2: the held-out window set must be FIXED, or every comparison dies."""
    out, ok = {}, True
    for k in FIXED_SET_INVARIANTS:
        vals = sorted({round(float(r[k]), 6) for _l, r in ev.values() if k in r})
        out[k] = {"distinct": len(vals), "values": vals,
                  "constant": len(vals) == 1}
        ok = ok and len(vals) == 1
    out["ALL_CONSTANT"] = ok
    out["meaning"] = (
        "constant => the same 160 windows at every step and across every "
        "relaunch, so the eval series is step-to-step comparable. If any value "
        "varies, EVERY cross-step eval comparison in this output is withdrawn.")
    return out


def gap_block(ev, tr, tagged, eval_every, drop_after_resume=0):
    """eval_X - mean(train X over (N-eval_every, N]).

    ``drop_after_resume`` drops that many train rows immediately after each
    launch boundary (the pre-registered robustness pass against post-resume
    spikes inflating the train side).
    """
    skip = set()
    if drop_after_resume:
        # the first `drop_after_resume` train rows of every launch AFTER the
        # first: the post-resume re-warm rows the robustness pass must exclude.
        counts = {}
        for lch, r in tagged:
            if kind(r) != "train":
                continue
            counts[lch] = counts.get(lch, 0) + 1
            if lch > 0 and counts[lch] <= drop_after_resume:
                skip.add(r["step"])
    rows = []
    for s in sorted(ev):
        ts = [t for t in tr if s - eval_every < t <= s and t not in skip]
        if not ts:
            continue
        row = {"step": s, "n_train_rows": len(ts)}
        for a, b in GAP_PAIRS:
            tm = sum(tr[t][1][a] for t in ts) / len(ts)
            row[f"train_{a}"] = tm
            row[f"gap_{a}"] = ev[s][1][b] - tm
        rows.append(row)
    return rows


def gap_summary(rows, eras):
    out = {}
    for name, lo, hi in eras:
        sel = [r for r in rows if lo <= r["step"] <= hi]
        out[name] = {"n": len(sel)}
        for a, _b in GAP_PAIRS:
            vals = [r[f"gap_{a}"] for r in sel]
            out[name][a] = (sum(vals) / len(vals)) if vals else None
    return out


def spikes(tagged, per_launch_steps, log_every=50):
    """R8: the first train row after a resume vs the pre-death level."""
    per = {}
    for lch, r in tagged:
        if kind(r) == "train":
            per.setdefault(lch, []).append(r)
    out = []
    for lch in sorted(per):
        rs = per[lch]
        ref = None
        if lch > 0 and (lch - 1) in per:
            prev = [r for r in per[lch - 1] if r["step"] <= rs[0]["step"]]
            if len(prev) >= 3:
                ref = statistics.median([r["loss"] for r in prev[-5:]])
        rec = None
        if ref is not None:
            for i, r in enumerate(rs):
                if r["loss"] <= ref:
                    rec = {"rows": i, "steps": i * log_every, "at_step": r["step"]}
                    break
        out.append({
            "launch": lch, "start_step": rs[0]["step"], "n_rows": len(rs),
            "first_losses": [r["loss"] for r in rs[:3]],
            "pre_boundary_median_loss": ref,
            "spike_ratio": (rs[0]["loss"] / ref) if ref else None,
            "recovered": rec,
        })
    return out


def pace(tagged, sup_relaunches, log_every=50):
    """R9: two independent clocks.

    probe A -- in-process ``elapsed_s`` per launch (skips the first row, which
    carries dataset init). probe B -- wall-clock deltas between ``supervisor.log``
    relaunch timestamps over the steps between the corresponding death steps.
    """
    per = {}
    for lch, r in tagged:
        if kind(r) == "train":
            per.setdefault(lch, []).append(r)
    probe_a = []
    for lch in sorted(per):
        rs = per[lch]
        if len(rs) < 3:
            probe_a.append({"launch": lch, "steps": [rs[0]["step"], rs[-1]["step"]],
                            "n_rows": len(rs), "s_per_step": None,
                            "reason": "fewer than 3 rows"})
            continue
        de = rs[-1]["elapsed_s"] - rs[1]["elapsed_s"]
        ds = rs[-1]["step"] - rs[1]["step"]
        probe_a.append({"launch": lch, "steps": [rs[0]["step"], rs[-1]["step"]],
                        "n_rows": len(rs), "span_s": de,
                        "s_per_step": de / ds if ds else None,
                        "startup_s_first_row": rs[0]["elapsed_s"]})
    probe_b = []
    for i in range(1, len(sup_relaunches)):
        a, b = sup_relaunches[i - 1], sup_relaunches[i]
        ta = datetime.strptime(a["ts"], "%Y-%m-%dT%H:%M:%SZ")
        tb = datetime.strptime(b["ts"], "%Y-%m-%dT%H:%M:%SZ")
        wall = (tb - ta).total_seconds()
        # resumed at (a.from_step - 500), died at b.from_step
        ds = b["from_step"] - (a["from_step"] - 500)
        if wall <= 0 or ds <= 0:
            continue
        probe_b.append({"from": a["ts"], "to": b["ts"], "wall_s": wall,
                        "steps": ds, "s_per_step_incl_startup": wall / ds,
                        "resume_step": a["from_step"] - 500,
                        "death_step": b["from_step"]})
    return {"probe_a_in_process_elapsed_s": probe_a,
            "probe_b_supervisor_wallclock": probe_b,
            "note": ("Two DIFFERENT clocks, not the same probe twice. "
                     "probe B includes relaunch startup, so it reads slightly "
                     "slower than probe A by construction.")}


def project_end(tagged, target_steps, launch_start_utc=None):
    """ESTIMATED epoch end from the CURRENT launch's own measured rate."""
    per = {}
    for lch, r in tagged:
        if kind(r) == "train":
            per.setdefault(lch, []).append(r)
    last_l = max(per)
    rs = per[last_l]
    if len(rs) < 3:
        return {"status": "UNAVAILABLE", "reason": "current launch too short"}
    de = rs[-1]["elapsed_s"] - rs[1]["elapsed_s"]
    ds = rs[-1]["step"] - rs[1]["step"]
    rate = de / ds
    remaining = target_steps - rs[-1]["step"]
    out = {"launch": last_l, "last_step": rs[-1]["step"],
           "last_elapsed_s": rs[-1]["elapsed_s"],
           "s_per_step": rate, "remaining_steps": remaining,
           "remaining_h": remaining * rate / 3600.0,
           "evidence": "ESTIMATED (assumes no further death and a stable rate)"}
    if launch_start_utc:
        t0 = datetime.strptime(launch_start_utc, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc)
        last_ts = t0 + timedelta(seconds=rs[-1]["elapsed_s"])
        end_ts = last_ts + timedelta(seconds=remaining * rate)
        out["last_row_utc"] = last_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        out["projected_end_utc"] = end_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    return out


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def build(metrics_path, supervisor_path=None, config_path=None,
          era_bounds=(16500, 17500), eval_every=500, log_every=50,
          target_steps=40284, launch_start_utc=None):
    rows = load_rows(metrics_path)
    tagged = split_launches(rows)
    ev, tr = canonical(tagged)
    n_launch = max(l for l, _ in tagged) + 1
    sup = parse_supervisor(supervisor_path)
    nav_s, leak_s = era_bounds
    eras = [("A_pre_nav_leaked", 0, nav_s),
            ("B_navswitched_still_leaked", nav_s + 1, leak_s),
            ("C_clean", leak_s + 1, 10 ** 9)]

    per_launch = {}
    for lch, r in tagged:
        if kind(r) == "train":
            per_launch.setdefault(lch, []).append(r["step"])
    launches = [{"launch": l, "first_step": min(v), "last_step": max(v),
                 "n_train_rows": len(v)} for l, v in sorted(per_launch.items())]
    rep = replay_accounting(tagged, log_every)
    for L in launches:
        L.update(rep.get(L["launch"], {"replayed_rows": 0, "replayed_steps": 0}))

    out = {
        "tool": "taniteval/tools/refcv3_metrics_read.py",
        "tier": "T0",
        "tier_note": ("in-training loss monitor over 160 FIXED held-out windows; "
                      "a WM/readout diagnostic. NEVER driving performance, and "
                      "NOT the four binding metric families."),
        "source": {"metrics": str(metrics_path),
                   "supervisor": str(supervisor_path) if supervisor_path else None,
                   "config": str(config_path) if config_path else None,
                   "rows_total": len(rows),
                   "rows_train": sum(1 for r in rows if kind(r) == "train"),
                   "rows_eval": sum(1 for r in rows if kind(r) == "eval"),
                   "rows_eval_error": sum(1 for r in rows
                                          if kind(r) == "eval_error")},
        "estimator_refusal": {
            "ci_available": False,
            "reason": ("each eval_* value is ALREADY the pooled mean over the "
                       "160 windows; metrics.jsonl carries no per-window values "
                       "and no episode index, and "
                       "taniteval.ci.episode_cluster_bootstrap requires both. No "
                       "estimator can produce an interval from this file."),
            "consequence": ("every era-level statement derived here is a "
                            "DIRECTION, not a verdict."),
        },
        "reconstruction": {
            "n_launches": n_launch,
            "launches": launches,
            "supervisor_relaunches": sup,
            "canonical_rule": "last-writer-wins per (step, kind)",
            "unique_eval_steps": len(ev),
            "unique_train_steps": len(tr),
            "replayed_train_rows": sum(v["replayed_rows"] for v in rep.values()),
            "replayed_steps_total": sum(v["replayed_steps"] for v in rep.values()),
        },
        "fixed_window_set_invariants": invariants(ev),
        "chance_lines": {"lat_tac_lon_tac_ln8": LN8,
                         "note": "8-class heads (tac_vocab_version v7.0)"},
        "eras": {}, "half_split": {},
        "train_eval_gap": {}, "post_resume_spikes": spikes(tagged, per_launch,
                                                           log_every),
        "pace": pace(tagged, sup, log_every),
        "projection": project_end(tagged, target_steps, launch_start_utc),
    }
    for name, lo, hi in eras:
        steps = sorted(s for s in ev if lo <= s <= hi)
        out["eras"][name] = era_block(ev, steps, EVAL_METRIC_COLS) if steps else {
            "n": 0}
        out["half_split"][name] = half_split(ev, steps, EVAL_METRIC_COLS)

    gap_rows = gap_block(ev, tr, tagged, eval_every, drop_after_resume=0)
    gap_rows_rob = gap_block(ev, tr, tagged, eval_every, drop_after_resume=2)
    out["train_eval_gap"] = {
        "per_step": gap_rows,
        "summary": gap_summary(gap_rows, eras),
        "summary_robustness_drop2_after_resume": gap_summary(gap_rows_rob, eras),
        "caveat": ("train rows are 20-window samples of the CURRENT batches; the "
                   "eval is a fixed 160-window set. The two are DIFFERENT "
                   "windows, so only the TREND of the gap is interpretable, "
                   "never its absolute level."),
    }
    out["_series"] = {"eval": {s: ev[s][1] for s in sorted(ev)},
                      "eval_launch": {s: ev[s][0] for s in sorted(ev)},
                      "train": {s: tr[s][1] for s in sorted(tr)},
                      "train_launch": {s: tr[s][0] for s in sorted(tr)}}
    return out


def write_csv(path, series, launches, cols):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "launch"] + list(cols))
        for s in sorted(series, key=int):
            r = series[s]
            w.writerow([s, launches[s]] + [r.get(c, "") for c in cols])


def _p(*a):
    print(*a, flush=True)


def report(d):
    _p(f"[refcv3-metrics-read] TIER {d['tier']} -- {d['tier_note']}")
    _p(f"  rows: {d['source']['rows_total']} "
       f"(train {d['source']['rows_train']}, eval {d['source']['rows_eval']}, "
       f"eval_error {d['source']['rows_eval_error']})")
    r = d["reconstruction"]
    _p(f"  launches {r['n_launches']}; unique eval steps {r['unique_eval_steps']}; "
       f"replayed {r['replayed_train_rows']} rows = {r['replayed_steps_total']} "
       f"steps recomputed")
    inv = d["fixed_window_set_invariants"]
    _p(f"  FIXED-WINDOW-SET check: ALL_CONSTANT={inv['ALL_CONSTANT']}")
    for k in FIXED_SET_INVARIANTS:
        _p(f"    {k:24s} distinct={inv[k]['distinct']} {inv[k]['values']}")
    if not inv["ALL_CONSTANT"]:
        _p("  !! WITHDRAWN: the window set moved; no cross-step comparison holds.")
    _p(f"  ESTIMATOR: ci_available={d['estimator_refusal']['ci_available']} "
       f"-- {d['estimator_refusal']['reason']}")
    for name, blk in d["eras"].items():
        if not blk.get("n"):
            continue
        _p(f"\n  == era {name}: n={blk['n']} window={blk['fit_window']}")
        _p(f"    {'col':24s} {'first':>9s} {'last':>9s} {'mean':>9s} "
           f"{'sd_step':>8s} {'slope/1k':>10s} {'R2':>6s} {'quotable':>9s}")
        for c, row in blk["cols"].items():
            sl = row["slope_per_1k_steps"]
            _p(f"    {c:24s} {row['first']:9.4f} {row['last']:9.4f} "
               f"{row['mean']:9.4f} "
               f"{(row['sd_step'] if row['sd_step'] else float('nan')):8.4f} "
               f"{(sl if sl is not None else float('nan')):10.5f} "
               f"{(row['r2'] if row['r2'] is not None else float('nan')):6.3f} "
               f"{str(row['quotable_rate']):>9s}")
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--supervisor", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--nav-switch-step", type=int, default=16500,
                    help="last step of the pre-nav-switch era (D-REFCV3-NAV-SWITCHED)")
    ap.add_argument("--leak-gate-step", type=int, default=17500,
                    help="last CONTAMINATED eval step (C-REFCV3-EVAL-PRIOR-LEAK)")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--target-steps", type=int, default=40284)
    ap.add_argument("--launch-start-utc", default=None,
                    help="UTC start of the CURRENT launch, for the end projection")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--eval-csv", default=None)
    ap.add_argument("--train-csv", default=None)
    a = ap.parse_args(argv)

    d = build(a.metrics, a.supervisor, a.config,
              era_bounds=(a.nav_switch_step, a.leak_gate_step),
              eval_every=a.eval_every, log_every=a.log_every,
              target_steps=a.target_steps, launch_start_utc=a.launch_start_utc)
    report(d)
    series = d.pop("_series")
    if a.eval_csv:
        cols = ["eval_windows", "eval_batches"] + list(EVAL_METRIC_COLS) + [
            "eval_slot_valid_frac", "eval_tac_label_v7", "eval_tac_label_rows",
            "eval_nav_injected"]
        write_csv(a.eval_csv, series["eval"], series["eval_launch"], cols)
        _p(f"  wrote {a.eval_csv}")
    if a.train_csv:
        cols = ["loss", "traj", "cls", "law", "route", "lat", "lon", "lat_tac",
                "lon_tac", "anchor_acc", "slot_valid_frac", "goal_tac",
                "goal2s_err_m", "sel_v3", "goal_gate", "goal_score_absmean",
                "goal_gate_grad", "nav_injected", "elapsed_s", "lr"]
        write_csv(a.train_csv, series["train"], series["train_launch"], cols)
        _p(f"  wrote {a.train_csv}")
    if a.out_json:
        Path(a.out_json).write_text(json.dumps(d, indent=1), encoding="utf-8")
        _p(f"  wrote {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
