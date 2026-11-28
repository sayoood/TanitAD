#!/usr/bin/env python3
"""⛔ SCORE the pre-registered survivor statistic on the HELD-OUT split.

Clause 4 of the Master Mind ruling of 2026-09-06: *"The replacement is a NEW hypothesis
with a NEW ID and its own bar, pre-registered, scored on data that did not select it."*

⛔ THIS TOOL REFUSES TO RUN UNLESS ITS PRE-REGISTRATION FILE EXISTS AND NAMES BOTH THE
STATISTIC AND THE BAR. The refusal is the mechanism that makes "pre-registered" checkable
rather than asserted: the bar cannot be typed in at the command line.

⛔ `G-REWARD` REMAINS FAILED ON ITS COMMITTED STATISTIC (THE RATE). This is a NEW
hypothesis with a NEW ID; it may never be presented as `G-REWARD` repaired.

Reads the rows banked by ``rl_statistic_mutation_rank.py`` -- the SAME geometry, so the
score cannot differ from the ranking by anything except the split.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("TANITAD_REPO") or os.path.dirname(os.path.dirname(HERE))

RANK = importlib.util.spec_from_file_location(
    "rl_statistic_mutation_rank_for_score",
    os.path.join(REPO, "stack", "scripts", "rl_statistic_mutation_rank.py"))
_m = importlib.util.module_from_spec(RANK)
sys.modules["rl_statistic_mutation_rank_for_score"] = _m
RANK.loader.exec_module(_m)
np = _m.np
CANDIDATES = _m.CANDIDATES
LADDER = _m.LADDER
gaps = _m.gaps
boot_stat = _m.boot_stat


#: ⛔ the pre-registration must carry these two machine-readable lines, so the bar this
#: tool applies is the bar that was COMMITTED, not one supplied at run time.
RE_STAT = re.compile(r"^[ 	]*PREREG_STATISTIC:\s*(\S+)\s*$", re.M)
RE_BAR = re.compile(r"^[ 	]*PREREG_BAR:\s*(\S+)\s*(<=|>=)\s*([-+0-9.eE]+)\s*$", re.M)
RE_ID = re.compile(r"^[ 	]*PREREG_HYPOTHESIS_ID:\s*(\S+)\s*$", re.M)
RE_RUNG = re.compile(r"^[ 	]*PREREG_RUNG:\s*(\S+)\s*$", re.M)
RE_CELL = re.compile(r"^[ 	]*PREREG_CELL:\s*(\S+)\s*$", re.M)
#: the banked G-REWARD all-window RATE on the pre-repair reward. ⭐ The CHANNEL control.
#: ⚠️ The banked composed MEAN is a STALE baseline (rewards._collision changed after the
#: 2026-09-05 bank) and is deliberately NOT used as a control.
BANKED_ALL_RATE = 0.441780259484316


def read_prereg(path):
    with open(path, "r", encoding="utf-8") as fh:
        txt = fh.read()
    m_stat, m_bar, m_id, m_rung = (RE_STAT.search(txt), RE_BAR.search(txt),
                                   RE_ID.search(txt), RE_RUNG.search(txt))
    m_cell = RE_CELL.search(txt)
    missing = [n for n, m in (("PREREG_HYPOTHESIS_ID", m_id),
                              ("PREREG_STATISTIC", m_stat),
                              ("PREREG_BAR", m_bar),
                              ("PREREG_RUNG", m_rung),
                              ("PREREG_CELL", m_cell)) if m is None]
    if missing:
        raise SystemExit("[statscore] the pre-registration %s does not declare %s -- "
                         "refusing (the bar may not come from the command line)"
                         % (path, ", ".join(missing)))
    stat = m_stat.group(1)
    if stat not in CANDIDATES:
        raise SystemExit("[statscore] pre-registered statistic %r is not a candidate"
                         % stat)
    if m_bar.group(1) != stat:
        raise SystemExit("[statscore] PREREG_BAR names %r but PREREG_STATISTIC is %r"
                         % (m_bar.group(1), stat))
    return {"hypothesis_id": m_id.group(1), "statistic": stat,
            "direction": m_bar.group(2), "bar": float(m_bar.group(3)),
            "rung": m_rung.group(1), "cell": m_cell.group(1)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="rows banked by the ranking tool")
    ap.add_argument("--prereg", required=True,
                    help="the pre-registration that declares the statistic and the bar")
    ap.add_argument("--split", default="score", choices=("score", "select", "all"))
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    pre = read_prereg(a.prereg)
    with open(a.rows, "r", encoding="utf-8") as fh:
        rows_all = json.load(fh)
    rows = ([r for r in rows_all if r["split"] == a.split] if a.split != "all"
            else list(rows_all))
    if not rows:
        raise SystemExit("[statscore] split %s is empty" % a.split)
    eids = np.array([r["eid"] for r in rows])
    tg = np.array([float(r.get("gt_time_gap_min_s", float("nan"))) for r in rows])
    fn = CANDIDATES[pre["statistic"]][0]
    g = gaps(rows, pre["cell"])

    ladder = {}
    for thr in LADDER:
        m = ((np.isfinite(tg) & (tg <= thr)) if np.isfinite(thr)
             else np.ones(len(rows), bool))
        label = "all" if not np.isfinite(thr) else "<=%.1fs" % thr
        n = int(m.sum())
        if n == 0:
            continue
        v = fn(g[m])
        lo, hi, sd = boot_stat(fn, g[m], eids[m], a.n_boot, a.seed)
        ladder[label] = {"n_windows": n,
                         "n_episodes": int(len(np.unique(eids[m]))),
                         "value": v, "ci": [lo, hi], "sd_boot": sd,
                         # every other candidate too -- reported, never the verdict
                         "others": {c: CANDIDATES[c][0](g[m]) for c in CANDIDATES}}

    # ⭐ CHANNEL CONTROL -- ⛔ PINNED TO `nopert_min`, NOT to the scored cell.
    # The banked 0.441780259484316 was measured on the PRE-REPAIR reward, so anchoring
    # it to whatever cell is being scored asks the wrong question and reads FAIL on a
    # healthy panel. (MEASURED 2026-09-06: the first version did exactly that and read
    # err 1.322e-01 on cell `ship` -- a defect in the CONTROL, found by running it.)
    # The scored cell's own retired-rate value is reported beside it as a DIAGNOSTIC.
    g_pin = gaps(rows_all, "nopert_min")
    chan_rate = CANDIDATES["S1_rate"][0](g_pin)
    channel = {"pinned_cell": "nopert_min",
               "all_window_rate": chan_rate, "banked": BANKED_ALL_RATE,
               "abs_err": abs(chan_rate - BANKED_ALL_RATE),
               "pass": bool(abs(chan_rate - BANKED_ALL_RATE) <= 1e-12),
               "scored_cell": pre["cell"],
               "scored_cell_all_window_rate": CANDIDATES["S1_rate"][0](
                   gaps(rows_all, pre["cell"])),
               "n_windows_all": len(rows_all),
               "note": "the banked composed MEAN is a STALE baseline "
                       "(rewards._collision changed after the 2026-09-05 bank) and is "
                       "deliberately NOT used as a control"}
    sel_ids = {r["wi"] for r in rows_all if r["split"] == "select"}
    sco_ids = {r["wi"] for r in rows_all if r["split"] == "score"}
    split_ctrl = {"n_select": len(sel_ids), "n_score": len(sco_ids),
                  "overlap": len(sel_ids & sco_ids),
                  "union_equals_panel": bool(len(sel_ids | sco_ids) == len(rows_all)),
                  "pass": bool(len(sel_ids & sco_ids) == 0
                               and len(sel_ids | sco_ids) == len(rows_all))}
    fr = np.array([sum(w * float(r["frozen__" + pre["cell"]].get(k, 0.0))
                       for k, w in _m.WEIGHTS.items()) for r in rows])
    hu = np.array([sum(w * float(r["human__" + pre["cell"]].get(k, 0.0))
                       for k, w in _m.WEIGHTS.items()) for r in rows])
    frozen_ctrl = {"frozen_ge_human_frac": float((fr >= hu - 1e-12).mean())}

    rung = pre["rung"]
    if rung not in ladder:
        raise SystemExit("[statscore] the pre-registered rung %r is not in the ladder"
                         % rung)
    cell = ladder[rung]
    if pre["direction"] == "<=":
        passed = cell["ci"][1] <= pre["bar"]
        point = cell["value"] <= pre["bar"]
    else:
        passed = cell["ci"][0] >= pre["bar"]
        point = cell["value"] >= pre["bar"]

    out = {
        "_what": "SCORE of the pre-registered survivor statistic on the HELD-OUT split",
        "_prereg": os.path.basename(a.prereg),
        "_not_a_gate": "G-REWARD remains FAILED on its committed statistic (the RATE). "
                       "This is a NEW hypothesis with a NEW ID and may never be "
                       "presented as G-REWARD repaired.",
        "_evidence_class": "MEASURED (ours)",
        "_tier": "T0 instrument probe, NON-PARITY RL-fit windows",
        "_estimator": "episode-cluster bootstrap, n_boot %d -- answers only 'would "
                      "another draw of EPISODES say this?' (H-ESTIM-SEED-1 open: "
                      "no training and no inference variance enters, because no arm "
                      "is trained and no planner samples here)" % a.n_boot,
        "prereg": pre, "split_scored": a.split,
        "n_windows": len(rows), "n_episodes": int(len(np.unique(eids))),
        "ladder": ladder,
        "controls": {"CHANNEL": channel, "SPLIT": split_ctrl,
                     "FROZEN": frozen_ctrl},
        "VERDICT": {
            "rung": rung, "statistic": pre["statistic"],
            "value": cell["value"], "ci": cell["ci"],
            "bar": pre["bar"], "direction": pre["direction"],
            "PASS_on_whole_interval": bool(passed),
            "PASS_on_point_estimate": bool(point),
        },
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print("== CONTROLS (must read known values)")
    print("   CHANNEL  all-window RATE on PINNED cell %s: %.15f  banked %.15f  "
          "err %.3e  pass=%s"
          % (channel["pinned_cell"], channel["all_window_rate"], channel["banked"],
             channel["abs_err"], channel["pass"]))
    print("   (diagnostic) the SCORED cell %s reads retired-rate %.15f"
          % (pre["cell"], channel["scored_cell_all_window_rate"]))
    print("   SPLIT    select %d / score %d  overlap %d  union==panel %s  pass=%s"
          % (split_ctrl["n_select"], split_ctrl["n_score"], split_ctrl["overlap"],
             split_ctrl["union_equals_panel"], split_ctrl["pass"]))
    print("   FROZEN   frozen >= human on the scored split: %.6f"
          % frozen_ctrl["frozen_ge_human_frac"])
    print()
    print("== %s -- %s on the %s split"
          % (pre["hypothesis_id"], pre["statistic"], a.split.upper()))
    print("%-9s %6s %5s %11s %-26s" % ("rung", "nwin", "neps", "value", "CI"))
    for label, c in ladder.items():
        print("%-9s %6d %5d %11.6f [%9.6f, %9.6f]"
              % (label, c["n_windows"], c["n_episodes"], c["value"],
                 c["ci"][0], c["ci"][1]))
    print("\n-- VERDICT at the PRE-REGISTERED rung %s: %s %s %.4f ?"
          % (rung, pre["statistic"], pre["direction"], pre["bar"]))
    print("   value %.6f  CI [%.6f, %.6f]" % (cell["value"], cell["ci"][0], cell["ci"][1]))
    print("   WHOLE-INTERVAL: %s      point estimate: %s"
          % ("PASS" if passed else "FAIL", "PASS" if point else "FAIL"))
    print("[statscore] wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
