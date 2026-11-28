#!/usr/bin/env python3
"""DATA-SCALE (PREREG_DATASCALE.md 4.3, 6, 9): the `cv5m` selection rule, its per-log out-of-fold table, and CTRL-INDEX.  Pure numpy, no torch, no real data.

`cv5m` = the parent EPH rule (5-fold log CV, `--select cv5`) PLUS a noise margin and a guard, decided from a PER-LOG out-of-fold table:
  * gain(s)  = OOF pooled tr_pdms(s) - tr_pdms(0), x100.   CI = the PAIRED log-cluster bootstrap over the N navtrain logs.
  * guard(s) = for each of NC, DAC, TTC, EP, C: the OOF sub-score delta(s) vs 0 must NOT be separated worse (CI-hi >= 0).
  * step s > 0 is ELIGIBLE iff CI-lo(gain(s)) > 0 AND the guard holds.  Chosen = the eligible step with the largest gain (ties earliest); none eligible -> step 0.
  * the parent's no-margin argmax (max pooled OOF PDMS, ties earliest, step 0 included) is also reported, descriptive only.

ESTIMATOR: `eval/p5c/p5c_read.py` `Boot` (10,000 draws, seed 20260927, 95 % percentile), UNCHANGED, never re-implemented.  `Boot` takes per-TOKEN values and a per-token log
array; the table holds per-LOG pooled means and token counts, so the wrapper expands each log's mean to its n tokens (np.repeat).  Boot's own `draws` then forms
sum_l n_l * mean_l / sum_l n_l over the drawn logs, which is the token-weighted mean it would form on the raw tokens (equal to floating-point rounding).  The decision is ALWAYS
computed from the table (`decide`), in the trainer and in any re-check, so a re-computation from `cv_select.json` reproduces the choice bit for bit (CTRL-CV(e)).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_P5C = HERE.parent / "eval" / "p5c"
if str(_P5C) not in sys.path:
    sys.path.insert(0, str(_P5C))
import p5c_read as P5C  # noqa: E402  (Boot, SEED, NBOOT: the estimator the card names)

METRICS = ("pdms", "nc", "dac", "ttc", "ep", "c")
SUBSCORES = ("nc", "dac", "ttc", "ep", "c")
GUARD = SUBSCORES
SEED, NBOOT = P5C.SEED, P5C.NBOOT
assert (SEED, NBOOT) == (20260927, 10000), "the card's bootstrap constants moved in p5c_read.py"

# the REGISTERED rule: G-DVB compares the trainer's built rule to this, so a mutation that drops a clause is visible to the gate
REGISTERED_RULE = {"margin": "ci_lo_gain_gt_0", "guard": list(GUARD), "guard_rule": "ci_hi_delta_ge_0", "n_boot": NBOOT, "seed": SEED, "ci": "95_percentile",
                   "estimator": "p5c_read.Boot paired log-cluster"}


def rule_for_arm(arm) -> dict:
    """The rule the trainer BUILDS.  `arm` is the launch gate's deliberate-regression arm (None in a launch):
    ds_no_margin drops the CI condition (a step is eligible on a positive POINT gain); ds_guard_no_ep drops EP from the guard."""
    r = {**REGISTERED_RULE, "guard": list(GUARD)}
    if arm == "ds_no_margin":
        r["margin"] = "point_gain_gt_0"
    if arm == "ds_guard_no_ep":
        r["guard"] = [m for m in GUARD if m != "ep"]
    return r


def build_table(logs, picks: dict, vals: dict) -> dict:
    """Per-LOG out-of-fold table.  logs [n] names; picks {step: [n] proposal index}; vals {metric: [n, 64]} (rates in 0..1).
    -> {log: {"n": tokens, "per_step": {str(step): {metric: pooled mean x100}}}} (float64 means over the log's tokens)."""
    logs = np.asarray(logs)
    ar = np.arange(len(logs))
    tab = {}
    for lg in sorted(set(logs.tolist())):
        idx = np.where(logs == lg)[0]
        tab[lg] = {"n": int(len(idx)), "per_step": {}}
        for s, k in sorted(picks.items()):
            tab[lg]["per_step"][str(int(s))] = {m: float(np.asarray(vals[m], np.float64)[ar, k][idx].mean() * 100.0) for m in METRICS}
    return tab


def _expand(tab: dict, ulogs: list, step: int, m: str):
    return np.concatenate([np.full(tab[lg]["n"], tab[lg]["per_step"][str(step)][m], np.float64) for lg in ulogs])


def make_boot(tab: dict, n_boot: int = NBOOT, seed: int = SEED):
    ulogs = sorted(tab)
    logs = np.concatenate([np.full(tab[lg]["n"], lg) for lg in ulogs])
    return P5C.Boot(logs, n_boot=n_boot, seed=seed), ulogs


def pooled(tab: dict, step: int, m: str) -> float:
    n = np.array([tab[lg]["n"] for lg in sorted(tab)], np.float64)
    v = np.array([tab[lg]["per_step"][str(step)][m] for lg in sorted(tab)], np.float64)
    return float((n * v).sum() / n.sum())


def decide(tab: dict, steps, rule: dict | None = None, boot=None) -> dict:
    """The cv5m decision from the per-log table.  Step 0 must be in `steps`.  Returns per-step gain / delta CIs, eligibility, the chosen step and the no-margin argmax."""
    rule = rule or REGISTERED_RULE
    steps = sorted(int(s) for s in steps)
    if 0 not in steps:
        raise SystemExit("FAILED: cv5m needs step 0 (C1's own head) in the saved steps")
    B, ulogs = (boot, sorted(tab)) if boot is not None else make_boot(tab, rule["n_boot"], rule["seed"])
    base = {m: _expand(tab, ulogs, 0, m) for m in METRICS}
    per = {}
    for s in steps:
        cur = {m: _expand(tab, ulogs, s, m) for m in METRICS}
        gain = B.ci(cur["pdms"] - base["pdms"])
        delta = {m: B.ci(cur[m] - base[m]) for m in SUBSCORES}
        viol = [m for m in rule["guard"] if delta[m]["hi"] < 0]
        guard_ok = not viol
        if rule["margin"] == "ci_lo_gain_gt_0":
            margin_ok = gain["lo"] > 0
        elif rule["margin"] == "point_gain_gt_0":
            margin_ok = gain["point"] > 0
        else:
            raise SystemExit(f"FAILED: unknown margin rule {rule['margin']!r}")
        per[str(s)] = {"gain": gain, "delta": delta, "pooled_oof_pdms": pooled(tab, s, "pdms"), "margin_ok": bool(margin_ok), "guard_ok": bool(guard_ok),
                       "guard_violations": viol, "eligible": bool(s > 0 and margin_ok and guard_ok)}
    elig = [s for s in steps if per[str(s)]["eligible"]]
    chosen = 0
    if elig:
        best = max(per[str(s)]["gain"]["point"] for s in elig)
        chosen = min(s for s in elig if per[str(s)]["gain"]["point"] == best)
    top = max(per[str(s)]["pooled_oof_pdms"] for s in steps)
    return {"rule": rule, "n_logs": len(ulogs), "n_tokens": int(sum(tab[lg]["n"] for lg in ulogs)), "per_step": per, "chosen_step": int(chosen),
            "chosen_is_untrained": chosen == 0, "eligible_steps": elig,
            "no_margin_argmax_step_from_table": int(min(s for s in steps if per[str(s)]["pooled_oof_pdms"] == top))}


def recheck(sel: dict) -> dict:
    """CTRL-CV(e): recompute cv5m from a written cv_select.json's per-log table and compare it to the recorded choice (and, per step, the gain and delta CIs: bit-equal)."""
    steps = sorted(int(s) for s in sel["per_step"])
    dec = decide(sel["per_log"], steps, sel["margin_rule"])
    rec = sel["margin"]
    same_ci = all(dec["per_step"][str(s)]["gain"] == rec[str(s)]["gain"] and dec["per_step"][str(s)]["delta"] == rec[str(s)]["delta"] for s in steps)
    return {"chosen_recomputed": dec["chosen_step"], "chosen_recorded": sel["chosen_step"], "match": dec["chosen_step"] == sel["chosen_step"], "ci_bit_equal": bool(same_ci)}


# ============================================================================================ CTRL-INDEX
def ctrl_index(index_185, parent_index) -> dict:
    """PREREG_DATASCALE 6 CTRL-INDEX: `index_185` holds every row of the parent index BIT-IDENTICAL (token, log, split and the sha256 of its file), PLUS new rows, and no log
    appears twice (no token twice; no log in both the parent's rows and the new rows).  Both args are index.csv paths (the SEALED form carries a sha256 per row).
    -> {"ok": bool, "problems": [...], "n_parent": int, "n_new": int, "n_logs": int}."""
    import safehead_data as D
    new, par = D.read_index(index_185), D.read_index(parent_index)         # read_index already refuses a duplicate token inside one index
    problems = []
    for nm, rows in (("index_185", new), ("parent", par)):
        if "sha256" not in rows[0]:
            problems.append(f"{nm} has no sha256 column (CTRL-INDEX compares files by sha256; seal the index first)")
    idx = {r["token"]: r for r in new}
    n_missing = n_diff = 0
    for r in par:
        q = idx.get(r["token"])
        if q is None:
            n_missing += 1
            continue
        if any(q.get(k) != r.get(k) for k in ("log", "split", "sha256")):
            n_diff += 1
    if n_missing:
        problems.append(f"{n_missing} parent token(s) are missing from index_185")
    if n_diff:
        problems.append(f"{n_diff} parent token(s) differ in log, split or file sha256 in index_185")
    ptok = {r["token"] for r in par}
    plogs = {r["log"] for r in par}
    newrows = [r for r in new if r["token"] not in ptok]
    if not newrows:
        problems.append("index_185 adds no row to the parent's")
    nlogs = {r["log"] for r in newrows}
    both = sorted(plogs & nlogs)
    if both:
        problems.append(f"{len(both)} log(s) appear in the parent's rows AND in the new rows (first {both[:3]}): a log must appear once")
    by_log: dict = {}
    for r in new:
        by_log.setdefault(r["log"], set()).add(r["split"])
    mixed = sorted(lg for lg, sp in by_log.items() if len(sp) > 1)
    if mixed:
        problems.append(f"{len(mixed)} log(s) carry more than one split (first {mixed[:3]}): a log is one unit of the cv folds")
    return {"ok": not problems, "problems": problems, "n_parent": len(par), "n_new": len(newrows), "n_logs": len(by_log)}
