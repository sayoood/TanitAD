"""SLOWWM read-out (PREREG_V8F_SLOWWM, frozen sha256 7c2775e3...94ca): PLANNABLE per arm and seed, the margin m, the floor and the committed outcome. Writes RESULT_SLOWWM.json / .md. Tier T0 on every row.

Rows. For each READ (PLANTEST = all windows of the form, prior-seeded, lambda 0; PLANTEST2 = SCORE half at the lambda TUNE chose, the ladder's own rule) and each CONTEXT (c1, c3), for each arm and seed, per family:
  LONGITUDINAL, LATERAL (n = windows and episodes in the row), estimator EC-BOOT = paired episode-cluster bootstrap over the episodes of the read set, B 2,000, seed 0, ratio of per-episode sums, 95 % percentile
  (plantest_lib.boot_*; it answers "another draw of episodes", not "another training run" -- that is the floor); label = plantest_lib.family_label (the harness's own rule), PLANNABLE = G-TRUE separated-better than
  HOLD and than G-RAND. TACTICAL: n = 0, no decision output on this rig. STRATEGIC: PAUSED (PI item 25, PI_DECISION_QUEUE.md), reported, never a pass.
Space. The deciding metrics are in TRAJECTORY space (the unicycle integration of the plan against that of the logged actions, plantest_lib.window_metrics): LONGITUDINAL = speed_err (m/s), LATERAL = heading_err (rad).
  The harness's action-space MAEs (acc_mae, yaw_mae) are read in parallel under ``space = action`` (`--space action` makes them the deciding rows). INTERPRETATION of "the deciding read is in trajectory space":
  the card does not name the two trajectory metrics; this choice is recorded in the result and is the one place a reader could differ.
Margin. m = mean(HOLD) - mean(G-TRUE) of the family metric (positive = G-TRUE better than HOLD), with the bootstrap interval of the paired difference. floor[fam, read, context] = |m(SLOW_rep) - m(SLOW_s0)|.
Outcomes (the card's four, evaluated per read x context cell; ``decide`` is a pure function, tested on planted tables in controls_read_slowwm.py). Cells that disagree are REPORTED as disagreeing; how the
  four cells combine is fixed by AMEND_SLOWWM_1.md (frozen 2026-11-11 14:20:08, before any basis or run, sha256 5d2270d4...1549): ``committed`` (A3 one cell, A4 across cells).
Usage: python report_slowwm.py [--runs DIR] [--ladder-runs DIR] [--out PREFIX] [--space traj|action] [--controls CONTROLS_READ_SLOWWM.json] [--ia0 IA0_SLOWWM.json]"""
import argparse
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

RUNS = os.environ.get("SLOWWM_PLAN_RUNS", "F:/TanitAD-scratch/slowwm/plan_runs")
LADDER_RUNS = "F:/TanitAD-scratch/ladder_plantest/runs"
FAMS = ("LONGITUDINAL", "LATERAL")
SPACES = {"traj": {"LONGITUDINAL": "speed_err", "LATERAL": "heading_err"}, "action": {"LONGITUDINAL": "acc_mae", "LATERAL": "yaw_mae"}}
UNIT = {"acc_mae": "m/s^2", "yaw_mae": "rad/s", "speed_err": "m/s", "heading_err": "rad"}
CH = {"LONGITUDINAL": 0, "LATERAL": 1}
READS, CTXS = ("PLANTEST", "PLANTEST2"), ("c1", "c3")
GROUPS = {"R35H": ("R35H_s0", "R35H_s1"), "SLOW": ("SLOW_s0", "SLOW_s1"), "PCA-B": ("PCA-B_s0", "PCA-B_s1")}
REP = "SLOW_rep_s2"
OUTCOMES = ("SLOW-PLANS", "BASIS-ONLY", "NO-EFFECT", "SLOW-COSTS")
ESTIMATOR = "EC-BOOT: paired episode-cluster bootstrap, B 2000, seed 0, ratio of per-episode sums, 95 % percentile (plantest_lib.boot_*); the training-run question is answered by the floor"
FAMILY_STATUS = {"TACTICAL": {"n": 0, "status": "NOT COMPUTABLE", "reason": "the planner harness has no decision output (it returns action plans); n = 0"},
                 "STRATEGIC": {"n": 0, "status": "PAUSED (PI item 25)", "reason": "PI_DECISION_QUEUE.md ITEM 25, 2026-09-19: the strategic layer is switched off temporarily; reported as paused, never as a pass"}}


def lam_s(l):
    return f"{l:g}"


# ------------------------------------------------------------------------------------------------------ the committed outcome table (pure)
def decide(m, plannable, floor):
    """m[group][seed][fam]: margin (HOLD minus G-TRUE, positive = better) for the groups R35H, SLOW, PCA-B at seeds 0 / 1; plannable[group][seed][fam]: bool; floor[fam]: |m(SLOW_rep) - m(SLOW_s0)|.
    Returns dict(status, per_family, outcomes). status INCOMPLETE (no outcome) when any needed margin or floor is None."""
    need = [m[g][s][f] for g in GROUPS for s in (0, 1) for f in FAMS] + [floor[f] for f in FAMS]
    if any(v is None for v in need):
        return dict(status="INCOMPLETE", per_family={}, outcomes=[])
    pf = {}
    for f in FAMS:
        fl = floor[f]
        both = lambda fn: all(fn(s) for s in (0, 1))
        pf[f] = dict(floor=fl,
                     slow_plannable_both=both(lambda s: bool(plannable["SLOW"][s][f])),
                     slow_beats_r35h=both(lambda s: m["SLOW"][s][f] - m["R35H"][s][f] > fl),
                     slow_beats_pcab=both(lambda s: m["SLOW"][s][f] - m["PCA-B"][s][f] > fl),
                     pcab_beats_r35h=both(lambda s: m["PCA-B"][s][f] - m["R35H"][s][f] > fl),
                     slow_pcab_within_floor=both(lambda s: abs(m["SLOW"][s][f] - m["PCA-B"][s][f]) <= fl),
                     slow_below_r35h=both(lambda s: m["R35H"][s][f] - m["SLOW"][s][f] > fl))
    out = []
    if any(p["slow_plannable_both"] and p["slow_beats_r35h"] and p["slow_beats_pcab"] for p in pf.values()):
        out.append("SLOW-PLANS")
    if any(p["slow_beats_r35h"] and p["pcab_beats_r35h"] and p["slow_pcab_within_floor"] for p in pf.values()):
        out.append("BASIS-ONLY")
    if not any(p["slow_beats_r35h"] or p["pcab_beats_r35h"] for p in pf.values()):
        out.append("NO-EFFECT")
    if any(p["slow_below_r35h"] for p in pf.values()):
        out.append("SLOW-COSTS")
    return dict(status="OK", per_family=pf, outcomes=out)


def combine_cells(per_cell):
    """per_cell {cell: decide(...)}. Agreement is reported, not forced: ``agree`` = the outcome set every cell shares (all cells must be OK)."""
    if not per_cell or any(d["status"] != "OK" for d in per_cell.values()):
        return dict(status="INCOMPLETE", agree=None, cells={k: v["outcomes"] for k, v in per_cell.items()})
    sets = [tuple(d["outcomes"]) for d in per_cell.values()]
    agree = len(set(sets)) == 1
    return dict(status="CELLS-AGREE" if agree else "CELLS-DISAGREE", agree=list(sets[0]) if agree else None, cells={k: v["outcomes"] for k, v in per_cell.items()})


# ------------------------------------------------------------------------------------------------------ rows
def ld(runs, kind, *parts):
    p = os.path.join(runs, "_".join((kind,) + tuple(str(x) for x in parts)) + ".npz")
    return dict(np.load(p)) if os.path.exists(p) else None


def load_meta(runs, name):
    p = os.path.join(runs, f"meta_{name}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


def sel_of(meta, form):
    ok = np.ones(len(meta["eid"]), bool) if form == "c1" else meta["hist_ok"].astype(bool)
    return dict(all=np.nonzero(ok)[0], tune=np.array([i for i in meta["tune_idx"] if ok[i]]), score=np.array([i for i in meta["score_idx"] if ok[i]]))


def arm_cell(P, Q, runs, name, meta, read, form, space, boot):
    """One arm, one cell: {fam: row} or {fam: None} when units are missing."""
    sel = sel_of(meta, form)
    A, v0 = meta["A"], meta["v0"]
    idx = sel["all"] if read == "PLANTEST" else sel["score"]
    uniq, Wt, pos = boot(read, form, meta["eid"][idx])
    hold_plan = P.hold_plan(A[idx])
    mh = P.window_metrics(hold_plan, A[idx], v0[idx])
    out = {}
    for fam in FAMS:
        if read == "PLANTEST":
            rt, rr, lam = ld(runs, "p1", name, form, "flat1", "true"), ld(runs, "p1", name, form, "flat1", "rand"), None
        else:
            chp = os.path.join(runs, f"choice_{name}_{form}_flat1.json")
            if not os.path.exists(chp):
                out[fam] = None
                continue
            lam = json.load(open(chp))["choice"][fam]
            rt, rr = ld(runs, "score", name, form, "flat1", "true", f"L{lam_s(lam)}"), ld(runs, "score", name, form, "flat1", "rand", f"L{lam_s(lam)}")
        if rt is None or rr is None:
            out[fam] = None
            continue
        assert np.array_equal(rt["idx"], idx) and np.array_equal(rr["idx"], idx), f"{name} {read} {form}: unit windows differ from the read set"
        mt, mr = P.window_metrics(rt["plan_raw"].astype(np.float64), A[idx], v0[idx]), P.window_metrics(rr["plan_raw"].astype(np.float64), A[idx], v0[idx])
        rows = {}
        for sp, qs in SPACES.items():
            q = qs[fam]
            d_hold, d_rand = P.boot_mean_diff(mt[q], mh[q], pos, Wt), P.boot_mean_diff(mt[q], mr[q], pos, Wt)
            sh, sr = P.sep_label(*d_hold), P.sep_label(*d_rand)
            rows[sp] = dict(metric=q, unit=UNIT[q], hold=P.boot_mean(mh[q], pos, Wt), true=P.boot_mean(mt[q], pos, Wt), rand=P.boot_mean(mr[q], pos, Wt), true_minus_hold=d_hold, true_minus_rand=d_rand,
                            sep_hold=sh, sep_rand=sr, label=P.family_label(sh, sr), m=-d_hold[0], m_ci=[-d_hold[2], -d_hold[1]])
        row = dict(lam=lam, n_windows=int(len(idx)), n_episodes=int(len(uniq)), tier="T0", estimator=ESTIMATOR, rows=rows)
        if read == "PLANTEST2":
            dt = float(Q.hold_distance(rt["plan_raw"], hold_plan, meta["std_raw"])[CH[fam]])
            row["d_true"], row["collapse"] = dt, bool(dt < Q.COLLAPSE_FRAC)
        out[fam] = row
    return out


def build(runs, ladder_runs, space):
    import lp_slowwm  # noqa: F401  (puts the gates on sys.path)
    import plantest_lib as P
    import plantest2_lib as Q
    cache = {}

    def boot(read, form, eids):
        k = (read, form)
        if k not in cache:
            cache[k] = P.boot_setup(eids)
        return cache[k]
    names = [n for g in GROUPS.values() for n in g] + [REP]
    src = {n: (ladder_runs if n.startswith("R35H") else runs) for n in names}
    metas = {n: load_meta(runs, n) for n in names}
    ref = next((m for m in metas.values() if m is not None), None)
    for n, m in metas.items():
        if m is not None and ref is not None:
            assert np.array_equal(m["eid"], ref["eid"]) and np.array_equal(m["tune_idx"], ref["tune_idx"]) and np.array_equal(m["A"], ref["A"]), f"{n}: window set differs from the other arms'"
    cells = {}
    for read in READS:
        for form in CTXS:
            cell = {}
            for n in names:
                if metas[n] is None:
                    cell[n] = {f: None for f in FAMS}
                    continue
                cell[n] = arm_cell(P, Q, src[n], n, metas[n], read, form, space, boot)
            cells[f"{read}|{form}"] = cell
    decided, tables = {}, {}
    for key, cell in cells.items():
        g = lambda n, f, fn: (None if cell[n][f] is None else fn(cell[n][f]["rows"][space]))
        m = {grp: {s: {f: g(n, f, lambda r: r["m"]) for f in FAMS} for s, n in enumerate(ns)} for grp, ns in GROUPS.items()}
        pl = {grp: {s: {f: g(n, f, lambda r: r["label"] == "PLANNABLE") for f in FAMS} for s, n in enumerate(ns)} for grp, ns in GROUPS.items()}
        ma, mb = {f: g(REP, f, lambda r: r["m"]) for f in FAMS}, {f: g("SLOW_s0", f, lambda r: r["m"]) for f in FAMS}
        floor = {f: (None if ma[f] is None or mb[f] is None else abs(ma[f] - mb[f])) for f in FAMS}
        tables[key] = dict(m=m, plannable=pl, floor=floor)
        decided[key] = decide(m, pl, floor)
    return dict(cells=cells, tables=tables, decided=decided, combined=combine_cells(decided), n_arms_with_meta=sum(v is not None for v in metas.values()))


ORDER = ("SLOW-PLANS", "BASIS-ONLY", "NO-EFFECT", "SLOW-COSTS")   # AMEND_SLOWWM_1 A4: most to least favourable


def cell_label(outcomes):
    """AMEND_SLOWWM_1 A3: exactly one outcome -> it; more than one -> MIXED; none -> NONE-OF-FOUR."""
    if len(outcomes) == 1:
        return outcomes[0]
    return "MIXED" if outcomes else "NONE-OF-FOUR"


def committed(combined):
    """AMEND_SLOWWM_1 A4 (frozen sha256 5d2270d4...1549). All four cells the same single outcome -> that outcome; otherwise CELLS-DISAGREE, and the
    committed next step is that of the least favourable outcome in any cell (NONE-OF-FOUR counts as NO-EFFECT, a MIXED cell as its least favourable member).
    Returns dict(outcome, next_step_of, cell_labels) or None when the cells are not all OK (INCOMPLETE)."""
    if combined["status"] == "INCOMPLETE":
        return None
    cells = combined["cells"]
    labels = {k: cell_label(v) for k, v in cells.items()}
    rank = lambda outs: max(ORDER.index(o) for o in outs) if outs else ORDER.index("NO-EFFECT")
    worst = ORDER[max(rank(v) for v in cells.values())]
    single = {tuple(v) for v in cells.values()}
    if len(cells) == len(READS) * len(CTXS) and len(single) == 1 and len(next(iter(single))) == 1:
        o = next(iter(single))[0]
        return dict(outcome=o, next_step_of=o, cell_labels=labels)
    return dict(outcome="CELLS-DISAGREE", next_step_of=worst, cell_labels=labels)


def final_outcome(controls, combined):
    """The read is unusable when a control failed or did not run; otherwise the outcome set the cells agree on, or the cells' status (CELLS-DISAGREE / INCOMPLETE)."""
    if not controls["usable"]:
        return "UNUSABLE: controls " + controls["status"]
    return combined["agree"] if combined["status"] == "CELLS-AGREE" else combined["status"]


def controls_status(path):
    if not path or not os.path.exists(path):
        return dict(status="NOT RUN", detail=f"{path} is absent: the outcome is not usable until controls a-d pass", usable=False)
    c = json.load(open(path, encoding="utf-8"))
    ok = bool(c.get("all_pass"))
    return dict(status="PASS" if ok else "FAIL", detail={k: v.get("pass") for k, v in c.items() if isinstance(v, dict) and "pass" in v}, usable=ok)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--ladder-runs", default=LADDER_RUNS)
    ap.add_argument("--out", default=os.path.join(HERE, "RESULT_SLOWWM"))
    ap.add_argument("--space", default="traj", choices=("traj", "action"))
    ap.add_argument("--controls", default=os.path.join(HERE, "CONTROLS_READ_SLOWWM.json"))
    ap.add_argument("--ia0", default=os.path.join(HERE, "IA0_SLOWWM.json"))
    a = ap.parse_args(argv)
    res = build(a.runs, a.ladder_runs, a.space)
    res.update(card="PREREG_V8F_SLOWWM.md sha256 7c2775e39f9b1271342a0c588780d3da8b4d63dfb56d06de9fef4c50bc8594ca", tier="T0", deciding_space=a.space, deciding_metrics=SPACES[a.space], families_not_read=FAMILY_STATUS,
               margin_definition="m = mean(HOLD) - mean(G-TRUE); positive = G-TRUE better; floor = |m(SLOW_rep) - m(SLOW_s0)| per read, context, family", controls=controls_status(a.controls))
    c = res["controls"]
    res["outcome"] = final_outcome(c, res["combined"])
    res["amendment"] = "AMEND_SLOWWM_1.md sha256 5d2270d40e5ae86f0b2b552bf5e9f1cc2b244e97e16f4ccbe400e77920df1549"
    res["committed"] = committed(res["combined"]) if c["usable"] else None
    if os.path.exists(a.ia0):
        z = json.load(open(a.ia0, encoding="utf-8"))
        res["ia0_beside"] = {k: z[k] for k in ("band_end_summary",) if k in z} or {"file": a.ia0}
    json.dump(res, open(a.out + ".json", "w", encoding="utf-8"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    open(a.out + ".md", "w", encoding="utf-8").write(markdown(res))
    print("outcome:", res["outcome"], "| controls:", c["status"], "| cells:", res["combined"]["cells"])
    return 0


def fmt(t):
    return "-" if t is None else f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"


def markdown(res):
    sp = res["deciding_space"]
    L = ["# RESULT_SLOWWM (PREREG_V8F_SLOWWM, T0)", "", f"Card: {res['card']}. Deciding space: **{sp}** ({res['deciding_metrics']}). {res['margin_definition']}.", f"Estimator on every row: {ESTIMATOR}.", "",
         f"**Controls: {res['controls']['status']}.** **Outcome: {res['outcome']}.** Cells: {res['combined']['cells']}", "",
         f"**Committed by {res.get('amendment', 'AMEND_SLOWWM_1.md')}: {res.get('committed')}**", "", "## Families not read", ""]
    L += [f"* {k}: n = {v['n']}, {v['status']}. {v['reason']}" for k, v in res["families_not_read"].items()] + [""]
    for key, cell in res["cells"].items():
        read, form = key.split("|")
        t, d = res["tables"][key], res["decided"][key]
        L += [f"## {read}, context {form[1]}", "", f"Floor: {t['floor']}. Decision status {d['status']}: {d['outcomes']}", "",
              "| arm | family | n windows / episodes | lambda | label | m (HOLD - G-TRUE) [CI] | G-TRUE - HOLD [CI] | G-TRUE - G-RAND [CI] | tier |", "|---|---|---|---|---|---|---|---|---|"]
        for n, fams in cell.items():
            for f in FAMS:
                r = fams[f]
                if r is None:
                    L.append(f"| {n} | {f} | - | - | INCOMPLETE | - | - | - | T0 |")
                    continue
                x = r["rows"][sp]
                L.append(f"| {n} | {f} | {r['n_windows']} / {r['n_episodes']} | {r['lam'] if r['lam'] is not None else '-'} | {x['label']}{' +HOLD-COLLAPSE' if r.get('collapse') else ''} | {x['m']:+.4f} [{x['m_ci'][0]:+.4f}, {x['m_ci'][1]:+.4f}] | {fmt(x['true_minus_hold'])} | {fmt(x['true_minus_rand'])} | T0 |")
        L.append("")
    if "ia0_beside" in res:
        L += ["## IA-0 skill per level (beside, no bar)", "", "```", json.dumps(res["ia0_beside"], indent=1, default=str)[:4000], "```", ""]
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
