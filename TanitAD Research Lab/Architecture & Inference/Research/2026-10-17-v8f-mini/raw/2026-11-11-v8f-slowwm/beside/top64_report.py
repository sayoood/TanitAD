"""B3 report (BESIDE_RULES_SLOWWM.md): for SLOW s0, s1 and rep, the margin m of the planner with the level-1 goal cost restricted to S (the top-64 mu directions) beside the SAME arm's full-width m (the deciding read's units),
per read (PLANTEST, PLANTEST2) x context (c1, c3) x family (LONGITUDINAL = speed_err, LATERAL = heading_err; the action-space MAEs are in the json), plus the beside numbers of each arm: rank(S), the share of eval z1 variance in S, and
the principal angles between S and the top-64 PCA directions of eval z1. NO bar, NO outcome label, no PLANNABLE label: numbers only. Tier T1-kin planner form; estimator EC-BOOT (paired episode-cluster bootstrap, B 2,000, seed 0, ratio of
per-episode sums, 95 % percentile: it answers "another draw of episodes", not "another training run"). m = mean(HOLD) - mean(G-TRUE), positive = G-TRUE better. Missing units print INCOMPLETE.
Usage: python top64_report.py [--b3-runs DIR] [--full-runs DIR] [--proj-dir DIR] [--out PREFIX]   -> REPORT_B3.json / .md"""
import argparse
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import top64_lib as T  # noqa: E402
import report_slowwm as R  # noqa: E402

B3_RUNS = os.environ.get("SLOWWM_B3_RUNS", "F:/TanitAD-scratch/slowwm/beside_top64_runs")
FULL_RUNS = "F:/TanitAD-scratch/slowwm/plan_runs"
ARMS = ("SLOW_s0", "SLOW_s1", "SLOW_rep_s2")
FAMS, SPACES = R.FAMS, R.SPACES


def _m(cell, fam, space):
    r = cell.get(fam)
    return None if r is None else r["rows"][space]


def build(b3_runs, full_runs, proj_dir, arms=ARMS):
    import plantest_lib as P
    import plantest2_lib as Q
    cache = {}

    def boot(read, form, eids):
        if (read, form) not in cache:
            cache[(read, form)] = P.boot_setup(eids)
        return cache[(read, form)]
    metas = {n: (R.load_meta(b3_runs, n), R.load_meta(full_runs, n)) for n in arms}
    for n, (mb, mf) in metas.items():
        if mb is not None and mf is not None:
            for k in ("eid", "tune_idx", "score_idx", "A", "hist_ok"):
                assert np.array_equal(mb[k], mf[k]), f"{n}: window set ({k}) of the B3 runs differs from the deciding read's"
    cells = {}
    for read in R.READS:
        for form in R.CTXS:
            cell = {}
            for n in arms:
                mb, mf = metas[n]
                cb = {f: None for f in FAMS} if mb is None else R.arm_cell(P, Q, b3_runs, n, mb, read, form, "traj", boot)
                cf = {f: None for f in FAMS} if mf is None else R.arm_cell(P, Q, full_runs, n, mf, read, form, "traj", boot)
                cell[n] = dict(top64=cb, full=cf)
            cells[f"{read}|{form}"] = cell
    table = {}
    for key, cell in cells.items():
        t = {}
        for n in arms:
            for f in FAMS:
                tb, tf = _m(cell[n]["top64"], f, "traj"), _m(cell[n]["full"], f, "traj")
                t[f"{n}|{f}"] = dict(m_top64=None if tb is None else tb["m"], m_top64_ci=None if tb is None else tb["m_ci"], m_full=None if tf is None else tf["m"], m_full_ci=None if tf is None else tf["m_ci"],
                                     top64_minus_full=None if tb is None or tf is None else tb["m"] - tf["m"],
                                     true_minus_rand_top64=None if tb is None else tb["true_minus_rand"], true_minus_rand_full=None if tf is None else tf["true_minus_rand"])
        rep, s0 = "SLOW_rep_s2", "SLOW_s0"
        for f in FAMS:
            g = lambda a, v: t[f"{a}|{f}"][v]
            t[f"rep_minus_s0|{f}"] = dict(top64=None if None in (g(rep, "m_top64"), g(s0, "m_top64")) else g(rep, "m_top64") - g(s0, "m_top64"),
                                          full=None if None in (g(rep, "m_full"), g(s0, "m_full")) else g(rep, "m_full") - g(s0, "m_full"))
        table[key] = t
    beside = {}
    for n in arms:
        p = os.path.join(proj_dir, f"proj_{n}.npz")
        if not os.path.exists(p):
            beside[n] = None
            continue
        pj = T.load_projector(p)
        b = pj["meta"]["beside"]
        ang = np.asarray(pj["principal_angles_deg"])
        beside[n] = dict(rank_S=pj["rank"], variance_share_in_S=b["variance_share_in_S"], variance_share_top64_pca=b["variance_share_top64_pca"], principal_angles_deg_summary=dict(min=float(ang.min()), median=float(np.median(ang)),
                         max=float(ang.max()), n=int(len(ang))), principal_angles_deg=ang.tolist(), n_eval_frames=b["n_eval_frames"], angles_S_vs_newest_frame_variant_deg_max=float(np.max(pj["angles_joint_vs_current_deg"])),
                         window_K=pj["meta"]["window_K"], n_train_frames=pj["meta"]["n_frames"], inputs_sha256=pj["meta"]["inputs_sha256"], projector_sha256=T.sha256_file(p), tier=pj["meta"]["tier"])
    return dict(protocol="B3", rules="BESIDE_RULES_SLOWWM.md", rules_sha256=T.RULES_SHA, tier=T.TIER, estimator=R.ESTIMATOR, deciding_metrics=SPACES["traj"],
                margin_definition="m = mean(HOLD) - mean(G-TRUE); positive = G-TRUE better; per read x context x family; top64 = level-1 goal cost restricted to P_S, full = the deciding read's full-width units of the same arm",
                no_bar_no_outcome_label=True, b3_runs=b3_runs, full_runs=full_runs, tables=table, beside=beside)


def fmt_m(m, ci):
    return "INCOMPLETE" if m is None else f"{m:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]"


def markdown(res, arms=ARMS):
    L = ["# REPORT_B3: planner with the level-1 goal cost restricted to SLOW's top-64 mu directions (beside, no bar, no outcome label)", "",
         f"Rules: BESIDE_RULES_SLOWWM.md sha256 {res['rules_sha256']}. Tier {res['tier']}. Estimator on every row: {res['estimator']}.", "", f"{res['margin_definition']}.",
         f"Deciding metrics: {res['deciding_metrics']}.", "", "## Beside numbers per arm", "",
         "| arm | rank(S) | share of eval z1 variance in S | share in the top-64 PCA directions | principal angles to the top-64 PCA directions (deg): min / median / max (n) | train frames |", "|---|---|---|---|---|---|"]
    for n in arms:
        b = res["beside"][n]
        if b is None:
            L.append(f"| {n} | INCOMPLETE | - | - | - | - |")
            continue
        a = b["principal_angles_deg_summary"]
        L.append(f"| {n} | {b['rank_S']} | {b['variance_share_in_S']:.4f} | {b['variance_share_top64_pca']:.4f} | {a['min']:.2f} / {a['median']:.2f} / {a['max']:.2f} ({a['n']}) | {b['n_train_frames']} |")
    for key, t in res["tables"].items():
        read, form = key.split("|")
        L += ["", f"## {read}, context {form[1]}", "", "| arm | family | m, top-64 cost [CI] | m, full width [CI] | top-64 minus full |", "|---|---|---|---|---|"]
        for n in arms:
            for f in FAMS:
                r = t[f"{n}|{f}"]
                d = "INCOMPLETE" if r["top64_minus_full"] is None else f"{r['top64_minus_full']:+.4f}"
                L.append(f"| {n} | {f} | {fmt_m(r['m_top64'], r['m_top64_ci'] or [0, 0])} | {fmt_m(r['m_full'], r['m_full_ci'] or [0, 0])} | {d} |")
        for f in FAMS:
            r = t[f"rep_minus_s0|{f}"]
            L.append(f"| SLOW_rep minus SLOW_s0 | {f} | {'-' if r['top64'] is None else format(r['top64'], '+.4f')} | {'-' if r['full'] is None else format(r['full'], '+.4f')} | |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--b3-runs", default=B3_RUNS)
    ap.add_argument("--full-runs", default=FULL_RUNS)
    ap.add_argument("--proj-dir", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "REPORT_B3"))
    a = ap.parse_args(argv)
    res = build(a.b3_runs, a.full_runs, a.proj_dir or a.b3_runs)
    json.dump(res, open(a.out + ".json", "w", encoding="utf-8"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    open(a.out + ".md", "w", encoding="utf-8").write(markdown(res))
    print("report written:", a.out + ".md", "| arms with a projector:", [n for n, b in res["beside"].items() if b])
    return 0


if __name__ == "__main__":
    sys.exit(main())
