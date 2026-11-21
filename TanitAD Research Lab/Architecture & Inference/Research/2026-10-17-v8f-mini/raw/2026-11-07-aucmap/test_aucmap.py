#!/usr/bin/env python3
"""test_aucmap.py -- controls + unit checks + deliberate-regression mutants (PREREG_AUCMAP.md section 11).  python -I test_aucmap.py [--quick]
Each check is a function (module, rig) -> None that raises AssertionError; every mutant must make at least one check fail (RED)."""
import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
import numpy as np
import aucmap as A

OUT = {}
ALL = np.ones(2399, bool)


def rows_ref(rig, arm, s, vs):
    return A.rr.rows_of(list(arm), list(vs), rig.eid, np.zeros(2399, int), 2000)[0]


def check_q0_is_base(A, rig):                     # C1
    on = A.select_top(np.arange(2399.0), ALL, 0.0)
    assert on.sum() == 0
    for s in A.SEEDS:
        r0 = rig.row_stat("LF100", s, 18, on, "R")
        assert r0["delta"] == 0.0 and r0["lo"] == 0.0 and r0["hi"] == 0.0
        rp = rig.row_stat("LF100", s, 18, on, "P")
        ref = rows_ref(rig, rig.R[s], s, rig.P[s])["M1 crawl share"]
        assert all(abs(rp[k] - ref[k]) < 1e-12 for k in ("delta", "lo", "hi"))


def check_q1_is_lever(A, rig):                    # C2 / C3
    on = A.select_top(np.arange(2399.0), ALL, 1.0)
    assert on.sum() == 2399
    for s in A.SEEDS:
        ref = rows_ref(rig, rig.X["LF100"][s], s, rig.R[s])
        for i, lab in enumerate(A.GD.LABELS):
            mine = rig.row_stat("LF100", s, i, on, "R")
            for k in ("delta", "lo", "hi"):
                assert abs(mine[k] - ref[lab][k]) < 1e-9, (s, lab, k, mine[k], ref[lab][k])
        refp = rows_ref(rig, rig.X["LF100"][s], s, rig.P[s])["M1 crawl share"]
        mp = rig.row_stat("LF100", s, 18, on, "P")
        assert abs(mp["delta"] - refp["delta"]) < 1e-9 and abs(mp["lo"] - refp["lo"]) < 1e-9
    e = A.evaluate(rig, "LF100", on)
    OUT["LF100_q1_share_s0"] = e["share_s"][0]
    assert abs(e["share_s"][0] - 0.899) < 0.005, e["share_s"]


def check_random_gate_matches_reader(A, rig):     # C3
    on = (np.random.default_rng(5).random(2399) < 0.4).astype(float)
    for s in A.SEEDS:
        mixed = [np.where(on > 0, rig.X["LF075"][s][i], rig.R[s][i]) for i in range(19)]
        ref, refp = rows_ref(rig, mixed, s, rig.R[s]), rows_ref(rig, mixed, s, rig.P[s])
        for i, lab in enumerate(A.GD.LABELS):
            m, mp = rig.row_stat("LF075", s, i, on, "R"), rig.row_stat("LF075", s, i, on, "P")
            assert all(abs(m[k] - ref[lab][k]) < 1e-9 for k in ("delta", "lo", "hi")), (lab, m, ref[lab])
            assert all(abs(mp[k] - refp[lab][k]) < 1e-9 for k in ("delta", "lo", "hi")), (lab, mp, refp[lab])


def check_selection(A, rig):
    sc = np.arange(100.0)
    scope = np.zeros(100, bool)
    scope[20:80] = True
    on = A.select_top(sc, scope, 0.5)
    assert on.sum() == 30 and on[:20].sum() == 0 and on[80:].sum() == 0 and on[50:80].sum() == 30
    assert A.select_top(sc, scope, 1.0).sum() == 60 and A.select_top(sc, scope, 0.0).sum() == 0


def check_auc(A, rig):                            # C4
    y = A.labels_from(rig.v0, rig.g)["GO"]
    for li, a in enumerate((0.68, 0.785, 0.85, 0.95)):
        r = [A.auc_mw(A.make_score(y, a, li + 1, d), y) for d in range(5)]
        assert abs(np.mean(r) - a) < 0.02, (a, r)
    assert A.auc_mw(A.make_score(y, 1.0, 7, 0), y) == 1.0
    assert abs(A.auc_mw(A.make_score(y, 1.0, 7, 0, permute=True), y) - 0.5) < 0.06


def check_label_matches_p1p(A, rig):               # C6 / C8
    post, gop, v0p = A.load_post(rig)
    sc = np.isfinite(post)
    y = A.labels_from(rig.v0, rig.g)["GO"]
    assert sc.sum() == 861 and np.array_equal(sc, rig.v0 < 8.0)
    lt6 = sc & (rig.v0 < 6.0)                          # AMEND_1: P1P `go` is limited to the probe population (v0 < 6); equality is asserted there only
    assert lt6.sum() == 608 and np.array_equal(y[lt6], gop[lt6]), int((y[lt6] != gop[lt6]).sum())
    assert not gop[sc & (rig.v0 >= 6.0)].any()
    assert np.nanmax(np.abs(v0p[sc] - rig.v0[sc])) < 1e-6
    OUT["anchor_auc_v0lt6_card_label"] = A.auc_mw(post[lt6], y[lt6])
    OUT["anchor_auc_v0lt8_card_label"] = A.auc_mw(post[sc], y[sc])
    OUT["anchor_auc_v0lt8_vs_P1P_go"] = A.auc_mw(post[sc], gop[sc])


def check_oracle_gate(A, rig):                     # C7
    y = A.labels_from(rig.v0, rig.g)["GO"]
    sc = A.make_score(y, 1.0, 7, 0)
    qg = float(y.mean())
    eg = A.evaluate(rig, "LF100", A.select_top(sc, ALL, qg))
    e1 = A.evaluate(rig, "LF100", A.select_top(sc, ALL, 1.0))
    e0 = A.evaluate(rig, "LF100", A.select_top(sc, ALL, 0.0))
    assert e0["share_mean"] == 0.0 and 0.0 < eg["share_mean"] < e1["share_mean"], (e0["share_mean"], eg["share_mean"], e1["share_mean"])
    OUT["oracle_gate_LF100"] = {"q": qg, "share_mean": eg["share_mean"], "ade6": eg["ade6_s"], "q1_share_mean": e1["share_mean"]}


def check_noskill_line(A, rig):                    # C5
    y = A.labels_from(rig.v0, rig.g)["GO"]
    full = A.evaluate(rig, "LF100", A.select_top(np.arange(2399.0), ALL, 1.0))["share_mean"]
    r = [A.evaluate(rig, "LF100", A.select_top(A.make_score(y, 0.5, 0, d), ALL, 0.5))["share_mean"] for d in range(10)]
    assert abs(np.mean(r) - 0.5 * full) < 0.08 * full + 0.02, (np.mean(r), 0.5 * full)
    OUT["noskill_half_gate_share"] = float(np.mean(r))
    OUT["ungated_LF100_share"] = full


def check_corner_known_values(A, rig):              # C9: the corner predicate can read True (X = base with the M1 row at P10) and can read False (an ACTIVE row pushed worse; ADE6 pushed past the guard)
    lever = "LF075"
    saved = {s: rig.X[lever][s].copy() for s in A.SEEDS}
    on = A.select_top(np.arange(2399.0), ALL, 1.0)
    try:
        def build(extra):
            for s in A.SEEDS:
                x = rig.R[s].copy()
                x[18] = rig.P[s][18]
                for i, add in extra.items():
                    x[i] = x[i] + add
                rig.X[lever][s] = x
        build({})
        e = A.evaluate(rig, lever, on)
        assert e["share_mean"] > 0.999 and e["M1P"] and e["M1FULL"] and e["GUARD"] and e["CORNER_FULL"] and e["CORNER_P"], e
        build({10: 5.0})                                   # cross-track MAE (ACTIVE, LATERAL) pushed worse
        e2 = A.evaluate(rig, lever, on)
        assert e2["M1FULL"] and not e2["GUARD"] and not e2["CORNER_FULL"], e2
        build({0: 0.2})                                    # ADE6 +0.2 m, past the +0.0604 m guard
        e3 = A.evaluate(rig, lever, on)
        assert e3["M1FULL"] and not e3["GUARD"] and not e3["CORNER_FULL"], e3
        OUT["C9_known_values"] = {"positive_corner_full": bool(e["CORNER_FULL"]), "active_row_worse_corner_full": bool(e2["CORNER_FULL"]), "ade6_past_guard_corner_full": bool(e3["CORNER_FULL"])}
    finally:
        for s in A.SEEDS:
            rig.X[lever][s] = saved[s]


CHECKS = [check_selection, check_auc, check_label_matches_p1p, check_q0_is_base, check_q1_is_lever, check_random_gate_matches_reader, check_oracle_gate, check_noskill_line, check_corner_known_values]


def run_checks(A, rig, checks=CHECKS):
    res = {}
    for c in checks:
        try:
            c(A, rig)
            res[c.__name__] = "PASS"
        except Exception as e:                      # an AssertionError or a crash both turn a mutant RED
            res[c.__name__] = f"FAIL: {type(e).__name__} {str(e)[:150]}"
    return res


def mutants(rig):
    sel, row_stat, mk = A.select_top, A.Rig.row_stat, A.make_score
    cases = [
        ("select_top inverted (lowest scores gate ON)", A, "select_top", lambda score, scope, q: sel(-score, scope, q)),
        ("scope mask dropped", A, "select_top", lambda score, scope, q: sel(score, np.ones_like(scope), q)),
        ("labels margin +2.0 instead of +1.0", A, "labels_from", lambda v0, g: {"GO": np.linalg.norm(g[:, 29] - g[:, 28], axis=-1) / 0.2 >= v0 + 2.0, "KEEP": v0 > -1}),
        ("shuffled-label arm not shuffled", A, "make_score", lambda y, auc, li, dr, permute=False: mk(y, auc, li, dr, False)),
        ("P10 contrast ignores the base excess", A.Rig, "row_stat", lambda self, lever, s, i, on, vs="R", dcache=None: row_stat(self, lever, s, i, on, "R", dcache)),
        ("GUARD ignores ACTIVE-family rows", A, "active_nonrow", lambda label: False),
        ("ADE6 guard disabled", A.SR, "ADE_GUARD_M", 99.0),
    ]
    out = {}
    for name, obj, attr, fn in cases:
        orig = getattr(obj, attr)
        setattr(obj, attr, fn)
        try:
            r = run_checks(A, rig)
        finally:
            setattr(obj, attr, orig)
        out[name] = {"RED": any(v.startswith("FAIL") for v in r.values()), "failed_checks": [k for k, v in r.items() if v.startswith("FAIL")]}
    return out


if __name__ == "__main__":
    t0 = time.time()
    A.init()
    rig = A.Rig(levers=["LF100", "LF075"])
    base = run_checks(A, rig)
    print("checks:", json.dumps(base, indent=1))
    mut = {} if "--quick" in sys.argv else mutants(rig)
    print("mutants:", json.dumps(mut, indent=1))
    os.makedirs(A.RAW, exist_ok=True)
    json.dump({"checks": base, "mutants": mut, "values": OUT, "wall_s": time.time() - t0}, open(f"{A.RAW}/CONTROLS{'_quick' if '--quick' in sys.argv else ''}.json", "w"), indent=1, default=float)
    ok = all(v == "PASS" for v in base.values()) and all(m["RED"] for m in mut.values())
    print("ALL GREEN, ALL MUTANTS RED" if ok else "PROBLEM", f"{time.time() - t0:.0f}s")
    sys.exit(0 if ok else 1)
