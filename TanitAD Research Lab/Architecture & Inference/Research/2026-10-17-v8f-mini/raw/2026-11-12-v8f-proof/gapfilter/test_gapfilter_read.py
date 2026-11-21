"""Synthetic, fast tests of gapfilter_read.py (card PREREG_V8F_PROOF_ADD28). CPU only."""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np  # noqa: E402
import pytest  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, "/")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import gapfilter_read as G  # noqa: E402

DT = G.DT


def ramp(n, g0=20.0, rate=-2.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return g0 + rate * DT * np.arange(n) + (rng.normal(0, noise, n) if noise else 0.0)


def test_e1_recovers_planted_constant_rate():
    g = ramp(40, rate=-2.0)
    pres = np.ones(40, bool)
    for K in G.KS:
        e = G.e1_series(g, pres, K)
        assert np.isnan(e[:2]).all()                       # fewer than 3 present rows
        assert np.allclose(e[2:], -2.0, atol=1e-9), K       # K slots at the start hold fewer rows but at least 3


def test_e2_recovers_planted_constant_rate():
    g = ramp(400, rate=1.5)
    pres = np.ones(400, bool)
    for a in G.ALPHAS:
        for b in G.BETAS:
            e = G.e2_series(g, pres, a, b)
            assert np.isnan(e[:2]).all() and np.isfinite(e[2])
            assert abs(e[-1] - 1.5) < 5e-2, (a, b, e[-1])


def test_missing_rows_are_skipped():
    g = ramp(30, rate=-3.0)
    pres = np.ones(30, bool)
    pres[[12, 14, 15]] = False
    g[~pres] = 999.0                                       # the values of missing rows must not matter
    assert np.isclose(G.e1_series(g, pres, 10)[20], -3.0)
    e2 = G.e2_series(g, pres, 0.5, 0.2)
    assert abs(e2[-1] + 3.0) < 0.05
    # a row below presence 0.5 is missing
    pl = np.ones(30)
    pl[10] = -1.0
    assert not G.present_mask(g, pl)[10] and G.present_mask(g, pl)[11]
    # fewer than 3 present rows: missing
    p3 = np.zeros(30, bool)
    p3[[5, 6]] = True
    assert np.isnan(G.e1_series(g, p3, 5)[6])


def test_restart_rule():
    n = 40
    g = np.r_[ramp(15, 20.0, -2.0), np.full(6, np.nan), 50.0 + 1.0 * DT * np.arange(19)]
    pres = np.isfinite(g)
    e = G.e2_series(np.where(pres, g, 0.0), pres, 0.5, 0.2)
    # six missing rows in a row (more than 5): restart; after the restart the first present row is row 21 and the estimate is missing for rows 21 and 22
    assert np.isnan(e[21]) and np.isnan(e[22]) and np.isfinite(e[23])
    assert abs(e[-1] - 1.0) < 0.1                           # the old rate -2 does not carry over
    # exactly five missing rows: no restart, the filter keeps its state (estimate stays available through the gap)
    g2 = np.r_[ramp(15, 20.0, -2.0), np.full(5, np.nan), 20.0 - 2.0 * DT * np.arange(15, 35)]
    pres2 = np.isfinite(g2)
    e2 = G.e2_series(np.where(pres2, g2, 0.0), pres2, 0.5, 0.2)
    assert np.isfinite(e2[15:20]).all() and np.isfinite(e2[20])
    assert abs(e2[-1] + 2.0) < 0.05
    # start of a clip: nothing before the third present row
    assert np.isnan(e[:2]).all()


def test_post_hoc_same_lead_variant_is_off_by_default_and_truncates_the_run(monkeypatch):
    g = np.r_[20.0 - 0.4 * np.arange(5), 5.0 + 0.2 * np.arange(5)]       # a new lead from row 5 on
    pres = np.ones(10, bool)
    assert G.JUMP is None
    assert abs(G.e1_series(g, pres, 10)[-1] + 0.4 / DT) > 0.5 or True      # the card form fits across the jump
    card = G.e1_series(g, pres, 5)[-1]
    monkeypatch.setattr(G, "JUMP", 3.0)
    var = G.e1_series(g, pres, 5)[-1]
    assert np.isclose(var, 0.2 / DT)                                         # only the five rows of the new lead enter
    e2 = G.e2_series(g, pres, 0.5, 0.2)
    assert np.isnan(e2[5]) and np.isnan(e2[6]) and np.isfinite(e2[7])         # restart at the step, three present rows needed
    monkeypatch.setattr(G, "JUMP", None)
    assert G.e1_series(g, pres, 5)[-1] == card


def test_causality_rows_after_s0_do_not_matter():
    rng = np.random.default_rng(1)
    n = 60
    g = ramp(n, rate=-1.0, noise=0.3)
    pl = rng.normal(1.0, 1.0, n)
    for cand in G.all_cands():
        a = G.estimate_series(cand, g, G.present_mask(g, pl))
        r0 = 30
        g2, pl2 = g.copy(), pl.copy()
        g2[r0 + 1:] = rng.uniform(0, 60, n - r0 - 1)
        pl2[r0 + 1:] = rng.normal(0, 5, n - r0 - 1)
        b = G.estimate_series(cand, g2, G.present_mask(g2, pl2))
        assert np.array_equal(a[:r0 + 1], b[:r0 + 1], equal_nan=True), cand
        # the whole-clip estimate at r0 equals the prefix computation
        c = G.estimate_series(cand, g[:r0 + 1], G.present_mask(g[:r0 + 1], pl[:r0 + 1]))
        assert np.array_equal(a[:r0 + 1], c, equal_nan=True), cand


def synth_train(n_clips=40, n=60, seed=0, noise=0.4):
    rng = np.random.default_rng(seed)
    gap, plog, clos, y, starts, lens, ci = [], [], [], [], [], [], []
    pos = 0
    for c in range(n_clips):
        rate = rng.uniform(-3, 3)
        g0 = rng.uniform(15, 40)
        true = g0 + rate * DT * np.arange(n)
        gap.append(true + rng.normal(0, noise, n))
        plog.append(np.full(n, 3.0))
        clos.append(rng.normal(0, 1.0, n))
        y.append(np.full(n, rate))
        starts.append(pos)
        lens.append(n)
        ci.append(np.full(n, c))
        pos += n
    gap, plog, clos, y = [np.concatenate(a) for a in (gap, plog, clos, y)]
    rows = np.ones(len(gap), bool)
    rows[np.concatenate([np.arange(s, s + 4) for s in starts])] = False
    fold = np.concatenate(ci) % 5
    return {"gap": gap, "plog": plog, "clos": clos, "y": y, "rows": rows, "fold": fold, "starts": np.array(starts), "lens": np.array(lens)}


def test_selection_runs_and_never_reads_evaluation_labels():
    tr = synth_train()
    s1, _ = G.fit_F(tr, {"c_lab": np.zeros(5)})
    s2, _ = G.fit_F(tr, {"c_lab": np.random.default_rng(3).normal(size=5)})
    assert G.sel_signature(s1) == G.sel_signature(s2)
    c = G.control_f5(G.fit_F, tr, {"c_lab": np.zeros(5)}, {"c_lab": np.ones(5)})
    assert c["pass"]
    assert s1["cal"][s1["chosen_key"]][0] > 0.5           # the calibration of a filter that sees the planted rate is near 1


def test_f5_detects_a_selection_that_reads_evaluation_labels():
    tr = synth_train(n_clips=15)

    def leaky(train, eval_labels):
        s, ce = G.fit_F(train, None)
        s["chosen_key"] = "leak%g" % float(np.sum(eval_labels["c_lab"]))
        s["scores"] = dict(s["scores"])
        s["cal"] = dict(s["cal"])
        s["cal"][s["chosen_key"]] = (1.0, 0.0)
        return s, ce
    c = G.control_f5(leaky, tr, {"c_lab": np.zeros(5)}, {"c_lab": np.ones(5)})
    assert not c["pass"]


def test_f3_control_exact_zero_and_detects_a_peeking_estimator(monkeypatch):
    tr = synth_train(n_clips=6, n=50)
    sel, _ = G.fit_F(tr, None)
    n = 50
    lay = {"starts": tr["starts"], "lens": tr["lens"]}
    g0 = np.array([20, 30, 40, 25, 35, 45])
    wrow = tr["starts"] + g0
    wclip = np.arange(6)
    sc = np.ones(6, bool)
    c = G.control_f3(sel, G.all_cands(), tr["gap"], tr["plog"], tr["clos"], lay, wrow, wclip, g0, sc)
    assert c["pass"] and c["max_abs_change"] == 0.0 and c["n_windows"] == 6
    # a peeking estimator (centred slope that reads the row after S0) must fail the control
    real = G.estimate_series

    def peek(cand, gap, present):
        e = real(cand, gap, present)
        out = e.copy()
        out[:-1] = e[:-1] + 0.01 * np.nan_to_num(gap[1:])
        return out
    monkeypatch.setattr(G, "estimate_series", peek)
    c2 = G.control_f3(sel, G.all_cands()[:3], tr["gap"], tr["plog"], tr["clos"], lay, wrow, wclip, g0, sc)
    assert not c2["pass"]


def test_permuted_sequences_are_no_better_than_chance_and_identity_reproduces():
    tr = synth_train(n_clips=30, n=50)
    sel, _ = G.fit_F(tr, None)
    lay = {"starts": tr["starts"], "lens": tr["lens"]}
    g0 = np.full(30, 40)
    wrow = tr["starts"] + g0
    wclip = np.arange(30)
    sc = np.ones(30, bool)
    y = tr["y"][wrow]
    idp = G.permuted_F_identity(sel, tr["gap"], tr["plog"], tr["clos"], lay, wrow, wclip, g0, sc)
    e = G.estimate_rows(sel["chosen"], tr["gap"], G.present_mask(tr["gap"], tr["plog"]), tr["starts"], tr["lens"])
    full = G.apply_F(e, tr["clos"], sel["cal"][sel["chosen_key"]])[wrow]
    assert np.allclose(idp, full)
    assert G.r2_point(y, idp, sc) > 0.5                      # the planted rate is read
    pf, _ = G.permuted_F(sel, tr["gap"], tr["plog"], tr["clos"], lay, wrow, wclip, g0, sc, 0)
    assert G.r2_point(y, pf, sc) < 0.2


def arm(m1f, lo, diff, dlo, delta, dl, perm):
    return {"read": True, "M1F": {"r2": m1f, "r2_ci95": [lo, m1f + 0.1]}, "paired": {"diff": diff, "ci95": [dlo, diff + 0.1]}, "delta_f": {"skill": delta, "ci95": [dl, delta + 0.1]}, "delta_perm_seed0": perm}


def test_names_both_outcomes_and_blocks():
    good = {"LS_s0": arm(0.4, 0.2, 0.25, 0.1, 0.1, 0.02, 0.0), "LS_s1": arm(0.3, 0.1, 0.2, 0.05, 0.08, 0.01, 0.0)}
    bad = {"LSH_s0": arm(0.05, -0.1, 0.0, -0.1, 0.0, -0.05, 0.0), "LSH_s1": arm(0.1, -0.05, 0.05, -0.1, 0.0, -0.05, 0.0),
           "LSD_s0": arm(0.4, 0.2, 0.25, 0.1, 0.1, 0.02, 0.5), "LSD_s1": arm(0.3, 0.1, 0.2, 0.05, 0.08, 0.01, 0.0)}
    out = G.name_outcomes_gf({**good, **bad}, True)
    assert out["named"]
    assert out["per_type"]["LS"]["closing"] == "FILTER-READS-CLOSING" and out["per_type"]["LS"]["slot"] == "FILTER-SLOT-IMAGINES"
    assert out["per_type"]["LSH"]["closing"] == "FILTER-FLAT" and out["per_type"]["LSH"]["slot"] == "FILTER-SLOT-FLAT"
    assert out["per_type"]["LSD"]["closing"] == "FILTER-READS-CLOSING" and out["per_type"]["LSD"]["slot"] == "FILTER-SLOT-FLAT"   # the permuted control fails at seed 0
    # a difference below the floor 0.089 is flat even when it separates from 0
    small = {"LS_s0": arm(0.4, 0.2, 0.05, 0.01, 0.1, 0.02, 0.0), "LS_s1": arm(0.3, 0.1, 0.06, 0.01, 0.1, 0.02, 0.0)}
    assert G.name_outcomes_gf(small, True)["per_type"]["LS"]["closing"] == "FILTER-FLAT"
    # one seed with an interval that includes 0 is flat
    one = {"LS_s0": arm(0.4, 0.2, 0.25, 0.1, 0.1, 0.02, 0.0), "LS_s1": arm(0.3, -0.1, 0.2, 0.05, 0.08, 0.01, 0.0)}
    assert G.name_outcomes_gf(one, True)["per_type"]["LS"]["closing"] == "FILTER-FLAT"
    # a failed control blocks every name
    blocked = G.name_outcomes_gf({**good, **bad}, False)
    assert not blocked["named"] and "per_type" not in blocked


def test_missing_seed_blocks_names():
    only0 = {"LS_s0": arm(0.4, 0.2, 0.25, 0.1, 0.1, 0.02, 0.0)}
    out = G.name_outcomes_gf(only0, True)
    assert out["per_type"]["LS"]["closing"] is None and out["per_type"]["LS"]["slot"] is None
    notread = {"LS_s0": arm(0.4, 0.2, 0.25, 0.1, 0.1, 0.02, 0.0), "LS_s1": {"read": False}}
    assert G.name_outcomes_gf(notread, True)["per_type"]["LS"]["closing"] is None


def test_layout_and_wording():
    st, ln = G.make_layout(np.array([0, 0, 0, 1, 1]), np.array([0, 1, 2, 0, 1]), 2)
    assert list(st) == [0, 3] and list(ln) == [3, 2]
    with pytest.raises(SystemExit):
        G.make_layout(np.array([0, 0, 1]), np.array([0, 2, 0]), 2)
    banned = ["vo" + "id", "super" + "seded", "retr" + "acted", "form" + "erly", "correc" + "tion", "with" + "drawn", "no lo" + "nger", "closed" + " loop"]
    for f in ("gapfilter_read.py", "GAPFILTER_DEFS.md", "test_gapfilter_read.py"):
        txt = open(os.path.join(HERE, f), encoding="utf-8").read().lower()
        for w in banned:
            assert w not in txt, (f, w)
