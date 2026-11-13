"""Tests for p5c_read.py (B7): known values, the card's verdict ladder, the qualifier table, and deliberate-regression (RED) arms.
Run: CUDA_VISIBLE_DEVICES=-1 python -m pytest test_p5c_read.py -q"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p5c_read as R   # noqa: E402

FAM_CI = os.path.join(HERE, "..", "..", "raw", "2026-11-12-instruments", "fam_ci", "fam_ci.py")


def mk_rows(arms: dict, n=120, n_logs=12):
    """arms: name -> {scorer: {metric: array in [0,1] or float}}; missing metrics default 1.0. Returns {token: row} as transp_score writes it."""
    rows = {}
    for i in range(n):
        t = f"t{i:04d}"
        r = {"log": f"log{i % n_logs:02d}", "token": t, "arms": {}}
        for a, sc in arms.items():
            r["arms"][a] = {}
            for s in R.SCORERS:
                d = {m: 1.0 for m in R.METRICS}
                for m, v in (sc.get(s) or {}).items():
                    d[m] = float(v[i]) if hasattr(v, "__len__") else float(v)
                r["arms"][a][s] = d
        rows[t] = r
    return rows


def noisy(n=120, n_logs=12, seed=3, sd=0.05, log_sd=0.03):
    rng = np.random.default_rng(seed)
    lg = rng.normal(0, log_sd, n_logs)
    return 0.9 + lg[np.arange(n) % n_logs] + rng.normal(0, sd, n)


def arms_with_shift(d_t=0.02, d_p=0.0, n=120):
    base = noisy(n)
    return {"C0": {"tracked": {"pdms": base}, "transparent": {"pdms": base}},
            "C1": {"tracked": {"pdms": base + d_t}, "transparent": {"pdms": base + d_p}},
            "base": {"tracked": {"pdms": base}, "transparent": {"pdms": base}}}


# ---------------------------------------------------------------------------------------------- constants and the draws
def test_B_is_the_cards_0_7031():
    assert R.B_BAR == 0.7031


def test_draws_equal_fam_ci_cluster_boot():
    """the instrument's CI is the SAME estimator and the SAME draws as fam_ci (so family CIs and these share draws)."""
    if not os.path.exists(FAM_CI):
        pytest.skip("fam_ci not in this tree")
    import importlib.util
    sp = importlib.util.spec_from_file_location("fam_ci_t", FAM_CI)
    fc = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(fc)
    rng = np.random.default_rng(5)
    v = rng.normal(0, 1, 120)
    logs = [f"log{i % 12:02d}" for i in range(120)]
    p, lo, hi, _ = fc.cluster_boot(v, logs)
    ci = R.Boot(logs).ci(v)
    assert (ci["point"], ci["lo"], ci["hi"]) == pytest.approx((p, lo, hi), rel=1e-12, abs=1e-12)   # summation order differs in the last ulp


def test_RED_per_token_bootstrap_is_narrower_than_log_cluster():
    """mutation: treating every token as its own cluster ignores the log structure and gives a too-narrow interval on clustered data."""
    n, L = 120, 12
    rng = np.random.default_rng(7)
    v = rng.normal(0, 1, L)[np.arange(n) % L] + rng.normal(0, 0.1, n)
    logs = [f"log{i % L:02d}" for i in range(n)]
    good = R.Boot(logs).ci(v)
    bad = R.Boot([f"tok{i}" for i in range(n)]).ci(v)
    assert (good["hi"] - good["lo"]) > 2 * (bad["hi"] - bad["lo"])


def test_TD_is_paired_per_draw_not_a_difference_of_intervals():
    rows = mk_rows(arms_with_shift(0.03, 0.01))
    res = R.run(rows, "C1", "C0", "base", rehearsal="t")
    h = res["d_headline"]
    assert h["TD"]["point"] == pytest.approx(h["d_tracked"]["point"] - h["d_T_transparent"]["point"])
    P = R.per_token(rows, ["C1", "C0"])
    bt = R.Boot(P["logs"])
    m = bt.draws(P["C1"]["tracked"]["pdms"] - P["C0"]["tracked"]["pdms"]) - bt.draws(P["C1"]["transparent"]["pdms"] - P["C0"]["transparent"]["pdms"])
    assert h["TD"]["lo"] == float(np.percentile(m, 2.5)) and h["TD"]["hi"] == float(np.percentile(m, 97.5))


# ---------------------------------------------------------------------------------------------- known-value reads
def test_constant_shift_known_values():
    rows = mk_rows({"C0": {"tracked": {"pdms": 0.90}, "transparent": {"pdms": 0.90}}, "C1": {"tracked": {"pdms": 0.92}, "transparent": {"pdms": 0.90}},
                    "base": {"tracked": {"pdms": 0.90}, "transparent": {"pdms": 0.90}}})
    res = R.run(rows, "C1", "C0", "base", rehearsal="t")
    h = res["d_headline"]
    assert h["d_tracked"] == pytest.approx({"point": 2.0, "lo": 2.0, "hi": 2.0})
    assert h["d_T_transparent"]["point"] == pytest.approx(0.0, abs=1e-12) and h["TD"]["point"] == pytest.approx(2.0)
    assert res["levels"]["C1"]["pdms"]["tracked"]["point"] == pytest.approx(92.0)
    assert res["qualifier"]["code"] == "TM"            # CI-lo(TD) > 0 and CI-lo(d_T) <= 0 (d_T = 0 exactly)


def test_P1b_is_strict_at_B():
    d = {"point": R.B_BAR, "lo": 0.1, "hi": 2.0}
    v = R.verdict_table(d, R.B_BAR, [], {"status": "PASS", "adverse": []}, True, True, None)
    assert v["verdict"] == "SWAMPED" and v["clauses"]["P1-b"] is False


def test_rates_and_mechanism_counts_known_values():
    n = 120
    dac_t = np.ones(n)
    dac_p = np.ones(n)
    dac_t[:12] = 0          # tracker-made: tracked 0, transparent 1  (12 tokens)
    dac_p[12:18] = 0        # tracker-saved: tracked 1, transparent 0  (6 tokens)
    rows = mk_rows({"C1": {"tracked": {"dac": dac_t}, "transparent": {"dac": dac_p}}, "C0": {}, "base": {}})
    res = R.run(rows, "C1", "C0", "base", rehearsal="t")
    r = res["rates"]["arms"]["C1"]
    assert r["tracker_made_dac_zero"]["ci"]["point"] == pytest.approx(10.0)
    assert r["tracker_saved_dac_zero"]["ci"]["point"] == pytest.approx(5.0)
    assert r["dac_zero"]["tracked"]["point"] == pytest.approx(10.0) and r["dac_zero"]["transparent"]["point"] == pytest.approx(5.0)
    assert res["rates"]["contrasts"]["C1-C0"]["tracker_made_dac_zero"]["ci"]["point"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------------------------- guards
def test_G1_adverse_subscore_gives_GUARD_FAIL_naming_it():
    n = 120
    a = arms_with_shift(0.05, 0.0)
    a["C1"]["tracked"]["ttc"] = 0.80       # TTC worse by 20 points in every token: separated adverse
    a["C1"]["transparent"]["ttc"] = 0.80
    rows = mk_rows(a)
    res = R.run(rows, "C1", "C0", "base", fam_ci={"frames": {"executed_tracked": {"families": {"longitudinal": {}, "lateral": {}}}}},
                controls=[], rehearsal="t")
    assert res["g1"]["adverse"] == ["ttc"]
    assert res["verdict"]["verdict"] == "REHEARSAL:GUARD-FAIL" and res["verdict"]["named"]["G1"] == ["ttc"]


def _fam(entries_tracked=None, entries_transp=None):
    mk = lambda e: {"families": {"longitudinal": e or {}, "lateral": {}}}   # noqa: E731
    return {"estimator": "synthetic", "frames": {"executed_tracked": mk(entries_tracked), "executed_transparent": mk(entries_transp)}}


def ent(lo, hi, point=None, A=None, B=None):
    return {"status": "OK", "diff_B_minus_A": {"point": point if point is not None else (lo + hi) / 2, "lo": lo, "hi": hi}, "A": A, "B": B}


def test_G2_reads_only_the_tracked_executed_frame():
    bad = {"speed_mae_mps": ent(0.1, 0.3)}                      # lower-is-better, diff lo > 0 -> C1 worse
    assert R.g2_from_famci(_fam(entries_tracked=bad))["status"] == "FAIL"
    ok_t = {"speed_mae_mps": ent(-0.1, 0.1)}                     # tracked frame with one comparable, non-adverse component
    assert R.g2_from_famci(_fam(entries_tracked=ok_t, entries_transp=bad))["status"] == "PASS"      # transparent frame is never a guard


def test_RED_G2_over_zero_components_is_not_evaluated_not_pass():
    """found 2026-11-13: a RAISED families block gave an empty fam_ci frame and G2 read PASS over nothing (Q-10)."""
    g = R.g2_from_famci(_fam())
    assert g["status"] == "NOT-EVALUATED" and g["components_compared"] == 0
    d = {"point": 2.0, "lo": 1.0, "hi": 3.0}
    assert R.verdict_table(d, R.B_BAR, [], g, True, True, None)["verdict"] == "INCOMPLETE"


def test_G2_polarity_flip_is_detected():
    higher = {"target_speed_acc.within_1.0_mps": ent(-0.3, -0.1)}   # accuracy-like: higher is better, diff hi < 0 -> worse
    assert R.g2_from_famci(_fam(entries_tracked=higher))["status"] == "FAIL"
    good = {"target_speed_acc.within_1.0_mps": ent(0.1, 0.3)}
    assert R.g2_from_famci(_fam(entries_tracked=good))["status"] == "PASS"
    assert R.fam_direction("accel_mae_mps2") == "lower"          # the 2026-09-28 substring bug must stay fixed


def test_G2_not_evaluated_blocks_a_positive_verdict():
    d = {"point": 2.0, "lo": 1.0, "hi": 3.0}
    v = R.verdict_table(d, R.B_BAR, [], R.g2_from_famci(None), True, True, None)
    assert v["verdict"] == "INCOMPLETE"


def test_strategic_is_paused_never_a_pass():
    res = R.run(mk_rows(arms_with_shift()), "C1", "C0", "base", rehearsal="t")
    assert res["strategic_family"]["status"] == "PAUSED"


# ---------------------------------------------------------------------------------------------- the verdict ladder (card 6.2)
OKG = {"status": "PASS", "adverse": []}


@pytest.mark.parametrize("d,g1,g2,c1v,ctl,p1c,want", [
    ({"point": 2.0, "lo": 1.0, "hi": 3.0}, [], OKG, True, True, None, "SEPARATED-SINGLE"),
    ({"point": 2.0, "lo": 1.0, "hi": 3.0}, [], OKG, True, True, {"pass": True}, "LEVER"),
    ({"point": 2.0, "lo": 1.0, "hi": 3.0}, [], OKG, True, True, {"pass": False}, "SWAMPED-BY-REPLICATE"),
    ({"point": 0.5, "lo": 0.1, "hi": 0.9}, [], OKG, True, True, None, "SWAMPED"),
    ({"point": 0.5, "lo": -0.4, "hi": 1.4}, [], OKG, True, True, None, "NULL"),
    ({"point": -2.0, "lo": -3.0, "hi": -1.0}, [], OKG, True, True, None, "REFUTED"),
    ({"point": 2.0, "lo": 1.0, "hi": 3.0}, ["ttc"], OKG, True, True, None, "GUARD-FAIL"),
    ({"point": 2.0, "lo": 1.0, "hi": 3.0}, [], {"status": "FAIL", "adverse": [{"component": "x"}]}, True, True, None, "GUARD-FAIL"),
    ({"point": 2.0, "lo": 1.0, "hi": 3.0}, [], OKG, False, True, None, "INVALID"),
    ({"point": 2.0, "lo": 1.0, "hi": 3.0}, [], OKG, True, False, None, "INVALID"),
    ({"point": 2.0, "lo": 1.0, "hi": 3.0}, [], OKG, True, None, None, "INCOMPLETE"),
])
def test_verdict_ladder(d, g1, g2, c1v, ctl, p1c, want):
    assert R.verdict_table(d, R.B_BAR, g1, g2, c1v, ctl, p1c)["verdict"] == want


def test_P1c_clause_numbers():
    rows = mk_rows({"C1": {"tracked": {"pdms": 0.93}}, "C0": {"tracked": {"pdms": 0.90}}, "C0r": {"tracked": {"pdms": 0.905}}, "base": {}})
    P = R.per_token(rows, ["C1", "C0", "C0r"])
    bt = R.Boot(P["logs"])
    p = R.p1c_clause(bt, P, "C1", "C0", "C0r", 3.0, R.B_BAR)
    assert p["F_p5c"] == pytest.approx(0.5) and p["threshold_max_B_2F"] == pytest.approx(1.0)
    assert p["d_r"]["point"] == pytest.approx(2.5) and p["pass"] is True
    rows2 = mk_rows({"C1": {"tracked": {"pdms": 0.93}}, "C0": {"tracked": {"pdms": 0.90}}, "C0r": {"tracked": {"pdms": 0.93}}, "base": {}})
    P2 = R.per_token(rows2, ["C1", "C0", "C0r"])
    p2 = R.p1c_clause(R.Boot(P2["logs"]), P2, "C1", "C0", "C0r", 3.0, R.B_BAR)
    assert p2["pass"] is False                       # d_r = 0: CI-lo(d_r) > 0 fails


def test_CTRL1_boundary():
    rows = mk_rows({"C0": {"tracked": {"pdms": 0.90}}, "base": {"tracked": {"pdms": 0.91}}, "C1": {"tracked": {"pdms": 0.90}}})
    res = R.run(rows, "C1", "C0", "base", rehearsal="t")
    assert res["ctrl1"]["valid"] is False and res["verdict"]["verdict"] == "REHEARSAL:INVALID"      # C0 - base = -1.0 < -B, hi < 0
    rows = mk_rows({"C0": {"tracked": {"pdms": 0.904}}, "base": {"tracked": {"pdms": 0.91}}, "C1": {"tracked": {"pdms": 0.90}}})
    assert R.run(rows, "C1", "C0", "base", rehearsal="t")["ctrl1"]["valid"] is True               # -0.6 is inside the floor
    rows = mk_rows({"C0": {"tracked": {"pdms": 0.92}}, "base": {"tracked": {"pdms": 0.90}}, "C1": {"tracked": {"pdms": 0.90}}})
    c = R.run(rows, "C1", "C0", "base", rehearsal="t")["ctrl1"]
    assert c["valid"] is True and "moves the base by itself" in c["finding"]


def test_failed_control_makes_INVALID(tmp_path):
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"pass": False}))
    res = R.run(mk_rows(arms_with_shift(0.05)), "C1", "C0", "base", fam_ci=_fam(), controls=[str(f)])
    assert res["verdict"]["verdict"] == "INVALID"


# ---------------------------------------------------------------------------------------------- qualifier table (card 7)
@pytest.mark.parametrize("td,dt,want", [
    ({"lo": 1, "hi": 3}, {"lo": -1, "hi": 1}, "TM"),
    ({"lo": 1, "hi": 3}, {"lo": -3, "hi": -1}, "TM+C"),
    ({"lo": -1, "hi": 1}, {"lo": 0.5, "hi": 2}, "TI"),
    ({"lo": 1, "hi": 3}, {"lo": 0.5, "hi": 2}, "TI"),
    ({"lo": -1, "hi": 1}, {"lo": -1, "hi": 1}, "TU"),
    ({"lo": -1, "hi": 1}, {"lo": -3, "hi": -1}, "TU"),
])
def test_qualifier(td, dt, want):
    assert R.qualifier(td, dt)["code"] == want


# ---------------------------------------------------------------------------------------------- refusals, determinism
def test_missing_arm_or_nonfinite_refused():
    rows = mk_rows(arms_with_shift())
    with pytest.raises(R.Refused):
        R.per_token(rows, ["C1", "nope"])
    rows["t0003"]["arms"]["C1"]["tracked"]["pdms"] = float("nan")
    with pytest.raises(R.Refused):
        R.per_token(rows, ["C1"])


def test_cli_refuses_without_controls_or_rehearsal(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in mk_rows(arms_with_shift()).values()))
    argv = ["--rows", str(p), "--c1", "C1", "--c0", "C0", "--base", "base", "--out", str(tmp_path / "o.json")]
    sys.argv = ["x"] + argv
    assert R.main() == 2
    sys.argv = ["x"] + argv + ["--rehearsal", "t", "--n-boot", "200"]
    assert R.main() == 0


def test_error_rows_refused(tmp_path):
    rows = list(mk_rows(arms_with_shift()).values())
    p = tmp_path / "rows.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n" + json.dumps({"log": "logX", "token": "tX", "error": "boom"}) + "\n")
    sys.argv = ["x", "--rows", str(p), "--c1", "C1", "--c0", "C0", "--base", "base", "--rehearsal", "t", "--out", str(tmp_path / "o.json")]
    assert R.main() == 2


def test_determinism_bit_identical():
    rows = mk_rows(arms_with_shift(0.03, 0.01))
    a = json.dumps(R.run(rows, "C1", "C0", "base", rehearsal="t"), sort_keys=True)
    b = json.dumps(R.run(rows, "C1", "C0", "base", rehearsal="t"), sort_keys=True)
    assert a == b


def test_rehearsal_label_prefixes_the_verdict():
    res = R.run(mk_rows(arms_with_shift(0.05)), "C1", "C0", "base", fam_ci=_fam(), rehearsal="stand-in arms")
    assert res["verdict"]["verdict"].startswith("REHEARSAL:") and "no P5-C" in res["REHEARSAL"]["claim"]
