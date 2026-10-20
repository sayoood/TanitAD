"""Tests of epadm_sim_s0.py on PLANTED data (no real pool, no label, no search output).  CPU only.
    cd code; CUDA_VISIBLE_DEVICES=-1 python -I -m pytest test_sim_s0.py -q -p no:cacheprovider --basetemp=<dir>
Needs EP_ADMISSION_DIR, EP_ADMISSION_R1_DIR (and EPADM_C1_DIR) as for the simulator itself."""
import hashlib
import inspect
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import epadm_sim_s0 as S                                                              # noqa: E402

R1, A, C1 = S.R1, S.A, S.C1
GEO = os.environ.get("S0_GEOMETRY") or os.path.join(HERE, "..", "geometry_s0.json")


def planted_base(n, seeds, rng, obs_zero=()):
    out = {}
    for s in seeds:
        v = {"Yu": rng.uniform(5, 9, n), "Su": rng.uniform(5, 9, n), "Ypc": rng.uniform(1, 3, n), "Spc": rng.uniform(1, 3, n), "Yec": rng.uniform(1, 3, n), "Sec": rng.uniform(1, 3, n)}
        v["y_p"], v["s_p"], v["y_e"], v["s_e"] = v["Yu"] + v["Ypc"], v["Su"] + v["Spc"], v["Yu"] + v["Yec"], v["Su"] + v["Sec"]
        for h in S.HEADS:
            v[f"OBS_{h}"] = np.zeros(n) if h in obs_zero else rng.integers(0, 3, n).astype(np.float64) + 0.0
            v[f"PRED_{h}"] = rng.uniform(0.2, 0.9, n)
        if "NC" not in obs_zero:
            v["OBS_NC"][0] = max(v["OBS_NC"][0], 1.0)
        out[s] = v
    return out


def draws(n, B=300):
    _, _, F = R1.libs()
    return F.boot_draws(n, B, 20260927)


# ------------------------------------------------------------------------------------------------------------------------------ pins, geometry, salts
def test_salts_are_new_and_distinct():
    assert S.S0_BASE_SALT != S.S0_OUTER_SALT
    assert S.S0_BASE_SALT not in S.USED_SALTS and S.S0_OUTER_SALT not in S.USED_SALTS
    assert set(S.USED_SALTS) >= {20261027, 20261028, 20261029, 20261019, 20261201, 20261202}


def test_geometry_sums_and_pin():
    for name, n in (("GS41", 41), ("GS117", 117), ("GS43", 43)):
        g = S.load_geometry(GEO, name)
        assert g["n"] == n and int(g["counts"].sum()) == S.TOKENS[name]
        assert (g["counts"] > 0).all()


def test_geometry_refuses_a_tampered_file(tmp_path):
    t = tmp_path / "g.json"
    t.write_bytes(open(GEO, "rb").read().replace(b'"n_tokens": 28', b'"n_tokens": 29', 1))
    with pytest.raises(SystemExit):
        S.load_geometry(str(t), "GS117")


def test_geometry_refuses_a_wrong_token_total(monkeypatch):
    monkeypatch.setitem(S.TOKENS, "GS41", 2781)
    with pytest.raises(SystemExit):
        S.load_geometry(GEO, "GS41")


def test_design_guard_refuses_without_a_matching_sidecar(tmp_path, monkeypatch):
    d = tmp_path / "SIM_DESIGN_S0.md"
    d.write_text("x")
    monkeypatch.setattr(S, "DESIGN", str(d))
    with pytest.raises(SystemExit):
        S.pin_design()
    (tmp_path / "SIM_DESIGN_S0.sha256").write_text("0" * 64)
    with pytest.raises(SystemExit):
        S.pin_design()
    (tmp_path / "SIM_DESIGN_S0.sha256").write_text(hashlib.sha256(b"x").hexdigest())
    assert S.pin_design() == hashlib.sha256(b"x").hexdigest()


def test_job_parser_refuses_unknown_and_x_with_t():
    for bad in ("X-GS41-T", "Z-GS41-A", "C-GS99-A", "C-GS41-B"):
        with pytest.raises(SystemExit):
            S.parse_job(bad)
    assert S.parse_job("C-GS117-T") == ("C", "GS117", "T")


# ------------------------------------------------------------------------------------------------------------------------------ the clone base
def fake_pool(rng, n_src=5, n_sets=60):
    log_idx = np.sort(rng.integers(0, n_src, n_sets))
    log_idx[:n_src] = np.arange(n_src)
    log_idx.sort()
    keys = ("y_p", "s_p", "y_e", "s_e", "Yu", "Su", "Ypc", "Spc", "Yec", "Sec", "OBS_NC", "PRED_NC", "OBS_DAC", "PRED_DAC", "OBS_TTC", "PRED_TTC")
    tok = {s: {k: rng.uniform(0, 1, n_sets) for k in keys} for s in S.SEEDS["C"]}
    return {"log_idx": log_idx, "log_names": [f"l{i}" for i in range(n_src)], "tok": tok, "n_sets": n_sets, "n_changed": {0: 5, 1: 6, 2: 7}}


def test_clone_integrity_holds_and_a_mixed_selection_is_refused():
    rng = np.random.default_rng(1)
    pool = fake_pool(rng)
    counts = np.array([3, 17, 1, 40, 8, 9])
    bc = S.base_c(pool, counts, 0, 0)
    assert S.check_clone(pool, bc, counts)
    assert all(np.bincount(bc["sel_log"], minlength=len(counts)) == counts)
    mixed = dict(bc)
    other = int((int(bc["src"][1]) + 1) % 5)
    sel = bc["sel"].copy()
    sel[bc["sel_log"] == 1][0:1]
    j = int(np.nonzero(bc["sel_log"] == 1)[0][0])
    sel[j] = int(np.nonzero(pool["log_idx"] == other)[0][0])                     # RED input: ONE set of simulated log 1 comes from another source log
    mixed["sel"] = sel
    assert not S.check_clone(pool, mixed, counts)
    short = dict(bc)
    short["sel"], short["sel_log"] = bc["sel"][:-1], bc["sel_log"][:-1]            # RED input: a missing set
    assert not S.check_clone(pool, short, counts)


def test_base_c_is_deterministic_and_the_seeds_share_the_draw():
    rng = np.random.default_rng(2)
    pool = fake_pool(rng)
    counts = np.array([4, 4, 4, 30])
    a, b = S.base_c(pool, counts, 5, 0), S.base_c(pool, counts, 5, 0)
    for s in S.SEEDS["C"]:
        for k in a["base"][s]:
            assert np.array_equal(a["base"][s][k], b["base"][s][k])
    c = S.base_c(pool, counts, 6, 0)
    assert not np.array_equal(a["sel"], c["sel"])
    assert not np.array_equal(S.base_c(pool, counts, 5, 1000)["sel"], a["sel"])           # the offset moves the draw


# ------------------------------------------------------------------------------------------------------------------------------ generators
def test_generator_T_keeps_the_logs_and_A_redraws_them():
    assert np.array_equal(S.outer_idx(3, 10, "T"), np.arange(10))
    assert not all(np.array_equal(S.outer_idx(r, 10, "A"), np.arange(10)) for r in range(5))
    assert len({tuple(S.outer_idx(r, 41, "A")) for r in range(20)}) == 20
    assert np.array_equal(S.outer_idx(7, 41, "A"), S.outer_idx(7, 41, "A"))


def test_generator_T_changes_the_estimator_input():
    rng = np.random.default_rng(3)
    base = planted_base(12, S.SEEDS["C"], rng)
    cond = C1.conditions()["CAL"]
    dr = draws(12)
    ra = S.replicate(cond, base, S.SEEDS["C"], "A", 0, dr)
    rt = S.replicate(cond, base, S.SEEDS["C"], "T", 0, dr)
    rt2 = S.replicate(cond, base, S.SEEDS["C"], "T", 9, dr)
    assert np.array_equal(rt["dm"], rt2["dm"], equal_nan=True)                       # T: nothing random with a fixed base
    assert not np.array_equal(ra["dm"], rt["dm"], equal_nan=True)


# ------------------------------------------------------------------------------------------------------------------------------ calibration, plants, identity, NaN policy
def test_calibrated_replicate_is_exactly_calibrated_at_the_base():
    rng = np.random.default_rng(4)
    base = planted_base(20, S.SEEDS["C"], rng)
    rec = S.replicate(C1.conditions()["CAL"], base, S.SEEDS["C"], "T", 0, draws(20))
    assert np.nanmax(np.abs(rec["br"] - 1.0)) <= 1e-12
    assert np.allclose(rec["mp"], 1.0) and np.allclose(rec["me"], 1.0)
    assert np.allclose(rec["dm"][:, 0], 0.0, atol=1e-12)


def test_planted_head_reads_rho_at_the_base():
    rng = np.random.default_rng(5)
    base = planted_base(20, S.SEEDS["C"], rng)
    rec = S.replicate(C1.conditions()["PLANT_A2_DAC_1.25"], base, S.SEEDS["C"], "T", 0, draws(20))
    j = S.HEADS.index("DAC")
    assert np.allclose(rec["br"][:, j], 1.25, atol=1e-12)
    for h in ("NC", "TTC"):
        assert np.allclose(rec["br"][:, S.HEADS.index(h)], 1.0, atol=1e-12)


def test_identity_candidate_reads_exactly_zero():
    rng = np.random.default_rng(6)
    base = planted_base(15, S.SEEDS["X"], rng)
    rec = S.replicate(C1.conditions()["IDENTITY"], base, S.SEEDS["X"], "A", 0, draws(15))
    assert (rec["dm"] == 0.0).all()


def test_no_observed_unsafe_pick_is_not_a_nan_under_the_real_sigma_convention():
    rng = np.random.default_rng(7)
    base = planted_base(20, S.SEEDS["C"], rng, obs_zero=("NC",))
    rec = S.replicate(C1.conditions()["CAL"], base, S.SEEDS["C"], "T", 0, draws(20))
    j = S.HEADS.index("NC")
    assert rec["k0"][:, j].all() and not rec["k0"][:, 1:].any()
    assert np.isfinite(rec["a2"][:, j, :]).all() and (rec["a2"][:, j, 0] == 0.0).all()      # ratio 0, lower CI 0 -> not a refusal at any margin
    assert not S.a2_refuse_module(rec["a2"], ["NC"], 1.05)
    assert S.failclosed_extra(rec["k0"], ["NC"]) and not S.failclosed_extra(rec["k0"], ["DAC", "TTC"])


def test_without_the_k0_fix_the_same_replicate_is_a_nan(monkeypatch):
    """RED arm of the NaN policy: the R1 builder alone (k = 0 wipes the prediction) gives 0/0, which the frozen module fails closed."""
    rng = np.random.default_rng(8)
    base = planted_base(20, S.SEEDS["C"], rng, obs_zero=("NC",))
    d = R1.build_condition(base[0], "cal")
    assert d["PRED_NC"].sum() == 0.0 and base[0]["OBS_NC"].sum() == 0.0
    _, _, F = R1.libs()
    ci = F.ratio_ci(d["OBS_NC"], d["PRED_NC"], draws(20))
    assert not np.isfinite(ci["lo"]) or not np.isfinite(ci["point"])


def test_planted_ep_shift_reads_the_delta():
    rng = np.random.default_rng(9)
    base = planted_base(30, S.SEEDS["C"], rng)
    rec = S.replicate(C1.conditions()["PLANT_EP_0.135"], base, S.SEEDS["C"], "T", 0, draws(30))
    assert np.allclose(rec["dm"][:, 0], 0.135, atol=1e-9)


# ------------------------------------------------------------------------------------------------------------------------------ decisions
def test_module_and_inline_decisions_agree_on_a_boundary_grid():
    pts, los = [-0.1, 0.0, 0.05, 0.0500001, 0.2, float("nan")], [-0.2, 0.0, 1e-12, 0.3, float("nan")]
    for p in pts:
        for lo in los:
            dm = np.array([[p, lo, 0.4], [0.0, -0.1, 0.1]])
            assert S.a1_refuse_module(dm) == S.a1_refuse_inline(dm), (p, lo)
    for m in (1.0, 1.05, 1.10):
        for lo in [0.5, 1.0, 1.05, 1.0500001, 1.10, 1.2, float("nan")]:
            a2 = np.full((2, 3, 3), 0.9)
            a2[0, 1] = (1.0, lo, 2.0)
            for heads in (["DAC"], ["NC", "DAC", "TTC"]):
                assert S.a2_refuse_module(a2, heads, m) == S.a2_refuse_inline(a2, heads, m), (m, lo, heads)


def test_decide_boundaries_and_mutation_arms():
    assert S.decide(0.05, 0.05, 0.70)["all_met"]
    assert not S.decide(0.0500001, 0.05, 0.70)["all_met"]
    assert not S.decide(0.05, 0.0500001, 0.70)["all_met"]
    assert not S.decide(0.05, 0.05, 0.6999)["all_met"]
    assert S.decide(0.5, 0.05, 0.9)["missed"] == ["A1_r1_false_refusal_wilson_upper_le_5pct"]
    src = inspect.getsource(S.decide)
    muts = [("a1_fr_upper <= BAR_A1_FR_UPPER", "a1_fr_upper < BAR_A1_FR_UPPER"), ("a2_fr_upper <= BAR_A2_FR_UPPER", "a2_fr_upper < BAR_A2_FR_UPPER"),
            ("dac_power_125 >= BAR_DAC_POWER_125", "dac_power_125 > BAR_DAC_POWER_125"), ("all(bars.values())", "any(bars.values())")]
    for old, new in muts:
        assert old in src
        ns = {"BAR_A1_FR_UPPER": 0.05, "BAR_A2_FR_UPPER": 0.05, "BAR_DAC_POWER_125": 0.70}
        exec(src.replace(old, new), ns)
        f = ns["decide"]
        caught = (f(0.05, 0.05, 0.70)["all_met"] is not True) or f(0.5, 0.05, 0.9)["all_met"] or f(0.05, 0.05, 0.6999)["all_met"] or f(0.05, 0.5, 0.9)["all_met"]
        assert caught, f"mutation survived: {old}"


def test_the_committed_constants_are_the_rulings():
    assert (S.BAR_A1_FR_UPPER, S.BAR_A2_FR_UPPER, S.BAR_DAC_POWER_125, S.BAR_MARGIN) == (0.05, 0.05, 0.70, 1.05)
    assert S.BOOT_B == 10000 and S.BOOT_SEED == 20260927
    assert S.MARGINS == (1.00, 1.05, 1.10)


def test_bar_status_table():
    assert S.bar_status(True, True) == "MET"
    assert S.bar_status(True, False) == "BASE-DEPENDENT"
    assert S.bar_status(False, True) == "MISSED" and S.bar_status(False, False) == "MISSED"


def test_summarize_counts_a_planted_refusal_rate():
    n, seeds = 40, S.SEEDS["C"]
    dm = np.zeros((n, 3, 3))
    dm[:, :, 1] = -0.1
    dm[:10, 0, 1] = 0.01                                   # 10 of 40 replicates fail A1 on seed 0 only
    rec = {"dm": dm, "a2": np.full((n, 3, 3, 3), np.nan), "mp": np.ones((n, 3)), "me": np.ones((n, 3)), "k0": np.zeros((n, 3, 3), bool), "br": np.full((n, 3, 3), np.nan)}
    s = S.summarize("X", {"a1": True, "a2": False, "heads": []}, rec, seeds)
    assert s["A1"]["r1_any_seed"]["k"] == 10 and s["A1"]["r1_per_seed"][0]["k"] == 10 and s["A1"]["r1_per_seed"][1]["k"] == 0
    assert s["pc4_module_vs_inline_disagreements"] == 0


def test_replicate_is_deterministic_and_uses_the_registered_draws():
    rng = np.random.default_rng(10)
    base = planted_base(25, S.SEEDS["C"], rng)
    cond = C1.conditions()["PLANT_A2_DAC_1.25"]
    dr = draws(25)
    r1, r2 = S.replicate(cond, base, S.SEEDS["C"], "A", 4, dr), S.replicate(cond, base, S.SEEDS["C"], "A", 4, dr)
    for k in r1:
        assert np.array_equal(r1[k], r2[k], equal_nan=True)
    r3 = S.replicate(cond, base, S.SEEDS["C"], "A", 4, dr, offset=1000)
    assert not np.array_equal(r1["a2"], r3["a2"], equal_nan=True)
