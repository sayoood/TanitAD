"""Tests of epadm_n1.py and epadm_sim_n1.py on PLANTED data (no real pool, no label, no search output).  CPU only.
    cd code; CUDA_VISIBLE_DEVICES=-1 python -I -m pytest test_n1.py -q -p no:cacheprovider --basetemp=<dir>
Needs EP_ADMISSION_DIR, EP_ADMISSION_R1_DIR and EPADM_C1_DIR as for the simulator itself.  Each guard has a RED input / mutation arm (mutation_check_n1.py rewrites the guard line and requires this suite to go RED)."""
import hashlib
import inspect
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import epadm_n1 as N                                                                   # noqa: E402
import epadm_sim_n1 as M                                                               # noqa: E402

S0, R1, A, C1, SB = M.S0, M.R1, M.A, M.C1, M.SB
HEADS = M.HEADS
GEO = os.environ.get("S0_GEOMETRY") or os.path.join(HERE, "..", "geometry_s0.json")


def ci(point, lo, hi):
    return {"point": point, "lo": lo, "hi": hi}


def planted_base(n, seeds, rng, obs_zero=(), no_changed=False, only_log=None):
    """Per-log planted sufficient statistics WITH the changed / unchanged split.  obs_zero: heads with no observed unsafe mass at the changed picks.  only_log: all changed picks sit in that one log."""
    out = {}
    for s in seeds:
        v = {"Yu": rng.uniform(5, 9, n), "Su": rng.uniform(5, 9, n), "Ypc": rng.uniform(1, 3, n), "Spc": rng.uniform(1, 3, n), "Yec": rng.uniform(1, 3, n), "Sec": rng.uniform(1, 3, n)}
        v["y_p"], v["s_p"], v["y_e"], v["s_e"] = v["Yu"] + v["Ypc"], v["Su"] + v["Spc"], v["Yu"] + v["Yec"], v["Su"] + v["Sec"]
        ch = rng.integers(3, 9, n).astype(np.float64)
        if no_changed:
            ch[:] = 0.0
        if only_log is not None:
            m = np.zeros(n)
            m[only_log] = ch[only_log] if ch[only_log] > 0 else 4.0
            ch = m
        v["CH"], v["NSETS"] = ch, np.full(n, 20.0)
        for h in HEADS:
            pc, pu = ch * rng.uniform(0.05, 0.3, n), (20.0 - ch) * rng.uniform(0.05, 0.3, n)
            oc = rng.integers(0, 3, n).astype(np.float64) * (ch > 0)
            ou = rng.integers(0, 3, n).astype(np.float64)
            if h in obs_zero:
                oc = np.zeros(n)
            v[f"OBSc_{h}"], v[f"PREDc_{h}"], v[f"OBSu_{h}"], v[f"PREDu_{h}"] = oc, pc, ou, pu
            v[f"OBS_{h}"], v[f"PRED_{h}"] = oc + ou, pc + pu
        out[s] = v
    return out


def draws(n, B=300):
    _, _, F = R1.libs()
    return F.boot_draws(n, B, 20260927)


def conds():
    return M.conditions()


def rep(cond, base, seeds, gen="A", r=0, n=None, idx=None, kfix=None):
    n = n or len(base[seeds[0]]["y_p"])
    return M.replicate(cond, base, seeds, gen, r, draws(n), idx=idx, kfix=kfix)


# ------------------------------------------------------------------------------------------------------------------------------ committed constants, salts, pins
def test_the_committed_constants_are_the_rulings():
    assert (M.BAR_A1_FR_UPPER, M.BAR_A2_FR_UPPER, M.BAR_DAC_POWER_125, M.BAR_MARGIN) == (0.05, 0.05, 0.70, 1.05)
    assert N.N1_MARGIN == 1.05
    assert M.BOOT_B == 10000 and M.BOOT_SEED == 20260927
    assert M.MARGINS == (1.00, 1.05, 1.10)
    assert M.decide is S0.decide and M.bar_status is S0.bar_status
    assert tuple(N.HEADS) == ("NC", "DAC", "TTC") == tuple(M.HEADS)


def test_salts_are_new_and_distinct():
    assert M.N1_BASE_SALT != M.N1_OUTER_SALT
    assert M.N1_BASE_SALT not in M.USED_SALTS and M.N1_OUTER_SALT not in M.USED_SALTS
    assert set(M.USED_SALTS) >= {20261027, 20261028, 20261029, 20261019, 20261201, 20261202, 20261301, 20261302}


def test_design_guard_refuses_without_a_matching_sidecar(tmp_path, monkeypatch):
    d = tmp_path / "SIM_DESIGN_N1.md"
    d.write_text("x")
    monkeypatch.setattr(M, "DESIGN", str(d))
    with pytest.raises(SystemExit):
        M.pin_design()
    (tmp_path / "SIM_DESIGN_N1.sha256").write_text("0" * 64)
    with pytest.raises(SystemExit):
        M.pin_design()
    (tmp_path / "SIM_DESIGN_N1.sha256").write_text(hashlib.sha256(b"x").hexdigest())
    assert M.pin_design() == hashlib.sha256(b"x").hexdigest()


def test_geometry_is_the_pinned_one():
    for name, n in (("GS41", 41), ("GS117", 117)):
        g = S0.load_geometry(GEO, name)
        assert g["n"] == n and int(g["counts"].sum()) == S0.TOKENS[name]


def test_job_parser_refuses_unknown_and_x_with_p_and_gs43():
    for bad in ("X-GS41-P", "Z-GS41-A", "C-GS99-A", "C-GS41-T", "C-GS43-A", "X-GS117-T"):
        with pytest.raises(SystemExit):
            M.parse_job(bad)
    assert M.parse_job("C-GS117-P") == ("C", "GS117", "P") and M.parse_job("X-GS41-A") == ("X", "GS41", "A")


# ------------------------------------------------------------------------------------------------------------------------------ the changed definition and the restriction
def test_changed_mask_is_the_slot_index_difference():
    pk_p, pk_c = np.array([3, 5, 7, 7]), np.array([3, 6, 7, 1])
    assert N.changed_mask(pk_p, pk_c).tolist() == [False, True, False, True]
    assert np.array_equal(N.changed_mask(pk_p, pk_c), A.libs()[0].pair_terms(np.zeros((4, 64)) + 0.5, np.zeros((4, 64)), pk_p, pk_c)["changed"])


def planted_bank(n_sets=30, n_logs=5, seed=0):
    rng = np.random.default_rng(seed)
    z6 = rng.normal(0, 1, (n_sets, 64, 6)).astype(np.float32)
    lab = (rng.random((n_sets, 64, 6)) < 0.9).astype(np.float32)
    log_idx = np.sort(np.arange(n_sets) % n_logs)
    return z6, lab, log_idx


def test_n1_excludes_unchanged_picks_and_all_ones_mask_would_not():
    """RED arm: poison the label at the UNCHANGED picks; N1 must not move.  The all-picks A2 (the same code with a mask of ones) does."""
    _, P, _ = A.libs()
    z6, lab, log_idx = planted_bank()
    zc = z6.copy()
    zc[:10, :, :] = z6[:10, ::-1, :]                                               # reverse the first 10 sets' fan: a different pick slot there
    pk_p, pk_c = P.rule_pick(z6), P.rule_pick(zc)
    ch = N.changed_mask(pk_p, pk_c)
    assert 0 < ch.sum() < len(ch)
    S = A.make_stats(log_idx, 5, B=200)
    base = N.n1_reliability(S, lab, zc, pk_p, pk_c)
    lab2 = lab.copy()
    ar = np.arange(len(pk_c))
    for col in (0, 1, 3):
        lab2[ar[~ch], pk_c[~ch], col] = 0.0                                        # every unchanged pick turns unsafe
    poisoned = N.n1_reliability(S, lab2, zc, pk_p, pk_c)
    for h in HEADS:
        assert R1.eq_ci(base["ci"][h], poisoned["ci"][h])
    an0, an1 = A.analyze_candidate(z6, zc, lab, S), A.analyze_candidate(z6, zc, lab2, S)
    assert not R1.eq_ci(an0["A2_reliability"]["DAC"], an1["A2_reliability"]["DAC"])
    lab3 = lab.copy()
    lab3[ar[ch], pk_c[ch], 1] = 0.0                                                # unsafe at the CHANGED picks does move N1
    assert not R1.eq_ci(base["ci"]["DAC"], N.n1_reliability(S, lab3, zc, pk_p, pk_c)["ci"]["DAC"])


def test_zero_changed_picks_refuses_to_compute():
    z6, lab, log_idx = planted_bank()
    _, P, _ = A.libs()
    pk = P.rule_pick(z6)
    S = A.make_stats(log_idx, 5, B=100)
    with pytest.raises(N.N1Undefined, match="zero changed picks"):
        N.n1_reliability(S, lab, z6, pk, pk)
    assert N.n1_verdict([None, None, None]) == "UNDEFINED-N1" != "ADMIT-N1"


def test_zero_predicted_mass_at_the_changed_picks_refuses_to_compute():
    z6, lab, log_idx = planted_bank()
    _, P, _ = A.libs()
    zc = z6.copy()
    zc[:8] = z6[:8, ::-1]
    zc[..., 1] = 60.0                                                              # sigma = 1.0 exactly for DAC everywhere: predicted unsafe mass 0
    pk_p, pk_c = P.rule_pick(z6), P.rule_pick(zc)
    assert N.changed_mask(pk_p, pk_c).sum() > 0
    S = A.make_stats(log_idx, 5, B=100)
    with pytest.raises(N.N1Undefined, match="predicted unsafe mass at the .* changed picks is 0"):
        N.n1_reliability(S, lab, zc, pk_p, pk_c)


def test_ci_logvec_refuses_a_zero_denominator_and_matches_ratio_ci_otherwise():
    d = draws(6, 50)
    with pytest.raises(N.N1Undefined, match="no changed pick in these logs"):
        N.n1_ci_logvec(np.ones(6), np.zeros(6), d)
    _, _, F = R1.libs()
    o, p = np.arange(1.0, 7.0), np.arange(2.0, 8.0)
    assert R1.eq_ci(N.n1_ci_logvec(o, p, d), F.ratio_ci(o, p, d))


# ------------------------------------------------------------------------------------------------------------------------------ the verdict and the 'any seed' rule
def test_verdict_any_seed_and_undefined_semantics():
    ok = {h: ci(1.0, 0.9, 1.1) for h in HEADS}
    bad = {**ok, "DAC": ci(1.3, 1.06, 1.5)}
    assert N.n1_verdict([ok, ok, ok]) == "ADMIT-N1"
    assert N.n1_verdict([ok, bad, ok]) == "FAIL-N1"                                 # ANY seed
    assert N.n1_verdict([ok, ok, bad]) == "FAIL-N1"
    assert N.n1_verdict([ok, None, ok]) == "UNDEFINED-N1"                           # undefined is NOT a pass
    assert N.n1_verdict([None, bad, ok]) == "FAIL-N1"                               # a definite refusal outranks an undefined seed
    assert N.n1_verdict([]) == "UNDEFINED-N1"
    assert N.n1_verdict([ok, ok, {"NC": ok["NC"], "DAC": ok["DAC"]}]) == "FAIL-N1"   # a missing head is not ok


def test_margin_boundary_is_closed_at_1_05_and_nan_fails_closed():
    edge = {h: ci(1.0, 1.05, 1.2) for h in HEADS}
    above = {h: ci(1.0, 1.0500001, 1.2) for h in HEADS}
    nan = {h: ci(1.0, float("nan"), float("nan")) for h in HEADS}
    assert N.n1_verdict([edge] * 3) == "ADMIT-N1"
    assert N.n1_verdict([above] * 3) == "FAIL-N1"
    assert N.n1_verdict([nan] * 3) == "FAIL-N1"
    assert N.n1_verdict([above] * 3, margin=1.10) == "ADMIT-N1" and N.n1_verdict([edge] * 3, margin=1.00) == "FAIL-N1"


def test_module_and_inline_verdicts_agree_on_a_grid():
    nan = float("nan")
    for lo in (0.5, 1.0, 1.05, 1.0500001, 1.10, 2.0, nan):
        for und_pat in ((False, False), (True, False), (True, True)):
            for heads in (["DAC"], list(HEADS)):
                n1_i = np.full((2, 3, 3), 0.9)
                n1_i[1, 1] = (1.0, lo, 3.0)
                und_i = np.zeros((2, 3), bool)
                und_i[0, :] = und_pat[0]
                und_i[1, 0] = und_pat[1]
                for m in (1.0, 1.05, 1.10):
                    assert M.n1_verdict_module(n1_i, und_i, heads, m) == M.n1_verdict_inline(n1_i, und_i, heads, m), (lo, und_pat, heads, m)


# ------------------------------------------------------------------------------------------------------------------------------ the bars as a pure function
def test_decide_boundaries_and_mutation_arms():
    assert M.decide(0.05, 0.05, 0.70)["all_met"]
    assert not M.decide(0.0500001, 0.05, 0.70)["all_met"] and not M.decide(0.05, 0.0500001, 0.70)["all_met"] and not M.decide(0.05, 0.05, 0.6999)["all_met"]
    src = inspect.getsource(S0.decide)
    for old, new in (("a1_fr_upper <= BAR_A1_FR_UPPER", "a1_fr_upper < BAR_A1_FR_UPPER"), ("a2_fr_upper <= BAR_A2_FR_UPPER", "a2_fr_upper < BAR_A2_FR_UPPER"),
                     ("dac_power_125 >= BAR_DAC_POWER_125", "dac_power_125 > BAR_DAC_POWER_125"), ("all(bars.values())", "any(bars.values())")):
        assert old in src
        ns = {"BAR_A1_FR_UPPER": 0.05, "BAR_A2_FR_UPPER": 0.05, "BAR_DAC_POWER_125": 0.70}
        exec(src.replace(old, new), ns)
        f = ns["decide"]
        assert (f(0.05, 0.05, 0.70)["all_met"] is not True) or f(0.5, 0.05, 0.9)["all_met"] or f(0.05, 0.05, 0.6999)["all_met"] or f(0.05, 0.5, 0.9)["all_met"], old


def test_bar_status_table():
    assert M.bar_status(True, True) == "MET" and M.bar_status(True, False) == "BASE-DEPENDENT"
    assert M.bar_status(False, True) == "MISSED" and M.bar_status(False, False) == "MISSED"


# ------------------------------------------------------------------------------------------------------------------------------ the replicate: calibration at the changed picks, plants, NaN policy
def test_cal_is_exactly_calibrated_at_the_changed_picks():
    rng = np.random.default_rng(4)
    n = 20
    base = planted_base(n, S0.SEEDS["C"], rng)
    rec = rep(conds()["CAL"], base, S0.SEEDS["C"], idx=np.arange(n))
    assert np.nanmax(np.abs(rec["br"] - 1.0)) <= 1e-12
    assert np.allclose(rec["n1"][:, :, 0], 1.0, atol=1e-12)                        # point = 1 with the logs not redrawn
    assert np.allclose(rec["mp"], 1.0) and np.allclose(rec["me"], 1.0) and np.allclose(rec["dm"][:, 0], 0.0, atol=1e-12)
    assert not rec["k0"].any() and not rec["und"].any()
    assert np.allclose(rec["nch"], [base[s]["CH"].sum() for s in S0.SEEDS["C"]])


def test_cal_is_not_calibrated_at_the_all_picks_ratio():
    """RED arm of the calibration class: calibrating with the ALL-picks scalar (the S0 / R1 form) leaves the changed class off 1 -- the N1 null must NOT be that."""
    rng = np.random.default_rng(14)
    base = planted_base(20, S0.SEEDS["C"], rng)
    v = base[0]
    k_all = v["OBS_DAC"].sum() / v["PRED_DAC"].sum()
    off = (v["OBSc_DAC"].sum()) / (k_all * v["PREDc_DAC"].sum())
    assert abs(off - 1.0) > 1e-3                                                   # the changed class is NOT at 1 under the all-picks scalar on this planted base
    rec = rep(conds()["CAL"], base, S0.SEEDS["C"], idx=np.arange(20))
    assert np.nanmax(np.abs(rec["br"] - 1.0)) <= 1e-12                            # ... and N1's null is at 1 by its own class scalar


def test_planted_head_reads_rho_at_the_changed_picks_and_only_that_head_is_computed():
    rng = np.random.default_rng(5)
    n = 20
    base = planted_base(n, S0.SEEDS["C"], rng)
    rec = rep(conds()["PLANT_N1_DAC_1.25"], base, S0.SEEDS["C"], idx=np.arange(n))
    j = HEADS.index("DAC")
    assert np.allclose(rec["br"][:, j], 1.25, atol=1e-12) and np.allclose(rec["n1"][:, j, 0], 1.25, atol=1e-12)
    assert np.isnan(rec["n1"][:, [0, 2]]).all()
    rec15 = rep(conds()["PLANT_N1_NC_1.5"], base, S0.SEEDS["C"], idx=np.arange(n))
    assert np.allclose(rec15["br"][:, 0], 1.5, atol=1e-12)


def test_plant_does_not_depend_on_the_unchanged_picks():
    rng = np.random.default_rng(15)
    n = 20
    base = planted_base(n, S0.SEEDS["C"], rng)
    r0 = rep(conds()["PLANT_N1_DAC_1.25"], base, S0.SEEDS["C"], r=3)
    base2 = {s: {**v, "OBSu_DAC": v["OBSu_DAC"] * 7 + 1.0, "PREDu_DAC": v["PREDu_DAC"] * 0.2} for s, v in base.items()}
    r1 = rep(conds()["PLANT_N1_DAC_1.25"], base2, S0.SEEDS["C"], r=3)
    assert np.array_equal(r0["n1"], r1["n1"], equal_nan=True)                      # N1 never reads the unchanged class


def test_a2_local_plant_is_diluted_below_rho_and_above_one():
    rng = np.random.default_rng(16)
    n = 20
    base = planted_base(n, S0.SEEDS["C"], rng)
    rec = rep(conds()["PLANT_A2L_DAC_1.25"], base, S0.SEEDS["C"], idx=np.arange(n))
    j = HEADS.index("DAC")
    for si, s in enumerate(S0.SEEDS["C"]):
        v = base[s]
        pu, pc = v["OBSu_DAC"].sum(), v["OBSc_DAC"].sum()
        want = (pu + pc) / (pu + pc / 1.25)                                          # both classes calibrated to 1, only the changed class planted
        assert abs(rec["a2l"][si, j, 0] - want) <= 1e-12
        assert 1.0 < rec["a2l"][si, j, 0] < 1.25


def test_no_observed_unsafe_mass_at_the_changed_picks_is_not_a_nan_under_the_real_sigma_convention():
    rng = np.random.default_rng(7)
    base = planted_base(20, S0.SEEDS["C"], rng, obs_zero=("NC",))
    rec = rep(conds()["CAL"], base, S0.SEEDS["C"])
    j = HEADS.index("NC")
    assert rec["k0"][:, j].all() and not rec["k0"][:, 1:].any() and not rec["und"].any()
    assert np.isfinite(rec["n1"][:, j, :]).all() and (rec["n1"][:, j, 0] == 0.0).all()
    assert M.n1_verdict_module(rec["n1"], rec["und"], ["NC"], 1.05) == "ADMIT-N1"


def test_without_the_k0_fix_the_same_replicate_is_undefined(monkeypatch):
    """RED arm of the NaN policy: scaling the prediction by k = 0 wipes it (0/0); N1 then refuses (undefined), which is NOT what a real sigma < 1 gives."""
    rng = np.random.default_rng(8)
    base = planted_base(20, S0.SEEDS["C"], rng, obs_zero=("NC",))
    v = base[0]
    assert M.calib(float(v["OBSc_NC"].sum()), float(v["PREDc_NC"].sum())) is None
    with pytest.raises(N.N1Undefined):
        N.n1_ci_logvec(v["OBSc_NC"], v["PREDc_NC"] * 0.0, draws(20))


def test_no_changed_pick_in_the_base_is_undefined_never_a_pass():
    rng = np.random.default_rng(9)
    base = planted_base(15, S0.SEEDS["C"], rng, no_changed=True)
    rec = rep(conds()["CAL"], base, S0.SEEDS["C"])
    assert rec["und"].all() and np.isnan(rec["n1"]).all() and (rec["nch"] == 0).all() and not rec["k0"].any()
    assert M.n1_verdict_module(rec["n1"][...], rec["und"], list(HEADS), 1.05) == "UNDEFINED-N1"


def test_no_changed_pick_in_the_REDRAWN_logs_is_undefined():
    rng = np.random.default_rng(10)
    n = 12
    base = planted_base(n, S0.SEEDS["C"], rng, only_log=0)
    idx = (np.arange(n) % (n - 1)) + 1                                              # length n, never log 0: the redraw misses the only log that holds changed picks
    rec = rep(conds()["CAL"], base, S0.SEEDS["C"], n=n, idx=idx)
    assert rec["und"].all() and not (rec["nch"] == 0).any()                         # defined at the base, undefined on the drawn logs
    rec_ok = rep(conds()["CAL"], base, S0.SEEDS["C"], n=n, idx=np.arange(n))
    assert not rec_ok["und"].any()


def test_identity_candidate_reads_zero_dm_and_an_undefined_n1():
    rng = np.random.default_rng(6)
    base = planted_base(15, S0.SEEDS["X"], rng)
    rec = rep(conds()["IDENTITY"], base, S0.SEEDS["X"])
    assert (rec["dm"] == 0.0).all() and rec["und"].all() and np.isnan(rec["n1"]).all() and (rec["nch"] == 0).all()
    assert M.n1_verdict_module(rec["n1"], rec["und"], list(HEADS), 1.05) == "UNDEFINED-N1"


def test_planted_ep_shift_reads_the_delta_and_a1_is_unchanged_by_n1():
    rng = np.random.default_rng(11)
    n = 30
    base = planted_base(n, S0.SEEDS["C"], rng)
    rec = rep(conds()["PLANT_EP_0.135"], base, S0.SEEDS["C"], idx=np.arange(n))
    assert np.allclose(rec["dm"][:, 0], 0.135, atol=1e-9)
    a1 = S0.replicate(C1.conditions()["PLANT_EP_0.135"], base, S0.SEEDS["C"], "T", 0, draws(n))
    assert np.array_equal(a1["dm"], rec["dm"], equal_nan=True)                     # the A1 inputs are the S0BANK simulator's, byte for byte


# ------------------------------------------------------------------------------------------------------------------------------ generators
def test_generator_A_redraws_logs_with_the_n1_salt_and_P_keeps_them():
    assert np.array_equal(M.outer_idx(3, 10, "P"), np.arange(10))
    assert len({tuple(M.outer_idx(r, 41, "A")) for r in range(20)}) == 20
    assert np.array_equal(M.outer_idx(7, 41, "A"), M.outer_idx(7, 41, "A"))
    assert not np.array_equal(M.outer_idx(7, 41, "A"), S0.outer_idx(7, 41, "A"))        # not the S0BANK draw


def test_generator_P_is_not_degenerate_while_A_is_calibrated_exactly():
    rng = np.random.default_rng(12)
    n = 25
    base = planted_base(n, S0.SEEDS["C"], rng)
    kfix = {(si, j): 1.0 for si in range(3) for j in range(3)}                     # a pool scalar that is NOT the replicate base's own
    cond = conds()["CAL"]
    ra = rep(cond, base, S0.SEEDS["C"], gen="A", idx=np.arange(n))
    rp = rep({**cond, "a1": False}, base, S0.SEEDS["C"], gen="P", kfix=kfix)
    assert np.nanmax(np.abs(ra["br"] - 1.0)) <= 1e-12                              # A: forced to 1 on its own base
    assert np.nanmax(np.abs(rp["br"] - 1.0)) > 1e-3                                  # P: the sample ratio varies around the POOL calibration
    rp2 = rep({**cond, "a1": False}, base, S0.SEEDS["C"], gen="P", r=9, kfix=kfix)
    assert np.array_equal(rp["n1"], rp2["n1"], equal_nan=True)                       # P: nothing random with a fixed base


def test_pool_k_is_the_pool_level_scalar():
    rng = np.random.default_rng(13)
    pool = {"tok": {s: {} for s in S0.SEEDS["C"]}}
    for s in S0.SEEDS["C"]:
        for h in HEADS:
            pool["tok"][s][f"OBSc_{h}"] = rng.integers(0, 2, 50).astype(float)
            pool["tok"][s][f"PREDc_{h}"] = rng.uniform(0.1, 0.5, 50)
    pool["tok"][1]["OBSc_NC"][:] = 0.0
    k = M.pool_k(pool)
    assert abs(k[(0, 1)] - pool["tok"][0]["OBSc_DAC"].sum() / pool["tok"][0]["PREDc_DAC"].sum()) < 1e-15
    assert k[(1, 0)] is None


# ------------------------------------------------------------------------------------------------------------------------------ the bases
def fake_pool(rng, n_src=5, n_sets=60):
    log_idx = np.sort(rng.integers(0, n_src, n_sets))
    log_idx[:n_src] = np.arange(n_src)
    log_idx.sort()
    keys = ("y_p", "s_p", "y_e", "s_e", "Yu", "Su", "Ypc", "Spc", "Yec", "Sec", "OBS_NC", "PRED_NC", "OBS_DAC", "PRED_DAC", "OBS_TTC", "PRED_TTC")
    tok = {s: {k: rng.uniform(0, 1, n_sets) for k in keys} for s in S0.SEEDS["C"]}
    for s in S0.SEEDS["C"]:
        ch = (rng.random(n_sets) < 0.3)
        tok[s].update(M.split_vectors({h: tok[s][f"OBS_{h}"] for h in HEADS}, {h: tok[s][f"PRED_{h}"] for h in HEADS}, ch))
    return {"log_idx": log_idx, "log_names": [f"l{i}" for i in range(n_src)], "tok": tok, "n_sets": n_sets, "n_changed": {0: 5, 1: 6, 2: 7}}


def test_split_vectors_partition_and_sum():
    rng = np.random.default_rng(20)
    ch = rng.random(40) < 0.4
    obs, pred = {h: rng.uniform(0, 1, 40) for h in HEADS}, {h: rng.uniform(0, 1, 40) for h in HEADS}
    sv = M.split_vectors(obs, pred, ch)
    for h in HEADS:
        assert np.allclose(sv[f"OBSc_{h}"] + sv[f"OBSu_{h}"], obs[h]) and np.allclose(sv[f"PREDc_{h}"] + sv[f"PREDu_{h}"], pred[h])
        assert (sv[f"OBSc_{h}"][~ch] == 0).all() and (sv[f"OBSu_{h}"][ch] == 0).all()
    assert sv["CH"].sum() == ch.sum()


def test_base_c_n1_is_deterministic_has_the_split_and_uses_its_own_salt(monkeypatch):
    rng = np.random.default_rng(2)
    pool = fake_pool(rng)
    counts = np.array([4, 4, 4, 30])
    a, b = M.base_c_n1(pool, counts, 5, 0), M.base_c_n1(pool, counts, 5, 0)
    for s in S0.SEEDS["C"]:
        for k in a["base"][s]:
            assert np.array_equal(a["base"][s][k], b["base"][s][k])
        assert "CH" in a["base"][s] and np.allclose(a["base"][s]["CH"].sum(), a["base"][s]["CH"].sum())
    assert S0.check_clone(pool, a, counts)
    assert not np.array_equal(M.base_c_n1(pool, counts, 6, 0)["sel"], a["sel"]) and not np.array_equal(M.base_c_n1(pool, counts, 5, 1000)["sel"], a["sel"])
    # fidelity of the copy: with the N1 salt patched into the S0 module, S0.base_c draws the SAME clone base; with the real salts it does not
    s0 = S0.base_c(pool, counts, 5, 0)
    assert not np.array_equal(s0["sel"], a["sel"])
    monkeypatch.setattr(S0, "S0_BASE_SALT", M.N1_BASE_SALT)
    s0p = S0.base_c(pool, counts, 5, 0)
    assert np.array_equal(s0p["sel"], a["sel"]) and np.array_equal(s0p["src"], a["src"])
    for k in ("y_p", "OBS_DAC", "OBS_NC", "PRED_TTC"):
        assert np.array_equal(s0p["base"][0][k], a["base"][0][k])


def test_basex_vectors_equal_a_brute_force_on_planted_arrays():
    rng = np.random.default_rng(21)
    N, nl = 80, 6
    z6 = rng.normal(0, 1, (N, 64, 6)).astype(np.float32)
    lab = (rng.random((N, 64, 6)) < 0.85).astype(np.float32)
    log_idx = np.sort(np.arange(N) % nl)
    oof = {0: rng.normal(0, 1, (N, 64)).astype(np.float32), 1: rng.normal(0, 1, (N, 64)).astype(np.float32)}
    out = M.basex_vectors(z6, lab, log_idx, oof, 123, 0.5, nl)
    _, P, _ = R1.libs()
    sub = np.random.default_rng(123).random(N) < 0.5
    for s in (0, 1):
        zc = z6.copy()
        zc[..., R1.EP] = np.where(sub[:, None], oof[s], z6[..., R1.EP])
        pk_p, pk_c = P.rule_pick(z6), P.rule_pick(zc)
        for l in range(nl):
            rows = np.nonzero(log_idx == l)[0]
            chg = [t for t in rows if pk_p[t] != pk_c[t]]
            assert out[s]["split"]["CH"][l] == len(chg)
            for h in HEADS:
                col = R1.HCOL[h]
                want = sum(1.0 - float(lab[t, pk_c[t], col]) for t in chg)
                assert abs(out[s]["split"][f"OBSc_{h}"][l] - want) < 1e-9
        assert out[s]["split"]["NSETS"].sum() == N
        for h in HEADS:
            assert np.allclose(out[s]["split"][f"OBSc_{h}"] + out[s]["split"][f"OBSu_{h}"], out[s]["allpick"][f"OBS_{h}"])
    other = M.basex_vectors(z6, lab, log_idx, oof, 124, 0.5, nl)                    # RED arm: another subset seed gives another Base X
    assert not np.array_equal(other[0]["split"]["CH"], out[0]["split"]["CH"])


def test_load_base_x_refuses_unless_pc9_passed(tmp_path):
    npz = tmp_path / "basex_n1.npz"
    np.savez(npz, a=np.zeros(3))
    with pytest.raises(SystemExit):
        M.load_base_x_n1("whatever.npz", str(npz), np.arange(3))                    # no json
    (tmp_path / "basex_n1.json").write_text(json.dumps({"pass": False}))
    with pytest.raises(SystemExit):
        M.load_base_x_n1("whatever.npz", str(npz), np.arange(3))
    (tmp_path / "basex_n1.json").write_text(json.dumps({"pass": True}))
    with pytest.raises(SystemExit):                                                 # now it reaches the stats pin and refuses the wrong file
        M.load_base_x_n1(str(npz), str(npz), np.arange(3))


# ------------------------------------------------------------------------------------------------------------------------------ summaries
def test_summarize_counts_planted_refusals_undefined_and_power_separately():
    n, S = 40, 3
    nan = np.nan
    n1 = np.full((n, S, 3, 3), nan)
    n1[:, :, 1, :] = (1.0, 0.9, 1.1)                                                # DAC fine everywhere
    n1[:10, 1, 1, 1] = 1.06                                                         # 10 replicates: DAC lower CI above 1.05 on seed 1
    und = np.zeros((n, S, 3), bool)
    und[10:14, 2, :] = True                                                         # 4 replicates: seed 2 undefined
    n1[10:14, 2, :, :] = nan
    rec = {"dm": np.full((n, S, 3), nan), "n1": n1, "a2l": np.full((n, S, 3, 3), nan), "mp": np.full((n, S), nan), "me": np.full((n, S), nan), "k0": np.zeros((n, S, 3), bool), "und": und,
           "br": np.full((n, S, 3), nan), "nch": np.full((n, S), 10.0), "eobs": np.ones((n, S, 3))}
    cond = {"a1": False, "n1": True, "a2l": False, "heads": ["DAC"], "kind": "plant_n1", "kw": {}}
    s = M.summarize("X", cond, rec, S0.SEEDS["C"])
    m = s["N1"]["by_margin"]["1.05"]
    assert m["any_head_any_seed_fail_only"]["k"] == 10 and m["undefined_verdicts"] == 4 and m["any_head_any_seed"]["k"] == 14      # FR counts FAIL + UNDEFINED
    assert m["per_head_any_seed"]["DAC"]["k"] == 10                                  # power counts the definite refusals only
    assert s["pc4_module_vs_inline_disagreements"] == 0


def test_bar_inputs_read_the_registered_margin_and_the_dac_head():
    def blk(fr_k, pw_k, n=100):
        r = lambda k: R1.rate(k, n)                                                  # noqa: E731
        return {"by_margin": {"1.05": {"any_head_any_seed": r(fr_k), "any_head_any_seed_failclosed": r(fr_k + 5), "per_head_any_seed": {"DAC": r(pw_k)}, "per_head_any_seed_failclosed": {"DAC": r(pw_k)}}}}
    summ = {"CAL": {"A1": {"r1_any_seed": R1.rate(3, 100)}, "N1": blk(2, 0)}, "PLANT_N1_DAC_1.25": {"N1": blk(0, 71)}}
    a1, a2, pw = M.bar_inputs(summ, "primary")
    assert (a1["k"], a2["k"], pw["k"]) == (3, 2, 71)
    assert M.bar_inputs(summ, "failclosed")[1]["k"] == 7


# ------------------------------------------------------------------------------------------------------------------------------ the statistic on synthetic ADAPTED banks (the gate's own loaders and Stats object)
@pytest.fixture(scope="module")
def banks(tmp_path_factory):
    import fixture_synth as FX
    root = tmp_path_factory.mktemp("n1banks")
    out = {}
    for tag, share, sd in (("all", 1.0, 5), ("none", 0.0, 6), ("part", 0.3, 3)):
        fx = FX.make_fixture(str(root / f"fx_{tag}"), n_tokens=96, n_logs=12, seed=sd, moved_share=share)
        d = str(root / f"bank_{tag}")
        SB.build(fx["bundle"], fx["search"], fx["labels"], d, {"token_sha256": fx["token_sha256"]})
        out[tag] = d
    return out


def test_n1_on_an_all_changed_bank_equals_the_frozen_a2(banks):
    _, P, _ = A.libs()
    p2, cands, _ = SB.load_with_gate(banks["all"])
    S = A.make_stats(p2["log_idx"], p2["n_logs"])
    pk_p = P.rule_pick(p2["z6"])
    for s in S0.SEEDS["C"]:
        pk_c = P.rule_pick(cands[s]["z6"])
        assert (pk_c != pk_p).all()
        mod = N.n1_reliability(S, p2["lab"], cands[s]["z6"], pk_p, pk_c)
        an = A.analyze_candidate(p2["z6"], cands[s]["z6"], p2["lab"], S)
        for h in HEADS:
            assert R1.eq_ci(mod["ci"][h], an["A2_reliability"][h]), (s, h)


def test_n1_on_a_partially_changed_bank_differs_from_the_frozen_a2(banks):
    _, P, _ = A.libs()
    p2, cands, _ = SB.load_with_gate(banks["part"])
    S = A.make_stats(p2["log_idx"], p2["n_logs"])
    pk_p = P.rule_pick(p2["z6"])
    pk_c = P.rule_pick(cands[0]["z6"])
    assert 0 < (pk_c != pk_p).sum() < len(pk_p)
    mod = N.n1_reliability(S, p2["lab"], cands[0]["z6"], pk_p, pk_c)
    an = A.analyze_candidate(p2["z6"], cands[0]["z6"], p2["lab"], S)
    assert not all(R1.eq_ci(mod["ci"][h], an["A2_reliability"][h]) for h in HEADS)


def test_n1_on_a_zero_changed_bank_refuses_and_the_bank_verdict_is_undefined(banks):
    _, P, _ = A.libs()
    p2, cands, _ = SB.load_with_gate(banks["none"])
    S = A.make_stats(p2["log_idx"], p2["n_logs"])
    pk_p = P.rule_pick(p2["z6"])
    for s in S0.SEEDS["C"]:
        with pytest.raises(N.N1Undefined, match="zero changed picks"):
            N.n1_reliability(S, p2["lab"], cands[s]["z6"], pk_p, P.rule_pick(cands[s]["z6"]))
    res = N.n1_from_bank(banks["none"])
    assert res["verdict"] == "UNDEFINED-N1" and all(v == 0 for v in res["n_changed_by_seed"].values())


def test_the_shortcut_equals_the_stats_path_on_an_adapted_bank(banks):
    res = M.pc3_n1(banks["all"], banks["none"], banks["part"])
    assert res["pass"] is True
    assert res["banks"]["none_changed"]["s0"]["refuses"] is True
