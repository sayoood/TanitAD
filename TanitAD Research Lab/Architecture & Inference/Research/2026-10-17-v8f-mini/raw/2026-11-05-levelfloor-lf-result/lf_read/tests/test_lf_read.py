"""test_lf_read.py -- the LF reader (code/lf_read.py): instruments, reach control, outcome labels, the headline, the flags, the tool-vs-card tick controls, the production controls on a SYNTHETIC out root, and the two known answers
(ka-null: the banked RIC0 arrays fed as RILF read share 0.0 and every contrast row exactly 0.0; ka-llr: the banked RILLR arrays fed as RILF reproduce LLR_READ.json's rows exactly).  Every test is a mutation target (run_mutations_lf.py).
Env override: LF_READ_PATH."""
import copy
import importlib.util
import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
ART = "E:/Projects/TanitAD-artifacts"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


LF = _load("lf_read_under_test", os.environ.get("LF_READ_PATH", f"{PK}/code/lf_read.py"))


@pytest.fixture(scope="module")
def inited():
    LF.init()
    return LF


def ticks(n=7197, **kw):
    """A consistent synthetic tick dump: v_hat 6 m/s, nothing binds, logged route speed 6 m/s, own speed 6 m/s."""
    gt = np.zeros((n, 30, 2))
    gt[:, :, 0] = (np.arange(1, 31) * 0.2 * 6.0)[None, :]
    t = {"lf_bind": np.zeros(n, bool), "lf_cap": np.zeros(n, bool), "lf_vf": np.full(n, 3.0, np.float32), "lf_vhat": np.full(n, 6.0, np.float32), "lf_pi": np.full(n, 0.2, np.float32),
         "lf_gap": np.full(n, 20.0, np.float32), "lf_lift": np.zeros(n, np.float32), "v0": np.full(n, 6.0), "gt": gt, "gt_ok": np.ones(n, bool)}
    t.update(kw)
    return t


def test_m1_tick_definition_is_the_pair_thresholds():
    v0 = np.array([4.0, 6.0, 4.0, 4.0, 2.0])
    vl = np.array([6.0, 6.0, 4.0, np.nan, 6.0])
    ok = np.array([True, True, True, False, True])
    assert LF.m1_ticks(v0, vl, ok).tolist() == [True, False, False, False, True]
    assert LF.card_crawl_ticks(v0, vl, ok).tolist() == [False, False, False, False, True]      # the card's band (own < half the logged speed) is narrower than M1's pair thresholds


def test_instruments_known_values_and_bind_inside_m1_ticks():
    n = 10
    t = ticks(n)
    t["v0"] = np.array([4.0] * 4 + [6.0] * 6)                    # ticks 0..3 are M1 crawl ticks (own 4 < 5.2 <= logged 6)
    t["lf_bind"] = np.array([True, True, False, False] + [True] + [False] * 5)
    t["lf_cap"] = np.array([True] + [False] * 9)
    t["lf_lift"] = np.array([2.0, 4.0, 0, 0, 1.0, 0, 0, 0, 0, 0], np.float32)
    r = LF.tick_instruments(t)
    assert r["n_m1_ticks"] == 4 and r["bind_rate_in_m1_ticks"] == {"n": 4, "rate": 0.5}
    assert r["bind_rate_overall"]["rate"] == 0.3 and r["cap_rate_overall"]["rate"] == 0.1 and r["cap_rate_in_m1_ticks"]["rate"] == 0.25
    assert abs(r["lift_where_bound_overall"]["mean_lift_mps"] - 7 / 3) < 1e-6 and abs(r["lift_where_bound_in_m1_ticks"]["mean_lift_mps"] - 3.0) < 1e-6
    vf = r["vf_vs_logged_overall"]
    assert abs(vf["mae_mps"] - 3.0) < 1e-6 and vf["share_vf_exceeds_logged"] == 0.0 and vf["n"] == n      # v_f 3.0 against a logged 6.0


def test_reach_control_one_percent_each_seed():
    mk = lambda r, n=100: {"bind_rate_in_m1_ticks": {"n": n, "rate": r}}
    assert LF.reach_control({"s0": mk(0.01), "s1": mk(0.5)})["PASS"] is True
    assert LF.reach_control({"s0": mk(0.0099), "s1": mk(0.5)})["PASS"] is False
    assert LF.reach_control({"s0": mk(0.5), "s1": mk(0.0)})["PASS"] is False          # BOTH seeds
    assert LF.reach_control({"s0": mk(None, 0), "s1": mk(0.5)})["PASS"] is False         # no M1 tick cannot evidence reach


def test_unreached_reads_no_rule(inited):
    bad = LF.reach_control({"s0": {"bind_rate_in_m1_ticks": {"n": 10, "rate": 0.0}}, "s1": {"bind_rate_in_m1_ticks": {"n": 10, "rate": 0.5}}})
    r = LF.read_dose("050", {}, 10, bad)                    # arrays {} are never touched: no rule is read
    assert r["rules"] is None and r["seeds"] is None and r["outcome"]["label"] == "F-UNREACHED" and r["outcome"]["counts_as_result"] is False


def blocks(P=False, G=True, dom=False, leak=False, worse=(), ade6=True, share=None):
    perg = {"ade6_delta": 0.0 if ade6 else 0.1, "ade6_within_guard": ade6, "non_crawl_separated_worse": list(worse)}
    return ({"PASS": P}, {"PASS": G, "per_seed": {"s0": perg, "s1": perg}},
            {"DOMINANT_COUNTED": dom, "by_seed": {"s0": {"separated_worse_rows": list(worse)}, "s1": {"separated_worse_rows": []}}}, {"LEAK_BORNE": leak})


def test_outcome_labels_every_branch():
    ok = {"PASS": True}
    P, G, D, L = blocks(dom=True, P=True)
    assert LF.decide_dose("050", ok, P, G, D, L)["label"] == "D-HOLDS"
    P, G, D, L = blocks(P=True, G=False, ade6=False)
    o = LF.decide_dose("050", ok, P, G, D, L)
    assert o["label"] == "P-HOLDS-G-OR-D-FAILS" and "0.375" in o["next"] and "ADE6" in o["harm_rows"]
    assert "0.625" in LF.decide_dose("075", ok, P, G, D, L)["next"]
    P, G, D, L = blocks(P=True, G=False, worse=("DK-HW",))
    assert "2.0 s time gap" in LF.decide_dose("050", ok, P, G, D, L)["next"]
    P, G, D, L = blocks(P=True, G=False, worse=("DK-HW", "speed MAE"))
    assert "0.375" in LF.decide_dose("050", ok, P, G, D, L)["next"]            # any ADE6 / speed MAE / along-track failure takes the dose branch
    P, G, D, L = blocks(P=True, G=False, worse=("heading MAE",))
    assert "Master Mind" in LF.decide_dose("050", ok, P, G, D, L)["next"]
    P, G, D, L = blocks(P=False)
    assert LF.decide_dose("050", ok, P, G, D, L, m1_moved=True)["label"] == "P-FAILS-M1-MOVED"
    assert LF.decide_dose("050", ok, P, G, D, L, m1_moved=False)["label"] == "P-FAILS-M1-NOT-MOVED"


def test_leak_borne_overrides_the_label_and_is_not_a_result():
    P, G, D, L = blocks(dom=True, P=True, leak=True)
    o = LF.decide_dose("075", {"PASS": True}, P, G, D, L)
    assert o["label"] == "LEAK-BORNE" and o["underlying"] == "D-HOLDS" and o["counts_as_result"] is False


def test_headline_names_the_dose_and_discloses_two_doses():
    mk = lambda lab, moved=False: {"outcome": {"label": lab, "underlying": lab, "next": "x", "counts_as_result": True}, "m1_moved": moved}
    j = LF.joint_reading({"050": mk("P-FAILS-M1-NOT-MOVED"), "075": mk("D-HOLDS")})
    assert j["branch"] == "COUNTED-DOMINANT" and "075" in j["headline"] and "050" not in j["headline"].split("Signal")[0] and "Two doses were read" in j["headline"]
    j = LF.joint_reading({"050": mk("P-FAILS-M1-MOVED", True), "075": mk("P-FAILS-M1-NOT-MOVED")})
    assert j["branch"] == "P-FAILS-ESTIMATOR-BOTTLENECK" and "Two doses were read" in j["headline"]
    j = LF.joint_reading({"050": mk("P-FAILS-M1-NOT-MOVED"), "075": mk("P-FAILS-M1-NOT-MOVED")})
    assert j["branch"] == "P-FAILS-M1-NOT-MOVED"
    j = LF.joint_reading({"050": mk("F-UNREACHED"), "075": mk("P-FAILS-M1-NOT-MOVED")})
    assert j["branch"] == "F-UNREACHED"


def test_rollouts_and_flags_copy_the_banked_base_plus_floor():
    r = LF.rollouts()
    assert [x[0] for x in r][:8] == ["RILF050_s0", "RILF050_s1", "RILF050C_s0", "RILF050C_s1", "RILF075_s0", "RILF075_s1", "RILF075C_s0", "RILF075C_s1"] and len(r) == 16
    assert [x[0] for x in r][8:12] == ["LF050_s0", "LF050_s1", "LF050C_s0", "LF050C_s1"]
    banked = json.load(open("F:/TanitAD-scratch/motion_c/ri/out/done_RIC0_s0.txt", encoding="utf-8"))["flags"]
    bankedc = json.load(open("F:/TanitAD-scratch/motion_c/ri/out/done_RIC0C_s0.txt", encoding="utf-8"))["flags"]
    banked_c0 = json.load(open("F:/TanitAD-scratch/recover_read/out/done_C0_s0.txt", encoding="utf-8"))["flags"]
    fl = LF.expected_flags("RILF", "050", 0)
    assert fl.startswith(banked + " --level-floor on --floor-kappa 0.5 --floor-ckpt ") and fl.endswith("/M2b_s0_recft_LLR/ckpt_last.pt")
    assert LF.expected_flags("RILFC", "075", 1).startswith(bankedc + " --level-floor on --floor-kappa 0.75 ")
    assert LF.expected_flags("LF", "075", 0).startswith(banked_c0 + " --level-floor on --floor-kappa 0.75 ")


def test_tick_controls_catch_a_tool_that_deviates_from_the_card():
    assert all(LF.tick_controls(ticks(), 0.5).values())
    for name, bad in (("v_f_between_0_and_kappa_v_hat", ticks(lf_vf=np.full(7197, 5.0, np.float32))), ("no_bind_below_the_3_mps_band", ticks(lf_vhat=np.full(7197, 2.0, np.float32), lf_bind=np.ones(7197, bool))),
                      ("cap_only_where_presence_at_least_half", ticks(lf_cap=np.ones(7197, bool))), ("lift_zero_where_unbound", ticks(lf_lift=np.full(7197, 1.0, np.float32))),
                      ("n_ticks_is_3x2399", ticks(100)), ("pi_in_unit_interval", ticks(lf_pi=np.full(7197, 1.5, np.float32)))):
        assert LF.tick_controls(bad, 0.5)[name] is False, name
    assert LF.tick_controls({k: v for k, v in ticks().items() if k != "lf_vf"}, 0.5) == {"fields_present": False}


def _synthetic_out_root(tmp, llr_sha_override=None, kappa_override=None, tool="a" * 64):
    os.makedirs(tmp, exist_ok=True)
    out = str(tmp / "out")
    pins = str(tmp / "PINS.txt")
    open(pins, "w", encoding="utf-8").write(f"patched_tool sha256 {tool}\n")
    for tag, base, dose, s, scene, feas in LF.rollouts():
        os.makedirs(f"{out}/{tag}/ticks", exist_ok=True)
        np.savez(f"{out}/{tag}/ticks/ticks_{tag}.npz", **ticks(7197, lf_vf=np.full(7197, 3.0 * LF.KAPPA[dose] / 0.5, np.float32)))
        hdr = {"scene": scene, "feasible": ({"mode": feas} if feas != "off" else None), "smooth": {"mode": LF.BASE_SMOOTH}, "n_windows": 2399, "kin_mode": "yaw", "level_floor": "on", "floor_kappa": LF.KAPPA[dose],
               "floor_ckpt": LF.floor_ckpt(s), "floor_ckpt_sha256": LF.PC.llr_sha_of(s)}
        if kappa_override and tag == kappa_override[0]:
            hdr["floor_kappa"] = kappa_override[1]
        if llr_sha_override and tag == llr_sha_override[0]:
            hdr["floor_ckpt_sha256"] = llr_sha_override[1]
        json.dump({"t1v8f": hdr}, open(f"{out}/{tag}/t1v8f_{tag}.json", "w", encoding="utf-8"))
        json.dump({"tag": tag, "flags": LF.expected_flags(base, dose, s), "ckpt": LF.c0_ckpt(s), "ckpt_sha256": LF.PC.C0_SHA[s], "tool_sha256": tool, "threads": 12}, open(f"{out}/done_{tag}.txt", "w", encoding="utf-8"))
    return out, pins


def test_production_controls_pass_on_a_consistent_out_root_and_refuse_each_deviation(inited, tmp_path):
    raw = f"{PK}/raw"
    out, pins = _synthetic_out_root(tmp_path)
    ctl = LF.run_controls(out, raw, pins)
    assert all(v["PASS"] for k, v in ctl.items()), {k: v for k, v in ctl.items() if not v["PASS"]}
    out, pins = _synthetic_out_root(tmp_path / "k", kappa_override=("RILF075_s1", 0.5))
    assert LF.run_controls(out, raw, pins)["rollout_RILF075_s1"]["PASS"] is False
    out, pins = _synthetic_out_root(tmp_path / "s", llr_sha_override=("LF050C_s0", "0" * 64))
    assert LF.run_controls(out, raw, pins)["rollout_LF050C_s0"]["PASS"] is False
    out, pins = _synthetic_out_root(tmp_path / "t")
    open(pins, "w", encoding="utf-8").write("patched_tool sha256 " + "b" * 64 + "\n")
    assert LF.run_controls(out, raw, pins)["rollout_RILF050_s0"]["PASS"] is False          # the done marker's tool sha256 is not the pinned patched_tool
    out, pins = _synthetic_out_root(tmp_path / "u")
    open(pins, "w", encoding="utf-8").write("# no key\n")
    assert LF.run_controls(out, raw, pins)["patched_tool_pin"]["PASS"] is False


def test_ka_null_runs_green_share_zero_and_every_row_exactly_zero(inited, tmp_path):
    assert LF.main(["--mode", "ka-null", "--raw", str(tmp_path), "--work-dir", str(tmp_path / "w")]) == 0
    d = json.load(open(f"{tmp_path}/KA_READER_ka-null.json", encoding="utf-8"))
    assert d["KA_PASS"] is True and len(d["ka_checks"]) >= 29


def test_ka_llr_runs_green_and_reproduces_llr_read_rows(inited, tmp_path):
    assert LF.main(["--mode", "ka-llr", "--raw", str(tmp_path), "--work-dir", str(tmp_path / "w")]) == 0
    d = json.load(open(f"{tmp_path}/KA_READER_ka-llr.json", encoding="utf-8"))
    assert d["KA_PASS"] is True and len(d["ka_checks"]) >= 45


def test_known_answer_files_in_the_package_are_pass():
    for m, n in (("ka-null", 29), ("ka-llr", 45)):
        d = json.load(open(f"{PK}/raw/KA_READER_{m}.json", encoding="utf-8"))
        assert d["KA_PASS"] is True and len(d["ka_checks"]) >= n


def test_a_negative_chord_lift_where_bound_is_reported_not_refused():
    """A3_RULING_2: where F binds, a negative chord lift is a property of the frozen operator (points moved forward along a bending path), reported beside, never a refusal."""
    t = ticks(lf_bind=np.ones(7197, bool), lf_lift=np.where(np.arange(7197) % 10 == 0, -0.2, 0.5).astype(np.float32))
    assert LF.tick_controls(t, 0.5)["lift_zero_where_unbound"] is True
    r = LF.tick_instruments(t)["lift_where_bound_overall"]
    assert r["n_negative_chord_lift"] == 720 and abs(r["share_negative_chord_lift"] - 720 / 7197) < 1e-9 and abs(r["min_lift_mps"] + 0.2) < 1e-6
