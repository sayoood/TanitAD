"""test_lfo_read.py -- the LFO reader (code/lfo_read.py): the oracle instruments, the tool-vs-card tick controls for the logged-speed input, the A4 outcome labels and their precedence, the rollouts and flags, the production controls on a
SYNTHETIC out root, and the two known answers (ka-lf: the banked RILF075 / RILF075C arrays fed in the LFO slot reproduce LF075's P / G / F numbers and rows exactly and read attribution 0.0 on every row; ka-null: the banked RIC0 /
RIC0C arrays fed in the LFO slot read share 0.0 and every contrast row exactly 0.0). Every test is a mutation target (run_mutations_lfo_read.py). Env override: LFO_READ_PATH."""
import importlib.util
import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


LO = _load("lfo_read_under_test", os.environ.get("LFO_READ_PATH", f"{PK}/code/lfo_read.py"))


@pytest.fixture(scope="module", autouse=True)
def inited():
    LO.init()
    return LO


N = 7197


def ticks(n=N, speed=6.0, **kw):
    """A consistent synthetic LFO tick dump: logged route speed ``speed`` (6 m/s) everywhere, v_hat := speed (the oracle), v_f = 0.75 * speed, nothing binds, the heads said 5 m/s."""
    gt = np.zeros((n, 30, 2))
    gt[:, :, 0] = (np.arange(1, 31) * 0.2 * speed)[None, :]
    t = {"lf_bind": np.zeros(n, bool), "lf_cap": np.zeros(n, bool), "lf_vf": np.full(n, 0.75 * speed, np.float32), "lf_vhat": np.full(n, speed, np.float32), "lf_pi": np.full(n, 0.2, np.float32),
         "lf_gap": np.full(n, 20.0, np.float32), "lf_lift": np.zeros(n, np.float32), "lf_vhat_heads": np.full(n, 5.0, np.float32), "v0": np.full(n, 6.0), "gt": gt, "gt_ok": np.ones(n, bool)}
    t.update(kw)
    return t


def with_no_route(t, idx):
    """Ticks ``idx`` have no complete logged route: gt NaN, gt_ok False, no floor (v_hat and v_f NaN)."""
    t = {k: np.array(v, copy=True) for k, v in t.items()}
    t["gt"][idx] = np.nan
    t["gt_ok"][idx] = False
    t["lf_vhat"][idx] = np.nan
    t["lf_vf"][idx] = np.nan
    return t


def test_oracle_block_known_values():
    idx = np.arange(0, N, 10)
    t = with_no_route(ticks(), idx)
    o = LO.oracle_block(t)
    assert o["n_no_route_ticks"] == len(idx) and abs(o["share_no_route_ticks"] - len(idx) / N) < 1e-12
    assert o["lf_vhat_equals_logged_speed_max_abs_mps"] < 1e-5 and o["lf_vhat_nan_exactly_where_no_route"] is True
    assert o["shortfall_check_max_abs_mps"] < 1e-5 and abs(o["mean_shortfall_vlog_minus_vf_uncapped_mps"] - 1.5) < 1e-5            # v_log - v_f = (1 - 0.75) * 6 = 1.5
    h = o["heads_estimate_on_this_arms_frames"]
    assert abs(h["mae_mps"] - 1.0) < 1e-5 and h["share_heads_exceed_logged"] == 0.0 and h["n"] == N - len(idx)
    bad = ticks(lf_vhat=np.full(N, 7.0, np.float32))
    assert LO.oracle_block(bad)["lf_vhat_equals_logged_speed_max_abs_mps"] == pytest.approx(1.0, abs=1e-6)
    r = LO.tick_instruments(ticks())                                                                  # the A3 instruments plus the oracle block
    assert r["n_ticks"] == N and r["oracle"]["n_no_route_ticks"] == 0 and "v_f_vs_logged_note" in r


def test_tick_controls_catch_a_tool_that_deviates_from_the_card():
    assert all(LO.tick_controls(ticks()).values())
    assert all(LO.tick_controls(with_no_route(ticks(), np.arange(0, N, 7))).values())
    for name, bad in (("v_hat_is_the_logged_speed_of_the_dumped_gt", ticks(lf_vhat=np.full(N, 7.0, np.float32))),
                      ("no_floor_where_no_complete_route", {**with_no_route(ticks(), np.arange(0, N, 7)), "lf_vf": np.full(N, 4.5, np.float32)}),
                      ("no_bind_below_the_3_mps_band", ticks(speed=2.0, lf_bind=np.ones(N, bool))),
                      ("v_f_is_kappa_v_log_where_the_cap_did_not_bind", ticks(lf_vf=np.full(N, 3.0, np.float32))),
                      ("v_f_between_0_and_kappa_v_hat", ticks(lf_vf=np.full(N, 5.5, np.float32))),
                      ("cap_only_where_presence_at_least_half", ticks(lf_cap=np.ones(N, bool))), ("lift_zero_where_unbound", ticks(lf_lift=np.full(N, 1.0, np.float32))),
                      ("n_ticks_is_3x2399", ticks(100)), ("pi_in_unit_interval", ticks(lf_pi=np.full(N, 1.5, np.float32))), ("pi_finite_every_tick", ticks(lf_pi=np.full(N, np.nan, np.float32)))):
        assert LO.tick_controls(bad)[name] is False, name
    assert LO.tick_controls({k: v for k, v in ticks().items() if k != "lf_vf"}) == {"fields_present": False}
    assert LO.tick_controls({k: v for k, v in ticks().items() if k != "lf_vhat_heads"}) == {"fields_present": False}


def _P(s0, s1):
    return {"PASS": bool(s0 and s1)}


def _G(ok, rows=(), ade_in=True):
    return {"PASS": ok, "per_seed": {"s0": {"ade6_delta": 0.01 if ade_in else 0.3, "ade6_within_guard": ade_in, "non_crawl_separated_worse": list(rows)}, "s1": {"ade6_delta": 0.01, "ade6_within_guard": True, "non_crawl_separated_worse": []}}}


def test_outcome_labels_every_branch_and_their_precedence():
    reach_ok, reach_bad = {"PASS": True}, {"PASS": False}
    d = LO.decide(reach_ok, _P(True, True), _G(True))
    assert d["label"] == "CEILING-CLEARS" and d["estimator_levers_signalled"] is True and "A5" in d["next"] and "degradation sweep" in d["next"] and "A40" in d["next"]
    d = LO.decide(reach_ok, _P(True, True), _G(False, ("DK-HW", "speed MAE"), ade_in=False))
    assert d["label"] == "CEILING-HARMS" and d["estimator_levers_signalled"] is False and d["rows"] == ["ADE6", "DK-HW", "speed MAE"] and "NOT signalled" in d["next"] and "operator variant" in d["next"]
    d = LO.decide(reach_ok, _P(True, False), _G(True))                                                # P_O fails at either seed
    assert d["label"] == "CEILING-SHORT" and d["estimator_levers_signalled"] is False and "T2 re-perception" in d["next"] and "kappa 1" in d["next"]
    d = LO.decide(reach_ok, _P(False, False), _G(False, ("ADE6",)))                                   # P_O fails: SHORT even when G_O fails too
    assert d["label"] == "CEILING-SHORT"
    d = LO.decide(reach_bad, _P(True, True), _G(True))                                                # the reach control comes first
    assert d["label"] == "F-UNREACHED" and d["estimator_levers_signalled"] is False and "instrument is checked first" in d["next"]
    assert LO.decide(None, _P(True, True), _G(True))["label"] == "CEILING-CLEARS"                    # known-answer runs have no reach control
    assert {LO.decide(reach_ok, _P(True, True), _G(True))["label"], LO.decide(reach_ok, _P(False, True), _G(True))["label"], LO.decide(reach_ok, _P(True, True), _G(False))["label"], LO.decide(reach_bad)["label"]} == set(LO.LABELS)


def test_failing_rows_name_the_harm():
    G = _G(False, ("DK-VIOL", "along-track MAE"), ade_in=False)
    assert LO.failing_rows(G) == ["ADE6", "DK-VIOL", "along-track MAE"]
    assert LO.failing_rows(_G(True)) == []


def test_rollouts_and_flags_copy_the_banked_base_plus_floor_plus_the_logged_source():
    r = LO.rollouts()
    assert [x[0] for x in r] == ["RILFO075_s0", "RILFO075_s1", "RILFO075C_s0", "RILFO075C_s1"]
    assert [(x[3], x[4]) for x in r] == [("reindexed", "off")] * 2 + [("reindexed", "kbox")] * 2
    banked = json.load(open("F:/TanitAD-scratch/motion_c/ri/out/done_RIC0_s0.txt", encoding="utf-8"))["flags"]
    bankedc = json.load(open("F:/TanitAD-scratch/motion_c/ri/out/done_RIC0C_s0.txt", encoding="utf-8"))["flags"]
    fl = LO.expected_flags("RILFO", 0)
    assert fl.startswith(banked + " --level-floor on --floor-kappa 0.75 --floor-ckpt ") and fl.endswith("/M2b_s0_recft_LLR/ckpt_last.pt --floor-vhat-source logged")
    flc = LO.expected_flags("RILFOC", 1)
    assert flc.startswith(bankedc + " --level-floor on --floor-kappa 0.75 --floor-ckpt ") and flc.endswith("--floor-vhat-source logged") and "/M2b_s1_recft_LLR/" in flc
    assert LO.name_of("RILFO") == "RILFO075" and LO.name_of("RILFOC") == "RILFO075C"


def _synthetic_out_root(tmp, mutate=None, tool="a" * 64, tick_kw=None):
    os.makedirs(tmp, exist_ok=True)
    out = str(tmp / "out")
    pins = str(tmp / "PINS.txt")
    open(pins, "w", encoding="utf-8").write(f"patched_tool={tool}\n")
    for tag, role, s, scene, feas in LO.rollouts():
        os.makedirs(f"{out}/{tag}/ticks", exist_ok=True)
        np.savez(f"{out}/{tag}/ticks/ticks_{tag}.npz", **ticks(**(tick_kw or {})))
        hdr = {"scene": scene, "feasible": ({"mode": feas} if feas != "off" else None), "smooth": {"mode": LO.BASE_SMOOTH}, "n_windows": 2399, "kin_mode": "yaw", "level_floor": "on", "floor_kappa": LO.KAPPA,
               "floor_ckpt": LO.floor_ckpt(s), "floor_ckpt_sha256": LO.PCO.llr_sha_of(s), "floor_vhat_source": "logged", "label": "CEILING", "level_floor_detail": {"vision_only": "NO. CEILING diagnostic"}}
        done = {"tag": tag, "flags": LO.expected_flags(role, s), "ckpt": f"{LO.RDB}/M2b_s{s}_recft_C0/ckpt_last.pt", "ckpt_sha256": LO.PCO.C0_SHA[s], "tool_sha256": tool, "threads": 12}
        if mutate and tag == mutate[0]:
            mutate[1](hdr, done)
        json.dump({"t1v8f": hdr}, open(f"{out}/{tag}/t1v8f_{tag}.json", "w", encoding="utf-8"))
        json.dump(done, open(f"{out}/done_{tag}.txt", "w", encoding="utf-8"))
    return out, pins


def _precheck_file(raw):
    os.makedirs(raw, exist_ok=True)
    json.dump({"PASS": True, "seeds": {f"s{s}": {"C0": {"ckpt_sha256": LO.PCO.C0_SHA[s]}, "LLR": {"ckpt_sha256": LO.PCO.llr_sha_of(s)}} for s in (0, 1)}}, open(f"{raw}/LFO_PRECHECK.json", "w", encoding="utf-8"))
    return raw


def test_production_controls_pass_on_a_consistent_out_root_and_refuse_each_deviation(inited, tmp_path):
    raw = _precheck_file(str(tmp_path / "raw"))
    out, pins = _synthetic_out_root(tmp_path / "ok")
    ctl = LO.run_controls(out, raw, pins)
    assert all(v["PASS"] for k, v in ctl.items()), {k: v for k, v in ctl.items() if not v["PASS"]}
    cases = {
        "RILFO075_s1": lambda h, d: h.__setitem__("floor_kappa", 0.5),
        "RILFO075C_s0": lambda h, d: h.__setitem__("floor_ckpt_sha256", "0" * 64),
        "RILFO075_s0": lambda h, d: h.__setitem__("floor_vhat_source", "heads"),
        "RILFO075C_s1": lambda h, d: h.pop("label"),
        "RILFO075_s1_flags": lambda h, d: d.__setitem__("flags", d["flags"].replace(" --floor-vhat-source logged", "")),
        "RILFO075C_s0_planner": lambda h, d: d.__setitem__("ckpt_sha256", "1" * 64),
        "RILFO075_s1_smoothflags": lambda h, d: d.__setitem__("flags", d["flags"].replace("--smooth sg72", "--smooth off")),
    }
    for key, fn in cases.items():
        tag = key if key in ("RILFO075_s1", "RILFO075C_s0", "RILFO075_s0", "RILFO075C_s1") else key.rsplit("_", 1)[0]
        out2, pins2 = _synthetic_out_root(tmp_path / f"m_{key}", mutate=(tag, fn))
        assert LO.run_controls(out2, raw, pins2)[f"rollout_{tag}"]["PASS"] is False, key
    out, pins = _synthetic_out_root(tmp_path / "t")
    open(pins, "w", encoding="utf-8").write("patched_tool=" + "b" * 64 + "\n")
    assert LO.run_controls(out, raw, pins)["rollout_RILFO075_s0"]["PASS"] is False                 # the done marker's tool sha256 is not the pinned patched_tool
    open(pins, "w", encoding="utf-8").write("# no key\n")
    assert LO.run_controls(out, raw, pins)["patched_tool_pin"]["PASS"] is False


def test_ka_lf_runs_green_and_reproduces_lf075_numbers_and_reads_attribution_zero(inited, tmp_path):
    assert LO.main(["--mode", "ka-lf", "--raw", str(tmp_path), "--work-dir", str(tmp_path / "w")]) == 0
    d = json.load(open(f"{tmp_path}/KA_READER_ka-lf.json", encoding="utf-8"))
    assert d["KA_PASS"] is True and len(d["ka_checks"]) >= 24 and d["arm"]["outcome"]["label"] == "CEILING-SHORT"
    assert any("attribution" in k and v for k, v in d["ka_checks"].items()) and d["arm"]["rules"]["ceiling_of_D"]["counted"] is False
    md = open(f"{tmp_path}/KA_READER_ka-lf_RESULTS.md", encoding="utf-8").read()
    assert "CEILING" in md and "never counted" in md and "ATTRIBUTION" in md and "KNOWN-ANSWER RUN" in md
    assert "DOMINANT_COUNTED" not in json.dumps(d["arm"]["rules"])


def test_ka_null_runs_green_share_zero_and_every_row_exactly_zero(inited, tmp_path):
    assert LO.main(["--mode", "ka-null", "--raw", str(tmp_path), "--work-dir", str(tmp_path / "w")]) == 0
    d = json.load(open(f"{tmp_path}/KA_READER_ka-null.json", encoding="utf-8"))
    assert d["KA_PASS"] is True and len(d["ka_checks"]) >= 17 and d["arm"]["outcome"]["label"] == "CEILING-SHORT"


def test_known_answer_files_in_the_package_are_pass():
    for m, n in (("ka-lf", 24), ("ka-null", 17)):
        d = json.load(open(f"{PK}/raw/KA_READER_{m}.json", encoding="utf-8"))
        assert d["KA_PASS"] is True and len(d["ka_checks"]) >= n


def test_a_negative_chord_lift_where_bound_is_reported_not_refused():
    """A3_RULING_2 carried over: where F binds, a negative chord lift is a property of the frozen operator, reported beside, never a refusal."""
    t = ticks(lf_bind=np.ones(N, bool), lf_lift=np.where(np.arange(N) % 10 == 0, -0.2, 0.5).astype(np.float32))
    assert LO.tick_controls(t)["lift_zero_where_unbound"] is True
    r = LO.tick_instruments(t)["lift_where_bound_overall"]
    assert r["n_negative_chord_lift"] == 720 and abs(r["share_negative_chord_lift"] - 720 / N) < 1e-9 and abs(r["min_lift_mps"] + 0.2) < 1e-6


def test_headline_carries_the_ceiling_label_and_never_a_counted_dominant():
    arm = {"outcome": LO.decide({"PASS": True}, _P(True, True), _G(True)), "rules": {"P_O": {"PASS": True, "per_seed": {"s0": {"share_closed": 0.6}, "s1": {"share_closed": 0.7}}},
                                                                                    "G_O": {"PASS": True, "per_seed": {"s0": {"ade6_delta": 0.01}, "s1": {"ade6_delta": 0.02}}}}}
    h = LO.headline(arm)
    assert h.startswith("CEILING-CLEARS (CEILING diagnostic)") and "NOT admissible at inference" in h and "not a counted DOMINANT" in h
    assert "CEILING: the level input is the human's logged route speed" in h and LO.CEILING_NOTE in h
    assert LO.headline({"outcome": LO.decide({"PASS": False}), "rules": None}).startswith("F-UNREACHED (CEILING diagnostic)")


def test_production_main_runs_end_to_end_on_synthetic_rollouts_and_real_arrays(inited, tmp_path, monkeypatch):
    """The production path of main(): controls on a synthetic out root, the instruments from its tick dumps, the reach control, the rules from the (existing) RILF075 arrays standing in for the LFO arrays, RESULTS.md. The extractor is replaced
    by the stand-in arrays (it needs real dumps); everything else is the production code."""
    raw = _precheck_file(str(tmp_path / "raw"))
    m1 = np.arange(N) % 10 == 0                                                                     # M1 ticks: own speed 4 < 5.2 <= logged speed 6; F binds on all of them
    out, pins = _synthetic_out_root(tmp_path / "o", tick_kw=dict(v0=np.where(m1, 4.0, 6.0), lf_bind=m1))
    monkeypatch.setattr(LO, "LFO_BUILD_PINS", pins)
    monkeypatch.setattr(LO, "extract_new", lambda out_root, arr_dir, workers: {f"{LO.name_of(r)}_s{s}": f"{LO.LF_ARR}/RILF075{'C' if r.endswith('C') else ''}_s{s}.npz" for _t, r, s, _sc, _f in LO.rollouts()})
    rc = LO.main(["--mode", "production", "--out-root", out, "--raw", raw, "--work-dir", str(tmp_path / "w"), "--results-md", str(tmp_path / "RESULTS.md"), "--n-boot", "100"])
    assert rc == 0
    d = json.load(open(f"{raw}/LFO_READ.json", encoding="utf-8"))
    assert d["label"] == "CEILING" and d["arm"]["outcome"]["label"] in LO.LABELS and d["arm"]["reach"]["PASS"] is True and all(v["PASS"] for v in d["controls"].values())
    assert set(d["instruments"]) == {"RILFO075_s0", "RILFO075_s1", "RILFO075C_s0", "RILFO075C_s1"} and d["instruments"]["RILFO075_s0"]["bind_rate_in_m1_ticks"]["rate"] == 1.0
    assert d["instruments_lf075_beside"] is None or set(d["instruments_lf075_beside"]) == {"RILF075_s0", "RILF075_s1"}
    md = open(f"{tmp_path}/RESULTS.md", encoding="utf-8").read()
    assert "CEILING" in md and "## Instruments" in md and "ATTRIBUTION" in md and "## Controls" in md and "never counted" in md and "Variance named" in md
    assert "KNOWN-ANSWER" not in md and "DOMINANT_COUNTED" not in json.dumps(d["arm"]["rules"])
    # a failing control REFUSES the read (exit 3) and leaves LFO_READ.json with the reason
    out2, pins2 = _synthetic_out_root(tmp_path / "o2", tool="b" * 64, tick_kw=dict(v0=np.where(m1, 4.0, 6.0), lf_bind=m1))
    monkeypatch.setattr(LO, "LFO_BUILD_PINS", pins2)
    open(pins2, "w", encoding="utf-8").write("patched_tool=" + "c" * 64 + "\n")
    raw2 = _precheck_file(str(tmp_path / "raw2"))
    assert LO.main(["--mode", "production", "--out-root", out2, "--raw", raw2, "--work-dir", str(tmp_path / "w2"), "--results-md", str(tmp_path / "R2.md"), "--n-boot", "100"]) == 3
    assert "controls failed" in json.load(open(f"{raw2}/LFO_READ.json", encoding="utf-8"))["refused"] and not os.path.exists(f"{tmp_path}/R2.md")
