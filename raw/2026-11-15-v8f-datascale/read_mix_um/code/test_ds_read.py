"""test_ds_read.py -- unit tests of the DSCALE10K block-1 read: every committed block-1 outcome reached by synthetic inputs, the cell / floor / reach logic, the bank and tree refusals, the argv builders.
Run:  python -m pytest -q -p no:cacheprovider --basetemp=F:/TanitAD-scratch/datascale/read/bt/t test_ds_read.py       (DS_SLOW=1 adds the two K-RE tests against banked results: torch + the read tree)"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds_lib as L  # noqa: E402
import ds_read as R  # noqa: E402
import ds_pull as P  # noqa: E402
import ds_run as RUN  # noqa: E402
import ds_t0 as T  # noqa: E402

CI = L.load_ci()
SLOW = os.environ.get("DS_SLOW") == "1"


def fs(gain=(), cost=()):
    """family statuses of one seed: every family 'none' except the named ones"""
    d = {f: {"status": "none", "gains": [], "costs": [], "missing": []} for f in R.FAMILIES}
    for f in gain:
        d[f]["status"] = "gain"
    for f in cost:
        d[f]["status"] = "cost"
    return d


# ---------------------------------------------------------------------------------------------------------------------------------------- the committed outcomes (card 6)
def test_no_reach_precedes_everything():
    assert R.outcome_label({0: fs(gain=["LATERAL", "TACTICAL"]), 1: fs(gain=["LATERAL", "TACTICAL"])}, reach_ok=False) == ["NO-REACH"]


@pytest.mark.parametrize("s0,s1,want", [
    (fs(gain=["LATERAL", "TACTICAL"]), fs(gain=["LATERAL", "TACTICAL", "STRATEGIC"]), "DATA-HELPS"),
    (fs(gain=["LATERAL"]), fs(gain=["LATERAL"]), "DATA-HELPS-PARTIAL"),
    (fs(), fs(), "DATA-NEUTRAL"),
    (fs(cost=["LATERAL"]), fs(cost=["LATERAL"]), "DATA-HURTS"),
    (fs(gain=["TACTICAL"], cost=["LATERAL"]), fs(gain=["TACTICAL"], cost=["LATERAL"]), "MIXED"),
    (fs(gain=["LATERAL"]), fs(), "SEEDS-DISAGREE"),                              # a gain at one seed only
    (fs(), fs(cost=["LONGITUDINAL"]), "SEEDS-DISAGREE"),                         # a cost at one seed only
    (fs(cost=["LATERAL"]), fs(cost=["TACTICAL"]), "SEEDS-DISAGREE"),             # costs in different families: not a both-seed cost of one family
    (fs(gain=["LATERAL", "TACTICAL"]), fs(gain=["LATERAL"], cost=["STRATEGIC"]), "SEEDS-DISAGREE"),   # a cost at one seed blocks DATA-HELPS
])
def test_every_outcome_row(s0, s1, want):
    assert R.outcome_label({0: s0, 1: s1}, True) == [want]


def test_blocked_and_one_seed():
    assert R.outcome_label({0: fs()}, True, blocked=["U_s0.A"]) == ["BLOCKED"]
    assert R.outcome_label({0: fs(gain=["LATERAL"])}, True) == ["NOT-YET"]       # block 1 has seed 0 only: no label


def test_seed0_reading_is_provisional_and_names_families():
    r = R.seed0_reading(fs(gain=["LATERAL", "TACTICAL"]), True)
    assert r["label"].startswith("NOT-YET") and r["provisional_if_seed_1_agrees"] == ["DATA-HELPS"] and r["families_gain"] == ["LATERAL", "TACTICAL"]
    assert R.seed0_reading(fs(), False)["provisional_if_seed_1_agrees"] == ["NO-REACH"]


def test_blocks_two_and_three_print_not_yet():
    assert set(L.NOT_YET_ARMS) >= {"B_s1", "U_s1", "UE_s0", "BL_s0", "X_s0"}


# ---------------------------------------------------------------------------------------------------------------------------------------- family status and the cell rule
def row(gain=False, cost=False, status="OK"):
    return {"status": status, "gain": gain, "cost": cost}


def test_family_status_cost_wins_and_missing_listed():
    h = R.HEADLINE["LONGITUDINAL"]
    assert R.family_status({h[0]: row(gain=True), h[1]: row(cost=True)}, h)["status"] == "cost"
    assert R.family_status({h[0]: row(gain=True), h[1]: row()}, h)["status"] == "gain"
    assert R.family_status({h[0]: row(), h[1]: row()}, h)["status"] == "none"
    st = R.family_status({h[0]: row(gain=True)}, h)
    assert st["status"] == "gain" and st["missing"] == [h[1]]
    assert R.family_status({}, h)["status"] == "NOT-YET"
    assert R.family_status({h[0]: row(status="NOT-APPLICABLE")}, h[:1])["status"] == "NOT-YET"


def test_headline_cells_are_the_cards():
    assert R.HEADLINE["LONGITUDINAL"] == ("operative_0_2s:speed_mae_mps", "operative_0_2s:dk_hw_abs_err_m")
    assert R.HEADLINE["LATERAL"] == ("operative_0_2s:yaw_rate_mae_degps", "operative_0_2s:curvature_mae_1pm")
    assert R.HEADLINE["TACTICAL"] == ("TAC-LAT", "TAC-LON") and R.HEADLINE["STRATEGIC"] == ("S1:pos_err_m@80", "S2:skill@80")
    assert all(c in R.CELLS for f in ("LONGITUDINAL", "LATERAL", "TACTICAL") for c in R.HEADLINE[f])


def cell(a_shift, floor, higher=False, n=600, seed=1):
    rng = np.random.default_rng(seed)
    eid = np.repeat(np.arange(40), n // 40)
    b = rng.normal(1.0, 0.3, n)
    a = b + a_shift + rng.normal(0, 0.02, n)
    return R.OR.cell_delta(CI, a, b, eid, higher, floor, 400)


def test_cell_rule_gain_cost_floor():
    assert cell(-0.2, 0.05)["gain"] and not cell(-0.2, 0.05)["cost"]                 # lower is better, separated, above the floor
    assert cell(+0.2, 0.05)["cost"] and not cell(+0.2, 0.05)["gain"]
    assert not cell(-0.2, 0.5)["gain"] and not cell(+0.2, 0.5)["cost"]               # |delta| below the floor: neither
    assert not cell(-0.0005, 0.0)["gain"]                                           # not separated: neither
    c = cell(-0.2, None)
    assert c["status"] == "OK" and not c["gain"] and not c["cost"]                   # no floor: never a gain or a cost (o2)
    assert cell(+0.2, 0.05, higher=True)["gain"] and cell(-0.2, 0.05, higher=True)["cost"]   # accuracy cells: higher is better


def test_strat_pair_uses_skill_higher_and_errors_lower():
    rng = np.random.default_rng(3)
    eid = np.repeat(np.arange(40), 15)
    base = {"S1:pos_err_m@80": rng.normal(5, 1, 600), "S2:skill@80": rng.normal(0.2, 0.1, 600)}
    arm = {"S1:pos_err_m@80": base["S1:pos_err_m@80"] - 1.0, "S2:skill@80": base["S2:skill@80"] + 0.3}
    r = R.strat_pair(CI, arm, base, eid, {"S1:pos_err_m@80": 0.1, "S2:skill@80": 0.05}, 300)
    assert r["S1:pos_err_m@80"]["gain"] and r["S2:skill@80"]["gain"]
    worse = {"S1:pos_err_m@80": base["S1:pos_err_m@80"] + 1.0, "S2:skill@80": base["S2:skill@80"] - 0.3}
    r = R.strat_pair(CI, worse, base, eid, {"S1:pos_err_m@80": 0.1, "S2:skill@80": 0.05}, 300)
    assert r["S1:pos_err_m@80"]["cost"] and r["S2:skill@80"]["cost"]


def test_floors_are_the_replicate_pair_abs_diff():
    rep = {c: np.full(10, 2.0) for c in R.CELLS}
    base = {c: np.full(10, 1.5) for c in R.CELLS}
    f = R.OR.floors_from(rep, base)
    assert all(abs(v - 0.5) < 1e-12 for v in f.values()) and set(f) == set(R.CELLS) and len(R.CELLS) == 24


# ---------------------------------------------------------------------------------------------------------------------------------------- T0: precondition and mechanism label
def d(point, sep):
    return {"point": point, "separated": sep}


def test_reach_rule():
    assert T.decide_reach(0.08, d(0.05, True))["label"] == "REACHED"
    assert T.decide_reach(0.049, d(0.05, True))["label"] == "NO-REACH"               # r below the +5 % bar
    assert T.decide_reach(0.08, d(0.05, False))["label"] == "NO-REACH"               # not separated
    assert T.decide_reach(0.08, d(-0.05, True))["label"] == "NO-REACH"               # the MSE delta sign contradicts r (sign rule)
    assert T.decide_reach(-0.08, d(-0.05, True))["label"] == "NO-REACH"              # R better than U is not a reach


def test_t_label_rule_with_floor():
    assert T.decide_t(-0.06, d(-0.04, True), 0.01)["label"] == "T-DATA-LIMITED"
    assert T.decide_t(-0.049, d(-0.04, True), 0.01)["label"] == "NOT T-DATA-LIMITED"   # r above the -5 % bar
    assert T.decide_t(-0.06, d(-0.04, False), 0.01)["label"] == "NOT T-DATA-LIMITED"
    assert T.decide_t(-0.06, d(+0.04, True), 0.01)["label"] == "NOT T-DATA-LIMITED"    # sign contradicts r
    assert T.decide_t(-0.06, d(-0.004, True), 0.01)["label"] == "NOT T-DATA-LIMITED"   # |delta| not above the stage-A T floor
    assert T.decide_t(-0.06, d(-0.04, True), None)["label"] == "NOT T-DATA-LIMITED"     # no floor: not decidable


def test_eval_name_convention():
    assert T.name_of("B_s0", "A") == "DSBs0A" and T.name_of("B_rep", "A") == "DSBrepA" and T.name_of("R_s0", "A") == "DSRs0A"


# ---------------------------------------------------------------------------------------------------------------------------------------- GTFIX: one constraint frame (gtfix/BUILD_GTFIX.md)
LATC = tuple(range(12))


def _frame_world(seed=0, n=80, nc=26, relabel=None, drop=None):
    """the same physical labels and the SAME physical T prediction, written by two checkpoints with different cons_mu / cons_sd (as the eval does: float32 codes, absent label -> 0,
    absmask zeroes GT-absent slots).  relabel=(i, j, dsd) moves U's label of window i slot j by dsd of B's SD; drop=(i, j) makes it absent in U only."""
    rng = np.random.default_rng(seed)
    has = rng.random(n) > 0.1
    lab = rng.normal(5.0, 3.0, (n, nc))
    lab[rng.random((n, nc)) < 0.4] = np.nan
    pred = np.where(np.isnan(lab), rng.normal(5.0, 3.0, (n, nc)), lab + rng.normal(0.0, 0.7, (n, nc)))
    mB, sB = rng.normal(5.0, 1.0, nc), rng.uniform(2.0, 4.0, nc)
    mU, sU = mB + rng.normal(0.0, 0.8, nc), sB * rng.uniform(0.8, 0.95, nc)
    labU = lab.copy()
    if relabel:
        i, j, dsd = relabel
        labU[i, j] += dsd * sB[j]
    if drop:
        labU[drop] = np.nan

    def arm(lb, m, sd):
        g = np.nan_to_num(((lb - m) / sd).astype(np.float32), nan=0.0).astype(np.float64)
        t = np.where(has[:, None] & np.isnan(lb), 0.0, ((pred - m) / sd).astype(np.float32).astype(np.float64))
        return {"has": has.copy(), "g_cons": g, "t_cons": t, "lat_code": None}
    i_pres = [(i, j) for i in range(n) for j in range(nc) if has[i] and not np.isnan(lab[i, j])]
    return arm(lab, mB, sB), arm(labU, mU, sU), {"B": (mB, sB), "U": (mU, sU)}, lab, pred, has, i_pres


def test_gtfix_same_physical_prediction_reads_r_zero_only_in_one_frame():
    B, U, S, lab, pred, has, _ = _frame_world()
    assert not np.array_equal(U["g_cons"][has], B["g_cons"][has])                  # the refusal's precondition: frames differ, so raw blocks differ
    r_native = T.latcons_rmse(U["t_cons"], U["g_cons"], has, LATC) / T.latcons_rmse(B["t_cons"], B["g_cons"], has, LATC) - 1.0
    assert abs(r_native) > 0.01                                                     # each arm in its own frame manufactures an r from the frame alone
    Z = {("B", 0): B, ("U", 0): U}
    seen = []
    rec = T.to_common_frame(Z, 0, "B", lambda n, s: S[n], lambda a, s, v: seen.append(a))
    assert rec["U"]["converted"] and seen == ["U"] and rec["U"]["max_gt_dev_sd"] < 1e-5
    assert np.array_equal(Z[("U", 0)]["g_cons"], B["g_cons"]) and Z[("U", 0)]["lat_code"] is None
    r = T.latcons_rmse(Z[("U", 0)]["t_cons"], B["g_cons"], has, LATC) / T.latcons_rmse(B["t_cons"], B["g_cons"], has, LATC) - 1.0
    assert abs(r) < 1e-5                                                            # identical physical predictions read r = 0 in the common frame
    mB, sB = S["B"]
    pres = has[:, None] & ~np.isnan(lab)
    assert np.abs(Z[("U", 0)]["t_cons"][pres] - ((pred - mB) / sB)[pres]).max() < 1e-5
    assert (Z[("U", 0)]["t_cons"][has[:, None] & np.isnan(lab)] == 0.0).all()      # absmask's absent-slot zero survives


def test_gtfix_refuses_a_genuine_relabel():
    _, _, _, _, _, _, ip = _frame_world()
    i, j = ip[7]
    B, U, S, *_ = _frame_world(relabel=(i, j, 0.01))
    with pytest.raises(RuntimeError, match=r"still differs on 1 windows / 1 slots"):
        T.to_common_frame({("B", 0): B, ("U", 0): U}, 0, "B", lambda n, s: S[n])


def test_gtfix_refuses_a_presence_change():
    _, _, _, _, _, _, ip = _frame_world()
    B, U, S, *_ = _frame_world(drop=ip[3])
    with pytest.raises(RuntimeError, match=r"GT presence differs .* on 1 windows / 1 slots"):
        T.to_common_frame({("B", 0): B, ("U", 0): U}, 0, "B", lambda n, s: S[n])


def test_gtfix_leaves_equal_blocks_and_same_frame_relabels_untouched():
    B, _, S, *_ = _frame_world()
    B2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in B.items()}

    def no_stats(n, s):
        raise AssertionError("stats read for bit-equal blocks")
    Z = {("B", 0): B, ("B2", 0): B2}
    g, t = B2["g_cons"], B2["t_cons"]
    assert T.to_common_frame(Z, 0, "B", no_stats) == {"B2": {"frame": "native", "converted": False}}
    assert Z[("B2", 0)]["g_cons"] is g and Z[("B2", 0)]["t_cons"] is t             # every earlier read (K-RE, B_rep, R vs U) is untouched
    _, _, _, _, _, _, ip = _frame_world()
    B, U, S, *_ = _frame_world(relabel=(*ip[2], 0.5))
    g = U["g_cons"]
    rec = T.to_common_frame({("B", 0): B, ("U", 0): U}, 0, "B", lambda n, s: S["B"])   # same frame, different block = different labels
    assert not rec["U"]["converted"] and U["g_cons"] is g                           # left for WC.present_error to refuse


def test_init_control_reads_params_equal():
    assert R.init_control({"record": {"B_s0_vs_U_s0_params_equal": True}})["reading"].endswith("data lever alone")
    assert "data plus init" in R.init_control({"record": {"B_s0_vs_U_s0_params_equal": False}})["reading"]
    assert R.init_control(None)["status"] == "NOT-GIVEN"


# ---------------------------------------------------------------------------------------------------------------------------------------- K-RE comparison logic
def test_kre_compare_exact_and_detects_tamper():
    rows = {"c": {"delta": 0.1, "lo": -0.1, "hi": 0.2, "separated": False, "floor": 0.05, "gain": False, "cost": False, "n_windows": 10, "n_episodes": 3}}
    n, bad = R.kre_compare(rows, json.loads(json.dumps(rows)))
    assert n == 9 and not bad
    t = json.loads(json.dumps(rows))
    t["c"]["lo"] = -0.1000001
    assert R.kre_compare(rows, t)[1]
    assert R.kre_compare(rows, {})[0] == 0                                            # nothing compared: the caller fails it


# ---------------------------------------------------------------------------------------------------------------------------------------- bank, tree, disk refusals
def mk_item(tmp, rec_over=None, arm="B_s0", stage="A"):
    d_ = tmp / L.item(arm, stage)
    d_.mkdir(parents=True)
    (d_ / "ckpt_last.pt").write_bytes(b"ckpt-bytes")
    rec = {"arm": arm, "stage": stage, "sha256": L.sha256_file(str(d_ / "ckpt_last.pt")), "source": "hf", "commit": L.COMMIT, "steps": L.STEPS[stage], "seed": 0}
    rec.update(rec_over or {})
    (d_ / "BANK_RECORD.json").write_text(json.dumps(rec))
    return d_


def test_bank_problem_branches(tmp_path):
    assert L.bank_problem("B_s0", "A", str(tmp_path)).startswith("REFUSED: checkpoint missing")
    d_ = mk_item(tmp_path)
    assert L.bank_problem("B_s0", "A", str(tmp_path)) is None
    for kw, frag in (({"arm": "U_s0"}, "names U_s0.A"), ({"commit": "e81bde01"}, "commit"), ({"steps": 3000}, "steps"), ({"source": "x"}, "sha256 / source"), ({"sha256": "0" * 64}, "differs")):
        dd = mk_item(tmp_path / ("x" + frag[:3]), kw)
        assert frag in L.bank_problem("B_s0", "A", str(tmp_path / ("x" + frag[:3])))
    (d_ / "BANK_RECORD.json").unlink()
    assert "BANK_RECORD.json missing" in L.bank_problem("B_s0", "A", str(tmp_path))


def test_pull_expected_sha_from_banker_not_from_file(tmp_path):
    sha = "ab" * 32
    bd = tmp_path / "bank"
    bd.mkdir()
    (bd / "B_s0.A.json").write_text(json.dumps({"sha256": sha, "hf_repo": "x/y", "hf_path": "B_s0/A/ckpt_last.pt"}))
    assert P.expected_sha("B_s0", "A", str(bd), lambda p: None)[:2] == (sha, "hf")
    (bd / "B_s0.A.json").write_text(json.dumps({"sha256": "zz", "hf_repo": "x/y", "hf_path": "p"}))
    assert P.expected_sha("B_s0", "A", str(bd), lambda p: None)[0] is None             # a malformed banker record is refused, not guessed
    pod = {f"{L.POD_ROOT}/bank_ok/B_s0.A.sha256": sha, f"{L.POD_ROOT}/bank_ready/B_s0.A.txt": f"/p {sha} 2026"}
    assert P.expected_sha("B_s0", "A", None, pod.get)[:2] == (sha, "pod")
    pod[f"{L.POD_ROOT}/bank_ready/B_s0.A.txt"] = f"/p {'cd' * 32} 2026"
    assert P.expected_sha("B_s0", "A", None, pod.get)[0] is None                       # bank_ok and bank_ready disagree
    assert P.expected_sha("B_s0", "A", None, lambda p: None)[0] is None                # not banked


def test_pull_verify_removes_a_tampered_copy(tmp_path):
    d_ = tmp_path / "B_s0.A"
    d_.mkdir()
    (d_ / "ckpt_last.pt").write_bytes(b"abc")
    good = L.sha256_file(str(d_ / "ckpt_last.pt"))
    assert P.verify_and_record("B_s0", "A", str(d_), "0" * 64, "hf", 19164).startswith("REFUSED")
    assert not (d_ / "ckpt_last.pt").exists() and not (d_ / "BANK_RECORD.json").exists()
    (d_ / "ckpt_last.pt").write_bytes(b"abc")
    assert P.verify_and_record("B_s0", "A", str(d_), good, "hf", 19164) is None
    assert json.load(open(d_ / "BANK_RECORD.json"))["sha256"] == good


def test_disk_gate():
    assert L.disk_problem(free=24.9) and L.disk_problem(free=25.0) is None and L.disk_problem(free=140) is None


def test_tree_pins_hold_or_refuse(tmp_path):
    assert L.tree_problem(str(tmp_path)) and "tree file missing" in L.tree_problem(str(tmp_path))
    if os.path.isdir(L.TREE):
        assert L.tree_problem() is None


# ---------------------------------------------------------------------------------------------------------------------------------------- dump and eval argv
def test_t1_argv_is_the_card_tier_without_nav_files():
    a, od, tag = RUN.t1_argv("U_s0", "B")
    s = " ".join(a)
    assert tag == "DS-U_s0-B_m10_causal_heads" and a[2] == "run" and "v8f_t1_eval.py" in a[1]
    for tok in ("--replan-steps 10", "--nav deadreckon", "--scene logged", "--limit causal", "--dump-heads", "--device cpu"):
        assert tok in s
    assert "--nav-text" not in s and "--max-windows" not in s


def test_mech_argvs_are_the_four_evals_of_the_instrument():
    j = dict(RUN.mech_argvs("B_s0", "A", "OUT"))
    assert list(j) == ["DSBs0A_s0", "DSBs0A_s0-tacabs", "DSBs0A_s0-tacorc", "DSBs0A_s0-tacorc-all-latcons"]
    assert j["DSBs0A_s0"][-2:] == ["--project", "--families"] and j["DSBs0A_s0-tacabs"][-2:] == ["--tac", "absmask"] and j["DSBs0A_s0-tacorc"][-2:] == ["--tac", "oracle"]
    assert j["DSBs0A_s0-tacorc-all-latcons"][-6:] == ["--tac", "oracle", "--tac-groups", "all-latcons", "--tac-alpha", "1"]
    assert all("v8f_mini_eval.py" in v[1] for v in j.values())


def test_route_argvs_variants():
    assert RUN.route_argv("U_s0", "B", "real", "o")[1][-2:] == ["--ages", "fresh"]
    assert RUN.route_argv("U_s0", "B", "donor", "o")[1][-1] == "--nav-shuffle" and RUN.route_argv("U_s0", "B", "zero", "o")[1][-1] == "--nav-withheld"


def test_variant_tree_map_is_explicit_and_pinned():
    """zero and realx run on the STRAT1 tree (e81bde01 / 008f48b7), real and donor on the code-of-record export; both trees are pinned in READ_PINS.json"""
    assert L.VARIANT_TREE == {"real": "read", "donor": "read", "zero": "zero", "realx": "zero", "donorx": "zero"}
    assert L.tree_of("real") == L.TREE and L.tree_of("donor") == L.TREE and L.tree_of("zero") == L.TREE_ZERO and L.tree_of("realx") == L.TREE_ZERO
    assert RUN.route_argv("U_s0", "B", "zero", "o")[1][1].replace("\\", "/").startswith(L.TREE_ZERO) and RUN.route_argv("U_s0", "B", "real", "o")[1][1].replace("\\", "/").startswith(L.TREE)
    pins = json.load(open(L.HERE + "/READ_PINS.json"))
    assert pins["tree_zero_sha256"]["stack/scripts/v8f_mini_eval.py"].startswith("008f48b7") and pins["tree_sha256"]["stack/scripts/v8f_mini_eval.py"].startswith("50c70ae8")
    if os.path.isdir(L.TREE_ZERO):
        assert L.tree_problem(L.TREE_ZERO, "tree_zero_sha256") is None


@pytest.mark.skipif(not os.path.isdir(L.TREE_ZERO) or not os.path.isdir(L.TREE), reason="needs both trees")
def test_every_route_variant_argv_is_accepted_by_its_evaluator(monkeypatch):
    """the reach test the argv-string test was not: each variant's argv goes through ITS tree's own argparse (dry parse)"""
    assert RUN.route_argv_problems() == []
    # mutation 1: the old routing (zero on the read tree, whose evaluator has no --nav-withheld) must be RED
    monkeypatch.setitem(L.VARIANT_TREE, "zero", "read")
    bad = RUN.route_argv_problems()
    assert bad and bad[0].startswith("zero:") and "--nav-withheld" in bad[0]
    monkeypatch.setitem(L.VARIANT_TREE, "zero", "zero")
    # mutation 2: zero mapped to an unknown flag must be RED
    monkeypatch.setitem(RUN.VARIANT_FLAGS, "zero", ["--nav-bogus-flag"])
    bad = RUN.route_argv_problems()
    assert bad and bad[0].startswith("zero:") and "--nav-bogus-flag" in bad[0]


def test_kevalx_compare_exact_and_detects_one_ulp(tmp_path):
    a = {"pred": np.arange(6, dtype=np.float32).reshape(2, 3), "eid": np.array([1, 2])}
    b = {k: v.copy() for k, v in a.items()}
    b["nav_variant"] = np.asarray("real")
    for n, d in (("a", a), ("b", b)):
        np.savez(tmp_path / f"{n}.npz", **d)
        json.dump({"x": 1.5, "n": {"y": [1, 2]}, "tag": n}, open(tmp_path / f"{n}.json", "w"))
    ok, diffs, extra = RUN.kevalx_compare(tmp_path / "a.npz", tmp_path / "a.json", tmp_path / "b.npz", tmp_path / "b.json")
    assert ok and diffs == [] and extra == ["nav_variant"]
    b["pred"] = np.nextafter(b["pred"], np.float32(10))            # one ulp: atol 0 must see it
    np.savez(tmp_path / "b.npz", **b)
    ok, diffs, _ = RUN.kevalx_compare(tmp_path / "a.npz", tmp_path / "a.json", tmp_path / "b.npz", tmp_path / "b.json")
    assert not ok and "pred" in diffs[0]


def test_reader_refuses_zero_passes_without_a_kevalx_pass(tmp_path):
    assert "missing" in R.kevalx_problem(str(tmp_path))
    items = {f"{a}.B {v}": {"identical": True} for a in ("B_s0", "U_s0", "B_rep") for v in ("real", "donor")}
    json.dump({"pass": True, "items": items}, open(tmp_path / "KEVALX.json", "w"))
    assert R.kevalx_problem(str(tmp_path)) is None
    items["U_s0.B real"]["identical"] = False
    json.dump({"pass": False, "items": items}, open(tmp_path / "KEVALX.json", "w"))
    assert R.kevalx_problem(str(tmp_path))


def test_preflight_names_every_unbanked_item(monkeypatch, tmp_path):
    monkeypatch.setattr(L, "CK", str(tmp_path))
    monkeypatch.setattr(RUN.L, "disk_problem", lambda *a, **k: None)
    monkeypatch.setattr(RUN.L, "tree_problem", lambda *a, **k: None)
    monkeypatch.setattr(RUN, "import_check", lambda: None)
    bad = RUN.preflight([("B_s0", "A"), ("R_s0", "A")])
    assert set(bad) == {"B_s0.A", "R_s0.A"}
    assert RUN.parse_items([]) == list(L.READ_ITEMS) and len(L.READ_ITEMS) == 7 and ("R_s0", "B") not in L.READ_ITEMS


def test_missing_dumps_write_not_yet(tmp_path):
    out = tmp_path / "r.json"
    rc = R.main(["--dumps", str(tmp_path / "none"), "--out", str(out)])
    r = json.load(open(out))
    assert rc == 1 and r["outcome"] == ["NOT-YET"] and "T1 dumps missing" in r["reason"] and "B_s1" in r["not_yet"]


# ---------------------------------------------------------------------------------------------------------------------------------------- K-RE against banked results (slow: torch + the read tree)
@pytest.mark.skipif(not SLOW, reason="DS_SLOW=1")
def test_kre_t0_dscale_banked():
    sys.path[:0] = [L.TREE + "/stack", L.TREE + "/taniteval"]
    k = R.kre_t0(HERE + "/kre", HERE + "/ref/dscale_read.json", 2000)
    assert k["pass"], k


@pytest.mark.skipif(not SLOW, reason="DS_SLOW=1")
def test_kre_t1_strat1_banked():
    sys.path[:0] = [L.TREE + "/stack", L.TREE + "/taniteval"]
    import obd_geom as G
    k = R.kre_t1(CI, G.build(), L.S1R + "/t1", L.S1R + "/out/RESULT_STRAT1.json", 2000)
    assert k["pass"], k
