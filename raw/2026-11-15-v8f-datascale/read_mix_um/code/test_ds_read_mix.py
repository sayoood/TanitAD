"""test_ds_read_mix.py -- unit tests of the DSCALE mixture-arm read (UM_s0): the deciding label of card 5.4 reached by PLANTED TAC-LAT rows (each label, both signs, Delta exactly at F_cell, the UM - BL upper end exactly at 0, mutual
exclusivity), the precedence BLOCKED > CONTROL-FAILED > NOT-YET > label, K-TACSRC under amendment UM-A1 on the REAL summary and replay record (and every field planted wrong), the UM-A1 line beside every UM label, the gate / Adam-state
check, the card / amendment / replay sha256 pins, the beside labels, the block "mix" constants and checkpoint roots, make_banker_mix accepted chain state, sync_refs, K-RE helpers and the launcher.
Run:  env -u DS_BLOCK python -m pytest -q -p no:cacheprovider --basetemp=F:/TanitAD-scratch/datascale/read_mix/bt/t test_ds_read_mix.py   (whole file, CUDA_VISIBLE_DEVICES=-1)"""
import itertools
import json
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds_lib as L  # noqa: E402
import ds_read_mix as M  # noqa: E402
import make_banker_mix as MB  # noqa: E402
import sync_refs as SY  # noqa: E402

PYX = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
CARD = "F:/TanitAD-scratch/datascale/card_mix"


def in_mix(code):
    r = subprocess.run([PYX, "-c", "import sys; sys.path.insert(0, " + repr(HERE) + "); " + code], env=dict(os.environ, DS_BLOCK="mix", CUDA_VISIBLE_DEVICES="-1"), capture_output=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr[-400:]
    return r.stdout.strip()


F = 0.0767


def row(delta, lo=None, hi=None, sep=None, floor=F):
    """a TAC-LAT row; the interval defaults to a width of 0.04 around delta; separated defaults to: the interval excludes 0"""
    lo = delta - 0.02 if lo is None else lo
    hi = delta + 0.02 if hi is None else hi
    return {"status": "OK", "delta": delta, "lo": lo, "hi": hi, "separated": (lo > 0 or hi < 0) if sep is None else sep, "floor": floor}


ZERO = row(0.0, 0.0, 0.0, False)


# ------------------------------------------------------------------------------------------------------------------------ the deciding label (card 5.4)
def test_confirmed_needs_a_gain_over_ue_and_um_bl_not_separated_worse():
    assert M.label_mix(row(0.0855), ZERO) == "DILUTION-CONFIRMED"                     # the BL stand-in answer
    assert M.label_mix(row(0.0855), row(-0.01, -0.05, 0.03)) == "DILUTION-CONFIRMED"
    assert M.label_mix(row(0.0855), row(0.05, 0.01, 0.09)) == "DILUTION-CONFIRMED"    # UM - BL better is still not separated worse


def test_um_bl_upper_end_exactly_zero_is_not_separated_worse():
    assert M.label_mix(row(0.0855), row(-0.04, -0.08, 0.0, True)) == "DILUTION-CONFIRMED"
    assert M.label_mix(row(0.0855), row(-0.04, -0.08, -1e-9, True)) == "DILUTION-PARTIAL"


def test_partial_when_the_gain_is_inside_f_or_um_bl_still_worse():
    assert M.label_mix(row(0.05, 0.02, 0.08), ZERO) == "DILUTION-PARTIAL"            # separated better, Delta <= F_cell
    assert M.label_mix(row(F, F - 0.02, F + 0.02), ZERO) == "DILUTION-PARTIAL"       # Delta exactly at F_cell is not MORE than F_cell
    assert M.label_mix(row(F + 1e-6, 0.02, 0.1), ZERO) == "DILUTION-CONFIRMED"
    assert M.label_mix(row(0.0855), row(-0.06, -0.09, -0.03)) == "DILUTION-PARTIAL"  # gain over UE but UM - BL separated worse


def test_not_dilution_when_um_ue_is_not_separated_either_way():
    assert M.label_mix(row(0.04, -0.01, 0.09), ZERO) == "NOT-DILUTION"
    assert M.label_mix(ZERO, ZERO) == "NOT-DILUTION"
    assert M.label_mix(row(-0.04, -0.09, 0.01), row(-0.3, -0.4, -0.2)) == "NOT-DILUTION"   # UM - BL does not matter when UM - UE is not separated


def test_rebalance_hurts_when_um_ue_is_separated_worse():
    assert M.label_mix(row(-0.05, -0.08, -0.02), ZERO) == "REBALANCE-HURTS"
    assert M.label_mix(row(-0.5), row(0.3)) == "REBALANCE-HURTS"


def test_unreadable_rows_are_not_yet_never_a_label():
    for bad in (None, {}, {"status": "NOT-APPLICABLE"}, row(float("nan")), {"status": "OK", "delta": 0.1, "lo": None, "hi": None, "separated": True, "floor": F}):
        assert M.label_mix(bad, ZERO) == "NOT-YET" and M.label_mix(row(0.1), bad) == "NOT-YET"


def test_labels_are_mutually_exclusive_and_cover_every_grid_case():
    seen = set()
    for d, w, hi_bl in itertools.product((-0.2, -0.0767, -0.02, 0.0, 0.02, 0.0767, 0.0768, 0.2), (0.01, 0.1), (-0.5, -1e-9, 0.0, 0.3)):
        lo, hi = d - w, d + w
        lab = M.label_mix(row(d, lo, hi), row(0.0, -1.0, hi_bl))
        assert lab in M.LABELS
        seen.add(lab)
        sep = lo > 0 or hi < 0
        if d > 0 and sep and d > F:
            assert lab == ("DILUTION-CONFIRMED" if hi_bl >= 0 else "DILUTION-PARTIAL")
        elif d > 0 and sep:
            assert lab == "DILUTION-PARTIAL"
        elif d < 0 and sep:
            assert lab == "REBALANCE-HURTS"
        else:
            assert lab == "NOT-DILUTION"
    assert seen == set(M.LABELS)


def test_row_helpers():
    assert M.row_gain(row(0.1)) and not M.row_gain(row(0.05, 0.01, 0.09)) and not M.row_gain(row(0.1, floor=None)) and not M.row_gain(row(-0.1))
    assert M.row_sep_better(row(0.05, 0.01, 0.09)) and not M.row_sep_better(row(0.05, -0.01, 0.09)) and M.row_sep_worse(row(-0.05, -0.09, -0.01)) and not M.row_sep_worse(row(-0.05, -0.09, 0.01))


# ------------------------------------------------------------------------------------------------------------------------ precedence
def test_precedence_blocked_over_control_failed_over_not_yet_over_label():
    for lab in M.LABELS + ("NOT-YET",):
        assert M.outcome_mix(lab, ["K-TACSRC"], ["K1"], ["x"]) == ["BLOCKED"]
        assert M.outcome_mix(lab, ["K-TACSRC"], [], []) == ["BLOCKED"]
    assert M.outcome_mix("DILUTION-CONFIRMED", None, ["K-RE"], ["ADE6"])[0].startswith("CONTROL-FAILED") and "not quotable" in M.outcome_mix("DILUTION-CONFIRMED", None, ["K-RE"], [])[0]
    assert M.outcome_mix("DILUTION-CONFIRMED", None, [], ["ADE6"]) == ["NOT-YET"] and M.outcome_mix(None, None, [], []) == ["NOT-YET"] and M.outcome_mix("NOT-YET", None, [], []) == ["NOT-YET"]
    for lab in M.LABELS:
        assert M.outcome_mix(lab, None, [], []) == [lab]
        assert M.outcome_mix(lab, [], [], []) == [lab]


def test_the_um_a1_line_is_beside_every_label_and_names_the_original_reading():
    t = M.tag_um("DILUTION-CONFIRMED")
    assert "DILUTION-CONFIRMED" in t and "read under UM-A1 (PI-accepted 2026-11-18)" in t and "10 > 5, BLOCKED as committed" in t
    assert M.A1_TAG == "read under UM-A1 (PI-accepted 2026-11-18)"


# ------------------------------------------------------------------------------------------------------------------------ K-TACSRC under UM-A1 (real files)
def _real():
    return (json.load(open(CARD + "/um_replay/UM_s0.A.summary.json", encoding="utf-8")), json.load(open(CARD + "/um_replay/um_empty_replay.json", encoding="utf-8")))


def test_ktacsrc_passes_under_um_a1_on_the_real_summary_and_fails_as_committed():
    s, r = _real()
    k = M.ktacsrc(s, r)
    assert k["pass_under_UM_A1"] and not k["problems_under_UM_A1"]
    assert not k["as_committed"]["pass"] and k["as_committed"]["line"] == "tac_src_empty_batches 10 > 5, BLOCKED as committed"
    assert k["fields"]["tac_src_empty_batches"] == 10 and k["fields"]["tac_src_rows_listed"] == 1223523 and k["fields"]["tac_src_rows_other"] == 2708437


@pytest.mark.parametrize("key,val", [("tac_src_w", 0.5), ("tac_src_w", None), ("tac_src_list_sha256", "0" * 64), ("tac_src_weight_share_listed", 0.99),
                                     ("tac_src_rows_other", 0), ("tac_src_rows_other", 2708438), ("tac_src_rows_listed", 1223524), ("tac_src_empty_batches", 11), ("tac_src_empty_batches", 9),
                                     ("tac_src_empty_batches", 0)])
def test_ktacsrc_every_field_planted_wrong_fails(key, val):
    s, r = _real()
    s = dict(s, **{key: val})
    assert not M.ktacsrc(s, r)["pass_under_UM_A1"]


def test_ktacsrc_a_missing_field_or_a_replay_that_differs_fails():
    s, r = _real()
    s2 = {k: v for k, v in s.items() if k != "tac_src_empty_batches"}
    assert not M.ktacsrc(s2, r)["pass_under_UM_A1"] and not M.ktacsrc(s, dict(r, empty_batches_replayed=9))["pass_under_UM_A1"] and not M.ktacsrc(s, {})["pass_under_UM_A1"] and not M.ktacsrc(None, r)["pass_under_UM_A1"]
    assert not M.ktacsrc(s, dict(r, rows_other_replayed=1))["pass_under_UM_A1"] and not M.ktacsrc(s, dict(r, rows_listed_replayed=1))["pass_under_UM_A1"]


def test_ktacsrc_as_committed_passes_only_with_a_small_count():
    s, r = _real()
    k = M.ktacsrc(dict(s, tac_src_empty_batches=5), dict(r, empty_batches_replayed=5))
    assert k["as_committed"]["pass"] and k["pass_under_UM_A1"]
    assert not M.ktacsrc(dict(s, tac_src_empty_batches=6), dict(r, empty_batches_replayed=6))["as_committed"]["pass"]


# ------------------------------------------------------------------------------------------------------------------------ gate, Adam state, pins
def test_gate_and_post_problems():
    good_checks = {"C-ARGV": {"pass": True, "problems": []}, "G-LIVE": {"pass": True}}
    good_post = {"registered": 252, "with_adam_state": 252, "problems": []}
    assert M.gate_problems(good_checks, good_post) == []
    assert any("G-LIVE" in p for p in M.gate_problems({"C-ARGV": {"pass": True}, "G-LIVE": {"pass": False}}, good_post))
    assert M.gate_problems({}, good_post) and M.gate_problems(None, good_post)
    for bad in ({"registered": 252, "with_adam_state": 251, "problems": []}, {"registered": 253, "with_adam_state": 253, "problems": []}, {"registered": 252, "with_adam_state": 252, "problems": ["x"]}, None):
        assert M.gate_problems(good_checks, bad)


def test_banked_evidence_files_are_a_pass():
    ev = HERE + "/evidence"
    assert os.path.isfile(ev + "/UM_s0.A.checks.json"), "run make_banker_mix.py first"
    assert M.gate_problems(json.load(open(ev + "/UM_s0.A.checks.json", encoding="utf-8")), json.load(open(ev + "/UM_s0.A.post.json", encoding="utf-8"))) == []


def test_rulings_pins_hold_on_the_real_card_and_refuse_a_changed_file(tmp_path):
    assert M.rulings_problem(CARD) is None
    d = tmp_path / "c"
    shutil.copytree(CARD, d, ignore=shutil.ignore_patterns("share_probe*"))
    assert M.rulings_problem(str(d)) is None
    with open(d / "AMENDMENT_UM_A1_KTACSRC.md", "ab") as f:
        f.write(b" ")
    assert "AMENDMENT" in M.rulings_problem(str(d))
    shutil.copy(CARD + "/AMENDMENT_UM_A1_KTACSRC.md", d / "AMENDMENT_UM_A1_KTACSRC.md")
    with open(d / "CARD_DSCALE_MIX.md", "ab") as f:
        f.write(b" ")
    assert "CARD" in M.rulings_problem(str(d))
    shutil.copy(CARD + "/CARD_DSCALE_MIX.md", d / "CARD_DSCALE_MIX.md")
    with open(d / "um_replay" / "um_empty_replay.json", "ab") as f:
        f.write(b" ")
    assert "um_empty_replay" in M.rulings_problem(str(d))
    os.remove(d / "um_replay" / "um_empty_replay.json")
    assert "missing" in M.rulings_problem(str(d))


def test_pairs_are_the_cards():
    assert M.PAIRS["UM-UE.A"] == ("UM_s0", "UE_s0") and M.PAIRS["UM-BL.A"] == ("UM_s0", "BL_s0") and M.PAIRS["UM-B_s0.A"] == ("UM_s0", "B_s0") and M.PAIRS["UM-U_s0.A"] == ("UM_s0", "U_s0")
    assert M.PAIRS["UE-BL.A"] == ("UE_s0", "BL_s0") and M.PAIRS["U-B_s0.A"] == ("U_s0", "B_s0") and M.PAIRS["BL-B_s0.A"] == ("BL_s0", "B_s0")
    assert M.KRE_PAIRS == ("UE-BL.A", "U-B_s0.A", "BL-B_s0.A") and M.CELL == "TAC-LAT" and M.LIST_SHA.startswith("37f4f9ee") and M.EMPTY_BOUND == 5
    assert set(M.MECH_REF) == set(M.PAIRS)


# ------------------------------------------------------------------------------------------------------------------------ beside labels
def _fam(status, costs=(), gains=()):
    return {"status": status, "costs": list(costs), "gains": list(gains), "missing": []}


def test_strat_kept_rule():
    assert M.strat_kept(_fam("none"), _fam("gain")) is True and M.strat_kept(_fam("gain"), _fam("gain")) is True
    assert M.strat_kept(_fam("cost"), _fam("gain")) is False and M.strat_kept(_fam("none"), _fam("none")) is False and M.strat_kept(_fam("none"), _fam("cost")) is False
    assert M.strat_kept(_fam("NOT-YET"), _fam("gain")) is None and M.strat_kept(_fam("none"), _fam("NOT-YET")) is None


def test_tac_lon_price_and_other_family_costs():
    assert M.tac_lon_price({"status": "OK", "cost": True}) and not M.tac_lon_price({"status": "OK", "cost": False}) and not M.tac_lon_price(None) and not M.tac_lon_price({"status": "NOT-APPLICABLE", "cost": True})
    fs = {"LONGITUDINAL": _fam("cost", ["operative_0_2s:speed_mae_mps"]), "LATERAL": _fam("none"), "TACTICAL": _fam("cost", ["TAC-LAT"]), "STRATEGIC": _fam("none")}
    assert M.other_family_costs(fs) == {"LONGITUDINAL": ["operative_0_2s:speed_mae_mps"]}


# ------------------------------------------------------------------------------------------------------------------------ constants and roots
def test_mix_constants():
    assert L.MIX_ITEMS == (("UM_s0", "A"),) and L.STEPS_ARM["UM_s0"] == 61568 and L.steps_of("UM_s0", "A") == 61568 and L.ARMS["UM_s0"] == 0 and L.STAGES_OF["UM_s0"] == ("A",)
    assert L.steps_of("UE_s0", "A") == 61568 and L.steps_of("B_s0", "A") == 19164 and L.MIX_PULL_ITEMS == (("UM_s0", "A"), ("UE_s0", "A"), ("BL_s0", "A"))
    import ds_t0 as T
    assert T.name_of("UM_s0") == "DSUMs0A" and len({T.name_of(a) for a in M.ARMS_ALL}) == 6


def test_mix_defaults_and_checkpoint_roots():
    out = in_mix("import ds_lib as L; print(L.BLOCK); print(L.DEFAULT_ITEMS); print(L.ck_root('UM_s0')); print(L.ck_root('UE_s0')); print(L.ck_root('BL_s0')); print(L.ck_root('B_s0')); print(L.ck_root('U_s0')); print(L.ck_root('B_rep'))").splitlines()
    h = os.path.normpath(HERE + "/ckpts").replace("\\", "/")
    assert out[0] == "mix" and out[1] == "(('UM_s0', 'A'),)"
    assert [x.replace("\\", "/") for x in out[2:5]] == [h, h, h] and [x.replace("\\", "/") for x in out[5:8]] == [L.B1_ROOT + "/ckpts"] * 3
    other = subprocess.run([PYX, "-c", "import sys; sys.path.insert(0, " + repr(HERE) + "); import ds_lib as L; print(len(L.DEFAULT_ITEMS)); print(L.BLOCK); print(L.ck_root('B_s0'))"], env={k: v for k, v in os.environ.items() if k != "DS_BLOCK"},
                           capture_output=True, encoding="utf-8").stdout.splitlines()
    assert other[0] == "7" and other[1] == "1" and other[2].replace("\\", "/") == os.path.normpath(HERE + "/ckpts").replace("\\", "/")


def _fake_ck(tmp_path, arm, steps, sha=None):
    d = tmp_path / L.item(arm, "A")
    d.mkdir(parents=True)
    (d / "ckpt_last.pt").write_bytes(b"ckpt-bytes-" + arm.encode())
    sha = sha or L.sha256_file(str(d / "ckpt_last.pt"))
    (d / "BANK_RECORD.json").write_text(json.dumps({"arm": arm, "stage": "A", "sha256": sha, "source": "hf", "commit": L.COMMIT, "steps": steps, "seed": 0}), encoding="utf-8")
    return str(tmp_path)


def test_bank_problem_wants_61568_steps_for_um(tmp_path):
    assert L.bank_problem("UM_s0", "A", _fake_ck(tmp_path / "ok", "UM_s0", 61568)) is None
    why = L.bank_problem("UM_s0", "A", _fake_ck(tmp_path / "short", "UM_s0", 19164))
    assert why and "61568" in why
    assert "sha256" in L.bank_problem("UM_s0", "A", _fake_ck(tmp_path / "bad", "UM_s0", 61568, sha="0" * 64))
    assert L.bank_problem("B_s0", "A", _fake_ck(tmp_path / "b61", "B_s0", 61568)) is not None


def test_route_and_dump_defaults_for_the_mix_block():
    out = in_mix("import ds_lib as L, ds_run as D; print(D.ROUTE_STAGE); print(D.parse_items([])); print(D.parse_items([], route=True)); j = D.route_jobs(D.parse_items([], route=True), 'o'); print(len(j)); "
                 "print(sorted(x[0] for x in j)); z = [x for x in j if x[0] == 'DS-UM_s0-A_zero'][0]; print(z[2] == L.TREE_ZERO, '--nav-withheld' in z[1]); r = [x for x in j if x[0] == 'DS-UM_s0-A_real'][0]; print(r[2] == L.TREE)").splitlines()
    assert out[0] == "A" and out[1] == "[('UM_s0', 'A')]" and out[2] == "[('UM_s0', 'A')]" and out[3] == "5"
    assert "DS-UM_s0-A_donorx" in out[4] and out[5] == "True True" and out[6] == "True"
    blk1 = subprocess.run([PYX, "-c", "import sys; sys.path.insert(0, " + repr(HERE) + "); import ds_run as D; print(D.ROUTE_STAGE)"], env={k: v for k, v in os.environ.items() if k != "DS_BLOCK"}, capture_output=True, encoding="utf-8").stdout.strip()
    assert blk1 == "B"


def test_pull_defaults_in_the_mix_block():
    out = in_mix("import ds_pull as P, ds_lib as L; print([L.item(a, s) for a, s in L.MIX_PULL_ITEMS]); print(P.GIT_BASH)").splitlines()
    assert out[0] == "['UM_s0.A', 'UE_s0.A', 'BL_s0.A']" and out[1].replace("\\", "/") == "C:/Program Files/Git/usr/bin/bash.exe"


# ------------------------------------------------------------------------------------------------------------------------ K-EVALX / route files
def _route_dir(p, arms, variants=("real", "donor", "zero", "realx", "donorx")):
    p.mkdir(parents=True, exist_ok=True)
    for a in arms:
        for v in variants:
            (p / f"eval_DS-{a}-A_{v}.npz").write_bytes(b"x")
    return str(p)


def _kx(p, arms, ok=True, drop=None):
    items = {f"{a}.A {v}": {"identical": ok} for a in arms for v in ("real", "donor")}
    if drop:
        items.pop(drop)
    (p / "KEVALX.json").write_text(json.dumps({"pass": ok, "items": items}), encoding="utf-8")


def test_kevalx_problem_covers_exactly_the_named_arms(tmp_path):
    assert M.kevalx_problem(str(tmp_path), ("UM_s0",)) == "KEVALX.json missing"
    _kx(tmp_path, ("UM_s0",))
    assert M.kevalx_problem(str(tmp_path), ("UM_s0",)) is None and "not a PASS" in M.kevalx_problem(str(tmp_path), M.ARMS_REF)
    _kx(tmp_path, ("UM_s0",), drop="UM_s0.A donor")
    assert "not a PASS" in M.kevalx_problem(str(tmp_path), ("UM_s0",))
    _kx(tmp_path, ("UM_s0",), ok=False)
    assert "not a PASS" in M.kevalx_problem(str(tmp_path), ("UM_s0",))
    (tmp_path / "r").mkdir()
    _kx(tmp_path / "r", M.ARMS_REF)
    assert M.kevalx_problem(str(tmp_path / "r"), M.ARMS_REF) is None


def test_route_missing_lists_every_absent_variant(tmp_path):
    d = _route_dir(tmp_path / "a", M.ARMS_REF)
    assert M.route_missing(d, M.ARMS_REF) == [] and len(M.route_missing(str(tmp_path / "none"), ("UM_s0",))) == 5
    os.remove(tmp_path / "a" / "eval_DS-BL_s0-A_donorx.npz")
    assert M.route_missing(d, M.ARMS_REF) == ["BL_s0.A donorx"]
    d2 = _route_dir(tmp_path / "b", M.ARMS_REF[:4])
    assert all(x.startswith("BL_s0") for x in M.route_missing(d2, M.ARMS_REF))


# ------------------------------------------------------------------------------------------------------------------------ K-RE helpers
def test_kre_ade6_and_floor_equal_and_differing():
    rec = {"B_s0": 2.1, "U_s0": 2.2, "F_A": 0.08}
    assert M.kre_ade6(rec, dict(rec, extra=9.9))["pass"]
    assert not M.kre_ade6(dict(rec, U_s0=2.2000001), rec)["pass"] and not M.kre_ade6(rec, {})["pass"]
    assert M.kre_floor({"a": 0.1, "b": 0.2}, {"a": 0.1, "b": 0.2})["pass"] and not M.kre_floor({"a": 0.1, "b": 0.3}, {"a": 0.1, "b": 0.2})["pass"] and not M.kre_floor({"a": 0.1}, {"a": 0.1, "b": 0.2})["pass"]


def _row2(gain=False, cost=False):
    return {"status": "OK", "gain": gain, "cost": cost, "delta": 0.1, "n_windows": 5}


def test_family_status_is_per_family_and_fam_rows_add_strategic_only_where_present():
    rows = {c: _row2() for f in M.FAMILIES for c in M.HEADLINE[f]}
    rows[M.HEADLINE["LATERAL"][0]] = _row2(gain=True)
    rows[M.HEADLINE["TACTICAL"][1]] = _row2(cost=True)
    st = M.pair_family_status(rows)
    assert st["LATERAL"]["status"] == "gain" and st["TACTICAL"]["status"] == "cost" and st["LONGITUDINAL"]["status"] == "none"
    del rows[M.HEADLINE["STRATEGIC"][0]], rows[M.HEADLINE["STRATEGIC"][1]]
    assert M.pair_family_status(rows)["STRATEGIC"]["status"] == "NOT-YET"
    pairs = {"UM-UE.A": {"a": _row2()}, "UE-BL.A": {"a": _row2()}}
    out = M.fam_rows_all(pairs, {"status": "OK", "pairs": {"UM-UE.A": {"S1:pos_err_m@80": _row2(gain=True)}}})
    assert "S1:pos_err_m@80" in out["UM-UE.A"] and set(out["UE-BL.A"]) == {"a"} and set(M.fam_rows_all(pairs, {"status": "NOT-YET"})["UM-UE.A"]) == {"a"}


def test_missing_dumps_write_a_tagged_non_label(tmp_path):
    out = tmp_path / "r.json"
    rc = M.main(["--dumps0", str(tmp_path / "x"), "--dumps3", str(tmp_path / "y"), "--dumpsm", str(tmp_path / "z"), "--out", str(out), "--b3-result", str(tmp_path / "nob3.json"),
                 "--um-summary", CARD + "/um_replay/UM_s0.A.summary.json", "--banker", str(tmp_path / "nobanker.json"), "--evidence", str(tmp_path / "noev")])
    r = json.load(open(out))
    assert rc == 1 and r["outcome"][0].startswith(("BLOCKED", "NOT-YET")) and "read under UM-A1" in r["outcome"][0] and "T1 dumps missing" in r["reason"] and "UM_s0.A" in r["reason"] and r["block"] == "mix"
    assert "K-TACSRC" in r["blocked_items"] or "gate / post" in r["blocked_items"]


# ------------------------------------------------------------------------------------------------------------------------ make_banker_mix: the ONE accepted chain state
GOOD_LOG = ("2026-11-18T12:43:21Z BLOCKUM queued (lock /root/gpu_queue.lock)\n12:43:21 block UM start (free 149 GB, root /root/dscale10k)\n"
            '12:43:36 um_checks pre: {"check": "pre", "pass": true, "problems": []}\n12:43:36 run UM_s0 stage A\n'
            'ZZ DSCALE10K BLOCKUM FAIL ZZ UM_s0 post check refused: {"check": "post", "pass": false, "problems": ["K-TACSRC tac_src_empty_batches 10 > 5"]} \n')


def test_chain_state_accepts_exactly_the_one_ktacsrc_refusal():
    assert MB.chain_state_problem(GOOD_LOG) is None


@pytest.mark.parametrize("edit,frag", [
    (lambda s: s.replace("ZZ DSCALE10K BLOCKUM FAIL ZZ", "ZZ DSCALE10K BLOCKUM DONE ZZ"), "DONE"),
    (lambda s: s + "ZZ DSCALE10K BLOCKUM DONE ZZ\n", "DONE"),
    (lambda s: s.replace('["K-TACSRC tac_src_empty_batches 10 > 5"]', '["K-TACSRC tac_src_empty_batches 10 > 5", "K-TACSRC tac_src_w 0.5"]'), "exactly"),
    (lambda s: s.replace('["K-TACSRC tac_src_empty_batches 10 > 5"]', '["K-TACSRC tac_src_empty_batches 11 > 5"]'), "exactly"),
    (lambda s: s.replace('["K-TACSRC tac_src_empty_batches 10 > 5"]', '["Adam state missing"]'), "exactly"),
    (lambda s: s.replace('"pass": false', '"pass": true'), "exactly"),
    (lambda s: s.replace('"check": "pre", "pass": true', '"check": "pre", "pass": false'), "pre"),
    (lambda s: s.replace("run UM_s0 stage A", "run nothing"), "stage A"),
    (lambda s: "\n".join(ln for ln in s.splitlines() if "ZZ" not in ln), "sentinel"),
    (lambda s: s + "ZZ DSCALE10K BLOCKUM FAIL ZZ again {}\n", "sentinels"),
    (lambda s: s.split("ZZ DSCALE10K")[0] + "ZZ DSCALE10K BLOCKUM FAIL ZZ no json here\n", "JSON"),
    (lambda s: "", "sentinel"),
])
def test_chain_state_refuses_every_other_state(edit, frag):
    why = MB.chain_state_problem(edit(GOOD_LOG))
    assert why and frag in why


def _led(**kw):
    r = {"tag": "UM_s0.A", "status": "BANKED", "repo": MB.REPO, "revision": "eae5836ec5877f765810aa470608f8b4d16b7aca", "sha256": "39de0b05" + "a" * 56, "t": "x",
         "files": {"UM_s0/A/ckpt_last.pt": [1, "39de0b05" + "a" * 56], "UM_s0/A/summary.json": [2, "b" * 64]}}
    r.update(kw)
    return r


def test_banker_record_and_hub_checks():
    assert MB.record_problem(_led()) is None and MB.record_problem(None) and MB.record_problem(_led(tag="UE_s0.A")) and MB.record_problem(_led(repo="x/y")) and MB.record_problem(_led(revision=""))
    assert MB.record_problem(_led(sha256="39de0b05" + "c" * 56))
    assert MB.record_problem(_led(sha256="0" * 64, files={"UM_s0/A/ckpt_last.pt": [1, "0" * 64], "UM_s0/A/summary.json": [2, "b" * 64]}))
    assert MB.record_problem(_led(files={"UM_s0/A/ckpt_last.pt": [1, "39de0b05" + "a" * 56]}))
    rec = _led()
    hub = {"UM_s0/A/ckpt_last.pt": rec["sha256"], "UM_s0/A/summary.json": "z"}
    assert MB.hub_problem(hub, True, rec) is None and MB.hub_problem(hub, False, rec) and MB.hub_problem({"UM_s0/A/summary.json": "z"}, True, rec) and MB.hub_problem(dict(hub, **{"UM_s0/A/ckpt_last.pt": "d" * 64}), True, rec)
    b = MB.banker_json(rec)
    assert b["sha256"] == rec["sha256"] and b["hf_path"] == "UM_s0/A/ckpt_last.pt" and b["hf_summary_path"] == "UM_s0/A/summary.json" and b["summary_sha256"] == "b" * 64 and b["hf_rev"] == rec["revision"]
    led = MB.parse_ledger(json.dumps(_led()) + "\ngarbage\n" + json.dumps(_led(status="PENDING", tag="X.A")))
    assert led["UM_s0.A"]["revision"].startswith("eae5836e") and "X.A" not in led


def test_make_banker_mix_main_writes_nothing_on_a_refused_chain(tmp_path):
    log = tmp_path / "chain.log"
    log.write_text(GOOD_LOG.replace("FAIL", "DONE"), encoding="utf-8")
    led = tmp_path / "led.jsonl"
    led.write_text(json.dumps(_led()), encoding="utf-8")
    args = ["--chain-log", str(log), "--ledger", str(led), "--no-hub", "--bank-dir", str(tmp_path / "b"), "--evidence", str(tmp_path / "e")]
    assert MB.main(args) == 1 and not (tmp_path / "b").exists()
    log.write_text(GOOD_LOG, encoding="utf-8")
    assert MB.main(args) == 0
    assert json.load(open(tmp_path / "b" / "UM_s0.A.json"))["sha256"] == _led()["sha256"]


# ------------------------------------------------------------------------------------------------------------------------ sync_refs
def test_sync_refs_plan_copies_verifies_and_refuses_a_differing_target(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    plan = SY.plan(str(src))
    assert len(plan) == 48 and any("DSUEs0A" in f for f in plan) and any("DSBLs0A" in f for f in plan) and any("DSRs0A" in f for f in plan) and not any("DSUMs0A" in f for f in plan)
    for f in plan:
        (src / f).write_bytes(f.encode())
    c, s, bad = SY.sync(str(src), str(dst))
    assert c == 48 and s == 0 and not bad
    c, s, bad = SY.sync(str(src), str(dst))
    assert c == 0 and s == 48 and not bad
    (dst / plan[0]).write_bytes(b"different")
    c, s, bad = SY.sync(str(src), str(dst))
    assert bad and "REFUSED" in bad[0] and (dst / plan[0]).read_bytes() == b"different"
    os.remove(src / plan[5])
    assert any("source missing" in b for b in SY.sync(str(src), str(tmp_path / "d2"))[2])


# ------------------------------------------------------------------------------------------------------------------------ the launcher
def test_launcher_keeps_the_bars_the_order_the_sentinels_and_the_checks():
    sh = open(HERE + "/launch_dscale_mix.sh", encoding="utf-8", newline="").read()
    assert '[ "$FREE" -lt 25 ]' in sh and "ZZ DSCALE MIX READ DONE ZZ" in sh and "ZZ DSCALE MIX READ FAIL ZZ" in sh and "DS_BLOCK=mix" in sh
    assert "$WC 5.5 30" in sh and 'ds_read_mix.py --out "$R/out/RESULT_DSCALE_MIX.json"' in sh and "env -u DS_BLOCK" in sh and "test_ds_read_mix.py test_ds_read.py" in sh
    order = [sh.index(x) for x in ("rulings_problem", "FREE=$(df", "pytest", "make_banker_mix.py >>", "ds_pull.py >>", "ds_pull.py --check-only", "sync_refs.py >>", "ds_run.py dumps", "ds_run.py mech", "ds_run.py route", "ds_read_mix.py --out")]
    assert order == sorted(order)
    assert "ds_pull.py --check-only --items UM_s0.A UE_s0.A BL_s0.A B_s0.A U_s0.A B_rep.A" in sh and "|| true" not in sh
    assert "RESULT_DSCALE10K_B3.json" in sh and "KEVALX.json" in sh
    assert "CreateFlags = 0x08000200" in open(HERE + "/launch_wmi_mix.ps1", encoding="utf-8").read()
    cmd = open(HERE + "/job_dscale_mix.cmd", encoding="utf-8").read()
    assert "Git" + chr(92) + "usr" + chr(92) + "bin" + chr(92) + "bash.exe" in cmd and "launch_dscale_mix.sh" in cmd
    assert chr(13) not in sh


def test_the_reader_keeps_the_pins():
    src = open(HERE + "/ds_read_mix.py", encoding="utf-8").read()
    for frag in ("0bc241f5c4e2292a56e4e8405305c76f8f18d82988c90485604d54f8e2702df0", "b2b2c4f0d189359e19d60ff5619ecfe36ab92b5c41d800d9630560ba6ce860d6", "37f4f9ee468e0872134074a82d63c2915458596fdbcf3d0fcf5b09f1fe0d8cc1"):
        assert frag in src


# ------------------------------------------------------------------------------------------------------------------------ the instrument finds the re-pulled UE / BL checkpoints (their copied evals name the deleted read3 paths)
def test_remap_verified_finds_only_a_sha_verified_checkpoint_of_this_folder(tmp_path, monkeypatch):
    import ds_t0 as T
    ck = tmp_path / "ck"
    monkeypatch.setattr(L, "CK", str(ck))
    _fake_ck(ck, "UE_s0", 61568)
    rec = "F:/TanitAD-scratch/datascale/read3/ckpts/UE_s0.A/ckpt_last.pt"
    got = T._remap_verified(rec.lower())
    assert got and got.replace("\\", "/").endswith("/ck/UE_s0.A/ckpt_last.pt") and os.path.isfile(got)
    assert T._remap_verified("F:/x/read3/ckpts/BL_s0.A/ckpt_last.pt") is None                                    # not pulled
    assert T._remap_verified("F:/x/read3/ckpts/XX_s0.A/ckpt_last.pt") is None and T._remap_verified("F:/x/read3/other/UE_s0.A/ckpt_last.pt") is None and T._remap_verified("F:/x/read3/ckpts/UE_s0.A/other.pt") is None
    with open(ck / "UE_s0.A" / "ckpt_last.pt", "ab") as f:
        f.write(b"x")
    assert T._remap_verified(rec) is None                                                                        # sha256 differs from the bank record: never used


# ------------------------------------------------------------------------------------------------------------------------ added after the first mutation run (each closes a survivor)
def test_mech_reference_frames_are_exactly_the_reference_arms_of_each_pair():
    assert M.MECH_REF == {"UM-UE.A": "UE_s0", "UM-BL.A": "BL_s0", "UM-B_s0.A": "B_s0", "UM-U_s0.A": "U_s0", "UE-BL.A": "BL_s0", "U-B_s0.A": "B_s0", "BL-B_s0.A": "B_s0"}
    assert all(M.MECH_REF[p] == M.PAIRS[p][1] for p in M.PAIRS)


def test_ktacsrc_rows_other_must_be_positive_even_when_the_replay_agrees():
    s, r = _real()
    k = M.ktacsrc(dict(s, tac_src_rows_other=0), dict(r, rows_other_replayed=0))
    assert not k["pass_under_UM_A1"] and any("rows_other" in p and "> 0" in p for p in k["problems_under_UM_A1"])


def test_post_check_needs_both_counts_to_be_252():
    ok = {"C": {"pass": True}}
    assert M.gate_problems(ok, {"registered": 252, "with_adam_state": 252, "problems": []}) == []
    assert M.gate_problems(ok, {"registered": 253, "with_adam_state": 252, "problems": []})
    assert M.gate_problems(ok, {"registered": 252, "with_adam_state": 253, "problems": []})


def test_a_lateral_family_cost_is_named():
    fs = {"LONGITUDINAL": _fam("none"), "LATERAL": _fam("cost", ["operative_0_2s:yaw_rate_mae_degps"]), "TACTICAL": _fam("none"), "STRATEGIC": _fam("none")}
    assert M.other_family_costs(fs) == {"LATERAL": ["operative_0_2s:yaw_rate_mae_degps"]}
    assert M.other_family_costs({"LONGITUDINAL": _fam("gain", [], ["x"]), "LATERAL": _fam("none"), "TACTICAL": _fam("cost", ["TAC-LAT"]), "STRATEGIC": _fam("cost", ["S1"])}) == {}


def test_the_pull_command_without_items_names_um_ue_and_bl_in_the_mix_block(tmp_path):
    r = subprocess.run([PYX, os.path.join(HERE, "ds_pull.py"), "--check-only", "--min-free", "0"], cwd=HERE, env=dict(os.environ, DS_BLOCK="mix", DS_CKPTS=str(tmp_path), CUDA_VISIBLE_DEVICES="-1"), capture_output=True, encoding="utf-8")
    items = [ln.split()[0] for ln in r.stdout.splitlines() if ln.strip() and ln.split()[0].endswith(".A")]
    assert items == ["UM_s0.A", "UE_s0.A", "BL_s0.A"] and r.returncode == 1                                 # nothing pulled into the empty folder: refused, and the three items are the ones named


def test_the_route_default_is_um_alone_in_the_mix_block():
    out = in_mix("import ds_lib as L; print(L.ROUTE_ITEMS_MIX); print(L.MIX_ITEMS)").splitlines()
    assert out[0] == out[1] == "(('UM_s0', 'A'),)"


def test_the_launcher_checks_the_block3_kevalx_record_fail_closed():
    sh = open(HERE + "/launch_dscale_mix.sh", encoding="utf-8", newline="").read()
    assert "grep -q '\"pass\": true' $D/read3/out/t0route/KEVALX.json || fail" in sh and "grep -q '\"pass\": true' $D/read/out/t0route/KEVALX.json || fail" in sh
