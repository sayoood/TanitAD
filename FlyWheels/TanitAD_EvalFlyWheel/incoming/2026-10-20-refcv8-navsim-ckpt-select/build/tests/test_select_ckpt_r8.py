"""LITERAL tests for ``code/select_ckpt_r8.py`` -- the D-R8-CKPT-RULE decision script (TANITAD VENV or any python with numpy+pytest).

Every expected value is a LITERAL (a number, a string or a boolean), never an expression over the code under test.  The worlds in
``worlds.py`` are built so the right answer is known by construction:

    constant +0.02 for 40k at both seeds        -> CHOOSE_EMA_40000, delta 0.02, interval [0.02, 0.02]
    one seed whose interval straddles 0         -> default
    margin inside 2x the inference floor        -> default (and 1x would have chosen 40k)
    stand-in / missing token / < 1000 tokens    -> INCOMPLETE, default, numbers withheld

``tests/test_mutation_select_ckpt_r8.py`` re-runs this file with ``R8SEL_MUTATION`` set; the tests named in its CASES table must go RED.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

import select_ckpt_r8 as S
import worlds as W

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(os.path.dirname(HERE), "code")


@pytest.fixture(scope="module")
def ci():
    return S.load_ci()


def decide(w, ci, **kw):
    kw.setdefault("beside", False)
    return S.decide(w.tables, w.sample, w.log_of, ci, **kw)


# ---------------------------------------------------------------------------------------------------------------------------
# the registered constants, as literals
# ---------------------------------------------------------------------------------------------------------------------------
def test_constants_are_the_registered_literals():
    assert S.N_BOOT == 2000
    assert S.ALPHA == 0.05
    assert S.MIN_TOKENS == 1000
    assert S.FLOOR_MULT == 2.0
    assert S.SEEDS == (0, 1)
    assert S.CHALLENGER == "ema40000" and S.DEFAULT == "ema48852"
    assert S.CKPT_MD5 == {"ema40000": "57807456bd7169800be159ff95871d22", "ema48852": "b4364029b68f29a33eeb4aebbbbd002b"}
    assert S.RT_DEPLOY_SHA256 == "3319b003fecc83577430f07dad762bce0a9a0b8594e9748e9a61bc4d264136b7"
    assert S.RT_FROZEN_SHA256 == "2aacc9ff32d3d2a5a18dda2d72ab5ce6c42d1c1268e9186912c5151bbe4d6d80"


def test_mutation_list_matches_the_module():
    assert set(S.MUTATIONS) == {"sign_flip", "seed0_only", "drop_floor_gate", "floor_1x", "margin_mean", "positional_join",
                                "token_cluster", "no_separation_needed", "standin_allowed", "min_tokens_off",
                                "missing_token_ignored", "identity_unchecked", "shuffle_control_identity", "controls_not_vetoing",
                                "alpha_90", "n_boot_200", "incomplete_chooses_40k"}


def test_ci_is_loaded_by_path_and_is_the_paired_cluster_bootstrap(ci):
    assert os.path.basename(ci.__r8sel_path__) == "ci.py"
    assert len(ci.__r8sel_sha256__) == 64
    assert hasattr(ci, "paired_episode_cluster_bootstrap")


# ---------------------------------------------------------------------------------------------------------------------------
# the decision, on worlds with known answers
# ---------------------------------------------------------------------------------------------------------------------------
def test_constant_better_chooses_40k(ci):
    r = decide(W.make_world(shift=(0.02, 0.02)), ci)
    assert r["verdict"] == "CHOOSE_EMA_40000" and r["choice"] == "ema40000"
    for s in ("0", "1"):
        p = r["per_seed"][s]
        assert abs(p["delta"] - 0.02) < 1e-12 and p["lo"] == 0.02 and p["hi"] == 0.02
        assert p["positive_and_excludes_zero"] is True and p["separated"] is True
    assert r["inference_floor_abs_delta_s0_minus_s1"] < 1e-12
    assert r["margin"]["exceeds"] is True
    assert r["n_tokens"] == 1040 and r["n_clusters"] == 13


def test_constant_worse_keeps_the_default(ci):
    r = decide(W.make_world(shift=(-0.02, -0.02)), ci)
    assert r["verdict"] == "DEFAULT_EMA_48852" and r["choice"] == "ema48852"
    assert abs(r["per_seed"]["0"]["delta"] - (-0.02)) < 1e-12
    assert r["per_seed"]["0"]["positive_and_excludes_zero"] is False
    assert r["per_seed"]["0"]["separated"] is True            # separated BELOW zero is still not a reason to choose 40k


def test_one_seed_whose_interval_straddles_zero_keeps_the_default(ci):
    # seed 1: delta is +0.02 but the log effects are large (sd 0.08): the interval is [-0.0108, 0.0541]
    r = decide(W.make_world(shift=(0.02, 0.02), log_sd=(0.0, 0.08)), ci)
    assert r["verdict"] == "DEFAULT_EMA_48852" and r["choice"] == "ema48852"
    assert r["per_seed"]["0"]["positive_and_excludes_zero"] is True
    assert r["per_seed"]["1"]["positive_and_excludes_zero"] is False
    assert r["per_seed"]["1"]["lo"] < 0 < r["per_seed"]["1"]["hi"]
    assert r["reasons"] == ["A_SEED_IS_NOT_POSITIVE_WITH_AN_INTERVAL_EXCLUDING_ZERO"]


def test_margin_inside_2x_the_floor_keeps_the_default(ci):
    # d0 = 0.030, d1 = 0.016 -> floor 0.014; margin 0.016 <= 2 x floor = 0.028 (but > 1 x floor 0.014)
    r = decide(W.make_world(shift=(0.030, 0.016)), ci)
    assert r["verdict"] == "DEFAULT_EMA_48852"
    assert abs(r["inference_floor_abs_delta_s0_minus_s1"] - 0.014) < 1e-9
    assert abs(r["margin"]["value"] - 0.016) < 1e-9 and abs(r["margin"]["needed_gt"] - 0.028) < 1e-9
    assert r["margin"]["exceeds"] is False
    assert r["reasons"] == ["MARGIN_NOT_ABOVE_2X_INFERENCE_FLOOR"]
    assert r["per_seed"]["0"]["positive_and_excludes_zero"] and r["per_seed"]["1"]["positive_and_excludes_zero"]


def test_the_margin_is_the_smaller_seed_not_the_mean(ci):
    # d0 = 0.0285, d1 = 0.0185 -> floor 0.010, 2x = 0.020: min 0.0185 fails, mean 0.0235 would pass
    r = decide(W.make_world(shift=(0.0285, 0.0185)), ci)
    assert r["verdict"] == "DEFAULT_EMA_48852"
    assert abs(r["margin"]["value"] - 0.0185) < 1e-9
    assert r["margin_mean_beside"]["exceeds_2x_floor"] is True
    assert r["margin"]["exceeds"] is False


def test_margin_above_2x_the_floor_chooses_40k(ci):
    r = decide(W.make_world(shift=(0.030, 0.026)), ci)       # floor 0.004, 2x = 0.008, margin 0.026
    assert r["verdict"] == "CHOOSE_EMA_40000"
    assert abs(r["inference_floor_abs_delta_s0_minus_s1"] - 0.004) < 1e-9
    assert r["margin"]["exceeds"] is True


def test_pairing_is_by_token_not_by_row_position(ci):
    # every arm's dict has a DIFFERENT insertion order; mis-pairing leaves the mean alone but widens the interval to [-0.0041, 0.034]
    w = W.make_world(shift=(0.015, 0.015), base_lo=0.02, base_hi=0.95, s1_noise=0.002)
    assert list(w.tables[("ema40000", 0)])[:3] != list(w.tables[("ema48852", 0)])[:3]
    r = decide(w, ci)
    assert r["verdict"] == "CHOOSE_EMA_40000"
    assert r["per_seed"]["0"]["lo"] == 0.015 and r["per_seed"]["0"]["hi"] == 0.015


def test_the_cluster_is_the_log_not_the_token(ci):
    # log effects sd 0.03 over 13 logs: the LOG-cluster interval is [-0.0016, 0.0228] (token-level would be [0.0086, 0.0113])
    r = decide(W.make_world(shift=(0.01, 0.01), log_sd=(0.03, 0.03)), ci)
    assert r["verdict"] == "DEFAULT_EMA_48852"
    assert r["per_seed"]["0"]["n_episodes_logs"] == 13
    assert r["per_seed"]["0"]["lo"] < 0


def test_the_interval_is_95_percent(ci):
    # 95 %: [-0.0018, 0.0262] straddles 0.  90 %: [+0.0007, ...] would not.
    r = decide(W.make_world(shift=(0.012, 0.012), log_sd=(0.025, 0.025), seed=4, share_log_effect=True), ci)
    assert r["verdict"] == "DEFAULT_EMA_48852"
    assert r["per_seed"]["0"]["lo"] == -0.0018 and r["per_seed"]["0"]["separated"] is False
    assert r["constants"]["ALPHA"] == 0.05


def test_the_bootstrap_is_b_2000_per_seed(ci):
    r = decide(W.make_world(), ci)
    assert r["per_seed"]["0"]["n_boot"] == 2000 and r["per_seed"]["1"]["n_boot"] == 2000
    assert r["per_seed"]["0"]["estimator"] == "paired_episode_cluster_bootstrap"
    assert r["estimator"].startswith("paired_episode_cluster_bootstrap (cluster = log_name)")


def test_the_decision_never_reads_a_sub_score(ci):
    w = W.subscore_world()
    r = S.decide(w.tables, w.sample, w.log_of, ci, beside=True)
    assert r["verdict"] == "CHOOSE_EMA_40000"
    sub = r["beside"]["subscore_paired_delta_40k_minus_48852"]["0"]
    assert sub["TTC"]["delta_x100"] == -10.0 and sub["EP"]["delta_x100"] == 30.0
    assert sub["NC"]["delta_x100"] == 0.0 and sub["DAC"]["delta_x100"] == 0.0 and sub["C"]["delta_x100"] == 0.0
    assert all(v["deciding"] is False for v in sub.values())
    assert r["per_seed"]["0"]["delta_x100"] == 8.3333
    assert "NEVER decide" in r["beside_note"]


# ---------------------------------------------------------------------------------------------------------------------------
# controls that must read known values
# ---------------------------------------------------------------------------------------------------------------------------
def test_48852_minus_48852_reads_exactly_zero(ci):
    r = decide(W.make_world(), ci)
    for s in ("0", "1"):
        z = r["controls"]["zero_48852_minus_48852"][s]
        assert z["delta"] == 0.0 and z["lo"] == 0.0 and z["hi"] == 0.0 and z["separated"] is False
        assert z["reads_exactly_zero"] is True
    assert r["controls"]["all_read_known_values"] is True


def test_a_token_shuffled_pairing_widens_the_interval(ci):
    r = decide(W.make_world(shift=(0.02, 0.02)), ci)
    c0 = r["controls"]["token_shuffled_pairing"]["0"]
    assert c0["width_true"] == 0.0 and c0["width_shuffled"] > 0.03
    assert c0["widens"] is True and c0["point_unchanged"] is True
    assert r["controls"]["token_shuffled_pairing"]["1"]["widens"] is True


def test_a_control_that_cannot_read_its_known_value_voids_the_choice(ci):
    # every 48,852 token has the SAME pdms: shuffling cannot widen anything, so the instrument is not trustworthy
    r = decide(W.make_world(shift=(0.02, 0.02), constant_base=0.5), ci)
    assert r["verdict"] == "INCOMPLETE_DEFAULT_EMA_48852" and r["choice"] == "ema48852"
    assert r["reasons"] == ["CONTROL_DID_NOT_READ_ITS_KNOWN_VALUE"]
    assert r["controls"]["token_shuffled_pairing"]["0"]["widens"] is False


# ---------------------------------------------------------------------------------------------------------------------------
# refusals (every one -> INCOMPLETE, the registered default holds, numbers withheld)
# ---------------------------------------------------------------------------------------------------------------------------
def _assert_incomplete(r, reason_prefix):
    assert r["verdict"] == "INCOMPLETE_DEFAULT_EMA_48852"
    assert r["choice"] == "ema48852"
    assert any(x.startswith(reason_prefix) for x in r["reasons"]), r["reasons"]
    assert "per_seed" not in r and "controls" not in r
    assert r["numbers"].startswith("WITHHELD")


def test_a_cv_standin_row_refuses(ci):
    w = W.make_world(shift=(0.02, 0.02))
    w.tables[("ema40000", 1)][w.tokens[5]]["standin"] = True
    _assert_incomplete(decide(w, ci), "CV_STANDIN_ROWS:ema40000_s1:1")


def test_an_unverifiable_standin_status_refuses(ci):
    w = W.make_world(shift=(0.02, 0.02))
    w.tables[("ema48852", 0)][w.tokens[7]]["standin"] = None
    _assert_incomplete(decide(w, ci), "STANDIN_STATUS_UNVERIFIED:ema48852_s0:1")


def test_fewer_than_1000_tokens_is_underpowered(ci):
    r = decide(W.make_world(n_logs=9, per_log=111), ci)          # 999 tokens
    _assert_incomplete(r, "UNDERPOWERED_LT_1000_TOKENS")
    assert r["underpowered"] is True and r["n_tokens"] == 999


def test_exactly_1000_tokens_is_enough(ci):
    r = decide(W.make_world(n_logs=10, per_log=100), ci)
    assert r["verdict"] == "CHOOSE_EMA_40000" and r["n_tokens"] == 1000


def test_a_token_missing_in_any_arm_refuses(ci):
    w = W.make_world()
    del w.tables[("ema48852", 1)][w.tokens[100]]
    _assert_incomplete(decide(w, ci), "TOKEN_MISSING_IN_ARM:ema48852_s1:1")


def test_token_sets_that_differ_refuse(ci):
    w = W.make_world()
    w.tables[("ema40000", 0)]["tok_EXTRA"] = dict(w.tables[("ema40000", 0)][w.tokens[0]])
    _assert_incomplete(decide(w, ci), "TOKEN_SET_DIFFERS:ema40000_s0:extra1")


def test_a_missing_arm_refuses(ci):
    w = W.make_world()
    del w.tables[("ema40000", 1)]
    _assert_incomplete(decide(w, ci), "ARM_ABSENT:ema40000_s1")


def test_a_single_log_is_not_a_cluster_bootstrap(ci):
    _assert_incomplete(decide(W.make_world(n_logs=1, per_log=1200), ci), "FEWER_THAN_2_CLUSTERS")


def test_a_token_without_a_log_name_refuses(ci):
    w = W.make_world()
    del w.log_of[w.tokens[3]]
    _assert_incomplete(decide(w, ci), "TOKEN_WITHOUT_LOG_NAME")


def test_an_invalid_or_nonfinite_row_refuses(ci):
    w = W.make_world()
    w.tables[("ema40000", 0)][w.tokens[9]]["valid"] = False
    _assert_incomplete(decide(w, ci), "ROW_INVALID_OR_NONFINITE:ema40000_s0")
    w = W.make_world()
    w.tables[("ema48852", 0)][w.tokens[9]]["PDMS"] = float("nan")
    _assert_incomplete(decide(w, ci), "ROW_INVALID_OR_NONFINITE:ema48852_s0")


def test_a_pdms_outside_0_1_or_breaking_the_identity_refuses(ci):
    w = W.make_world()
    w.tables[("ema40000", 0)][w.tokens[2]]["PDMS"] = 1.2
    r = decide(w, ci)
    assert r["verdict"] == "INCOMPLETE_DEFAULT_EMA_48852"
    assert "PDMS_OUT_OF_RANGE:ema40000_s0" in r["reasons"] and "PDMS_IDENTITY_VIOLATED:ema40000_s0:1" in r["reasons"]


def test_every_incomplete_run_keeps_the_default(ci):
    w = W.make_world(shift=(0.02, 0.02))                          # would CHOOSE 40k if complete
    w.tables[("ema40000", 0)][w.tokens[1]]["standin"] = True
    r = decide(w, ci)
    assert r["choice"] == "ema48852" and r["verdict"] == "INCOMPLETE_DEFAULT_EMA_48852"


def test_identical_arms_keep_the_default(ci):
    r = decide(W.make_world(shift=(0.0, 0.0)), ci)
    assert r["verdict"] == "DEFAULT_EMA_48852"
    assert r["per_seed"]["0"]["delta"] == 0.0 and r["per_seed"]["0"]["positive_and_excludes_zero"] is False


# ---------------------------------------------------------------------------------------------------------------------------
# beside rows
# ---------------------------------------------------------------------------------------------------------------------------
def test_reference_rows_are_printed_beside_and_never_decide(ci):
    w = W.make_world(shift=(0.02, 0.02))
    stop = {t: W._row(0.0) for t in w.tokens}                     # STOP scores 0 here
    cv = {t: W._row(0.25) for t in w.tokens[:-1]}                 # a reference over a DIFFERENT token set: unusable, harmless
    r = S.decide(w.tables, w.sample, w.log_of, ci, references={"STOP": stop, "CV": cv}, beside=True)
    assert r["verdict"] == "CHOOSE_EMA_40000"
    assert r["beside"]["references"]["STOP"]["usable"] is True and r["beside"]["references"]["STOP"]["PDMS"]["mean_x100"] == 0.0
    assert r["beside"]["references"]["CV"]["usable"] is False


# ---------------------------------------------------------------------------------------------------------------------------
# file level: the scorer + bridge artifacts as the real run leaves them
# ---------------------------------------------------------------------------------------------------------------------------
def _from_spec(spec_path, ci):
    sample, tables, refs, problems, prov = S.load_spec(spec_path)
    return S.decide(tables, sample, dict(sample.get("token_log", {})), ci, pre_problems=problems, references=refs, prov=prov, beside=False)


@pytest.fixture()
def files(tmp_path):
    w = W.make_world(shift=(0.02, 0.02))
    spec = W.write_world(tmp_path / "run", w)
    return w, str(tmp_path / "run"), str(tmp_path / "run" / "spec.json"), spec


def test_files_roundtrip_chooses_40k(files, ci):
    w, root, spec_path, _ = files
    r = _from_spec(spec_path, ci)
    assert r["verdict"] == "CHOOSE_EMA_40000"
    assert r["inputs"]["arms"]["ema40000_s0"]["identity"]["ckpt_md5"] == "57807456bd7169800be159ff95871d22"
    assert r["inputs"]["arms"]["ema48852_s1"]["identity"]["step"] == 48852


def _edit_json(path, fn):
    d = json.load(open(path, encoding="utf-8"))
    fn(d)
    json.dump(d, open(path, "w", encoding="utf-8"))


def test_a_swapped_checkpoint_directory_refuses(files, ci):
    """The 40k bridge dir's forward manifest claims the 48,852 md5: a mislabelled arm must not be read as the other checkpoint."""
    w, root, spec_path, spec = files
    _edit_json(f"{root}/ema40000/bridge/seam_R8_LEGAL.manifest.json", lambda d: d["model"].update(ckpt_md5=W.CKPT_MD5["ema48852"], step=48852))
    r = _from_spec(spec_path, ci)
    assert r["verdict"] == "INCOMPLETE_DEFAULT_EMA_48852"
    assert any(x.startswith("ema40000_s0:ARM_IDENTITY_MISMATCH") for x in r["reasons"])


def test_a_cpu_arm_refuses(files, ci):
    w, root, spec_path, spec = files
    _edit_json(f"{root}/ema48852/bridge/seam_R8_LEGAL_s1.manifest.json", lambda d: d.update(device=["cpu"]))
    r = _from_spec(spec_path, ci)
    assert r["verdict"] == "INCOMPLETE_DEFAULT_EMA_48852"
    assert any(x.startswith("ema48852_s1:ARM_NOT_ON_THOR_CUDA") for x in r["reasons"])


@pytest.mark.parametrize("what,edit,reason", [
    ("rt_from_wrong_rows", lambda root: _edit_json(f"{root}/ema40000/bridge/seam_R8_RT_s1.manifest.json",
                                                   lambda d: d.update(derived_from="rows_R8_LEGAL.jsonl")), "ema40000_s1:RT_DERIVED_FROM_rows_R8_LEGAL.jsonl"),
    ("rt_table", lambda root: _edit_json(f"{root}/ema40000/bridge/seam_R8_RT.manifest.json",
                                         lambda d: d["retime"].update(table_sha256="0" * 64)), "ema40000_s0:RT_TABLE_NOT_THE_REGISTERED_DEPLOY_TABLE"),
    ("navtest_split", lambda root: _edit_json(f"{root}/ema48852/bridge/seam_R8_LEGAL.manifest.json",
                                              lambda d: d.update(split="navtest")), "ema48852_s0:ARM_SPLIT_NOT_NAVTRAIN"),
    ("kpr_fail", lambda root: _edit_json(f"{root}/ema48852/bridge/seam_R8_LEGAL.manifest.json",
                                         lambda d: d["KPR"].update(verdict="FAIL")), "ema48852_s0:ARM_KPR_FAIL"),
    ("wrong_seed", lambda root: _edit_json(f"{root}/ema40000/bridge/seam_R8_LEGAL_s1.manifest.json",
                                           lambda d: d["spec"].update(seed=0)), "ema40000_s1:ARM_SEED_MISMATCH:0"),
    ("partial_manifest", lambda root: _edit_json(f"{root}/ema40000/bridge/seam_R8_RT.manifest.json",
                                                 lambda d: d.update(partial=True)), "ema40000_s0:MANIFEST_PARTIAL_OR_STANDIN"),
    ("counts_fail", lambda root: _edit_json(f"{root}/scores/r8sel_ema40000_s0/r8sel_ema40000_s0.counts.json",
                                            lambda d: d.update(status="FAIL")), "ema40000_s0:SCORER_COUNTS_STATUS_FAIL"),
    ("counts_n_expected", lambda root: _edit_json(f"{root}/scores/r8sel_ema40000_s0/r8sel_ema40000_s0.counts.json",
                                                  lambda d: d.update(n_expected=1039)), "ema40000_s0:SCORER_N_EXPECTED_1039_NE_1040"),
    ("binding_seam_sha", lambda root: _edit_json(f"{root}/scores/r8sel_ema48852_s0/r8sel_ema48852_s0.binding.json",
                                                 lambda d: d.update(seam_sha256="1" * 64)), "ema48852_s0:BINDING_SEAM_SHA_MISMATCH"),
    ("binding_sample_sha", lambda root: _edit_json(f"{root}/scores/r8sel_ema48852_s1/r8sel_ema48852_s1.binding.json",
                                                   lambda d: d.update(sample_sha256_sorted_tokens="2" * 64)), "ema48852_s1:BINDING_SAMPLE_SHA_MISMATCH"),
    ("binding_split", lambda root: _edit_json(f"{root}/scores/r8sel_ema48852_s1/r8sel_ema48852_s1.binding.json",
                                              lambda d: d.update(split="navtest")), "ema48852_s1:BINDING_SPLIT_navtest"),
])
def test_file_level_refusals(files, ci, what, edit, reason):
    w, root, spec_path, spec = files
    edit(root)
    r = _from_spec(spec_path, ci)
    assert r["verdict"] == "INCOMPLETE_DEFAULT_EMA_48852", what
    assert reason in r["reasons"], (what, r["reasons"])


def test_a_csv_edited_after_scoring_refuses(files, ci):
    """The binding records the csv sha256 at scoring time; any later edit (a hand-fixed row) breaks it."""
    w, root, spec_path, spec = files
    p = f"{root}/scores/r8sel_ema40000_s0/r8sel_ema40000_s0.csv"
    s = open(p, encoding="utf-8").read().replace(",True,", ",True,", 1) + "\n"
    open(p, "w", encoding="utf-8").write(s)
    r = _from_spec(spec_path, ci)
    assert "ema40000_s0:BINDING_CSV_SHA_MISMATCH" in r["reasons"]


def test_a_cv_standin_in_the_seam_file_refuses(files, ci):
    w, root, spec_path, spec = files
    toks = sorted(w.tokens)
    W.write_seam(f"{root}/ema40000/bridge/seam_R8_RT.npz", None, toks, "R8_RT", standin={toks[10]})
    _edit_json(f"{root}/scores/r8sel_ema40000_s0/r8sel_ema40000_s0.binding.json",
               lambda d: d.update(seam_sha256=W._sha(f"{root}/ema40000/bridge/seam_R8_RT.npz")))     # keep the binding honest
    r = _from_spec(spec_path, ci)
    assert "CV_STANDIN_ROWS:ema40000_s0:1" in r["reasons"]


def test_seam_and_csv_token_sets_must_agree(files, ci):
    w, root, spec_path, spec = files
    toks = sorted(w.tokens)[:-1]
    W.write_seam(f"{root}/ema48852/bridge/seam_R8_RT.npz", None, toks, "R8_RT")
    _edit_json(f"{root}/scores/r8sel_ema48852_s0/r8sel_ema48852_s0.binding.json",
               lambda d: d.update(seam_sha256=W._sha(f"{root}/ema48852/bridge/seam_R8_RT.npz")))
    r = _from_spec(spec_path, ci)
    assert "ema48852_s0:SEAM_CSV_TOKEN_SETS_DIFFER" in r["reasons"]


def test_csv_with_a_duplicate_or_missing_token_or_bad_row_refuses(files, ci):
    w, root, spec_path, spec = files
    p = f"{root}/scores/r8sel_ema48852_s1/r8sel_ema48852_s1.csv"
    lines = open(p, encoding="utf-8").read().splitlines()
    lines.insert(3, lines[2])                                     # a duplicate token row
    open(p, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    _edit_json(f"{root}/scores/r8sel_ema48852_s1/r8sel_ema48852_s1.binding.json", lambda d: d.update(csv_sha256=W._sha(p)))
    r = _from_spec(spec_path, ci)
    assert any(x.startswith("ema48852_s1:DUPLICATE_TOKEN:") for x in r["reasons"])


@pytest.mark.parametrize("status,reason", [("UNDERPOWERED", "SAMPLE_STATUS_UNDERPOWERED"),
                                           ("NOT_FIXED__NO_READABLE_NAVTRAIN_TOKEN__ACQUISITION_NEEDS_PI_APPROVAL",
                                            "SAMPLE_STATUS_NOT_FIXED__NO_READABLE_NAVTRAIN_TOKEN__ACQUISITION_NEEDS_PI_APPROVAL")])
def test_a_sample_that_is_not_fixed_refuses(files, ci, status, reason):
    w, root, spec_path, spec = files
    _edit_json(spec["sample"], lambda d: d.update(status=status))
    assert reason in _from_spec(spec_path, ci)["reasons"]


def test_a_sample_whose_sha256_does_not_rederive_refuses(files, ci):
    w, root, spec_path, spec = files
    _edit_json(spec["sample"], lambda d: d.update(sha256_sorted_tokens="3" * 64))
    assert "SAMPLE_SHA256_DOES_NOT_REDERIVE" in _from_spec(spec_path, ci)["reasons"]


# ---------------------------------------------------------------------------------------------------------------------------
# the CLI, at the real process boundary
# ---------------------------------------------------------------------------------------------------------------------------
@pytest.mark.e2e
def test_cli_layout_root_writes_the_decision_and_the_marker(files, tmp_path):
    w, root, spec_path, spec = files
    out = str(tmp_path / "DECISION.json")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.pop("R8SEL_MUTATION", None)
    p = subprocess.run([sys.executable, os.path.join(CODE, "select_ckpt_r8.py"), "--layout-root", root, "--sample", spec["sample"],
                        "--out", out, "--md", str(tmp_path / "DECISION.md")], capture_output=True, text=True, env=env)
    assert "ZZSELECT verdict=CHOOSE_EMA_40000 choice=ema40000 reasons=0" in p.stdout, p.stdout + p.stderr
    d = json.load(open(out, encoding="utf-8"))
    assert d["choice"] == "ema40000" and d["schema"] == "r8-ckpt-select/1"
    assert d["ci_py"]["sha256"] and d["constants"]["N_BOOT"] == 2000
    assert open(str(tmp_path / "DECISION.md"), encoding="utf-8").read().startswith("# D-R8-CKPT-RULE decision -- CHOOSE_EMA_40000")


@pytest.mark.e2e
def test_the_cli_refuses_while_a_mutation_is_set(files, tmp_path):
    w, root, spec_path, spec = files
    out = str(tmp_path / "DECISION.json")
    env = dict(os.environ, R8SEL_MUTATION="floor_1x", PYTHONIOENCODING="utf-8")
    p = subprocess.run([sys.executable, os.path.join(CODE, "select_ckpt_r8.py"), "--spec", spec_path, "--out", out],
                       capture_output=True, text=True, env=env)
    assert p.returncode != 0 and not os.path.exists(out)
    assert "mutation-test environment never produces a decision" in (p.stdout + p.stderr)


@pytest.mark.e2e
def test_spec_from_layout_names_the_files_the_runbook_produces():
    spec = S.spec_from_layout("/x/sel", "/x/NAVTRAIN_SAMPLE_TOKENS.json")
    assert spec["arms"]["ema40000_s1"] == {
        "csv": "/x/sel/scores/r8sel_ema40000_s1/r8sel_ema40000_s1.csv",
        "counts": "/x/sel/scores/r8sel_ema40000_s1/r8sel_ema40000_s1.counts.json",
        "binding": "/x/sel/scores/r8sel_ema40000_s1/r8sel_ema40000_s1.binding.json",
        "seam": "/x/sel/ema40000/bridge/seam_R8_RT_s1.npz", "bridge_dir": "/x/sel/ema40000/bridge"}
    assert sorted(spec["references"]) == ["CV", "PRIOR_ha0p", "STOP"]
    assert spec["references"]["PRIOR_ha0p"]["seam"] == "/x/sel/ema48852/bridge/seam_PRIOR_ha0p.npz"
