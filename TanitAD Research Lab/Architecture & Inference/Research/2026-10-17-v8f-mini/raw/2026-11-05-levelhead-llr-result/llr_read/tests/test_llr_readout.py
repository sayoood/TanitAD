"""test_llr_readout.py -- the token-wiring O-sensitivity read-out (code/llr_readout.py) on the SYNTHETIC LLR-shaped checkpoint (make_synthetic_llr.py; random lvl_head / lead_head / e_aux, so the numbers mean nothing: the controls
and the plumbing are what is tested).  Env override: LLR_READOUT_PATH.  Run with --basetemp on F:."""
import importlib.util
import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
SYN = "F:/TanitAD-scratch/motion_c/llr_read/pt/synth/M2b_s0_recft_LLR/ckpt_last.pt"


def _load(path):
    spec = importlib.util.spec_from_file_location("llr_readout_under_test", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


R = _load(os.environ.get("LLR_READOUT_PATH", f"{PK}/code/llr_readout.py"))


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    assert os.path.isfile(SYN), "run code/make_synthetic_llr.py first"
    out = str(tmp_path_factory.mktemp("ro") / "ro_s0.json")
    rc = R.main(["--seed", "0", "--max-windows", "64", "--threads", "8", "--batch", "32", "--ckpt", SYN, "--out", out])
    return rc, json.load(open(out, encoding="utf-8"))


def test_the_end_to_end_run_passes_every_control_and_reports_both_inputs(smoke):
    rc, r = smoke
    assert rc == 0 and r["control_failures"] == [] and r["n_windows"] == 64 and r["ckpt_cfg"]["aux_cond"] == "token"
    c = r["controls"]
    assert c["zero_shift_change_max_abs"] == 0.0 and c["presence_set_reaches_column1_exact"] and c["gap_set_reaches_column2_exact"] and c["level_v_equals_hook_vhat_max_abs_dev"] == 0.0
    assert c["level_shift_reaches_column0_max_abs_dev_from_0p3"] < 1e-5 and c["level_shift_leaves_columns_1_2_max_abs_dev"] == 0.0 and c["presence_set_leaves_columns_0_2_max_abs_dev"] == 0.0
    assert c["reach_control_big_e_aux_output_perturbation"]["mean_abs_plan_speed_change_ms"] > 0.0
    for blk in (r["level"]["plus3"], r["level"]["minus3"], r["lead"]["flip_1_minus_0"], r["lead"]["presence_1_vs_baseline"], r["beside"]["e_aux_output_zeroed"]):
        assert set(blk) >= {"mean_signed_change_ms", "mean_abs_change_ms"} and len(blk["mean_signed_change_ms"]["ci95"]) == 2
    assert r["level"]["label"] in ("O-SENSITIVE", "WEAK", "O-INSENSITIVE") and r["lead"]["label_by_analogy"] in ("O-SENSITIVE", "WEAK", "O-INSENSITIVE")
    assert "BY ANALOGY" in r["lead"]["label_rule"]


def test_a_random_e_aux_moves_the_plan_when_the_input_is_changed(smoke):
    _, r = smoke
    assert r["lead"]["flip_1_minus_0"]["mean_abs_change_ms"]["point"] > 0.0 and r["level"]["plus3"]["mean_abs_change_ms"]["point"] > 0.0           # the plumbing reaches the plan


def test_control_failures_name_each_broken_control(smoke):
    _, r = smoke
    base = json.loads(json.dumps(r))
    assert R.control_failures(base) == []
    cases = {"zero shift": lambda d: d["controls"].__setitem__("zero_shift_change_max_abs", 1e-3),
             "e_aux input is not": lambda d: d["controls"].__setitem__("a_in_equals_vhat10_sigmoid_logit_sigmoid_gaplogit_where_frame_ok_max_abs_dev", 0.1),
             "not zero where the frame": lambda d: d["controls"].__setitem__("a_in_zero_where_frame_not_ok_max_abs", 0.3),
             "column 0 alone": lambda d: d["controls"].__setitem__("level_shift_leaves_columns_1_2_max_abs_dev", 0.3),
             "presence set did not reach": lambda d: d["controls"].__setitem__("presence_set_reaches_column1_exact", False),
             "presence set did not reach ": lambda d: d["controls"].__setitem__("presence_set_leaves_columns_0_2_max_abs_dev", 0.2),
             "gap set": lambda d: d["controls"].__setitem__("gap_set_reaches_column2_exact", False),
             "forward hook": lambda d: d["controls"].__setitem__("level_v_equals_hook_vhat_max_abs_dev", 1e-3),
             "reach control": lambda d: d["controls"]["reach_control_big_e_aux_output_perturbation"].__setitem__("mean_abs_plan_speed_change_ms", 0.0),
             "banked LLR rollout": lambda d: d.__setitem__("banked_tick0_check", {"status": "computed", "max_abs_dev_m": 0.5})}
    for needle, fn in cases.items():
        d = json.loads(json.dumps(base))
        fn(d)
        f = R.control_failures(d)
        assert f and any(needle.strip() in x for x in f), (needle, f)


def test_labels_use_a1s_thresholds_and_the_plan_statistic_is_the_step_length_mean():
    assert R.sens_label(0.5) == "O-SENSITIVE" and R.sens_label(0.4999) == "WEAK" and R.sens_label(0.1) == "WEAK" and R.sens_label(0.0999) == "O-INSENSITIVE" and (R.SENS_HI, R.SENS_LO) == (0.5, 0.1)
    traj = np.zeros((1, 60, 2))
    traj[0, :, 0] = np.arange(1, 61) * 0.5                                   # 0.5 m per 0.1 s = 5 m/s
    assert abs(float(R.plan_speed_0_3(traj)[0]) - 5.0) < 1e-9
    assert R.LEVEL_SCALE == 10.0 and R.SHIFT_MS / R.LEVEL_SCALE == 0.3 and (R.GAP_LO, R.GAP_HI) == (0.25, 0.75)


def test_merge_writes_the_table_and_refuses_failed_controls(tmp_path, smoke, monkeypatch):
    _, r = smoke
    for s in (0, 1):
        d = json.loads(json.dumps(r))
        d["seed"], d["full_eval_set"] = s, True
        json.dump(d, open(tmp_path / f"LLR_READOUT_s{s}.json", "w", encoding="utf-8"))
    monkeypatch.setattr(R, "PK", str(tmp_path))
    assert R.merge(str(tmp_path)) == 0
    md = open(tmp_path / "SENSITIVITY.md", encoding="utf-8").read()
    assert "Controls: ALL PASS" in md and "presence flip 1 minus 0" in md and "BY ANALOGY" in md and "No wiring decision" in md
    assert json.load(open(tmp_path / "LLR_READOUT.json", encoding="utf-8"))["PASS"] is True
    d = json.loads(json.dumps(r))
    d["full_eval_set"], d["control_failures"] = False, ["zero shift did not read exactly 0.0"]
    json.dump(d, open(tmp_path / "LLR_READOUT_s1.json", "w", encoding="utf-8"))
    assert R.merge(str(tmp_path)) == 1 and "FAILED" in open(tmp_path / "SENSITIVITY.md", encoding="utf-8").read()


def test_the_readout_refuses_a_non_token_or_lead_less_checkpoint(tmp_path):
    with pytest.raises(AssertionError, match="level_head / lead_head / aux_cond"):
        R.run_seed(0, 32, 4, 32, ckpt="E:/Projects/TanitAD-artifacts/v8f-recover/M2b_s0_recft_CLR/ckpt_last.pt")
