"""Tests of k5 in parts (note PREREG_V8F_PROOF_ADD38_N2): the known-good probe passes, and each part has a deliberate regression that fails it."""
import copy
import json

import pytest

import lv2_read as R

GOOD = {"ckpt": "/c/ckpt.pt", "same_window_list": True, "d_actions_max_abs": 0.0, "a_rollout_fn_on_banked_inputs_max_abs": 2.0e-05,
        "b_start_state_pod_vs_banked": {"cosine_mean": 0.99967, "rms_diff": 0.0237, "rms_state": 0.949},
        "b_future_state_pod_vs_banked": {"cosine_mean": 0.99968, "rms_diff": 0.0236, "rms_state": 0.952},
        "c_nearest_frame_to_banked_start_is_4_share": 1.0, "c_nearest_frame_to_banked_future_is_14_share": 1.0,
        "rollout_pod_vs_banked": {"max_abs": 0.2008550763130188}}
DUMP = {"k5_max_abs": 0.2008550763130188, "meta": {"ckpt": "/c/ckpt.pt"}}


def _write(tmp_path, q, name="p.json"):
    f = tmp_path / name
    f.write_text("a log line before the json" + chr(10) + json.dumps(q) + chr(10), encoding="utf-8")
    return str(f)


def test_good_probe_passes(tmp_path):
    c = R.control_k5_parts([DUMP, DUMP], [_write(tmp_path, GOOD, "a.json"), _write(tmp_path, GOOD, "b.json")])
    assert c["pass"] is True and c["name"] == "k5p" and all(r["pass"] for r in c["per_seed"])


MUT = {"rollout function differs": ("a_rollout_fn_on_banked_inputs_max_abs", 1e-2),
       "actions differ": ("d_actions_max_abs", 1e-6),
       "another window list": ("same_window_list", False),
       "start frame off": ("c_nearest_frame_to_banked_start_is_4_share", 0.5),
       "future frame off": ("c_nearest_frame_to_banked_future_is_14_share", 0.98),
       "another checkpoint": ("ckpt", "/c/other.pt")}


@pytest.mark.parametrize("what", sorted(MUT))
def test_each_part_fails_on_its_regression(tmp_path, what):
    q = copy.deepcopy(GOOD)
    k, v = MUT[what]
    q[k] = v
    c = R.control_k5_parts([DUMP, DUMP], [_write(tmp_path, GOOD, "a.json"), _write(tmp_path, q, "b.json")])
    assert c["pass"] is False and c["per_seed"][0]["pass"] is True and c["per_seed"][1]["pass"] is False


def test_state_disagreement_and_untied_probe_fail(tmp_path):
    q = copy.deepcopy(GOOD); q["b_start_state_pod_vs_banked"]["cosine_mean"] = 0.99
    assert R.control_k5_parts([DUMP], [_write(tmp_path, q)])["pass"] is False
    q = copy.deepcopy(GOOD); q["b_future_state_pod_vs_banked"]["cosine_mean"] = 0.9989
    assert R.control_k5_parts([DUMP], [_write(tmp_path, q)])["pass"] is False
    q = copy.deepcopy(GOOD); q["rollout_pod_vs_banked"]["max_abs"] = 0.2008
    assert R.control_k5_parts([DUMP], [_write(tmp_path, q)])["pass"] is False


def test_one_probe_per_dump(tmp_path):
    with pytest.raises(SystemExit):
        R.control_k5_parts([DUMP, DUMP], [_write(tmp_path, GOOD)])
