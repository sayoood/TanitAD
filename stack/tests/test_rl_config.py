"""Config validation — especially the methods we CANNOT supply a signal for.

A config that lets you select an unsupplied method without saying so is how a
run produces a confident, meaningless number. `dpo` is refused on purpose.
"""

from __future__ import annotations

import pytest

from tanitad.rl.config import METHODS, REQUIREMENTS, PostTrainConfig


def test_default_config_validates():
    PostTrainConfig().validate()


def test_dpo_is_REFUSED_because_we_have_no_preference_pairs():
    """⛔ The honest REJECT. A demonstration corpus has no negative class."""
    with pytest.raises(ValueError, match="REFUSED"):
        PostTrainConfig(method="dpo").validate()


def test_dpo_requirement_records_why_and_the_measured_cost():
    req = REQUIREMENTS["dpo"]
    assert req["have"] is False
    assert "preference" in req["needs"].lower()
    assert "0.0974" in req["why"], (
        "the refusal must carry the MEASURED cost of the manufactured "
        "alternative, not just an opinion")


def test_grpo_and_awr_are_supplied():
    for m in ("grpo", "awr"):
        assert REQUIREMENTS[m]["have"] is True
        PostTrainConfig(method=m).validate()


def test_every_method_has_a_requirement_row():
    assert set(REQUIREMENTS) == set(METHODS)
    for m, r in REQUIREMENTS.items():
        assert {"needs", "have", "why"} <= set(r), m


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="method must be one of"):
        PostTrainConfig(method="ppo").validate()


def test_grpo_group_of_one_is_rejected_at_config_time():
    """Caught early, not at the first silent no-op update."""
    with pytest.raises(ValueError, match="group_size >= 2"):
        PostTrainConfig(method="grpo", group_size=1).validate()


def test_bad_normalize_and_noise_mode_rejected():
    with pytest.raises(ValueError, match="normalize"):
        PostTrainConfig(normalize="zscore").validate()
    with pytest.raises(ValueError, match="noise_mode"):
        PostTrainConfig(noise_mode="gaussian").validate()


def test_defaults_encode_the_published_and_corrected_choices():
    cfg = PostTrainConfig()
    assert cfg.freeze_trunk is True, "frozen trunk is the mandatory default"
    assert cfg.noise_mode == "multiplicative", "DDv2 measured 90.1 vs 89.7"
    assert cfg.normalize == "none", "Dr. GRPO: do not divide by the group std"
    assert cfg.kl_coef == 0.0, "the IL loss is the anchor, not a KL"


def test_to_dict_records_every_knob_and_the_requirement():
    d = PostTrainConfig().to_dict()
    for k in ("method", "group_size", "normalize", "noise_mode", "noise_scale",
              "reward_weights", "lr", "steps", "seed", "freeze_trunk",
              "w_imitation", "kl_coef"):
        assert k in d, k
    assert d["_requirements"]["have"] is True
    assert "_evidence_class" in d and "_tier" in d


def test_empty_reward_is_rejected():
    with pytest.raises(ValueError, match="no components"):
        PostTrainConfig(reward_weights={}).validate()
