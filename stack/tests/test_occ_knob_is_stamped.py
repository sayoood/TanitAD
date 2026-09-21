"""A behaviour knob that leaves no trace in the run record — pinned by MUTATION.

⛔ WHY THIS FILE EXISTS, AND IT IS A DEBT I CREATED. `6a472d1` added
`AgentSlotDecoder.occ_from_geometry`, which changes what the model EMITS at inference, and added it
as a bare attribute with **no config field and no stamp entry**. `AgentSeamConfig.as_dict`'s own
docstring says what that costs:

    a run record that cannot rebuild its own model config is not a run record

⇒ a run that enabled the flag could not be rebuilt from `config.json`, and its `occ` numbers would
be **unattributable between the learned slice and the derived read** — the anchor-units failure in a
new costume, where a correct number is quoted outside the scope that gives it meaning.
⚠️ The existing provenance test (`test_P2_every_knob_is_recoverable_from_the_stamp_BY_VALUE`) could
not have caught this: it walks **argparse** dests, and this knob has no CLI flag. A guard is only as
wide as the thing it enumerates.

⭐ SO THE PIN HERE IS THE KEY SET AS A LITERAL, NOT A PROPERTY OF THE CODE. Asserting
"every field of the dataclass appears in as_dict" would be an expression over the code under test
and would pass for any future knob added to both at once — including one added to neither. A
frozen literal forces a DELIBERATE edit, which is the whole point: the same discipline that the
five-site gate value earned.

Companion: `mutate_occ_stamp.py` reintroduces each real defect — dropped stamp entry, and the
"declared but never plumbed" seam defect — and requires these to go RED.
"""
from __future__ import annotations

import pytest
import torch

from tanitad.models.agent_slots import occ_logit_from_centre
from tanitad.refs.refc_agents import AgentSeamConfig, build_agent_head

# ⛔ FROZEN LITERALS. Adding a knob to either seam means adding it here, on purpose.
AGENT_SEAM_STAMP_KEYS = {
    "enable", "oracle", "oracle_sigma_range_m", "oracle_miss_rate",
    "queries", "d_model", "depth", "n_heads", "enforce_band",
    "w_project", "w_ground", "presence_gate", "presence_hard",
    "occ_from_geometry",
    "n_classes", "classes", "n_queries_default_upstream",
}
PERCEPTION_STAMP_KEYS = {
    "w_map", "w_box3d", "d_bev", "n_queries", "d_model", "bev_tokens_hw",
    "heights_m", "stride", "use_bev_in_box_head", "occ_from_geometry",
    "bev_encoder",
}


def test_agent_seam_stamp_key_set_is_exactly_the_frozen_literal():
    got = set(AgentSeamConfig().as_dict())
    missing = AGENT_SEAM_STAMP_KEYS - got
    extra = got - AGENT_SEAM_STAMP_KEYS
    assert not missing, (
        f"AgentSeamConfig.as_dict no longer stamps {sorted(missing)}. A knob that shapes this "
        f"seam and is absent from the record makes the run unrebuildable.")
    assert not extra, (
        f"AgentSeamConfig.as_dict gained {sorted(extra)} without this pin being updated. If that "
        f"is a real new knob, add it here DELIBERATELY — that is what this literal is for.")


def test_perception_stamp_key_set_is_exactly_the_frozen_literal():
    from tanitad.models.refcv6_perception_branch import (BEVEncoderConfig,
                                                         PerceptionBranchConfig)
    cfg = PerceptionBranchConfig(w_map=1.0, w_box3d=1.0,
                                 bev_cfg=BEVEncoderConfig(d_in=128))
    got = set(cfg.as_dict())
    assert not (PERCEPTION_STAMP_KEYS - got), (
        f"PerceptionBranchConfig.as_dict no longer stamps "
        f"{sorted(PERCEPTION_STAMP_KEYS - got)}")
    assert not (got - PERCEPTION_STAMP_KEYS), (
        f"PerceptionBranchConfig.as_dict gained {sorted(got - PERCEPTION_STAMP_KEYS)} "
        f"without this pin being updated")


@pytest.mark.parametrize("flag", [False, True])
def test_the_stamp_reports_the_value_that_was_actually_set(flag):
    """⛔ Presence in the stamp is not enough — it must carry the VALUE. A stamp hardcoded to the
    default would satisfy a key-set check and still misreport every enabled run."""
    assert AgentSeamConfig(occ_from_geometry=flag).as_dict()["occ_from_geometry"] is flag


@pytest.mark.parametrize("flag", [False, True])
def test_the_config_field_ACTUALLY_REACHES_the_built_agent_head(flag):
    """⛔⛔ DECLARED IS NOT PLUMBED — the refcv6 seam defect verbatim (six channels declared, two
    never wired). A field that is stamped but never applied is WORSE than an unstamped one: the
    record then asserts a behaviour the model does not have."""
    head = build_agent_head(
        AgentSeamConfig(enable=True, queries=4, d_model=32, depth=1, n_heads=4,
                        enforce_band=False, occ_from_geometry=flag),
        d_memory=32, n_memory=8)
    assert head.occ_from_geometry is flag, (
        "build_agent_head did not apply cfg.occ_from_geometry to the head it returned")
    raw = torch.randn(1, head.n_queries, head.head.out_features)
    out = head.decode(raw)
    derived = occ_logit_from_centre(out["box"][..., 0], out["box"][..., 1])
    if flag:
        assert torch.allclose(out["occ_logit"], derived, atol=1e-6)
    else:
        assert not torch.allclose(out["occ_logit"], derived, atol=1e-6)


@pytest.mark.parametrize("flag", [False, True])
def test_the_config_field_ACTUALLY_REACHES_the_built_box3d_head(flag):
    """The 3-D head is the one `s1_pass` scores, so it is the one that matters most.

    ⚠️ AN EARLIER VERSION OF THIS TEST ONLY CHECKED `as_dict` WHILE BEING NAMED
    `ACTUALLY_REACHES_the_built_box3d_head`. It passed, and it would have passed with the plumbing
    line deleted — a test whose NAME claims more than its body checks, which is worse than a
    missing test because it reads as coverage. It now builds the branch and reads the attribute
    off the decoder that was actually constructed.
    """
    from tanitad.models.refcv6_perception_branch import (BEVEncoderConfig,
                                                         PerceptionBranch,
                                                         PerceptionBranchConfig)
    cfg = PerceptionBranchConfig(w_map=0.0, w_box3d=1.0, d_model=32, n_queries=4,
                                 bev_cfg=BEVEncoderConfig(d_in=128),
                                 enforce_param_band=False, occ_from_geometry=flag)
    assert cfg.as_dict()["occ_from_geometry"] is flag
    br = PerceptionBranch(cfg, d_image=128, image_hw=(26, 64))
    assert br.box_dec is not None, "w_box3d > 0 must build the box decoder"
    assert br.box_dec.occ_from_geometry is flag, (
        "PerceptionBranch did not apply cfg.occ_from_geometry to the Box3DSlotDecoder it built — "
        "the stamp would then assert a behaviour the scored head does not have")
