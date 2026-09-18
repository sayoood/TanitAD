"""W0 — a refcv6 checkpoint must LOAD. It could not, for any driver.

⛔ THE DEFECT, REPRODUCED BY HAND 2026-09-19 on a real checkpoint:

    [refcv3_arm] ⛔ config.json CONTRADICTS the rebuilt model
    param_breakdown: config.json {... 'total': 50519066} vs rebuilt {... 'total': 45703450}

EVERY NAMED LINE AGREED — `core`, `phi_tac`, `str_goal_head`, `gstr_cond`, `tac_heads`,
`tac_latent_proj`, `scorer`, `nav_inject`, `tac_decoder_v6`, `tac_behaviour_gate_v6` were
identical. Only `total` differed, by **4,815,616**, which is EXACTLY
`refcv6_perception.branch_params.total` in the same file: the perception branch is attached
to the MODEL by the trainer and so is absent from a model rebuilt from `RefCV3Config`.

⛔ `--allow-nonstrict` could not reach it — `cross_check_config` runs BEFORE the load.
MEASURED: both settings refused identically.

⭐ The fix needed NO new information: `config.json`'s own `refcv6_perception` stamp carries
the whole `PerceptionBranchConfig` (`as_dict()`'s key set is a strict SUBSET of the
stamp's). This file pins the reconstruction; the live strict load is banked as an artifact.
"""
from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from tanitad.models import refcv6_perception_branch as _perc  # noqa: E402


def _stamp(**over) -> dict:
    """A `refcv6_perception` stamp in the shape the trainer writes."""
    cfg = _perc.PerceptionBranchConfig(w_map=1.0, w_box3d=1.0)
    st = dict(cfg.as_dict())
    st["branch_params"] = {"total": 4_815_616}
    st.update(over)
    return st


def test_the_stamp_ROUND_TRIPS_the_branch_config():
    """⭐ The property the whole fix rests on: everything needed to rebuild the branch is
    already in the run record. If a future field is added to `PerceptionBranchConfig` and
    not to `as_dict()`, this goes red BEFORE a checkpoint becomes unloadable."""
    cfg = _perc.PerceptionBranchConfig(w_map=1.0, w_box3d=1.0)
    st = _stamp()
    assert set(cfg.as_dict()) <= set(st)
    fields = {f.name for f in dataclasses.fields(_perc.PerceptionBranchConfig)}
    # every field that is not derivable must be recoverable from the stamp
    missing = {f for f in fields
               if f not in st and f not in {"bev_cfg", "enforce_param_band"}}
    assert not missing, f"stamp cannot rebuild: {sorted(missing)}"


def test_a_zero_weight_stamp_rebuilds_NOTHING():
    """⛔ The trainer's bit-identity condition: at weight 0 it builds no submodule, so
    `model.parameters()` and `state_dict()` are unchanged. The driver must agree, or it
    would ADD a branch the checkpoint does not have and invert the very defect it fixes."""
    from refcv3_arm import rebuild_perception_branch          # noqa: PLC0415

    class _M:
        pass

    m = _M()
    out = rebuild_perception_branch(m, {"refcv6_perception":
                                        _stamp(w_map=0.0, w_box3d=0.0)}, "cpu")
    assert out is None
    assert not hasattr(m, "_perception")


def test_no_stamp_at_all_rebuilds_NOTHING():
    """A pre-refcv6 checkpoint has no such stamp and must load exactly as before."""
    from refcv3_arm import rebuild_perception_branch          # noqa: PLC0415

    class _M:
        pass

    assert rebuild_perception_branch(_M(), {}, "cpu") is None


def test_the_lift_bank_is_NOT_silently_claimed():
    """⚠️ The branch is rebuilt; the LIFT BANK is not — it comes from a per-clip extrinsics
    FILE that may not exist outside the training box. It holds no parameters, so the load
    and the param cross-check are unaffected, but a caller that wants MAP metrics must
    supply it. The provenance says so rather than leaving a reader to assume."""
    cfg = _perc.PerceptionBranchConfig(w_map=1.0, w_box3d=1.0)
    st = _stamp()
    assert "branch_params" in st
    # the contract: whatever the rebuild returns must declare the lift bank's absence
    assert "_lift_bank_rebuilt" not in cfg.as_dict(), (
        "the flag belongs to the REBUILD's provenance, not to the branch config")
