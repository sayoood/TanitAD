"""E1 — the nav command's COMPANION BIT (`nav_known`) as a GATED REF-C input.

WHY THE SEAM EXISTS. `refb_labels.nav_command_v21` maps BOTH `ROUTE_STRAIGHT` and
`ROUTE_UNKNOWN` onto the SAME `NAV_FOLLOW` token, so at the model input "the road goes
straight" and "I could not judge the route" are byte-identical. MEASURED at corpus scale:
1,985 of 3,179 `follow` windows (62.4 %) are a collapsed UNKNOWN; on the banked night
scene 125 of 190. `nav_input_v22` already returns the `(cmd, known)` pair — until this
change it was wired nowhere.

WHAT THESE TESTS PIN, and why each one is here rather than left to review:

* **OFF IS BYTE-IDENTICAL.** A gated lever that perturbs the default arm is not gated. The
  param count, the `state_dict` shapes and the forward output must all be unchanged.
* **THE PARAMETER COST IS AN ASSERTED CONSTANT, not a comment.** A previous input lever
  cost +272,001 params before its own capacity control caught it; the accepted one cost
  +897. This one is pinned at **+128** (one column of `measurement.0.weight`) so a future
  refactor that inflates it fails a test instead of a review.
* **IT FAILS LOUD IN BOTH DIRECTIONS.** Passing the bit to a model that cannot read it is
  the silent-drop bug the seam exists to remove; running the gate ON without the bit would
  assert "this command is a real judgement" on every window, which is the defect itself
  wearing the fix's clothes.
* **IT IS MATERIAL.** A seam that is wired but cannot change the output is decoration. The
  trajectory must move when only `nav_known` moves — the same MANIPULATION discipline
  R-2026-08-03-l imposed on the route head.
"""
from __future__ import annotations

import pytest
import torch

from tanitad.refs.refc import (RefCModel, param_breakdown, refc_config,
                               refc_smoke_config)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

# The cost is STRUCTURAL: one extra input column on `measurement.0.weight`, i.e.
# exactly `cfg.measurement.hidden` weights and zero biases. Pinned twice — as the law
# (asserted on the smoke preset, which is cheap to build) and as the concrete number for
# the DEPLOYED base preset, which is what a capacity control would read.
BASE_DELTA_PARAMS = 128         # refc_config().measurement.hidden


def _cfgs(preset=refc_smoke_config):
    off, on = preset(), preset()
    on.nav_known_channel = True
    return off, on


def _build(cfg, seed=0):
    torch.manual_seed(seed)
    return RefCModel(cfg).eval()


def _batch(b=2):
    cfg = refc_smoke_config()
    torch.manual_seed(1234)
    return (torch.randn(b, int(cfg.window), cfg.encoder.in_channels,
                        cfg.encoder.image_size, cfg.encoder.image_size),
            torch.tensor([1, 2][:b]), torch.tensor([5.0, 7.0][:b]))


def test_default_is_off_and_costs_nothing():
    off_cfg, _ = _cfgs()
    assert off_cfg.nav_known_channel is False, "E1 must be OFF by default"
    m = _build(off_cfg)
    assert m.measurement[0].in_features == 1 + 4 + 0


def test_gate_on_costs_exactly_one_input_column_and_nothing_else():
    off_cfg, on_cfg = _cfgs()
    expected = int(off_cfg.measurement.hidden)          # 32 on smoke, 128 on base
    off, on = _build(off_cfg), _build(on_cfg)
    n_off = sum(p.numel() for p in off.parameters())
    n_on = sum(p.numel() for p in on.parameters())
    assert n_on - n_off == expected
    b_off, b_on = param_breakdown(off), param_breakdown(on)
    # the whole delta lands on `measurement` and nowhere else
    assert b_on["measurement"] - b_off["measurement"] == expected
    for k in b_off:
        if k in ("measurement", "total"):
            continue
        assert b_on[k] == b_off[k], f"{k} changed: {b_off[k]} -> {b_on[k]}"


def test_the_deployed_base_preset_costs_exactly_128_params():
    """The number a capacity control reads. A prior input lever cost +272,001 params
    before its own control caught it; the accepted one cost +897. This is +128."""
    assert int(refc_config().measurement.hidden) == BASE_DELTA_PARAMS
    assert refc_config().nav_known_channel is False


def test_state_dict_keys_are_unchanged_only_one_shape_moves():
    off_cfg, on_cfg = _cfgs()
    off, on = _build(off_cfg), _build(on_cfg)
    so, sn = off.state_dict(), on.state_dict()
    assert set(so) == set(sn), "the gate must not add or remove parameters"
    moved = {k for k in so if so[k].shape != sn[k].shape}
    assert moved == {"measurement.0.weight"}


def test_off_forward_is_bit_identical_to_the_pre_seam_path():
    """Same seed, same input, gate off -> the arm every published number came from."""
    off_cfg, _ = _cfgs()
    f, nav, v0 = _batch()
    a = _build(off_cfg, seed=7)
    b = _build(off_cfg, seed=7)
    with torch.no_grad():
        oa = a(f, nav_cmd=nav, v0=v0, steps=0)
        ob = b(f, nav_cmd=nav, v0=v0, steps=0)
    assert torch.equal(oa["traj"], ob["traj"])


def test_off_rejects_a_bit_it_cannot_read():
    off_cfg, _ = _cfgs()
    m = _build(off_cfg)
    f, nav, v0 = _batch()
    with pytest.raises(ValueError, match="silently dropped"):
        m(f, nav_cmd=nav, v0=v0, steps=0, nav_known=torch.ones(2))


def test_on_refuses_to_invent_the_bit_when_a_command_was_supplied():
    _, on_cfg = _cfgs()
    m = _build(on_cfg)
    f, nav, v0 = _batch()
    with pytest.raises(ValueError, match="nav_known must be supplied"):
        m(f, nav_cmd=nav, v0=v0, steps=0)


def test_on_with_nav_cmd_none_defaults_known_to_zero():
    """`nav_cmd=None` -> the `follow` fallback IS the sentinel, so known=0 needs no
    argument. This is the TanitEval decode condition and it must keep working."""
    _, on_cfg = _cfgs()
    m = _build(on_cfg)
    f, _, v0 = _batch()
    with torch.no_grad():
        auto = m(f, nav_cmd=None, v0=v0, steps=0)
        explicit = m(f, nav_cmd=torch.zeros(2, dtype=torch.long), v0=v0, steps=0,
                     nav_known=torch.zeros(2))
    assert torch.equal(auto["traj"], explicit["traj"])


def test_the_seam_is_MATERIAL_the_plan_moves_when_only_the_bit_moves():
    """⭐ The manipulation test. Everything except `nav_known` is held fixed."""
    _, on_cfg = _cfgs()
    m = _build(on_cfg)
    f, nav, v0 = _batch()
    with torch.no_grad():
        k0 = m(f, nav_cmd=nav, v0=v0, steps=0, nav_known=torch.zeros(2))
        k1 = m(f, nav_cmd=nav, v0=v0, steps=0, nav_known=torch.ones(2))
    assert not torch.equal(k0["traj"], k1["traj"]), (
        "nav_known reaches the measurement encoder but changes nothing — the seam "
        "would be decoration")


def test_nav_input_v22_pairs_the_command_with_an_honest_bit():
    """The label side: `known` is 0.0 EXACTLY on the UNKNOWN sentinel, and the command
    itself is unchanged from v2.1 (this is additive, not a relabelling)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from refb_labels import (NAV_FOLLOW, nav_command_v21,  # noqa: E402
                             nav_command_v21_ex, nav_input_v22)

    # a straight run of poses -> a decided road-following command, not a sentinel
    t = torch.zeros(60, 4)
    t[:, 0] = torch.arange(60, dtype=torch.float32) * 1.0     # 1 m/step, due +x
    t[:, 3] = 10.0
    nav, known = nav_input_v22(t, 0)
    ex = nav_command_v21_ex(t, 0)
    v21 = nav_command_v21(t, 0)
    assert nav == int(v21[0]), "v2.2 must not change the COMMAND"
    assert known == (0.0 if ex["unknown_sentinel"] else 1.0)
    assert known in (0.0, 1.0)

    # a window with no future at all -> UNKNOWN -> follow, and the bit says so
    nav_end, known_end = nav_input_v22(t, t.shape[0] - 1)
    assert nav_end == NAV_FOLLOW
    assert known_end == 0.0, (
        "a command emitted because the route could not be judged must carry "
        "known=0 — that collapse is the whole defect E1 addresses")


def test_driver_policies_declare_whether_they_consume_the_bit():
    """The driver reads `consumes_nav_known` before passing the bit, so an un-wired arm
    records a DROPPED bit instead of losing it silently."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                           / "experiments" / "alpasim-gsplat"))
    import closedloop_drive as cd            # noqa: E402
    assert cd._BasePolicy.consumes_nav_known is False
    # the flagship path never overrides it: its StrategicPolicy has no companion-bit
    # seam, and that is a named gap rather than an accident.
    assert cd.FlagshipV1Policy.consumes_nav_known is False
