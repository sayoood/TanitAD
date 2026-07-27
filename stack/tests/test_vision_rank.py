"""Vision enters at rank ~= 16 — tanitad/models/vision_rank.py + the v4 wiring.

The brief was two-part: make the rank a FIRST-CLASS PARAMETER defaulting to 16,
and make raw-2048 concatenation IMPOSSIBLE TO SELECT BY ACCIDENT. The second half
is the one that needs adversarial tests, so every accident mode is driven here —
a default, a missing config key, a ``0``, a ``None``, a rank at ``state_dim`` and
a rank above it — not just the deliberate override.

Also pinned: the raw path, when explicitly allowed, is a BIT-EXACT identity with
zero parameters (the reproduction claim), and the projection is DECODE-SIDE, so
``encoder_touching_levers <= 2`` is unaffected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tanitad.models.flagship_v15 import V15Config  # noqa: E402
from tanitad.models.flagship_v4 import (FlagshipV4Head, V4Config,  # noqa: E402
                                        param_breakdown, v4_config)
from tanitad.models.vision_rank import (  # noqa: E402
    DEFAULT_VISION_RANK, DEGRADATION_ONSET_K, DOSE_RESPONSE, LEGACY_RAW_REASON,
    RAW_STATE_DIM, RawVisionRankRefused, VisionRankProjection,
    resolve_vision_rank)


# --------------------------------------------------------- first-class knob --
def test_the_rank_is_a_first_class_parameter_defaulting_to_16():
    assert DEFAULT_VISION_RANK == 16
    assert V15Config().vision_rank == 16
    assert V4Config().vision_rank == 16
    assert v4_config().vision_rank == 16
    # and it is settable without touching any other field
    assert V4Config(vision_rank=32).vision_rank == 32


def test_the_dose_response_travels_with_the_lever():
    """A rank quoted without its curve is the kind of bare number this program
    has retracted before. The measured ladder ships with the code."""
    for k in ("ego_only", "k16", "k64", "k256", "k2048"):
        assert isinstance(DOSE_RESPONSE[k], float)
    assert DOSE_RESPONSE["k2048_separated_vs_chance"] is False
    # the narrow claim is on the record, so the lever cannot be misquoted as a gain
    assert "NOT 'vision adds value'" in DOSE_RESPONSE["claim"]
    assert DOSE_RESPONSE["k16"] > DOSE_RESPONSE["k64"] > DOSE_RESPONSE["k2048"]


# ------------------------------------------- raw-2048 cannot be an accident --
@pytest.mark.parametrize("bad", [0, -1, None, RAW_STATE_DIM, RAW_STATE_DIM + 1,
                                 4096])
def test_every_accident_mode_that_means_RAW_is_refused(bad):
    """⭐ The guard, driven with each way a caller lands on raw WITHOUT meaning to.

    A missing config key deserialises to None; a 'disabled' knob is written as 0;
    'use the whole state' is written as the state's own width or something larger.
    All of them are the arm MEASURED not separated from chance.
    """
    with pytest.raises(RawVisionRankRefused, match="RAW flat vision path"):
        resolve_vision_rank(bad, RAW_STATE_DIM)


def test_the_flag_ALONE_is_not_enough_a_written_reason_is_required():
    """A boolean can be flipped absent-mindedly; a sentence cannot."""
    with pytest.raises(RawVisionRankRefused):
        resolve_vision_rank(0, RAW_STATE_DIM, allow_raw=True)            # no reason
    with pytest.raises(RawVisionRankRefused):
        resolve_vision_rank(0, RAW_STATE_DIM, allow_raw=True, reason="   ")
    with pytest.raises(RawVisionRankRefused):
        resolve_vision_rank(0, RAW_STATE_DIM, reason="I want raw")       # no flag
    # both together, and only then
    assert resolve_vision_rank(0, RAW_STATE_DIM, allow_raw=True,
                               reason=LEGACY_RAW_REASON) == RAW_STATE_DIM


def test_the_config_refuses_raw_at_CONSTRUCTION_not_at_step_1():
    """A bad rank must cost zero GPU seconds, not be found on a pod at hour 3."""
    with pytest.raises(RawVisionRankRefused):
        V4Config(vision_rank=0)
    with pytest.raises(RawVisionRankRefused):
        V4Config(vision_rank=2048)
    cfg = V4Config(vision_rank=0, allow_raw_vision=True,
                   vision_rank_reason=LEGACY_RAW_REASON)
    assert cfg.vision_rank == cfg.state_dim


def test_a_rank_at_the_degradation_onset_warns_on_the_record(capsys):
    """k=64 is admissible but MEASURED worse (3.000x vs 3.685x). It must not be
    silent — a step down the ladder belongs in the run's own log."""
    assert resolve_vision_rank(DEGRADATION_ONSET_K, RAW_STATE_DIM) == 64
    out = capsys.readouterr().out
    assert "degradation onset" in out and "64" in out

    resolve_vision_rank(16, RAW_STATE_DIM)
    assert capsys.readouterr().out == "", "rank 16 should be silent"


# ------------------------------------------------- the projection's contract --
def test_projection_shapes_and_that_rank_16_is_what_the_reader_sees():
    p = VisionRankProjection(RAW_STATE_DIM, 16)
    assert p.out_dim == 16 and p.is_raw is False
    x = torch.randn(5, RAW_STATE_DIM)
    assert p(x).shape == (5, 16)


def test_the_ALLOWED_raw_path_is_a_bit_exact_identity_with_zero_parameters():
    """The reproduction claim: an arm that opted into raw must be indistinguishable
    from having no projection at all. Proven as Δ == 0.0, not asserted."""
    p = VisionRankProjection(RAW_STATE_DIM, RAW_STATE_DIM)
    assert p.is_raw and p.out_dim == RAW_STATE_DIM
    assert sum(q.numel() for q in p.parameters()) == 0
    x = torch.randn(4, RAW_STATE_DIM)
    d = (p(x) - x).abs().max().item()
    assert d == 0.0, f"raw path is not bit-exact: max|Δ| = {d}"
    assert torch.equal(p(x), x)


def test_basis_seeding_round_trips_and_a_wrong_shape_raises():
    p = VisionRankProjection(64, 8)
    assert not bool(p.basis_loaded)
    W = torch.linalg.qr(torch.randn(64, 8))[0]              # [64, 8] orthonormal
    p.init_from_basis(W)
    assert bool(p.basis_loaded)
    x = torch.randn(3, 64)
    assert torch.allclose(p(x), x @ W, atol=1e-5)

    with pytest.raises(ValueError, match=r"\(64, 8\)"):
        p.init_from_basis(torch.randn(64, 9))
    with pytest.raises(RawVisionRankRefused, match="RAW identity"):
        VisionRankProjection(64, 64).init_from_basis(torch.randn(64, 64))


# --------------------------------------------------------- the v4 head wiring --
def _head(**kw):
    cfg = V4Config(state_dim=2048, cond_imagination=False, window=4,
                   horizons=tuple(range(1, 5)), n_anchors=8, **kw)
    cfg.decoder.d, cfg.decoder.layers, cfg.decoder.n_heads = 32, 1, 2
    cfg.d_token = 32
    return FlagshipV4Head(cfg)


def test_the_v4_factorised_heads_read_vision_at_rank_16_not_at_2048():
    """⭐ The wiring proof. The factorised LAT/LON/DIST heads are v4's FLAT reader
    — they took ``states[:, -1]`` straight into a Linear, which is the exact shape
    the swamping dose-response was measured on."""
    h = _head()
    assert h.vision_rank_proj.out_dim == 16
    assert h.lat_head[0].in_features == 16, (
        f"the factorised reader still takes {h.lat_head[0].in_features} inputs — "
        f"the rank lever is not attached")
    for m in (h.lon_head, h.dist_head):
        assert m[0].in_features == 16

    # the raw control, explicitly opted into: the reader is 2048-wide again
    raw = _head(vision_rank=0, allow_raw_vision=True,
                vision_rank_reason=LEGACY_RAW_REASON)
    assert raw.vision_rank_proj.is_raw
    assert raw.lat_head[0].in_features == 2048


def test_the_projection_is_DECODE_SIDE_so_the_encoder_lever_count_is_unchanged():
    """v4 is at 2 of 2 encoder-touching levers and ``encoder_touching_levers <= 2``
    is a KILL secondary. The projection must live in the HEAD."""
    h = _head()
    owners = {n.split(".")[0] for n, _ in h.named_parameters()
              if "vision_rank_proj" in n}
    assert owners == {"vision_rank_proj"}
    # nothing named like a trunk module appears anywhere in the head
    assert not any(n.startswith(("encoder.", "predictor."))
                   for n, _ in h.named_parameters())
    # and the projection is counted, so a param audit cannot miss it
    assert param_breakdown(h)["vision_rank_proj"] > 0
    raw = _head(vision_rank=0, allow_raw_vision=True,
                vision_rank_reason=LEGACY_RAW_REASON)
    assert param_breakdown(raw)["vision_rank_proj"] == 0


def test_the_head_forward_still_runs_end_to_end_at_rank_16():
    h = _head()
    states = torch.randn(3, 4, 2048)
    out = h(states, torch.full((3,), 8.0), vt_band=torch.zeros(3, dtype=torch.long),
            vt_speed=torch.full((3,), 8.0), route=torch.zeros(3, dtype=torch.long),
            route_graded=torch.zeros(3))
    assert out["wp_seq"].shape == (3, 4, 2)
    assert out["lat_logits"].shape[0] == 3
    assert torch.isfinite(out["sel_score"]).all()


def test_a_legacy_config_dict_without_the_key_routes_through_the_named_override():
    """The compat path a pre-lever checkpoint takes, and it self-documents."""
    hc = {"state_dim": 2048}
    assert "vision_rank" not in hc
    hc.update(vision_rank=RAW_STATE_DIM, allow_raw_vision=True,
              vision_rank_reason=LEGACY_RAW_REASON)
    cfg = V4Config(**hc)
    assert cfg.vision_rank == 2048
    assert "legacy checkpoint" in cfg.vision_rank_reason
