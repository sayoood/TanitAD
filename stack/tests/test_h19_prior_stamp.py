"""H19-STAMP-1 — a run must RECORD which side of the ``man5`` guard it took.

⛔⛔ THE DEFECT. ``RefCV3Model._hook`` sets ``man5 = None`` under any non-``kin3``
tactical vocabulary. It is GUARDED, so it never raises; ``refc.py``'s
``reweight = maneuver_logits if maneuver_logits is not None else man_logits``
silently substitutes the CORE's own aux head; and until 2026-09-06 **no artifact
of any kind recorded that this had happened**. refcv4b's own ``config.json``
(MEASURED, two dev-box copies, 6,416 B, ``tac_vocab_version = "v7.0"``) carries
ten ``param_breakdown`` keys and a ``registered_delta`` and not one H19 field, so
a v7.0 arm and a kin3 arm are indistinguishable on this axis from the record.

⭐ WHAT IS ACTUALLY LOST — and it is NOT the prior. MEASURED 2026-09-06, CPU
smoke rung, zero GPU (``Research/2026-09-06-h19-prior/``). Perturbing ONLY
``lat_head_tac``/``lon_head_tac`` and reading what the DECODER receives:

    decoder maneuver_logits   kin3 MOVED 24.4632   v7.0 UNCHANGED 0.0
    decoder lat_prior         kin3 MOVED 21.8972   v7.0 UNCHANGED 0.0
    anchor_logits             kin3 MOVED 18.0365   v7.0 UNCHANGED 0.0

The prior is LIVE under both; the TACTICAL feed is cut, and the cut is an EXACT
ZERO rather than a degradation. Both halves are pinned below, because a stamp
that let a reader conclude "H19 is off" would trade one wrong record for another.

⛔ TWO-SIDED THROUGHOUT. Every behavioural test carries the opposite build in the
SAME BREATH. A stamp stuck in either position fails here — which is the whole
point: the sibling defect shipped the same night was a manifest string that read
identically on the real and the unverified arms, and therefore measured nothing.

⭐ REACHED, NOT MERELY CORRECT. ``test_the_flag_reaches_a_real_forward`` asserts
the bool arrives in the model's OWN output on an ordinary ``model(frames)`` call.
A guard wired only into ``preflight`` covers one launch path of two —
``refc_v3_train.main`` runs ``preflight`` only under ``--preflight`` and
otherwise calls ``train()`` directly.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/stack/tests
_STACK = os.path.dirname(_HERE)                             # <repo>/stack
if _STACK not in sys.path:
    sys.path.insert(0, _STACK)

from tanitad.refs import refc, refc_v3 as v3                # noqa: E402

VOCABS = ("kin3", "v7.0")


def _build(vocab: str, *, hier: bool = True):
    """A cheap CPU build at the requested tactical vocabulary."""
    cfg = v3.refc_v3_smoke_config(hier=hier)
    cfg.core.encoder = refc.CNNEncoderConfig(
        in_channels=cfg.core.encoder.in_channels, image_size=64,
        base_width=8, blocks=(1, 1, 1, 1))
    cfg.tac_vocab_version = vocab
    m = v3.RefCV3Model(cfg)
    m.eval()
    return m, cfg


def _inputs(cfg, b: int = 2):
    enc = cfg.core.encoder
    return (torch.randn(b, cfg.core.window, enc.in_channels, enc.image_size,
                        enc.image_width or enc.image_size),
            torch.full((b,), 10.0))


def _forward_spying_on_the_decoder(m, frames, v0):
    """Run a normal forward, capturing what the DECODER was handed.

    The decoder's argument is the only place the question can be answered
    honestly: the hook's emission and the core's fallback both arrive there.
    """
    seen: dict = {}
    real = m.core.decoder.forward

    def spy(*a, **kw):
        seen["maneuver_logits"] = kw.get("maneuver_logits")
        seen["lat_prior"] = kw.get("lat_prior")
        return real(*a, **kw)

    m.core.decoder.forward = spy
    try:
        with torch.no_grad():
            out = m(frames, v0=v0, steps=0)
    finally:
        m.core.decoder.forward = real
    return out, seen


# ==========================================================================
# 1 — THE STAMP, BOTH DIRECTIONS IN ONE BREATH
# ==========================================================================
def test_stamp_is_two_sided():
    """⛔ THE CONTRACT. ``applied`` on kin3, ``dropped(...)`` on v7.0 — and the
    two must DIFFER, which is the assertion the sibling defect failed."""
    kin, _ = _build("kin3")
    v7, _ = _build("v7.0")
    s_kin, s_v7 = v3.h19_prior_stamp(kin), v3.h19_prior_stamp(v7)

    assert s_kin["h19_prior"] == "applied", s_kin
    assert s_v7["h19_prior"].startswith("dropped("), s_v7
    assert s_kin["h19_prior"] != s_v7["h19_prior"], \
        "H19-STAMP-1 FAILED: the stamp reads the same on both vocabularies, " \
        "so it measures nothing — which is precisely the bug it exists for"

    # ⭐ The reason travels WITH the verdict: a stamp saying only "dropped" is
    # a second thing for a future reader to go and look up.
    assert "v7.0" in s_v7["h19_prior"] and "POSITIONAL" in s_v7["h19_prior"]
    assert "8-wide" in s_v7["h19_prior"], s_v7["h19_prior"]

    # ...and the SOURCE field names what still feeds the prior, in both.
    assert s_kin["h19_prior_source"] == "tactical_z_tac_via_man5"
    assert s_v7["h19_prior_source"] == "core_aux_kin3"


def test_flat_arm_is_stamped_n_a_and_never_dropped():
    """⭐ A flat arm has no tactical brain, so there is no tactical feed to
    lose. ``n/a`` and ``dropped`` have different remedies and must not collide.

    CONTROL, same breath: the hier build at the SAME vocabulary DOES read
    ``dropped``, so the ``n/a`` above is scoped to the flat arm and is not a
    stamp that has simply stopped working."""
    flat, _ = _build("v7.0", hier=False)
    s_flat = v3.h19_prior_stamp(flat)
    assert s_flat["h19_prior"].startswith("n/a("), s_flat
    assert "dropped" not in s_flat["h19_prior"]

    hier, _ = _build("v7.0", hier=True)
    assert v3.h19_prior_stamp(hier)["h19_prior"].startswith("dropped(")


# ==========================================================================
# 2 — REACHED: the flag arrives on an ordinary forward, not only in preflight
# ==========================================================================
@pytest.mark.parametrize("vocab", VOCABS)
def test_the_flag_reaches_a_real_forward(vocab):
    """⭐⭐ WIRING, NOT CORRECTNESS. ``model(frames)`` — no preflight, no
    special entry point — must carry the bool out in its own dict."""
    m, cfg = _build(vocab)
    frames, v0 = _inputs(cfg)
    with torch.no_grad():
        out = m(frames, v0=v0, steps=0)
    assert "h19_tactical_feed" in out, \
        "the stamp does not reach the forward output; a record wired only " \
        "into preflight covers one launch path of two"
    assert out["h19_tactical_feed"] is (vocab == "kin3")


def test_the_runtime_flag_and_the_record_stamp_cannot_disagree():
    """⭐ ONE PREDICATE, ONE CONSUMER. The forward's bool and the record's
    string are two READOUTS of one guard, never two copies of one condition."""
    for vocab in VOCABS:
        m, cfg = _build(vocab)
        frames, v0 = _inputs(cfg)
        with torch.no_grad():
            out = m(frames, v0=v0, steps=0)
        applied = v3.h19_prior_stamp(m)["h19_prior"] == "applied"
        assert out["h19_tactical_feed"] is applied, vocab


# ==========================================================================
# 3 — THE ANTI-MISREADING CONTROL: the prior itself is LIVE under BOTH
# ==========================================================================
@pytest.mark.parametrize("vocab", VOCABS)
def test_h19_prior_is_live_under_both_vocabularies(vocab):
    """⛔ "dropped" NAMES THE TACTICAL FEED, NOT THE PRIOR. ``refc.py`` falls
    back to the core's own 5-way, so the decoder receives a valid
    ``maneuver_logits [B, 5]`` either way. If this ever fails, the stamp's
    wording is wrong and must change with it."""
    m, cfg = _build(vocab)
    frames, v0 = _inputs(cfg)
    _, seen = _forward_spying_on_the_decoder(m, frames, v0)
    ml = seen["maneuver_logits"]
    assert ml is not None, \
        "the H19 prior really IS dark here — the stamp's wording under-states it"
    assert ml.shape[-1] == refc.N_MANEUVERS
    # The graft that carries it is named from the BUILT decoder, so a factored
    # build (`maneuver_to_anchor is None`) cannot be misread as "no H19".
    assert v3.h19_prior_stamp(m)["h19_graft"] != "unknown(no decoder)"


# ==========================================================================
# 4 — THE MEASURED LOSS: an EXACT ZERO, pinned by mutation
# ==========================================================================
@pytest.mark.parametrize("vocab", VOCABS)
def test_tactical_heads_reach_the_anchor_prior_only_under_kin3(vocab):
    """⭐⭐ THE NUMBER THE STAMP EXISTS TO MAKE LEGIBLE. Move ONLY the tactical
    action heads and read what the decoder receives.

    ⛔ Under ``v7.0`` the answer is EXACTLY 0.0 on three tensors — the 8-wide v7
    heads are trained by CE and reach the trajectory decoder through E7 and E9
    only, never through the anchor prior. Under ``kin3`` all three move. Both
    directions are asserted, so a seam stuck either open or shut fails.
    """
    torch.manual_seed(0)
    m, cfg = _build(vocab)
    frames, v0 = _inputs(cfg)
    out_a, seen_a = _forward_spying_on_the_decoder(m, frames, v0)
    base = (seen_a["maneuver_logits"].clone(), seen_a["lat_prior"].clone(),
            out_a["anchor_logits"].clone())

    with torch.no_grad():
        for h in (m.lat_head_tac, m.lon_head_tac):
            h.weight.add_(torch.randn_like(h.weight) * 5.0)
            h.bias.add_(torch.randn_like(h.bias) * 5.0)
    out_b, seen_b = _forward_spying_on_the_decoder(m, frames, v0)
    after = (seen_b["maneuver_logits"], seen_b["lat_prior"],
             out_b["anchor_logits"])
    deltas = [float((x - y).abs().max()) for x, y in zip(base, after)]

    if vocab == "kin3":
        assert all(d > 1e-3 for d in deltas), (
            "the tactical brain no longer reaches the anchor prior under kin3 "
            "either — the seam is gone, not merely unstamped", deltas)
    else:
        assert deltas == [0.0, 0.0, 0.0], (
            "a v7.0 build now moves the anchor prior from its tactical heads; "
            "the stamp's 'dropped' reading is stale", deltas)
