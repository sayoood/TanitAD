"""F-14 / SPEED_BAND derivation — THE BLOCKER, PINNED SO IT CANNOT ROT.

⛔ F-14 IS NOT IMPLEMENTED, DELIBERATELY. Its spec names two derivation inputs
and BOTH are unavailable on this corpus — one of them FORBIDDEN rather than
merely missing. This file pins the arithmetic so a later agent does not
re-derive C87 from scratch, and so the blocker cannot become a stale
"UNAVAILABLE" line nobody revisits (the failure class CLAUDE.md names twice).

THE SPEC, quoted (two independent locations):

  * ``…/2026-08-07-hierarchical-wm-redesign/HIERARCHY_VOCABULARY.md:87`` —
    *"`SPEED_BAND` | v_lo, v_hi | LON axis — SET BY THE TACTICAL LAYER (PI
    decision 2026-08-11): target speed is a tactical responsibility, computed
    from traffic-sign inputs (VLM/OCR speed-limit fields) and prior speed
    information (corridor speed statistics), bounded by the strategic layer's
    `REDUCE_TO` only as an upper envelope"* — quoted verbatim in ``v6.py``.
  * ``…/2026-08-16-diagram-conformance/DIAGRAM_CONFORMANCE.md:114`` —
    *"owns target speed (signs + priors) | PARTIAL | … Supervision/derivation
    NOT BUILT: no sign/OCR prior enters any loss or label path in v6 … Fix
    F-14"*, and ``:219``.

WHY IT IS NOT BUILDABLE (each half MEASURED, and each pinned below):

  1. **sign/OCR is FORBIDDEN, not missing.** The sign channel is released only
     as a PRESENCE flag; sign KIND and TEXT stay forbidden (RETRACTION_LOG C87)
     and the G1 sign-text gate is CLOSED at 0/31. A speed-limit prior needs
     exactly `kind == "speed"` and `text` — the two forbidden fields. And the
     two highest-scoring FALSE positives are a dashboard `30` roundel (0.927)
     and a hoarding (0.778), both scoring ABOVE true signs, so a threshold
     keeps the harmful errors. ⭐ A dashboard roundel is the EGO SPEEDOMETER:
     the sign path's worst failure here is an EGO ECHO arriving through the
     vision channel.
  2. **corridor speed priors need a corridor**, and this corpus has no map or
     lane graph — asserted in code, not only in prose.

AND THE ADMISSIBLE-LOOKING SUBSTITUTE (``vtarget_guarded``) IS HINDSIGHT EGO
GEOMETRY, so wiring it and calling it F-14 would swap a regulatory prior for a
behavioural one. The bar it would have to clear is pinned here too.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

_TS = (REPO / "TanitAD Research Hub" / "Architecture & Inference" /
       "Implementation" / "incoming" / "2026-08-04-target-speed")
_V6 = ROOT / "tanitad" / "models" / "v6.py"


# ===========================================================================
# 1. The absence is EXECUTABLE, not quoted — two independent paths
# ===========================================================================

_SIGN_TOKENS = ("sign_kind", "sign_text", "speed_limit", '"signs"', "signs[")


def test_NO_sign_or_OCR_field_reaches_any_v6_loss_or_label_path():
    """Probe 1 — the three files that own v6's losses and label joins."""
    for rel in ("scripts/train_v6_staged.py", "tanitad/models/v6.py",
                "scripts/s2_labels.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        # the blocker COMMENT names these tokens; strip it before searching, or
        # the pin would match its own documentation (the self-matching-filter
        # trap, in a test's costume).
        body = "\n".join(ln for ln in src.splitlines()
                         if not ln.lstrip().startswith(("#", "*", '"""')))
        for tok in _SIGN_TOKENS:
            assert tok not in body, f"{rel} now reads {tok!r}"


def test_NO_sign_derived_goal_token_is_emitted_by_the_tactical_vocabulary():
    """Probe 2 — a different path to the same absence: the vocabulary itself.

    ``STOP_POINT`` carries a ``reason`` slot whose enum includes ``"sign"``,
    and that is a REASON CODE, not a sign reading — it must not be mistaken for
    a sign-derived channel. ``SPEED_BAND`` carries no categorical slot at all.
    """
    from tanitad.models.v6 import (GOAL_CAT_ARG_TOKENS, STOP_REASONS,
                                   TACTICAL_GOAL_TOKENS)
    assert "SPEED_BAND" in TACTICAL_GOAL_TOKENS
    assert "SPEED_BAND" not in GOAL_CAT_ARG_TOKENS, (
        "SPEED_BAND acquired a categorical arg channel — if that channel is a "
        "sign reading, C87 forbids it")
    assert "sign" in STOP_REASONS  # a reason code, deliberately unchanged


def test_the_v6_SPEED_BAND_blocker_text_is_present_and_names_both_halves():
    """⛔ The blocker lives where the next implementer looks. If it is deleted,
    this fails — that is the whole point (the `test_physicalai_feature_readset`
    idiom: pin the fact, name the document to update)."""
    src = _V6.read_text(encoding="utf-8")
    blk = src.split("F-14 BLOCKER", 1)
    assert len(blk) == 2, "the F-14 blocker note was removed from v6.py"
    # the note is a wrapped `#:` comment block — normalise the wrapping before
    # searching, or this pin fails on a reflow rather than on a deletion.
    note = " ".join(blk[1][:3500].replace("#:", " ").split())
    for needle in ("KIND and TEXT stay forbidden", "CLOSED at 0/31",
                   "dashboard", "no map or lane graph", "vtarget_guarded",
                   "F10_F14_CELLS.md"):
        assert needle in note, f"the blocker no longer states: {needle}"


def test_the_corpus_has_no_corridor_to_take_a_speed_prior_FROM():
    """Asserted in CODE, not only in the dataset card — the second probe."""
    src = " ".join((REPO / "taniteval" / "taniteval" / "corridor.py").read_text(
        encoding="utf-8").split())
    assert "never a topology" in src
    assert "There is no map, no lane graph and no junction annotation" in src


# ===========================================================================
# 2. The admissibility verdicts, read from the BANKED artifact
# ===========================================================================

def _verdicts() -> dict:
    p = _TS / "raw" / "vt_admissibility.json"
    if not p.exists():
        pytest.skip(f"banked admissibility artifact absent: {p}")
    return json.loads(p.read_text(encoding="utf-8"))["verdicts"]


def test_a_SUPPLIED_target_speed_is_INADMISSIBLE_and_only_the_PREDICTED_is_not():
    """⛔ The shape any SPEED_BAND wiring must respect. Read off the banked
    verdicts rather than re-quoted from prose."""
    v = _verdicts()
    assert v["vt_guarded_SUPPLIED"]["verdict"] == "INADMISSIBLE"
    assert v["vt_oracle_SUPPLIED"]["verdict"] == "INADMISSIBLE"
    assert v["vt_guarded_AS_LABEL"]["verdict"] == "ADMISSIBLE"
    assert v["vt_predicted_from_image_and_v0"]["verdict"] == "ADMISSIBLE"


def test_the_situation_classifier_counterexample_tripwire_still_FIRES():
    """The 2026-08-03 binding rule's own counterexample: a goal carrying the
    situation classifier's output must be refused."""
    v = _verdicts()
    key = "COUNTEREXAMPLE_goal_carrying_the_situation_classifier"
    assert v[key]["verdict"] == "INADMISSIBLE"


# ===========================================================================
# 3. The BAR a SPEED_BAND head must clear — the ego-echo control
# ===========================================================================

def test_the_free_hold_v0_baseline_BEATS_every_learned_ego_arm():
    """⭐ THE ECHO CONTROL, pinned. On ego inputs, literally repeating the
    current speed's band scores 0.4066 and beats every trained arm — so a
    SPEED_BAND head is a DEAD PARAMETER unless it clears 0.4066 from VISION.

    Any future "the tactical layer sets target speed" claim that does not quote
    this number against itself is quoting an echo.
    """
    p = _TS / "raw" / "vt_four_families.json"
    if not p.exists():
        pytest.skip(f"banked four-family panel absent: {p}")
    ts = json.loads(p.read_text(encoding="utf-8"))["LONGITUDINAL"][
        "target_speed"]
    arms = ts["arms"]
    free = arms["hold_v0"]["band_top1"]
    assert free == pytest.approx(0.4066, abs=1e-4)
    assert arms["past_ridge"]["band_top1"] == pytest.approx(0.3673, abs=1e-4)
    assert ts["band_classifier"]["band_top1_argmax"] == pytest.approx(
        0.2465, abs=1e-4)
    for name, arm in arms.items():
        if name == "hold_v0":
            continue
        assert arm["band_top1"] < free, (
            f"{name} now beats the free baseline — the F-14 blocker's "
            f"'dead parameter' arithmetic has changed and must be re-read")


def test_setting_the_target_speed_is_HARDER_behind_a_lead():
    """The stratum that says where the lever actually is — and it is not where
    a speed-limit sign would help."""
    p = _TS / "raw" / "vt_four_families.json"
    if not p.exists():
        pytest.skip("banked four-family panel absent")
    bl = json.loads(p.read_text(encoding="utf-8"))["LONGITUDINAL"][
        "target_speed"]["by_lead_state"]
    lead = bl["LEAD"]["arms"]["hold_v0"]["band_top1"]
    no_lead = bl["NO_LEAD"]["arms"]["hold_v0"]["band_top1"]
    assert lead == pytest.approx(0.2995, abs=1e-4)
    assert no_lead == pytest.approx(0.4684, abs=1e-4)
    assert lead < no_lead


# ===========================================================================
# 4. The three MEASURED errors in DIAGRAM_CONFORMANCE's F-14 row
# ===========================================================================

_VTB = _TS / "code" / "vt_band_from_vision.py"


def test_the_named_F14_source_contains_NO_sign_OCR_or_corridor_content():
    """⛔ ERROR 1. ``DIAGRAM_CONFORMANCE.md:219`` calls this file *"the sign/OCR
    + corridor-prior source"*. It is a leave-one-episode-out LINEAR READOUT
    asking whether the pooled IMAGE feature predicts the VTARGET band better
    than repeating v0's band. It mentions none of those things."""
    if not _VTB.exists():
        pytest.skip(f"{_VTB} absent")
    txt = _VTB.read_text(encoding="utf-8").lower()
    for tok in ("sign", "ocr", "corridor", "speed limit", "speed_limit"):
        assert tok not in txt, f"{_VTB.name} now mentions {tok!r}"
    assert "identity_v0_band" in txt and "pooled" in txt


def test_the_named_F14_source_IS_tracked_contrary_to_the_row():
    """⛔ ERROR 2. The row calls it *"currently untracked incoming"*."""
    if not _VTB.exists():
        pytest.skip(f"{_VTB} absent")
    import subprocess
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--", str(_VTB.relative_to(REPO))],
        cwd=str(REPO), capture_output=True, text=True)
    assert out.stdout.strip(), (
        "vt_band_from_vision.py is NOT tracked — the conformance row's claim "
        "would then be correct and this pin must be retired")


def test_the_decisive_vision_measurement_has_NEVER_BEEN_RUN():
    """⛔ ERROR 3, and it is the one that matters: the probe that decides
    whether F-14 has ANY deployable form was never executed. Its output would
    carry ``identity_v0_band``; no banked JSON in the target-speed stream does.

    ⚠️ THIS PIN IS DESIGNED TO RETIRE ITSELF. When someone runs the probe, this
    fails — and the correct response is to delete it and re-read the blocker,
    not to loosen it.
    """
    raw = _TS / "raw"
    if not raw.exists():
        pytest.skip("target-speed raw dir absent")
    for j in raw.glob("*.json"):
        assert "identity_v0_band" not in j.read_text(encoding="utf-8"), (
            f"{j.name} carries the vision-band probe's output — the F-14 "
            f"blocker's 'never run' claim is now FALSE. Re-read it.")
