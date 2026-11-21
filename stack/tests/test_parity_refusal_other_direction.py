"""The refusal message's OTHER-DIRECTION count must name the OTHER direction.

⛔ WHY THIS FILE EXISTS. `guard_corpus_build` records BOTH overlap counts whatever the
role, on purpose -- "the fact a build cannot know it needs is exactly the fact that must
ride along uninvited". The refusal message is where a human meets that fact, and its
number ternary was INVERTED relative to its label ternary: both branches printed the
HAZARD count (already on the line above) under the OTHER direction's name.

MEASURED 2026-09-18 on a real refusal over 139 clip ids: the message said
"11 in the val deployment" while the same call's record says `in_deployed_val: 0`. The 11
are in the parity TRAIN split. An operator would have hunted the wrong leak.

⚠️ It survived because the line is reached ONLY ON A REFUSAL -- the path nobody exercises
until something is already wrong -- and nothing asserted it. A message read only in an
emergency is the one that most needs pinning.

⭐ Both arms use DISTINCT counts (7 vs 3) so an inverted implementation cannot pass by
coincidence. Equal counts are what would have hidden it.
"""
from __future__ import annotations

import pytest

from tanitad.data import parity


TRAIN_IDS = [f"t{i:03d}" for i in range(7)]      # 7 -- in the parity TRAIN split
VAL_IDS = [f"v{i:03d}" for i in range(3)]        # 3 -- in the deployed VAL
ALL_IDS = TRAIN_IDS + VAL_IDS


@pytest.fixture()
def oracles(monkeypatch):
    """Both membership oracles stubbed, so the test pins the MESSAGE, not the corpus."""
    monkeypatch.setattr(parity, "clips_in_parity_train",
                        lambda ids, path=None: [i for i in ids if i in set(TRAIN_IDS)])
    monkeypatch.setattr(parity, "clips_in_deployed_val",
                        lambda ids, path=None: [i for i in ids if i in set(VAL_IDS)])


def _refusal_text(role: str) -> str:
    with pytest.raises(parity.ParityViolation) as ex:
        parity.guard_corpus_build(ALL_IDS, label="UNIT", role=role, mode="refuse")
    return str(ex.value)


def test_heldout_role_reports_the_VAL_count_as_the_other_direction(oracles):
    """role=eval: the hazard is the parity-train overlap (7); the OTHER direction is
    the deployed-val overlap (3)."""
    msg = _refusal_text("eval")
    assert "disqualifying    : 7" in msg, msg
    assert "other direction : 3 in the val deployment" in msg, msg
    # ⛔ the defect, stated as the thing that must NOT appear
    assert "other direction : 7" not in msg, msg


def test_supervision_role_reports_the_TRAIN_count_as_the_other_direction(oracles):
    """Undeclared role: the hazard is the deployed-val overlap (3); the OTHER
    direction is the parity-train overlap (7)."""
    msg = _refusal_text("")
    assert "disqualifying    : 3" in msg, msg
    assert "other direction : 7 in the parity train split" in msg, msg
    assert "other direction : 3" not in msg, msg


def test_the_record_and_the_message_agree(oracles):
    """⭐ The control that makes the two tests above meaningful: the RECORD was always
    right, and it is the message that drifted from it. If they ever disagree again, the
    message is the one to fix -- the record is what callers write into manifests."""
    #: ⚠️ `mode="exclude"` — the only non-raising mode. "keep" is NOT accepted
    #: (`guard_corpus_build` allows only "refuse"/"exclude"), which this test
    #: discovered the hard way and which had also reached two CLIs as an offered
    #: choice. Excluding filters the hazard but still records BOTH counts.
    kept, rec = parity.guard_corpus_build(ALL_IDS, label="UNIT", role="eval",
                                          mode="exclude")
    assert rec["in_parity_train"] == 7
    assert rec["in_deployed_val"] == 3
    msg = _refusal_text("eval")
    assert f"disqualifying    : {rec['in_parity_train']}" in msg
    assert f"other direction : {rec['in_deployed_val']} " in msg
