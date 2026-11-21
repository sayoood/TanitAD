"""The participation floor may never again be quoted as a corpus-free scalar.

MEASURED 2026-08-23 -- frozen DINOv3 ViT-L/16, patch tokens mean-pooled per
frame, through `spectrum_report` itself, n=1440 in every row:

    12 physicalai-val clips   (the corpus 8.56 is sourced to)   ->   5.756
   130-clip lead corpus       (the corpus 40.77 is sourced to)  ->  20.228 +- 0.327
   130-clip lead corpus, full sample n=5617                     ->  20.516

Neither published number survives.  The code says 8.56; E_TRUNK_3_LADDER.md says
40.77; our own instrument says 5.756 and 20.23 depending only on WHICH CLIPS the
1440 frames come from.  That 3.51x is EPISODE DIVERSITY at fixed encoder, fixed
d, fixed n and fixed instrument (H-RANK-23) -- and sample size is NOT the
confound (H-RANK-21 REFUTED: the estimator reads 0.97-1.00x of closed-form truth
at n=1440 for the concentrated spectra we actually have).

⛔ THE FAILURE THIS FILE EXISTS TO PREVENT is the one the repo has already paid
for four times in other costumes (`df` on a pod, `free` on Thor, cgroup
`usage_in_bytes`, `step_s`): a number that is true for ONE scope, quoted where
that scope does not apply.  Here it is worse than a wrong reading, because the
number DECIDES: champ30k is on record as FAILING the O6 collapse gate at
6.489 < 8.56 -- against a value no live instrument reproduces, measured on a
different corpus, at half the ambient dimension.

⚠️ These tests deliberately do NOT assert that champ30k passes.  It is d=2048 and
the DINOv3 column is d=1024; participation scales with the ambient dimension, so
"6.489 beats 5.756" is just as inadmissible as the original FAIL.  The honest
state is UNKNOWN until a matched-d reference exists, and the tests below pin
exactly that: the provenance, not a verdict.
"""
from __future__ import annotations

import pytest

from tanitad.models.v6 import (O6_FLOOR_IS_CORPUS_AND_DIM_SPECIFIC,
                               O6_PARTICIPATION_FLOOR,
                               O6_PARTICIPATION_REFERENCES)

VAL12 = "dinov3_vitl16_meanpooled/physicalai-val-12clips/n1440/d1024"
LEAD1440 = "dinov3_vitl16_meanpooled/lead-130clips/n1440/d1024"
LEAD5617 = "dinov3_vitl16_meanpooled/lead-130clips/n5617/d1024"


def test_every_reference_names_its_corpus_its_n_and_its_dimension():
    """A bare number is exactly the defect. Each key must carry all three."""
    assert O6_PARTICIPATION_REFERENCES, "the measured references were dropped"
    for key in O6_PARTICIPATION_REFERENCES:
        encoder, corpus, n, d = key.split("/")
        assert encoder and corpus
        assert n.startswith("n") and n[1:].isdigit(), f"{key}: no sample size"
        assert d.startswith("d") and d[1:].isdigit(), f"{key}: no ambient dimension"


def test_the_code_floor_is_NOT_reproduced_on_its_own_corpus():
    """8.56 vs a measured 5.756 on the very clips it is sourced to."""
    got = O6_PARTICIPATION_REFERENCES[VAL12]
    assert got == pytest.approx(5.756, abs=0.01)
    assert abs(got - O6_PARTICIPATION_FLOOR) / O6_PARTICIPATION_FLOOR > 0.25, (
        "if these ever agree, the floor has been re-derived -- update this test "
        "and the docstring together, and say which measurement moved")


def test_episode_diversity_moves_participation_3_5x_at_FIXED_n_and_d():
    """The mechanism behind the whole 8.56 / 20.52 / 40.77 family."""
    val, lead = O6_PARTICIPATION_REFERENCES[VAL12], O6_PARTICIPATION_REFERENCES[LEAD1440]
    assert lead / val == pytest.approx(3.51, abs=0.1), (
        "same encoder, same d=1024, same n=1440, same instrument -- only the "
        "clips differ. This ratio IS the portability defect")


def test_sample_size_is_NOT_the_confound():
    """H-RANK-21: flat in n on the real bank, so n cannot explain the gap."""
    a = O6_PARTICIPATION_REFERENCES[LEAD1440]
    b = O6_PARTICIPATION_REFERENCES[LEAD5617]
    assert b / a == pytest.approx(1.0, abs=0.10), (
        "participation is flat in n over 1440->5617 on the real DINOv3 bank; a "
        "large ratio here would reopen H-RANK-21")


def test_the_floor_is_flagged_as_corpus_and_dim_specific():
    assert O6_FLOOR_IS_CORPUS_AND_DIM_SPECIFIC is True, (
        "flipping this to False asserts the floor is portable across corpora "
        "and dimensions, which is REFUTED by the three measured references above")


def test_the_measured_references_are_NOT_dimensionally_comparable_to_z_op():
    """The trap on the other side: 'champ30k 6.489 > DINOv3 5.756' is also invalid."""
    for key in O6_PARTICIPATION_REFERENCES:
        assert key.endswith("/d1024"), (
            "every banked reference is d=1024 (DINOv3 ViT-L). z_op is d=2048, so "
            "no arm may be declared to BEAT these either -- add a matched-d "
            "reference before making a comparison in either direction")
