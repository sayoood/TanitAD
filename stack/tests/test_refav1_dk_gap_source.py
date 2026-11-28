"""BAR-4 of `D-REFAV1-DK-DECODED`: the distance-keeping cost's GAP-SOURCE
vocabulary and its provenance stamp, each guard PROVEN BY MUTATION.

⛔ WHY MUTATION AND NOT INSPECTION. An AST census once read 0 suspects on BOTH
the fixed and the broken trainer, so "the guard is in the file" is not evidence
that the guard *rejects* anything. Every test below therefore comes in a pair:

  1. the bad input RAISES (the failure branch is REACHABLE), and
  2. with the guard's own datum mutated so it should no longer object, the SAME
     bad input is ACCEPTED.

Half (2) is what proves the refusal came from the guard under test and not from
an unrelated error on the way past it.

⭐ THE THING BEING PROTECTED. `gap_source` is stamped verbatim into every dump.
Before 2026-09-06 it was a free-form string, so a typo, a stale flag or an
optimistic label would have been BANKED AS FACT -- and the difference between
`oracle_label` (a CEILING) and `decoded` (a vision-only capability) is the whole
result. The vocabulary is closed so that a source nobody declared cannot be
recorded as one that was.
"""
import copy

import pytest
import torch

from tanitad.refs.refav1_lon_cost import (
    GAP_SOURCES, GAP_SOURCE_KIND, GAP_SOURCE_ORACLE, GAP_SOURCE_DECODED,
    GAP_SOURCE_DECODED_GAP, GAP_SOURCE_UNSET, GAP_SOURCE_UNIT_TEST,
    DistanceKeepingSpec, GapHeadProvenance, distance_keeping_cost)
import tanitad.refs.refav1_lon_cost as LC


SHA = "a" * 64


def _prov(**kw):
    d = dict(head_path="head.pt", head_sha256=SHA, head_version="dkgap-v1",
             ckpt_sha256="b" * 64, ckpt_step=21109,
             fit_corpus="v7.2-eval-141", fit_scheme="5fold-crossfit-clip",
             n_fit_rows=1000, n_fit_clips=100, pool=(8, 20), grid=(16, 40), win=4)
    d.update(kw)
    return GapHeadProvenance(**d)


# --------------------------------------------------------------------------- #
# 1. the vocabulary is CLOSED                                                   #
# --------------------------------------------------------------------------- #
def test_a_unrecognised_source_is_refused():
    with pytest.raises(ValueError, match="vocabulary is CLOSED"):
        DistanceKeepingSpec(w_dk=1.0, gap_source="banana")


def test_a_MUTATION_widening_the_vocabulary_accepts_it():
    """⭐ THE MUTATION HALF: the refusal above must come from the vocabulary
    check, not from anything else on the way. Widen the vocabulary and the
    identical construction succeeds."""
    old_srcs, old_kind = LC.GAP_SOURCES, LC.GAP_SOURCE_KIND
    try:
        LC.GAP_SOURCES = tuple(old_srcs) + ("banana",)
        LC.GAP_SOURCE_KIND = dict(old_kind)
        LC.GAP_SOURCE_KIND["banana"] = {"gate": "none", "gap": "none",
                                        "vision_only": False, "is_ceiling": False,
                                        "deployable": False, "needs_head": False}
        spec = DistanceKeepingSpec(w_dk=1.0, gap_source="banana")
        assert spec.gap_source == "banana"          # ACCEPTED once widened
    finally:
        LC.GAP_SOURCES, LC.GAP_SOURCE_KIND = old_srcs, old_kind
    # and the guard is back: a same-breath control that MUST refuse again
    with pytest.raises(ValueError):
        DistanceKeepingSpec(w_dk=1.0, gap_source="banana")


def test_a_every_declared_source_has_a_kind():
    assert set(GAP_SOURCES) == set(GAP_SOURCE_KIND)
    for s, k in GAP_SOURCE_KIND.items():
        assert set(k) == {"gate", "gap", "vision_only", "is_ceiling",
                          "deployable", "needs_head"}, s


# --------------------------------------------------------------------------- #
# 2. an ARMED term must DECLARE its source                                      #
# --------------------------------------------------------------------------- #
def test_b_armed_with_unset_source_is_refused():
    with pytest.raises(ValueError, match="must declare its"):
        DistanceKeepingSpec(w_dk=1e-5, gap_source=GAP_SOURCE_UNSET)


def test_b_unarmed_with_unset_source_is_fine():
    """⭐ SAME-BREATH CONTROL: the refusal is about ARMING, not about 'unset'.
    The shipped default path must stay constructible."""
    spec = DistanceKeepingSpec()                     # w_dk = 0, source unset
    assert spec.armed is False and spec.gap_source == GAP_SOURCE_UNSET


def test_b_MUTATION_dropping_the_arm_check_accepts_it():
    spec = DistanceKeepingSpec.__new__(DistanceKeepingSpec)
    object.__setattr__(spec, "w_dk", 1e-5)
    object.__setattr__(spec, "tau_target_s", 1.5)
    object.__setattr__(spec, "d0_m", 5.0)
    object.__setattr__(spec, "lead_speed_closure", LC.LEAD_SPEED_STEADY)
    object.__setattr__(spec, "gap_source", GAP_SOURCE_UNSET)
    object.__setattr__(spec, "gap_head", None)
    # bypassing __post_init__ is exactly what "no guard" looks like:
    assert spec.armed is True and spec.gap_source == GAP_SOURCE_UNSET
    # ⛔ and the cost would happily price it -- which is the danger:
    c = torch.zeros(2, 5, 2, dtype=torch.float64)
    out = distance_keeping_cost(c, v0=20.0, gap0_m=3.0, dt=0.2, spec=spec)
    assert float(out.sum()) > 0.0


# --------------------------------------------------------------------------- #
# 3. the stamp must MATCH the source, both directions                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("src", [GAP_SOURCE_DECODED, GAP_SOURCE_DECODED_GAP])
def test_c_decoded_source_without_provenance_is_refused(src):
    with pytest.raises(ValueError, match="REQUIRES a GapHeadProvenance"):
        DistanceKeepingSpec(w_dk=1e-5, gap_source=src)


@pytest.mark.parametrize("src", [GAP_SOURCE_DECODED, GAP_SOURCE_DECODED_GAP])
def test_c_decoded_source_with_provenance_is_accepted(src):
    """SAME-BREATH CONTROL for the test above: the refusal is about the MISSING
    stamp, so supplying one must succeed."""
    spec = DistanceKeepingSpec(w_dk=1e-5, gap_source=src, gap_head=_prov())
    assert spec.gap_head is not None


def test_c_oracle_source_carrying_a_head_is_refused():
    """⛔ THE MORE DANGEROUS DIRECTION: an ORACLE arm stamped with a head reads
    like a vision result."""
    with pytest.raises(ValueError, match="must\n?\\s*NOT carry head provenance|"
                                         "must NOT carry head provenance"):
        DistanceKeepingSpec(w_dk=1e-5, gap_source=GAP_SOURCE_ORACLE,
                            gap_head=_prov())


def test_c_MUTATION_flipping_needs_head_accepts_the_unstamped_decoded_arm():
    old = copy.deepcopy(LC.GAP_SOURCE_KIND)
    try:
        LC.GAP_SOURCE_KIND[GAP_SOURCE_DECODED]["needs_head"] = False
        spec = DistanceKeepingSpec(w_dk=1e-5, gap_source=GAP_SOURCE_DECODED)
        assert spec.gap_head is None                 # ACCEPTED once mutated
    finally:
        LC.GAP_SOURCE_KIND.clear()
        LC.GAP_SOURCE_KIND.update(old)
    with pytest.raises(ValueError):                  # guard restored
        DistanceKeepingSpec(w_dk=1e-5, gap_source=GAP_SOURCE_DECODED)


# --------------------------------------------------------------------------- #
# 4. the provenance itself refuses to be empty or short                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field", ["head_path", "head_sha256", "head_version",
                                   "ckpt_sha256", "fit_corpus", "fit_scheme"])
def test_d_blank_provenance_field_is_refused(field):
    with pytest.raises(ValueError, match="missing"):
        _prov(**{field: ""})


def test_d_short_sha_is_refused():
    """⛔ A short or empty hash is how a 'verified' comparison passes on two
    FAILED reads -- the 2026-09-04 blob-comparison hole, in stamp form."""
    with pytest.raises(ValueError, match="64-char sha256"):
        _prov(head_sha256="deadbeef")


def test_d_full_sha_is_accepted():
    assert len(_prov(head_sha256="c" * 64).head_sha256) == 64


# --------------------------------------------------------------------------- #
# 5. what a dump can be read for -- the record carries the MEANING              #
# --------------------------------------------------------------------------- #
def test_e_only_decoded_is_vision_only():
    assert DistanceKeepingSpec(w_dk=1e-5, gap_source=GAP_SOURCE_DECODED,
                               gap_head=_prov()).vision_only is True
    # ⛔ the middle rung takes its ARMING GATE from the label and is NOT
    # vision-only however good its gap is:
    assert DistanceKeepingSpec(w_dk=1e-5, gap_source=GAP_SOURCE_DECODED_GAP,
                               gap_head=_prov()).vision_only is False
    assert DistanceKeepingSpec(w_dk=1e-5,
                               gap_source=GAP_SOURCE_ORACLE).vision_only is False


def test_e_oracle_and_decoded_gap_are_ceilings_and_decoded_is_not():
    assert DistanceKeepingSpec(w_dk=1e-5,
                               gap_source=GAP_SOURCE_ORACLE).is_ceiling is True
    assert DistanceKeepingSpec(w_dk=1e-5, gap_source=GAP_SOURCE_DECODED_GAP,
                               gap_head=_prov()).is_ceiling is True
    assert DistanceKeepingSpec(w_dk=1e-5, gap_source=GAP_SOURCE_DECODED,
                               gap_head=_prov()).is_ceiling is False


def test_e_record_stamps_source_kind_and_head_identity():
    r = DistanceKeepingSpec(w_dk=1e-5, gap_source=GAP_SOURCE_DECODED,
                            gap_head=_prov()).record()
    assert r["gap_source"] == GAP_SOURCE_DECODED
    assert r["vision_only"] is True and r["is_ceiling"] is False
    assert r["gap_source_kind"]["gate"] == "decoded"
    assert r["gap_head"]["head_sha256"] == SHA
    assert r["gap_head"]["ckpt_step"] == 21109
    assert r["gap_head"]["fit_scheme"] == "5fold-crossfit-clip"
    # the oracle record must NOT look like a head record -- same breath
    r0 = DistanceKeepingSpec(w_dk=1e-5, gap_source=GAP_SOURCE_ORACLE).record()
    assert r0["gap_head"] is None and r0["vision_only"] is False


def test_e_unit_test_source_is_stamped_non_deployable():
    """⭐ The escape hatch the arithmetic tests use is itself stamped, so a dump
    carrying it can never be read as a run."""
    r = DistanceKeepingSpec(w_dk=1.0, gap_source=GAP_SOURCE_UNIT_TEST).record()
    assert r["gap_source_kind"]["deployable"] is False
    assert r["vision_only"] is False


# --------------------------------------------------------------------------- #
# 6. NONE of this changed the arithmetic -- the parity control                   #
# --------------------------------------------------------------------------- #
def test_f_cost_is_unchanged_by_the_source_it_declares():
    """⛔ The stamp is METADATA. If declaring a source moved a number, every
    banked comparison would be confounded by its own label."""
    c = torch.randn(6, 10, 2, dtype=torch.float64) * 0.3
    kw = dict(v0=18.0, gap0_m=12.0, dt=0.2)
    a = distance_keeping_cost(c, spec=DistanceKeepingSpec(
        w_dk=3.0, gap_source=GAP_SOURCE_ORACLE), **kw)
    b = distance_keeping_cost(c, spec=DistanceKeepingSpec(
        w_dk=3.0, gap_source=GAP_SOURCE_DECODED, gap_head=_prov()), **kw)
    assert torch.equal(a, b)
    # same-breath control that MUST differ, so equality above is not vacuous
    d = distance_keeping_cost(c, spec=DistanceKeepingSpec(
        w_dk=6.0, gap_source=GAP_SOURCE_ORACLE), **kw)
    assert not torch.equal(a, d)


def test_f_unarmed_is_still_exactly_zero_under_every_source():
    c = torch.randn(4, 8, 2, dtype=torch.float64)
    for src in (GAP_SOURCE_UNSET, GAP_SOURCE_ORACLE, GAP_SOURCE_UNIT_TEST):
        out = distance_keeping_cost(c, v0=20.0, gap0_m=1.0, dt=0.2,
                                    spec=DistanceKeepingSpec(w_dk=0.0,
                                                             gap_source=src))
        assert torch.equal(out, torch.zeros(4, dtype=torch.float64))
