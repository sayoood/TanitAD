"""DLR OpenDRIVE registry guard — the licence is PER RECORD DOI, not per publisher.

These tests exist because of a MEASURED trap, not a hypothetical one: the **same
authors' same test bed** ships under two different licences —
``zenodo.7056722`` (sample) is ``CC-BY-4.0`` while ``zenodo.18507692`` (full) is
``CC-BY-NC-SA-4.0``. A publisher-level rule would route the NC-SA map into a
commercial shard, which is a licence **violation**, not a risk tolerance.

Following the ``e1c_selftest`` pattern that the program has found actually works:
**every guard here is driven with input designed to make it FAIL**, and the test
asserts the failing behaviour is produced. A guard that is only ever tested with
input it accepts has not been tested.
"""
from __future__ import annotations

import pytest

from tanitad.lake.schema import (
    SOURCE_REGISTRY,
    dlr_opendrive_source_key,
    _DLR_OPENDRIVE_CC_BY_DOIS,
    _DLR_OPENDRIVE_NC_DOIS,
)

_SAMPLE_DOI = "10.5281/zenodo.7056722"    # Test Bed Lower Saxony — SAMPLE, CC-BY
_FULL_NC_DOI = "10.5281/zenodo.18507692"  # Test Bed Lower Saxony — FULL, NC-SA


def test_both_keys_registered_with_the_licences_measured_on_zenodo():
    ship = SOURCE_REGISTRY["dlr_opendrive"]
    assert ship.license_class == "owned-safe"
    assert ship.license_name == "CC-BY-4.0"
    assert ship.share_alike is False        # commercial-OK AND not copyleft
    assert ship.is_synthetic is False       # real surveyed roads, not a sim town

    nc = SOURCE_REGISTRY["dlr_opendrive_nc"]
    assert nc.license_class == "nc-research"
    assert nc.license_name == "CC-BY-NC-SA-4.0"
    assert nc.share_alike is True


def test_the_trap_itself_same_testbed_two_licences_resolve_differently():
    """THE failing-input case. Same authors, same test bed, opposite tiers."""
    assert dlr_opendrive_source_key(_SAMPLE_DOI) == "dlr_opendrive"
    assert dlr_opendrive_source_key(_FULL_NC_DOI) == "dlr_opendrive_nc"
    assert dlr_opendrive_source_key(_SAMPLE_DOI) != dlr_opendrive_source_key(_FULL_NC_DOI)


def test_the_nc_record_can_never_reach_a_commercial_tier():
    """What value would make this guard FAIL? This one — so it is asserted."""
    key = dlr_opendrive_source_key(_FULL_NC_DOI)
    assert SOURCE_REGISTRY[key].license_class == "nc-research"
    assert SOURCE_REGISTRY[key].license_class != "owned-safe"
    assert _FULL_NC_DOI not in _DLR_OPENDRIVE_CC_BY_DOIS


def test_unknown_doi_raises_rather_than_defaulting():
    """A default has to pick a side; the permissive side is a licence violation."""
    with pytest.raises(KeyError):
        dlr_opendrive_source_key("10.5281/zenodo.99999999")
    with pytest.raises(KeyError):
        dlr_opendrive_source_key("")


def test_doi_url_prefix_and_case_are_tolerated():
    assert dlr_opendrive_source_key(
        "https://doi.org/10.5281/ZENODO.7056722") == "dlr_opendrive"


def test_the_two_allowlists_are_disjoint():
    assert not (_DLR_OPENDRIVE_CC_BY_DOIS & _DLR_OPENDRIVE_NC_DOIS)
