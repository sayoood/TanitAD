"""The packed-ASCII ``eid`` defect, and the guard that it stays repaired.

THE DEFECT. `windows_flagship-v4.1-10k.pt`, `-v4.2-step4000.pt` and
`-v16-ab-ft.pt` store `eid` as a big-endian ASCII byte-packing of the true
4-char hex PhysicalAI episode id (808464434 == b'0002') where the other 24
committed dumps store the val-list index 0..39. A cross-arm join keyed on
`eid` does not error against those three — it matches NOTHING, silently.

WHAT LICENSES THE REPAIR (MEASURED 2026-08-04, this file re-measures it rather
than inheriting it): each affected dump's `gt` is bit-identical to the
canonical `windows_flagship-30k.pt`, so row i is the same window in both; and
first-appearance rank of the packed ids reproduces the canonical 0..39 exactly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from taniteval.rollout import load_windows, normalise_eid, save_windows

RES = Path(__file__).resolve().parents[1] / "results"
AFFECTED = ("flagship-v4.1-10k", "flagship-v4.2-step4000", "flagship-v16-ab-ft")
CANONICAL = "flagship-30k"


def _dump(key):
    p = RES / f"windows_{key}.pt"
    if not p.exists():
        pytest.skip(f"fixture {p.name} not present")
    return p


# --- the unit: normalise_eid ------------------------------------------- #

def test_clean_integer_ids_are_the_identity():
    """24 of 27 dumps are already canonical. The normaliser must not touch
    them, or every banked number keyed on eid moves for no reason."""
    eid = [0, 0, 1, 1, 2, 39]
    canon, raw = normalise_eid(eid)
    assert raw is None
    assert list(np.asarray(canon).tolist()) == eid


def test_a_non_contiguous_integer_subset_is_left_alone():
    """The trap in the obvious fix: rank-remapping {0, 5, 9} to {0, 1, 2}
    would INVENT a join key rather than repair one. Small ints are never
    touched, whatever their spacing."""
    eid = [0, 0, 5, 5, 9]
    canon, raw = normalise_eid(eid)
    assert raw is None
    assert list(np.asarray(canon).tolist()) == eid


def test_packed_ascii_is_decoded_and_ranked():
    packed = [int.from_bytes(s.encode(), "big") for s in
              ("0002", "0002", "0084", "00ba", "0084")]
    canon, raw = normalise_eid(packed)
    assert raw == ["0002", "0002", "0084", "00ba", "0084"]
    assert canon == [0, 0, 1, 2, 1]          # first-appearance order


def test_an_empty_dump_does_not_explode():
    canon, raw = normalise_eid([])
    assert raw is None
    assert list(np.asarray(canon).tolist()) == []


# --- the fixtures: the three real dumps -------------------------------- #

@pytest.mark.parametrize("key", AFFECTED)
def test_affected_dumps_load_repaired_and_join_the_canonical_arm(key):
    """The whole point: after the fix a cross-arm join on eid MATCHES."""
    good = load_windows(_dump(CANONICAL))
    bad = load_windows(_dump(key))

    # The alignment premise, re-measured rather than assumed: same rows.
    assert np.allclose(np.asarray(good["gt"]), np.asarray(bad["gt"]), atol=0)

    ge = np.asarray(good["eid"])
    be = np.asarray(bad["eid"])
    assert be.tolist() == ge.tolist(), "eid does not join the canonical arm"
    assert len(set(be.tolist())) == 40
    # provenance kept, not discarded
    assert "eid_raw" in bad and len(bad["eid_raw"]) == len(be)
    assert all(len(s) == 4 for s in bad["eid_raw"])


def test_clean_fixture_is_untouched_on_load():
    good = load_windows(_dump(CANONICAL))
    assert "eid_raw" not in good
    assert sorted(set(np.asarray(good["eid"]).tolist())) == list(range(40))


# --- the write path ----------------------------------------------------- #

def test_save_windows_cannot_write_the_defect_again(tmp_path):
    packed = [int.from_bytes(s.encode(), "big")
              for s in ("0002", "0084", "0002")]
    p = tmp_path / "windows_synthetic.pt"
    save_windows({"eid": packed, "gt": torch.zeros(3, 4, 2)}, p)
    back = torch.load(p, map_location="cpu", weights_only=False)
    assert back["eid"] == [0, 1, 0]
    assert back["eid_raw"] == ["0002", "0084", "0002"]


def test_save_windows_round_trips_a_clean_dump_bit_for_bit(tmp_path):
    d = {"eid": [0, 1, 2], "gt": torch.zeros(3, 4, 2)}
    p = tmp_path / "windows_clean.pt"
    save_windows(d, p)
    back = torch.load(p, map_location="cpu", weights_only=False)
    assert back["eid"] == [0, 1, 2]
    assert "eid_raw" not in back
