"""S2 labels v3 — the 80 REASSIGNED a_str rows are DECLINED, and nothing else moved.

WHY THIS EXISTS. `labels/SUPERSEDED.json` recorded an open defect in its own
successor: removing the refuted lane-change gate did not DROP its 80
``PREPARE_LANE_CHANGE`` rows, it REASSIGNED them to 71 ``HOLD_CORRIDOR`` +
9 ``REDUCE_TO`` — manufacturing a different confident claim in place of a wrong
one. `s2_labels` grew a per-family abstain mask for exactly this, and it had
never been emitted: MEASURED, `labels_v2` carries ``a_str`` abstains **0**.

The v3 set declines those rows. These tests pin the two things that make that
change trustworthy:
  * it is CORRECT — exactly the 80 joined rows, matching the count the finder
    documented independently;
  * it is SURGICAL — g_str is untouched, and every non-declined a_str row is
    byte-identical to v2. A label rebuild that quietly moved anything else
    would be the same class of defect it is fixing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stack" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import s2_labels as SL                                        # noqa: E402

_INCOMING = (ROOT / "TanitAD Research Lab" / "Data Engineering" /
             "Implementation" / "incoming")
_V1 = _INCOMING / "2026-08-16-s2-v1-labels" / "labels"
_V2 = _INCOMING / "2026-08-16-s2-v1-labels" / "review" / "labels_v2"
_V3 = _INCOMING / "2026-08-23-s2-abstain-v3" / "labels_v3"

SPLITS = (("aug120", "train"), ("w120val", "eval"))
#: MEASURED by joining v1 and v2 on clip_id, and independently equal to the
#: count `SUPERSEDED.json` recorded from the loader on both directories.
EXPECTED_ABSTAINS = {"aug120": 19, "w120val": 61}
EXPECTED_TOTAL = 80


def _rows(p: Path) -> dict:
    return {r["clip_id"]: r for r in
            (json.loads(l) for l in
             p.read_text(encoding="utf-8").splitlines() if l.strip())}


def _need(*dirs):
    for d in dirs:
        if not d.is_dir():
            pytest.skip(f"{d} not present (pod checkout carries no Research "
                        f"Hub) — this suite pins an artifact, not a code path")


# --------------------------------------------------------------------------
# correct
# --------------------------------------------------------------------------

@pytest.mark.parametrize("split,role", SPLITS)
def test_v3_declines_exactly_the_documented_rows(split, role):
    _need(_V3)
    ls = SL.load_s2_labels(_V3 / f"s2_labels_{split}.jsonl", role=role)
    census = ls.abstain_census()
    assert ls.has_abstain is True
    assert census["a_str"] == EXPECTED_ABSTAINS[split]
    # ⛔ g_str must NEVER abstain here: its NONE_ABSTAIN is a supervised
    # target, a different claim entirely. A g_str abstain would mean the
    # rebuild touched the wrong family.
    assert census["g_str"] == 0


def test_the_total_matches_the_independently_documented_count():
    _need(_V3)
    total = sum(SL.load_s2_labels(_V3 / f"s2_labels_{s}.jsonl",
                                  role=r).abstain_census()["a_str"]
                for s, r in SPLITS)
    assert total == EXPECTED_TOTAL


@pytest.mark.parametrize("split,role", SPLITS)
def test_the_declined_rows_are_the_ones_the_refuted_gate_fired_on(split, role):
    """The identity of the 80 is READ OFF the v1/v2 diff, never re-inferred."""
    _need(_V1, _V2, _V3)
    v1 = _rows(_V1 / f"s2_labels_{split}.jsonl")
    v2 = _rows(_V2 / f"s2_labels_{split}.jsonl")
    v3 = _rows(_V3 / f"s2_labels_{split}.jsonl")

    gate_fired = {c for c, r in v1.items()
                  if r.get("a_str", {}).get("token") == "PREPARE_LANE_CHANGE"}
    reassigned = {c for c in gate_fired
                  if v2.get(c, {}).get("a_str", {}).get("token")
                  != "PREPARE_LANE_CHANGE"}
    declined = {c for c, r in v3.items() if r.get("a_str", {}).get("abstain")}
    assert declined == reassigned, "v3 declined a different set than the join"


# --------------------------------------------------------------------------
# surgical
# --------------------------------------------------------------------------

@pytest.mark.parametrize("split,role", SPLITS)
def test_g_str_is_byte_identical_to_v2(split, role):
    _need(_V2, _V3)
    v2, v3 = (_rows(_V2 / f"s2_labels_{split}.jsonl"),
              _rows(_V3 / f"s2_labels_{split}.jsonl"))
    assert set(v2) == set(v3), "the record SET changed — v3 is not a rebuild"
    for c in v2:
        assert v2[c]["g_str"] == v3[c]["g_str"], f"g_str moved on {c}"


@pytest.mark.parametrize("split,role", SPLITS)
def test_every_non_declined_a_str_row_is_byte_identical_to_v2(split, role):
    _need(_V2, _V3)
    v2, v3 = (_rows(_V2 / f"s2_labels_{split}.jsonl"),
              _rows(_V3 / f"s2_labels_{split}.jsonl"))
    for c in v2:
        if v3[c]["a_str"].get("abstain"):
            continue
        assert v2[c]["a_str"] == v3[c]["a_str"], f"a_str moved on {c}"


@pytest.mark.parametrize("split,role", SPLITS)
def test_an_abstaining_block_carries_no_target_of_any_kind(split, role):
    """`_check_block` refuses a token/args/arg_mask beside an abstain — a
    target sitting next to a declination is the exact ambiguity that lets a
    consumer read one as the other. Pinned here too, because this file is
    where a future rebuild will be checked."""
    _need(_V3)
    v3 = _rows(_V3 / f"s2_labels_{split}.jsonl")
    n = 0
    for c, r in v3.items():
        blk = r["a_str"]
        if not blk.get("abstain"):
            continue
        n += 1
        assert "token" not in blk, f"{c}: abstain carries a token"
        assert "args" not in blk, f"{c}: abstain carries args"
        assert "arg_mask" not in blk, f"{c}: abstain carries an arg_mask"
        assert blk.get("reason"), f"{c}: abstain with no recorded reason"
        assert blk.get("superseded_token_v1") == "PREPARE_LANE_CHANGE"
    assert n == EXPECTED_ABSTAINS[split]


# --------------------------------------------------------------------------
# the inertness claim, on the branch that is NOT the default
# --------------------------------------------------------------------------

def test_v2_emits_no_family_masks_and_v3_does():
    """The whole safety argument for the abstain mask is that it is
    default-OFF and provably inert: a set with no declination emits the
    incumbent seven-key batch. Both branches are exercised so neither can rot.
    """
    _need(_V2, _V3)
    v2 = SL.load_s2_labels(_V2 / "s2_labels_aug120.jsonl", role="train")
    v3 = SL.load_s2_labels(_V3 / "s2_labels_aug120.jsonl", role="train")
    assert v2.has_abstain is False
    assert v3.has_abstain is True


def test_v3_is_not_yet_the_canonical_set():
    """⚠️ Deliberately pins the CURRENT state, not the desired one.

    `S2_CANONICAL_LABELS_REL` still names labels_v2. Flipping it changes what
    a live training arm is supervised by, so it is escalated to the PI rather
    than taken here. When that decision lands, THIS test is the one that must
    be updated — which is the point: the pointer cannot move without someone
    reading this rationale.
    """
    assert SL.S2_CANONICAL_LABELS_REL.endswith("labels_v2")
