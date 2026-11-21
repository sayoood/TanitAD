"""Both agent-join readers carry ``with_track_ids``, or the eval z/h supervision dies.

⛔ THE DEFECT THIS PINS, MEASURED 2026-09-18 by the 139-clip coverage pass.
`refc_v3_train` builds a `JoinFileReader` **twice** -- once for train (:6111) and once for
eval (:6310) -- and only the train one passed `with_track_ids`. `lookup_track_ids`'s own
docstring says it returns **None** when the reader was not built that way, so on the eval
split `__getitem__` fell through to `zh_targets(t)`'s **all-False** mask and the eval
reported

    box3d_z = 0.0   box3d_h = 0.0   with   n_z = 0   n_h = 0

over **900 windows on both halves** -- while the TRAIN side of the SAME runs had
`box3d_n_z == box3d_n_matched` **exactly** (23=23, 41=41, 34=34).

⚠️ **A zero that reads as PERFECT.** Nothing in the log said the term had gone dark; only
reading `n_z` beside the value revealed it. That is the `tac_goal` inversion
`_map_item`'s docstring was written to prevent, arriving through a different door.

⭐ The invariant below is a PAIRING one, because the defect class is "two construction
sites, one missing a flag": every reader that is built `with_rates` must also be built
`with_track_ids`. Counting one without the other is what let them drift apart.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

SRC = (ROOT / "scripts" / "refc_v3_train.py").read_text(encoding="utf-8")


def test_every_reader_built_with_rates_is_also_built_with_track_ids():
    n_rates = SRC.count("with_rates=")
    n_tids = SRC.count("with_track_ids=")
    assert n_rates >= 2, "expected a train AND an eval reader construction"
    assert n_tids == n_rates, (
        f"{n_rates} reader constructions pass with_rates but only {n_tids} pass "
        "with_track_ids -- the one that does not will return None from "
        "lookup_track_ids, and its split's box3d z/h supervision silently vanishes "
        "while reporting 0.0")


def test_the_EVAL_reader_specifically_carries_it():
    """⛔ The pairing count alone would pass if BOTH readers dropped the flag."""
    i = SRC.index("_e_rd = _JFR(")
    block = SRC[i:i + 400]
    assert "with_track_ids=" in block, (
        "the EVAL reader construction must pass with_track_ids; without it "
        "eval_box3d_n_z and eval_box3d_n_h read 0 while eval_box3d_n_matched is "
        "in the tens")


def test_lookup_track_ids_documents_that_the_flag_gates_it():
    """The contract the two tests above rest on, read from the reader itself."""
    p8 = (ROOT / "scripts" / "train_p8_occupancy.py").read_text(encoding="utf-8")
    i = p8.index("def lookup_track_ids")
    doc = p8[i:i + 600]
    assert "with_track_ids=True" in doc, (
        "lookup_track_ids must document that it returns None without the flag; if "
        "that contract changed, these tests are pinning the wrong thing")


def test_a_zero_z_loss_is_only_readable_beside_its_n():
    """⚠️ The reporting rule this defect earned, pinned so it cannot be dropped.

    `box3d_set_loss` reports `n["z"]`, and an absent 3-D label is a MASK, never a
    zero-fill. A run that logs `box3d_z` without `box3d_n_z` is unreadable.
    """
    assert '"cz": batch.get(' in SRC and '"zh_mask": (torch.zeros_like' in SRC, (
        "the mask-not-zero-fill contract must stay in the target assembly")
    assert "box3d_n_z" in SRC or "n_z" in SRC, (
        "the per-term count must be logged beside the loss value")
