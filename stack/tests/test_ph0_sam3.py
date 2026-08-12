"""CPU tests for Engine C (SAM3) — the parts that hold without the weights.

⛔ WHY THIS FILE EXISTS. The previous version of this engine carried the note
"API verified on pod4, not guessed". It was a SIGNATURE check, not an EXECUTION
check — the repo was gated so it could not be run — and when access was granted
the very first run died: `SAM3InteractiveImagePredictor` wants a
`Sam3TrackerBase`, not the `Sam3Image` the image builder returns. These tests
pin the pure geometry/serialisation that a gated repo can never excuse.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))


def test_boxes_are_converted_to_xywh():
    """⛔ MEASURED FROM THE SOURCE: `add_prompt` passes `bounding_boxes`
    straight into `boxes_xywh=`. Our B3 groundings are xyxy. An unconverted box
    still segments SOMETHING, so this bug would never announce itself — only a
    test can hold it."""
    from ph0_sam3 import xyxy_to_xywh
    assert xyxy_to_xywh([10, 20, 60, 100]) == [10.0, 20.0, 50.0, 80.0]
    # a box touching the origin must not become negative
    assert xyxy_to_xywh([0, 0, 5, 7]) == [0.0, 0.0, 5.0, 7.0]


def test_xywh_conversion_is_not_the_identity():
    """Guards the one-character regression where w,h are left as x1,y1."""
    from ph0_sam3 import xyxy_to_xywh
    b = [100, 50, 140, 90]
    assert xyxy_to_xywh(b) != [float(v) for v in b]


def _mask(h, w, rows):
    import numpy as np
    m = np.zeros((h, w), bool)
    for r, a, b in rows:
        m[r, a:b] = True
    return m


def test_rows_rle_roundtrips_exactly():
    from ph0_sam3 import _rows_rle
    import numpy as np
    rng = np.random.default_rng(0)
    m = rng.random((20, 30)) > 0.6
    runs = _rows_rle(m)
    back = np.zeros_like(m)
    for r, a, b in runs:
        back[r, a:b] = True
    assert (back == m).all()


def test_rows_rle_splits_disjoint_runs_in_one_row():
    """Two blobs on the same row must be TWO runs, not one spanning the gap."""
    from ph0_sam3 import _rows_rle
    m = _mask(3, 20, [(1, 2, 5), (1, 10, 14)])
    runs = [r for r in _rows_rle(m) if r[0] == 1]
    assert runs == [[1, 2, 5], [1, 10, 14]]


def test_box_mask_agreement_reports_two_numbers_not_one():
    """⚠️ A mask spilling outside its prompt and a mask filling one corner are
    OPPOSITE failures calling for opposite fixes; a single averaged score hides
    which happened. The contract is that both are reported."""
    from ph0_sam3 import box_mask_agreement
    # mask exactly fills the box -> both 1.0
    m = _mask(20, 20, [(r, 5, 10) for r in range(5, 10)])
    a = box_mask_agreement(m, [5, 5, 10, 10])
    assert a["frac_mask_in_box"] == 1.0 and a["frac_box_covered"] == 1.0
    # mask twice the box, half inside -> spill: in_box 0.5, covered 1.0
    m2 = _mask(20, 20, [(r, 5, 10) for r in range(5, 15)])
    a2 = box_mask_agreement(m2, [5, 5, 10, 10])
    assert a2["frac_mask_in_box"] == pytest.approx(0.5)
    assert a2["frac_box_covered"] == 1.0
    # mask in one corner of a big box -> in_box 1.0, covered small
    a3 = box_mask_agreement(m, [0, 0, 20, 20])
    assert a3["frac_mask_in_box"] == 1.0 and a3["frac_box_covered"] < 0.1


def test_box_mask_agreement_handles_empty_mask_without_dividing_by_zero():
    from ph0_sam3 import box_mask_agreement
    import numpy as np
    a = box_mask_agreement(np.zeros((10, 10), bool), [1, 1, 5, 5])
    assert a["frac_mask_in_box"] == 0.0 and a["mask_area_px"] == 0


def test_agent_concepts_cover_the_unfilled_tactical_slots():
    """The `text` mode exists to supply AGENT SLOTS the tactical vocabulary
    needs and PH0 extracts nowhere else: GAP_TARGET, EVADE_IN_CORRIDOR,
    WAIT_FOR_ONCOMING, TRAFFIC_LIGHT_REACT, FOLLOW(time_gap_s)."""
    from ph0_sam3 import AGENT_CONCEPTS
    for need in ("car", "pedestrian", "cyclist", "traffic light"):
        assert need in AGENT_CONCEPTS


def _outputs(n, h=4, w=6):
    """The REAL propagate payload, MEASURED from a live run 2026-08-12."""
    import numpy as np
    return {"out_obj_ids": np.arange(n),
            "out_probs": np.full(n, 0.75),
            "out_boxes_xywh": np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (n, 1)),
            "out_binary_masks": np.ones((n, h, w), bool),
            "frame_stats": {}}


def test_read_outputs_reads_the_real_keys_not_the_guessed_ones():
    """⛔ The first reader guessed `pred_masks`/`masks` and silently returned
    nothing on every frame of every clip. The real keys are these."""
    from ph0_sam3 import read_outputs
    recs = read_outputs(_outputs(2))
    assert len(recs) == 2
    r = recs[0]
    assert r["obj_id"] == 0 and r["prob"] == 0.75
    assert r["box_xywh"] == [1.0, 2.0, 3.0, 4.0]
    assert r["box_xyxy"] == [1.0, 2.0, 4.0, 6.0]        # x+w, y+h
    assert r["mask_area_px"] == 24 and r["mask_hw"] == [4, 6]


def test_zero_detections_is_a_result_not_an_error():
    """⚠️ N=0 means SAM3 saw nothing; it must not be conflated with a broken
    reader. The first run printed `0 frames` on all 8 clips and the two causes
    were indistinguishable — which is why n_det is recorded per frame."""
    from ph0_sam3 import read_outputs
    assert read_outputs(_outputs(0)) == []


def test_read_outputs_rejects_a_payload_it_does_not_understand():
    from ph0_sam3 import read_outputs
    import numpy as np
    assert read_outputs({"pred_masks": np.ones((1, 2, 2))}) == []
    assert read_outputs(None) == []
    assert read_outputs([1, 2, 3]) == []


# =========================================================================== #
# The documented Sam3Processor path (validated against the official README)    #
# =========================================================================== #
def test_box_iou_is_symmetric_and_bounded():
    from ph0_sam3 import _box_iou
    a, b = [0, 0, 10, 10], [5, 5, 15, 15]
    assert _box_iou(a, b) == pytest.approx(_box_iou(b, a))
    assert 0.0 <= _box_iou(a, b) <= 1.0
    assert _box_iou(a, a) == pytest.approx(1.0)
    assert _box_iou(a, [20, 20, 30, 30]) == 0.0      # disjoint
    assert _box_iou(a, [0, 0, 0, 0]) == 0.0          # degenerate, no div-by-0


def test_detect_returns_frame_pixel_boxes_and_scores():
    """⭐ MEASURED on pod4: the README path returns boxes in ORIGINAL frame
    coordinates (a 'sky' box spanned [0.6, 0.1, 447.2, 85.8] on a 448-wide
    frame), which is why no rescaling happens here."""
    from ph0_sam3 import detect
    import numpy as np

    class FakeProc:
        def set_image(self, img):
            return {"state": 1}

        def set_text_prompt(self, state, prompt):
            return {"scores": np.array([0.83, 0.20]),
                    "boxes": np.array([[263.2, 74.2, 273.2, 84.4],
                                       [1.0, 2.0, 3.0, 4.0]]),
                    "masks": np.ones((2, 3, 3), bool)}

    recs = detect(FakeProc(), object(), "traffic sign")
    assert len(recs) == 2
    assert recs[0]["concept"] == "traffic sign"
    assert recs[0]["score"] == 0.83
    assert recs[0]["box_xyxy"] == [263.2, 74.2, 273.2, 84.4]
    assert recs[0]["mask_area_px"] == 9


def test_min_score_defaults_to_zero_so_nothing_is_silently_filtered():
    """⚠️ A threshold chosen before the score distribution is known is a
    decision dressed as a default. Filtering must be downstream and visible."""
    from ph0_sam3 import detect
    import numpy as np

    class FakeProc:
        def set_image(self, img):
            return {}

        def set_text_prompt(self, state, prompt):
            return {"scores": np.array([0.05]),
                    "boxes": np.array([[0.0, 0.0, 1.0, 1.0]]), "masks": None}

    assert len(detect(FakeProc(), object(), "car")) == 1
    assert len(detect(FakeProc(), object(), "car", min_score=0.5)) == 0


def test_empty_detection_is_an_abstention_not_an_error():
    """⛔ MEASURED: on a real clip frame SAM3 returned tree 16 / sky 1 / road 1 /
    traffic sign 1, and car 0 / person 0 — because that frame HAS no car. The
    zeros are correct. An earlier run printed '0 frames' everywhere and could
    not distinguish this from a broken reader."""
    from ph0_sam3 import detect
    import numpy as np

    class FakeProc:
        def set_image(self, img):
            return {}

        def set_text_prompt(self, state, prompt):
            return {"scores": np.array([]), "boxes": np.array([]),
                    "masks": None}

    assert detect(FakeProc(), object(), "car") == []
