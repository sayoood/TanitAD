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


def test_cross_check_runs_the_exact_frame_the_vlm_grounded_on():
    """⛔ THE CONFOUND THIS FIXES. The first version ran only strided frames and
    snapped each VLM box to the nearest one — comparing engine B and engine C on
    frames up to ~3.5 s of driving apart. The resulting 0/8 "agreement" was a
    property of the SNAPPING, not of the VLM's boxes, and reporting it as a
    grounding score would have been the `--v2` conflation in miniature.

    The frame list must now contain every grounded frame EXACTLY."""
    import inspect
    from ph0_sam3 import run_clip_frames
    src = inspect.getsource(run_clip_frames)
    # the strided set is UNIONED with the grounded frames, not replaced by it
    assert "todo = sorted(set(range(0, len(frames)" in src
    assert 'int(v.get("frame_idx", 0)) for v in vlm_boxes' in src
    # and the lookup no longer divides-and-multiplies by the stride
    assert "// max(1, frame_stride)" not in src
    assert "EXACT frame, never snapped" in src


def test_cross_check_records_enough_to_audit_a_zero():
    """A 0/N match must be readable as EITHER 'SAM3 saw no sign on that frame'
    OR 'both saw a sign and they disagree'. Those call for opposite fixes, so
    the count of SAM3 signs on the grounded frame is recorded alongside."""
    import inspect
    from ph0_sam3 import run_clip_frames
    src = inspect.getsource(run_clip_frames)
    for k in ("n_sam3_signs_on_frame", "sam3_frame_idx", "frame_aligned"):
        assert k in src


# =========================================================================== #
# C77 — the dtype fix and the LIVENESS positive control                       #
# =========================================================================== #
class _RaisingProc:
    """A processor in exactly the C77 state: it runs, and raises on every
    concept. The 115 backfilled clips of 2026-08-16 were made of this."""

    def set_image(self, img):
        return {}

    def set_text_prompt(self, state, prompt):
        raise RuntimeError("mat1 and mat2 must have the same dtype, "
                           "but got BFloat16 and Float")


class _LiveProc:
    """A working processor: `road`/`sky` hit, and the agent concepts are
    legitimately empty on this frame (an open road)."""

    def set_image(self, img):
        return {}

    def set_text_prompt(self, state, prompt):
        import numpy as np
        if prompt in ("road", "sky"):
            return {"scores": np.array([0.94]),
                    "boxes": np.array([[0.0, 0.0, 10.0, 10.0]]),
                    "masks": None}
        return {"scores": np.array([]), "boxes": np.array([]), "masks": None}


def _frames(n=9, h=8, w=12):
    import numpy as np
    return [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(n)]


def test_liveness_concepts_are_disjoint_from_the_measured_vocabulary():
    """⛔ A positive control that is also one of the quantities being measured
    is circular — it would make the run's own output its own proof. road/sky
    are deliberately NOT agent slots and never enter `per_concept_hits`."""
    from ph0_sam3 import AGENT_CONCEPTS, LIVENESS_CONCEPTS
    assert LIVENESS_CONCEPTS, "the control may not be empty"
    assert not (set(LIVENESS_CONCEPTS) & set(AGENT_CONCEPTS))


def test_liveness_probe_calls_a_dead_engine_dead():
    """⭐ THE C77 FIX. Every AGENT concept may legitimately be 0, so all-zero
    agent counts prove nothing. road and sky cannot BOTH be 0 on a
    forward-facing driving frame ⇒ their zero is an ALARM, and the cause is
    recorded next to it."""
    from ph0_sam3 import liveness_probe
    dead = liveness_probe(_RaisingProc(), object())
    assert dead["live"] is False
    assert dead["n_det"] == {"road": 0, "sky": 0}
    assert "BFloat16" in dead["errors"]["road"]

    live = liveness_probe(_LiveProc(), object())
    assert live["live"] is True and live["n_det"]["road"] == 1
    assert "errors" not in live


def test_run_clip_frames_banks_the_alarm_a_structural_check_would_miss():
    """⛔ THE EXACT C77 ARTIFACT: 5-7 frames run, schema valid, clip_id right,
    `frames` populated — and zero detections. The record must now carry, in
    its own summary keys, the three quantities that settle it: how many
    detections, how many ERRORS, and whether the positive control fired."""
    from ph0_sam3 import run_clip_frames
    out = run_clip_frames(_RaisingProc(), _frames(), ["car", "pedestrian"],
                          [], frame_stride=4)
    assert out["n_frames_run"] == 3               # the container looks fine
    assert out["n_det_total"] == 0                # ... and it is empty
    assert out["n_err_total"] == 6                # 3 frames x 2 concepts
    assert out["err_kinds"] == {"RuntimeError": 6}
    assert out["liveness"]["live"] is False


def test_run_clip_frames_distinguishes_an_empty_scene_from_a_dead_engine():
    """The other half of the same test: agent concepts all zero, engine FINE.
    Before the control these two records were indistinguishable — which is
    precisely how 115 clips of nothing passed review."""
    from ph0_sam3 import run_clip_frames
    out = run_clip_frames(_LiveProc(), _frames(), ["car", "pedestrian"], [],
                          frame_stride=4)
    assert out["n_det_total"] == 0 and out["n_err_total"] == 0
    assert out["liveness"]["live"] is True
    assert out["per_concept_hits"] == {"car": 0, "pedestrian": 0}
    # the control stays OUT of the measured vocabulary
    assert "road" not in out["per_concept_hits"]


def test_liveness_probe_can_be_turned_off_only_explicitly():
    from ph0_sam3 import run_clip_frames
    out = run_clip_frames(_LiveProc(), _frames(), ["car"], [],
                          frame_stride=4, liveness=False)
    assert out["liveness"] is None


def test_dtype_agreement_patch_targets_vitdet_and_keeps_the_input_dtype():
    """⛔ THE ROOT CAUSE, pinned. `perflib/fused.py::addmm_act` casts bias,
    input AND weight to bfloat16; `vitdet.py:74`'s fc2 stays fp32 ⇒
    `mat1 and mat2 must have the same dtype`. The replacement keeps the fused
    kernel and drops the casts.

    ⚠️ It must rebind the name in `sam3.model.vitdet`, NOT in
    `sam3.perflib.fused` — vitdet does `from ... import addmm_act` at import
    time, so patching the source module is a silent no-op."""
    import sys
    import types
    import torch
    from ph0_sam3 import install_dtype_agreement

    def vendor(activation, linear, mat1):            # the real one, in spirit
        return mat1.to(torch.bfloat16)

    saved = {k: sys.modules.get(k)
             for k in ("sam3", "sam3.model", "sam3.model.vitdet")}
    try:
        sam3 = sys.modules.setdefault("sam3", types.ModuleType("sam3"))
        mdl = sys.modules.setdefault("sam3.model",
                                     types.ModuleType("sam3.model"))
        vit = types.ModuleType("sam3.model.vitdet")
        vit.addmm_act = vendor
        sys.modules["sam3.model.vitdet"] = vit
        sam3.model, mdl.vitdet = mdl, vit

        info = install_dtype_agreement()
        assert info["applied"] is True
        assert info["target"] == "sam3.model.vitdet.addmm_act"
        assert vit.addmm_act is not vendor
        assert getattr(vit.addmm_act, "_tanitad_dtype_safe", False)

        lin = torch.nn.Linear(4, 3)
        x = torch.randn(2, 5, 4)
        y = vit.addmm_act(torch.nn.GELU, lin, x)
        assert y.dtype == torch.float32, "no silent precision downgrade"
        assert y.shape == (2, 5, 3), "the [..., F] shape must survive"
        ref = torch.nn.functional.gelu(lin(x), approximate="tanh")
        assert torch.allclose(y, ref, atol=2e-3)

        # idempotent: a second install must not wrap the wrapper
        again = install_dtype_agreement()
        assert again["applied"] is True and "already" in again["reason"]
        assert vit.addmm_act._vendor is vendor
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_build_processor_installs_the_fix_before_the_weights_load():
    """A fix applied after the first forward is no fix. Pinned by source: the
    install call precedes `build_sam3_image_model` and its provenance is
    banked into the meta dict every run report quotes."""
    import inspect
    from ph0_sam3 import build_processor
    # the docstring quotes the README call, so compare positions in the BODY
    body = inspect.getsource(build_processor).split('"""')[-1]
    assert body.index("install_dtype_agreement()") < body.index(
        "build_sam3_image_model(")
    assert '"dtype_fix": dtype_fix' in body


# =========================================================================== #
# encode-once: the 4.4x, and the guarantee that it stays a refactor           #
# =========================================================================== #
class _CountingProc:
    """Counts image encodes and prompt evaluations separately."""

    def __init__(self, raise_on=()):
        self.n_encode = 0
        self.n_prompt = 0
        self.raise_on = set(raise_on)

    def set_image(self, img):
        self.n_encode += 1
        if "__image__" in self.raise_on:
            raise RuntimeError("encode blew up")
        return {"enc": self.n_encode}

    def set_text_prompt(self, state, prompt):
        import numpy as np
        self.n_prompt += 1
        assert "enc" in state, "the prompt must be scored against a state"
        if prompt in self.raise_on:
            raise RuntimeError("mat1 and mat2 must have the same dtype")
        return {"scores": np.array([0.9]),
                "boxes": np.array([[1.0, 2.0, 3.0, 4.0]]), "masks": None}


def test_detect_many_encodes_the_frame_once_not_once_per_concept():
    """⛔ THE 4.4x. `run_clip_frames` used to call `detect` per concept, and
    `detect` encodes; a 7-concept vocabulary therefore ran the ViT trunk SEVEN
    times on the identical frame. MEASURED banked `wall_s` 97-98 s for a
    6-frame clip; one encode per frame makes the same clip ~22 s."""
    from ph0_sam3 import detect_many
    p = _CountingProc()
    dets = detect_many(p, object(), ["car", "bus", "pedestrian"])
    assert p.n_encode == 1 and p.n_prompt == 3
    assert [d["concept"] for d in dets] == ["car", "bus", "pedestrian"]


def test_run_clip_frames_encodes_once_per_frame():
    from ph0_sam3 import run_clip_frames
    p = _CountingProc()
    out = run_clip_frames(p, _frames(n=9), ["car", "bus", "pedestrian"], [],
                          frame_stride=4, liveness=False)
    assert out["n_frames_run"] == 3
    assert p.n_encode == 3, "one encode per RUN FRAME, not per concept"
    assert p.n_prompt == 9


def test_a_failed_encode_is_recorded_once_per_concept():
    """⚠️ The C77 census must stay complete whichever half broke: if the shared
    encode dies, every concept on that frame still gets its own error row —
    otherwise one frame's crash would look like one crash instead of seven."""
    from ph0_sam3 import detect_many
    p = _CountingProc(raise_on=("__image__",))
    dets = detect_many(p, object(), ["car", "bus"])
    assert len(dets) == 2
    assert all("error" in d for d in dets)
    assert {d["concept"] for d in dets} == {"car", "bus"}


def test_one_bad_concept_does_not_lose_the_others():
    from ph0_sam3 import detect_many
    p = _CountingProc(raise_on=("bus",))
    dets = detect_many(p, object(), ["car", "bus", "pedestrian"])
    assert p.n_encode == 1
    ok = [d for d in dets if "score" in d]
    bad = [d for d in dets if "error" in d]
    assert [d["concept"] for d in ok] == ["car", "pedestrian"]
    assert [d["concept"] for d in bad] == ["bus"]


def test_detect_and_detect_many_agree_on_the_same_state():
    """`detect_many` must be a REFACTOR, not a new scoring path: one concept
    through either entry point yields the identical record."""
    from ph0_sam3 import detect, detect_many
    a = detect(_CountingProc(), object(), "car")
    b = detect_many(_CountingProc(), object(), ["car"])
    assert a == b


def test_live_is_ANY_control_not_ALL_because_sky_can_be_occluded():
    """⛔ CORRECTED BY THE DATA, 2026-08-16. The first version required EVERY
    control concept to fire; clip `24b6948f` returned `road 2 · sky 0` under an
    underpass and was flagged dead while the engine was plainly working (22
    detections on that clip). The control's question is *"is the engine
    producing at all?"* — one control detection answers it. Requiring all of
    them re-imports the scene-dependence the control exists to escape."""
    from ph0_sam3 import liveness_probe

    class _SkyOccluded:
        def set_image(self, img):
            return {}

        def set_text_prompt(self, state, prompt):
            import numpy as np
            if prompt == "road":
                return {"scores": np.array([0.9, 0.8]),
                        "boxes": np.array([[0, 0, 1, 1], [1, 1, 2, 2]]),
                        "masks": None}
            return {"scores": np.array([]), "boxes": np.array([]),
                    "masks": None}

    r = liveness_probe(_SkyOccluded(), object())
    assert r["n_det"] == {"road": 2, "sky": 0}
    assert r["live"] is True, "an occluded sky is not a dead engine"
    assert r["all_fired"] is False, "the stricter scene reading stays available"


def test_a_dead_engine_is_still_dead_under_the_any_rule():
    from ph0_sam3 import liveness_probe
    r = liveness_probe(_RaisingProc(), object())
    assert r["live"] is False and r["all_fired"] is False


def test_aug120_pipeline_reads_the_census_not_the_return_code():
    """⛔ C77's first half: `SAM3_RC=0` was read as full coverage. The second
    half is worse — a run that raised on every concept of every frame ALSO
    returns 0-shaped success with well-formed records. The batch driver must
    print the census and say so when the liveness control did not fire.

    Source-level by necessity: `aug120_pipeline.py` runs its batch loop at
    import time, so it cannot be imported in a test."""
    import os
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))), "scripts", "aug120_pipeline.py")
    src = open(p, encoding="utf-8").read()
    assert "SAM3_CENSUS" in src and "SAM3_CENSUS_FAIL" in src
    assert "SAM3_CENSUS_MISSING" in src, (
        "a pre-census sam3.json must be called out, not silently accepted")
    # the census must be read BEFORE the folder is pushed as coverage
    assert src.index("SAM3_CENSUS") < src.index("upload_folder")
    # and --n must still be explicit (the original 115-clip gap)
    assert '"--n", str(len(batch))' in src
