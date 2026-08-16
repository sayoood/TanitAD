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
    from ph0_sam3 import is_live
    dead = liveness_probe(_RaisingProc(), object())
    assert is_live(dead) is False
    assert dead["n_det"] == {"road": 0, "sky": 0}
    assert "BFloat16" in dead["errors"]["road"]

    live = liveness_probe(_LiveProc(), object())
    assert is_live(live) is True and live["n_det"]["road"] == 1
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
    from ph0_sam3 import is_live
    assert is_live(out["liveness"]) is False


def test_run_clip_frames_distinguishes_an_empty_scene_from_a_dead_engine():
    """The other half of the same test: agent concepts all zero, engine FINE.
    Before the control these two records were indistinguishable — which is
    precisely how 115 clips of nothing passed review."""
    from ph0_sam3 import run_clip_frames
    out = run_clip_frames(_LiveProc(), _frames(), ["car", "pedestrian"], [],
                          frame_stride=4)
    assert out["n_det_total"] == 0 and out["n_err_total"] == 0
    from ph0_sam3 import is_live
    assert is_live(out["liveness"]) is True
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

    from ph0_sam3 import is_live
    r = liveness_probe(_SkyOccluded(), object())
    assert r["n_det"] == {"road": 2, "sky": 0}
    assert is_live(r) is True, "an occluded sky is not a dead engine"


def test_a_dead_engine_is_still_dead_under_the_any_rule():
    from ph0_sam3 import is_live, liveness_probe
    r = liveness_probe(_RaisingProc(), object())
    assert is_live(r) is False


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


def test_no_derived_boolean_is_stored_in_the_record():
    """⛔ THE FIELD IS DELETED, NOT JUST RECOMPUTED. A `live` boolean lived in
    the schema for half a corpus and its rule changed mid-corpus (all -> any).
    MEASURED consequence, found on disk by the far-side census: clip
    `24b6948f` carried `live: False` **contradicting its own `n_det`
    {road: 2, sky: 0}** — a healthy underpass scene that every consumer
    reading the flag would score as the one dead-engine failure.

    The counts are the primitive; the verdict is a cache; a cache of a rule
    that changed is a trap with a long fuse. A field that cannot be stale
    beats a field that must be kept in sync."""
    from ph0_sam3 import liveness_probe, run_clip_frames
    r = liveness_probe(_LiveProc(), object())
    assert "live" not in r and "all_fired" not in r
    assert set(r) <= {"concepts", "n_det", "errors"}
    out = run_clip_frames(_LiveProc(), _frames(), ["car"], [], frame_stride=4)
    assert "live" not in out["liveness"]
    assert "all_fired" not in out["liveness"]


# =========================================================================== #
# The (1, H, W) mask shape — the defect that made v1's whole RLE undrawable    #
# =========================================================================== #
def test_the_real_mask_shape_is_N_1_H_W_and_the_rle_must_survive_it():
    """⛔ MEASURED ON A LIVE T4, 2026-08-16: `Sam3Processor.set_text_prompt`
    returns `masks` shaped **[N, 1, H, W]**. `_rows_rle` used to `enumerate` it
    directly, which yields ONE item whose `row` is the entire (H, W) plane, so
    `np.flatnonzero` returned FLATTENED indices and every banked run came out
    as `[0, flat_start, flat_end)`.

    It passed every check anyone ran: the run lengths still summed to
    `mask_area_px` (120 on the measured clip), the JSON was well-formed, the
    field was present. What it could not do was redraw the mask — and
    `ph0_rich_overlay.draw_masks` does `over[r, a:b]`, so it painted row 0 with
    an out-of-range slice, which numpy clips to nothing.

    A (1, H, W) mask must produce the SAME runs as its (H, W) squeeze."""
    from ph0_sam3 import _rows_rle
    import numpy as np
    m = np.zeros((7, 10), bool)
    m[2, 3:6] = True
    m[5, 0:2] = True
    flat = _rows_rle(m)
    assert flat == [[2, 3, 6], [5, 0, 2]]
    assert _rows_rle(m[None, ...]) == flat, "(1, H, W) must not flatten"
    assert _rows_rle(m[None, None, ...]) == flat


def test_a_mask_that_is_not_two_dimensional_raises_instead_of_guessing():
    """⚠️ The permissive version of this is what caused the defect. A shape it
    cannot interpret must FAIL, not emit plausible numbers."""
    from ph0_sam3 import as_2d_mask
    import numpy as np
    assert as_2d_mask(np.zeros((1, 4, 5), bool)).shape == (4, 5)
    with pytest.raises(ValueError):
        as_2d_mask(np.zeros((3, 4, 5), bool))


def test_the_renderer_shouts_when_a_run_falls_outside_the_frame():
    """⛔ THE HALF OF C85 THAT LET IT SURVIVE. `draw_masks` did `over[r, a:b]`
    and numpy CLIPS an out-of-range slice to nothing, so a flattened record
    rendered a clean, plausible, empty picture — and that picture was banked as
    an overlay video and reviewed. A renderer that draws nothing must SAY it
    drew nothing."""
    from ph0_sam3 import draw_masks
    import numpy as np
    from PIL import Image
    img = Image.fromarray(np.zeros((20, 30, 3), np.uint8))
    good = draw_masks(img, [{"rle_rows": [[5, 2, 8]], "box": [2, 5, 8, 6]}])
    flat = draw_masks(img, [{"rle_rows": [[0, 152, 158]],   # 5*30+2 flattened
                             "box": [2, 5, 8, 6]}])
    a, b = np.asarray(good), np.asarray(flat)
    assert (a[:, :, 0] != b[:, :, 0]).any(), "the two must not render alike"
    # the alarm is a red border on the flattened one, and only on it
    assert b[0, 0, 0] > 200 and b[0, 0, 1] < 60
    assert not (a[0, 0, 0] > 200 and a[0, 0, 1] < 60)


def test_detection_records_carry_the_mask_shape_they_were_drawn_from():
    """Without `mask_hw` a consumer cannot tell a correctly-encoded run from a
    flattened one — 36794 is a legal column index in SOME frame size. The
    banked height/width is what makes the encoding self-checking."""
    from ph0_sam3 import detect
    import numpy as np

    class FakeProc:
        def set_image(self, img):
            return {}

        def set_text_prompt(self, state, prompt):
            m = np.zeros((1, 1, 9, 20), bool)          # the REAL vendor shape
            m[0, 0, 4, 5:9] = True
            return {"scores": np.array([0.7]),
                    "boxes": np.array([[5.0, 4.0, 9.0, 5.0]]), "masks": m}

    r = detect(FakeProc(), object(), "car")[0]
    assert r["mask_hw"] == [9, 20]
    assert r["rle_rows"] == [[4, 5, 9]]
    assert r["mask_area_px"] == 4
    assert max(run[2] for run in r["rle_rows"]) <= r["mask_hw"][1]


# =========================================================================== #
# SCHEMA v2 — contours, oriented extents, the scene channel, the ego lane     #
# =========================================================================== #
def test_contour_area_equals_the_mask_area_exactly_before_simplification():
    """⛔ THE REASON THE TRACE IS ON THE PIXEL-CORNER LATTICE. A polygon through
    boundary-pixel CENTRES under-counts area by ~half the perimeter — a 3x3
    blob would come out at 4 px^2 against a true 9. Our objects are small
    (MEASURED median `car` box 188 px^2, `traffic light` 34 px^2), so that bias
    would dominate every oriented extent and would LOOK like a snug fit.

    At tol=0 the polygon's area must equal the pixel count exactly."""
    from ph0_sam3 import contour_of_mask
    m = _mask(9, 9, [(r, 3, 6) for r in range(3, 6)])       # a 3x3 block
    c = contour_of_mask(m, tol_px=0.0)
    assert c["contour_area_px"] == 9 == int(m.sum())
    assert c["contour_n_loops"] == 1
    # an L shape: still exact, and it needs more than 4 corners
    m2 = _mask(9, 9, [(3, 2, 7), (4, 2, 7), (5, 2, 4), (6, 2, 4)])
    c2 = contour_of_mask(m2, tol_px=0.0)
    assert c2["contour_area_px"] == int(m2.sum())
    assert len(c2["contour_xy"]) // 2 == 6                  # 6 corners, no more


def test_contour_is_a_flat_integer_list_on_the_corner_lattice():
    from ph0_sam3 import contour_of_mask
    c = contour_of_mask(_mask(6, 6, [(r, 1, 4) for r in range(1, 4)]),
                        tol_px=0.0)
    xy = c["contour_xy"]
    assert len(xy) % 2 == 0 and all(isinstance(v, int) for v in xy)
    xs, ys = xy[0::2], xy[1::2]
    # corner lattice: coordinates run 0..W and 0..H INCLUSIVE
    assert min(xs) == 1 and max(xs) == 4 and min(ys) == 1 and max(ys) == 4


def test_contour_survives_a_hole_and_keeps_the_outer_loop():
    """A donut has TWO loops. Only the outer one is the contour, and the record
    says how many there were so a fragmented detection stays visible."""
    from ph0_sam3 import contour_of_mask
    import numpy as np
    m = np.zeros((11, 11), bool)
    m[2:9, 2:9] = True
    m[4:7, 4:7] = False
    c = contour_of_mask(m, tol_px=0.0)
    assert c["contour_n_loops"] == 2
    # the OUTER polygon encloses the hole too, so it is bigger than the mask —
    # which is exactly why the RLE is still the primitive
    assert c["contour_area_px"] == 49 > int(m.sum()) == 40


def test_contour_never_replaces_the_rle_in_a_detection_record():
    """⛔ THE CONTOUR IS LOSSY AND ADDITIVE. `rle_rows` and `mask_area_px` must
    be present and unchanged next to it — a consumer redrawing the mask reads
    the RLE, never the polygon."""
    from ph0_sam3 import detect
    import numpy as np

    class FakeProc:
        def set_image(self, img):
            return {}

        def set_text_prompt(self, state, prompt):
            m = np.zeros((1, 12, 12), bool)
            m[0, 3:9, 4:8] = True
            return {"scores": np.array([0.9]),
                    "boxes": np.array([[4.0, 3.0, 8.0, 9.0]]), "masks": m}

    r = detect(FakeProc(), object(), "car")[0]
    assert r["mask_area_px"] == 24
    assert r["rle_rows"] and r["contour_xy"]
    assert r["contour_area_px"] == 24
    assert r["contour_tol_px"] == 1.0


def test_contours_can_be_turned_off_without_touching_the_mask():
    from ph0_sam3 import detect
    import numpy as np

    class FakeProc:
        def set_image(self, img):
            return {}

        def set_text_prompt(self, state, prompt):
            return {"scores": np.array([0.9]),
                    "boxes": np.array([[0.0, 0.0, 2.0, 2.0]]),
                    "masks": np.ones((1, 4, 4), bool)}

    r = detect(FakeProc(), object(), "car", contours=False)[0]
    assert r["mask_area_px"] == 16 and r["rle_rows"]
    assert "contour_xy" not in r and "obb_cxcylwa" not in r


def test_simplification_is_bounded_by_its_tolerance_and_reports_it():
    """The tolerance is a CONTRACT: no vertex of the true boundary may end up
    further than `tol` from the simplified polygon. And when the point cap
    forces a coarser tolerance, the record carries the one ACTUALLY used —
    not the one that was asked for."""
    from ph0_sam3 import contour_of_mask
    import numpy as np
    rng = np.random.default_rng(3)
    m = np.zeros((60, 60), bool)
    m[10:50, 10:50] = True
    # ragged edge -> a long boundary that the cap must bite on
    for r in range(10, 50):
        m[r, 50:50 + int(rng.integers(0, 6))] = True
    tight = contour_of_mask(m, tol_px=0.0, max_pts=10_000)
    capped = contour_of_mask(m, tol_px=0.0, max_pts=12)
    assert len(tight["contour_xy"]) // 2 > 12
    assert len(capped["contour_xy"]) // 2 <= 12
    assert capped["contour_tol_px"] > tight["contour_tol_px"]


def test_empty_mask_yields_no_contour_and_is_not_an_error():
    from ph0_sam3 import contour_of_mask
    import numpy as np
    assert contour_of_mask(np.zeros((5, 5), bool)) == {}


def test_oriented_extent_recovers_an_angle_a_box_cannot_express():
    """⭐ THE WHOLE POINT OF THE CONTOUR. The agent-slot decoder's target is
    `(cx, cy, yaw, l, w)`; `box_xyxy` has NO angle, so a rotated object and an
    upright one share a box. A diagonal bar must come back at ~45 degrees with
    a long side much longer than the short one."""
    from ph0_sam3 import contour_of_mask, oriented_extent
    import numpy as np
    m = np.zeros((60, 60), bool)
    for i in range(6, 50):
        m[i, i:i + 4] = True
    c = contour_of_mask(m, tol_px=0.5)
    cx, cy, lng, shrt, deg = oriented_extent(c["contour_xy"])
    assert lng > 4 * shrt
    assert 40.0 < deg < 50.0
    assert 20 < cx < 40 and 20 < cy < 40
    # ... and the axis-aligned box of the SAME mask is square: no angle in it
    ys, xs = np.nonzero(m)
    assert abs((xs.max() - xs.min()) - (ys.max() - ys.min())) <= 4


def test_oriented_extent_of_an_axis_aligned_rectangle_is_that_rectangle():
    from ph0_sam3 import contour_of_mask, oriented_extent
    m = _mask(20, 30, [(r, 4, 24) for r in range(6, 12)])   # 20 wide, 6 tall
    cx, cy, lng, shrt, deg = oriented_extent(
        contour_of_mask(m, tol_px=0.0)["contour_xy"])
    assert lng == pytest.approx(20.0) and shrt == pytest.approx(6.0)
    assert cx == pytest.approx(14.0) and cy == pytest.approx(9.0)
    assert deg in (0.0, 180.0) or deg < 1e-6


def test_oriented_extent_angle_is_a_direction_not_a_yaw():
    """A 180-degree flip is the SAME extent, so the angle folds into [0,180).
    Pinned because a consumer reading it as a heading would silently get a
    quantity that is only defined up to that flip."""
    from ph0_sam3 import oriented_extent
    a = oriented_extent([0, 0, 10, 0, 10, 4, 0, 4])
    b = oriented_extent([10, 4, 0, 4, 0, 0, 10, 0])         # reversed winding
    assert a[2:] == b[2:] and 0.0 <= a[4] < 180.0


def test_scene_concepts_are_a_separate_channel_from_the_agent_contract():
    """⛔ THE CONTRACT. `ph1_fuse` builds object tracks out of `frames[*].det`
    and every run report sums `per_concept_hits` / `n_det_total`. Lane markings
    are not objects and must not enter either — otherwise a schema addition
    silently moves numbers three documents quote."""
    from ph0_sam3 import run_clip_frames

    class _P:
        def set_image(self, img):
            return {}

        def set_text_prompt(self, state, prompt):
            import numpy as np
            n = 2 if prompt == "lane marking" else 1
            return {"scores": np.full(n, 0.8),
                    "boxes": np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (n, 1)),
                    "masks": None}

    out = run_clip_frames(_P(), _frames(), ["car"], [], frame_stride=4,
                          scene_concepts=["lane marking", "road curb"],
                          liveness=False)
    assert out["schema_version"] == 2
    assert out["per_concept_hits"] == {"car": 3}            # 3 frames
    assert "lane marking" not in out["per_concept_hits"]
    assert out["per_scene_hits"] == {"lane marking": 6, "road curb": 3}
    assert out["n_det_total"] == 3, "the agent total must not move"
    assert out["n_scene_det_total"] == 9
    f0 = out["frames"]["0"]
    assert [d["concept"] for d in f0["det"]] == ["car"]
    assert {d["concept"] for d in f0["scene"]} == {"lane marking", "road curb"}


def test_road_is_the_control_and_is_never_a_scene_class():
    """⛔ A CONTROL DRAWN FROM THE MEASURED VOCABULARY IS CIRCULAR. `road` is
    the C77 positive control; putting it in SCENE_CONCEPTS would make "the
    engine is live" and "the scene channel produced something" the SAME event.
    The PI asked for a road class and gets one — with full geometry, from the
    control's own frame, banked outside every measured total."""
    from ph0_sam3 import (AGENT_CONCEPTS, LIVENESS_CONCEPTS, SCENE_CONCEPTS,
                          liveness_probe)
    assert not (set(SCENE_CONCEPTS) & set(LIVENESS_CONCEPTS))
    assert not (set(SCENE_CONCEPTS) & set(AGENT_CONCEPTS))
    assert "road" in LIVENESS_CONCEPTS and "road" not in SCENE_CONCEPTS
    r = liveness_probe(_LiveProc(), object(), keep_det=True)
    assert [d["concept"] for d in r["det"]] == ["road", "sky"]
    assert "live" not in r and "all_fired" not in r        # still no cache


def test_scene_concepts_ride_the_same_encode_as_the_agent_concepts():
    """⛔ THE 4.21x MUST NOT BE SPENT ON THE NEW CLASSES. Scoring the scene
    vocabulary in a second `detect_many` call would re-run the ViT trunk a
    second time per frame — the exact defect the encode-once fix removed, in a
    new costume, and it would have read as 'the new classes are expensive'."""
    from ph0_sam3 import run_clip_frames
    p = _CountingProc()
    out = run_clip_frames(p, _frames(n=9), ["car", "bus"], [], frame_stride=4,
                          scene_concepts=["lane marking", "road curb"],
                          liveness=False)
    assert out["n_frames_run"] == 3
    assert p.n_encode == 3, "one encode per FRAME — not one per channel"
    assert p.n_prompt == 12                                # 3 frames x 4


def test_the_error_census_covers_both_channels():
    """⚠️ `n_err_total` now spans agent AND scene, on purpose: an error census
    that silently omitted a channel would be C77's own defect rebuilt. The v1
    quantity is still readable as `n_err_agent`."""
    from ph0_sam3 import run_clip_frames
    out = run_clip_frames(_RaisingProc(), _frames(), ["car"], [],
                          frame_stride=4, scene_concepts=["lane marking"],
                          liveness=False)
    assert out["n_err_agent"] == 3 and out["n_err_scene"] == 3
    assert out["n_err_total"] == 6
    assert out["err_kinds"] == {"RuntimeError": 6}


def test_a_v1_consumer_reads_a_v2_record_unchanged():
    """The additive-schema promise, pinned. Every key a v1 consumer touches is
    present and means what it meant; the new ones are extra."""
    from ph0_sam3 import run_clip_frames
    v1 = run_clip_frames(_LiveProc(), _frames(), ["car"], [], frame_stride=4,
                         scene_concepts=None)
    v2 = run_clip_frames(_LiveProc(), _frames(), ["car"], [], frame_stride=4,
                         scene_concepts=["lane marking"])
    for k in ("frames", "per_concept_hits", "n_frames_run", "n_det_total",
              "n_err_total", "err_kinds", "liveness", "vlm_cross_check"):
        assert k in v1 and k in v2
    assert v1["per_concept_hits"] == v2["per_concept_hits"]
    assert v1["n_det_total"] == v2["n_det_total"]
    for fk, fv in v2["frames"].items():
        assert fv["det"] == v1["frames"][fk]["det"]
    assert "schema_version" not in v1                       # v1 stays v1


def test_ego_lane_is_derived_and_says_so():
    """⛔ 'EGO LANE' IS NOT PROMPTED FOR. It is a RELATION between the ego and
    the boundaries, not an appearance — the ego's lane pixels look exactly like
    the next lane's. A prompt would return a plausible unfalsifiable mask."""
    from ph0_sam3 import SCENE_CONCEPTS, derive_ego_lane
    assert not any("ego" in c for c in SCENE_CONCEPTS)
    dets = [{"concept": "lane marking", "score": 0.7,
             "rle_rows": [[95, 30, 33], [99, 28, 31]]},
            {"concept": "lane marking", "score": 0.6,
             "rle_rows": [[97, 70, 73], [99, 72, 75]]},
            {"concept": "road curb", "score": 0.8,
             "rle_rows": [[99, 4, 8]]}]
    out = derive_ego_lane(dets, (100, 100))
    assert out["class"] == "DERIVED-ESTIMATED"
    assert out["derived_from"] == ["lane marking", "road curb"]
    assert out["n_left"] == 2 and out["n_right"] == 1
    # ⛔ 0-based FROM THE RIGHT — `ego_lane_idx`'s own definition in
    # `s2_derive.LANE_CONTEXT_INPUTS` ("ego's 0-based lane index from the
    # right"). A second convention for the same quantity is how two correct
    # numbers become one wrong one, and the first version of this function had
    # it backwards.
    assert out["lane_idx_est"] == 0
    assert out["lane_width_px"] == pytest.approx(43.5, abs=0.6)
    assert out["n_lanes_est"] == 2                 # 3 boundaries -> 2 gaps


def test_ego_lane_says_which_of_the_four_lane_inputs_it_does_NOT_supply():
    """⛔ THE OVERCLAIM THIS BLOCKS. `lane_change_requirement()` needs four
    inputs; two of them (`route_lane_idx`, `lane_continues`) need lane
    TOPOLOGY, which no camera frame contains. A vision boundary estimator
    supplies `ego_lane_idx` and `n_lanes_same_direction` and NOT those two — so
    `PREPARE_LANE_CHANGE` stays blocked, and the record says so itself rather
    than leaving the reader to infer it."""
    from ph0_sam3 import derive_ego_lane
    out = derive_ego_lane([], (100, 100))
    assert out["supplies"] == ["ego_lane_idx", "n_lanes_same_direction"]
    assert out["does_not_supply"] == ["route_lane_idx", "lane_continues"]
    assert "FROM THE RIGHT" in out["index_convention"]


def test_ego_lane_refuses_when_it_is_not_bounded_on_both_sides():
    """A one-sided read is not a lane. It must return the REASON, not a number
    that looks like a measurement."""
    from ph0_sam3 import derive_ego_lane
    out = derive_ego_lane([{"concept": "lane marking", "score": 0.7,
                            "rle_rows": [[99, 10, 13]]}], (100, 100))
    assert out["lane_idx_est"] is None and out["lane_width_px"] is None
    assert "not bounded" in out["reason"]
    empty = derive_ego_lane([], (100, 100))
    assert empty["n_boundary_det"] == 0 and empty["lane_idx_est"] is None


def test_ego_lane_merges_the_dashes_of_one_painted_line():
    """SAM3 returns a dashed line as one detection PER DASH. Three dashes of
    the same boundary are ONE boundary — counting them as three would inflate
    `lane_idx_est`, which is the number the blocked PREPARE_LANE_CHANGE work
    would consume."""
    from ph0_sam3 import derive_ego_lane
    dets = [{"concept": "lane marking", "score": 0.7,
             "rle_rows": [[99, 30 + k, 32 + k]]} for k in (0, 1, 2)]
    dets.append({"concept": "lane marking", "score": 0.7,
                 "rle_rows": [[99, 70, 72]]})
    out = derive_ego_lane(dets, (100, 100))
    assert len(out["boundaries"]) == 2 and out["boundaries"][0]["n"] == 3


def test_ego_lane_ignores_the_far_field():
    """Perspective collapses the lanes together up the image; a marking whose
    footpoint is above the near-field band says nothing about which lane the
    ego is in."""
    from ph0_sam3 import derive_ego_lane
    far = [{"concept": "lane marking", "score": 0.7,
            "rle_rows": [[10, 40, 42]]},
           {"concept": "lane marking", "score": 0.7,
            "rle_rows": [[12, 60, 62]]}]
    out = derive_ego_lane(far, (100, 100))
    assert out["n_boundary_det"] == 0
    assert out["reason"] == "no boundary detection in the near field"


def test_concept_kinds_say_which_counts_are_object_counts():
    """⚠️ `per_scene_hits['lane marking'] = 14` is NOT 'fourteen lane markings'
    — it is fourteen separately grounded painted segments. A count of STUFF is
    not an object count, and the record must say which it is."""
    from ph0_sam3 import AGENT_CONCEPTS, CONCEPT_KIND, SCENE_CONCEPTS
    for c in AGENT_CONCEPTS:
        assert CONCEPT_KIND[c] == "thing"
    for c in SCENE_CONCEPTS:
        assert CONCEPT_KIND[c].startswith("stuff")
    assert CONCEPT_KIND["road"] == "stuff_region"


def test_build_processor_verifies_the_threshold_it_asked_for():
    """⛔ THE FLOOR IS DESTRUCTIVE AND INVISIBLE. Anything below
    `confidence_threshold` never reaches the record, and a corpus built at the
    wrong floor cannot be told from a correct one by looking at it — only by
    re-detecting. The kwarg was read from vendor SOURCE and never executed
    (exactly the check that failed on this engine before), so it must be read
    back off the object and the run must refuse if it did not take."""
    import inspect
    from ph0_sam3 import build_processor
    body = inspect.getsource(build_processor).split('"""')[-1]
    assert 'getattr(proc, "confidence_threshold", None)' in body
    assert "REFUSED" in body
    assert body.index("confidence_threshold=conf_threshold") < body.index(
        "REFUSED")


def test_is_live_ignores_a_stale_stored_flag_and_trusts_the_counts():
    """Pre-2026-08-16 records still carry the boolean. It must be IGNORED —
    including when it lies in the dangerous direction (a healthy clip stored
    as dead), which is the exact record the census found."""
    from ph0_sam3 import is_live
    stale_false = {"n_det": {"road": 2, "sky": 0}, "live": False}
    stale_true = {"n_det": {"road": 0, "sky": 0}, "live": True}
    assert is_live(stale_false) is True, "road 2 means the engine ran"
    assert is_live(stale_true) is False, "all-zero counts mean it did not"
    assert is_live(None) is False and is_live({}) is False
