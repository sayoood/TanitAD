"""Engine C — SAM3, the PIXEL engine of the PI's decided stack.

engine A = algorithmic integrated ego path (NUMBERS) · engine B = Qwen3.5-9B
(SYMBOLS + OCR) · **engine C = SAM3 (PIXELS)**. This is C.

⭐ TWO PROMPT MODES, AND THE SECOND ONE IS THE VALUABLE ONE.

  `boxes`  — SAM3 is prompted with the VLM's own B3 sign boxes and refines them
             to pixel-accurate masklets. The box↔mask agreement is a measurable
             grounding signal where a VLM box alone is an unverifiable claim.

  `text`   — SAM3 is prompted with CONCEPT TEXT ("car", "pedestrian", …) and
             finds the objects ITSELF. ⭐ This is the one that closes a real gap:
             the tactical vocabulary needs AGENT SLOTS (`GAP_TARGET(agent_slot,
             time_gap_s)`, `EVADE_IN_CORRIDOR(obstacle_slot)`,
             `WAIT_FOR_ONCOMING(oncoming_slot)`, `TRAFFIC_LIGHT_REACT(light_slot)`,
             `FOLLOW(time_gap_s)`) and PH0 currently extracts NO agents at all.
             It is also INDEPENDENT of the VLM, so agreement between the two is
             evidence rather than a refinement of one engine by itself.

`propagate_in_video` carries each prompt across frames, so the output is TRACKS,
not per-frame blobs — which is what a time-gap or an oncoming-slot needs.

⛔ THREE THINGS HERE WERE MEASURED THE HARD WAY, ALL ON 2026-08-12:

 1. **`SAM3InteractiveImagePredictor` is the WRONG predictor for this.** Its
    annotation is `sam_model: Sam3TrackerBase` and it reads `model.image_size`;
    handed a `Sam3Image` from `build_sam3_image_model` it dies with
    ``AttributeError: 'Sam3Image' object has no attribute 'image_size'``.
    ⚠️ I had recorded "API verified, not guessed" for that pairing. It was a
    SIGNATURE check, not an EXECUTION check — the repo was gated so I could not
    run it, and I wrote the weaker check down using the stronger word. Guessing
    that `enable_inst_interactivity=True` would fix it was wrong too: same error.
 2. **Boxes are XYWH here, not XYXY** (`add_prompt` → `boxes_xywh=`). Passing
    xyxy would have produced plausible-looking masks that were silently wrong.
 3. **The gate was real and access is now granted.** While gated,
    `/api/models/facebook/sam3` returned 200 while every file 403'd — metadata
    being public says NOTHING about file access, and I reported this repo as
    "available" twice on that basis. MEASURED after Meta's approval:
    `hf_hub_download("facebook/sam3","config.json")` -> 25 843 bytes.
    (`sam3-large` / `sam3-base` are 404 and do not exist.)

API (read from the installed package source, `sam3/model/sam3_video_predictor.py`):
  Sam3VideoPredictor(checkpoint_path=None, bpe_path=..., video_loader_type="cv2")
    .start_session(resource_path=<mp4>)            -> {"session_id": ...}
    .add_prompt(session_id, frame_idx, text=..., bounding_boxes=[[x,y,w,h]], obj_id=...)
    .propagate_in_video(session_id, propagation_direction, start_frame_idx,
                        max_frame_num_to_track)    -> yields {"frame_index", "outputs"}
    .close_session(session_id)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

MASK_COLOURS = [(42, 120, 214), (235, 104, 52), (74, 58, 167),
                (26, 148, 106), (196, 62, 140), (176, 132, 20)]

#: ⛔ THE SCHEMA IS VERSIONED BECAUSE IT IS NOW EXTENDED, AND THE EXTENSION IS
#: STRICTLY ADDITIVE. v1 records carry no `schema_version` at all; v2 adds
#: contours + oriented extents per detection, a SEPARATE scene-concept channel,
#: and a derived ego-lane block. Every v1 key keeps its v1 meaning — `det`,
#: `per_concept_hits` and `n_det_total` still describe the AGENT vocabulary and
#: nothing else — so `ph1_fuse.py`, `ph0_rich_overlay.py`, `aug120_pipeline.py`
#: and `content_census` read a v2 record without a change and without a silent
#: shift in any number they already quote. Read the version, never infer it
#: from which keys happen to be present.
SCHEMA_VERSION = 2

#: RDP tolerance in PIXELS. 1.0 px is one mask pixel: the simplified polygon may
#: not stray further from the true boundary than the boundary's own
#: quantisation. Measured area error at this tolerance: `SAM3_EXTRACTION_V2.md`.
#: (Defined up here, not beside `contour_of_mask`, because `_score`'s signature
#: needs it at def time — a default argument is evaluated at import.)
CONTOUR_TOL_PX_DEFAULT = 1.0
#: Hard cap on points per contour. A polygon is a COMPACT summary — an
#: unsimplified boundary is just the mask again in a more expensive encoding,
#: and the mask is already banked as RLE. On overflow the tolerance is doubled
#: until the cap holds, and the tolerance actually used is banked per detection.
CONTOUR_MAX_PTS_DEFAULT = 48

# The concept vocabulary for the `text` mode. Each entry is a tactical-vocabulary
# slot that PH0 cannot currently fill — see the module docstring. Deliberately
# CONCRETE nouns: SAM3 grounds objects, not abstractions like "hazard".
AGENT_CONCEPTS = ["car", "truck", "bus", "pedestrian", "cyclist",
                  "traffic light", "traffic sign"]

# ⭐ THE SCENE VOCABULARY — a DIFFERENT KIND OF THING, kept structurally apart
#   (PI, 2026-08-16: *"include additional classes like guardrails, road
#   markings, road, ego lane, road curbs"*).
#
# ⛔ WHY IT IS A SEPARATE LIST AND A SEPARATE RECORD SLOT, not seven more
# entries in AGENT_CONCEPTS. `AGENT_CONCEPTS` is a CONTRACT: `ph1_fuse.py`
# builds per-object tracks out of `frames[*].det` and filters hazards on
# `t["concept"] in ("car","truck","bus","pedestrian",...)`, and
# `per_concept_hits` is what `content_census` and every run report sum. Pouring
# lane markings into that list would (a) create dozens of one-frame "tracks"
# per clip that mean nothing as objects, and (b) silently move `n_det_total` —
# the number three documents quote — without any consumer noticing. So scene
# detections live in `frames[*].scene` / `per_scene_hits`, and the agent
# contract is byte-for-byte what it was.
#
# ⚠️ AND THEY ARE NOT "THINGS". A car is a countable instance; a lane marking is
# a piece of STUFF that SAM3 happens to return chopped into instances (one per
# painted dash), and a road/guardrail is one extended region whose instance
# count is an artifact of occlusion, not a fact about the world. `CONCEPT_KIND`
# records which is which so a consumer never reads `per_scene_hits["lane
# marking"] = 14` as "14 lane markings" — it is *"14 painted segments were
# separately grounded"*. Counts of stuff are not object counts.
#
# ⭐ WHY THESE THREE FIRST, and it is not a perception argument. The PI ruled
# that `PREPARE_LANE_CHANGE` must be derived from CONTEXT, and that work is
# BLOCKED on two inputs that do not exist — `route_lane_idx` (which lane the ego
# is in) and `lane_continues` — because nothing in the programme perceives lane
# structure at all (`…/2026-08-16-s2-v1-labels/review/PI_REVIEW_FINDINGS.md`,
# §BINDING). Lane markings and curbs ARE that missing input: the markings give
# the lane boundaries, the curb gives the road edge that anchors the count, and
# `derive_ego_lane` turns them into `lane_idx_est` + a lane-width estimate.
# ⇒ These three carry a blocked decision, so they lead the list; the rest of the
# PI's scene vocabulary follows them.
#
# ⛔ `road` IS DELIBERATELY ABSENT FROM THIS LIST — see LIVENESS_CONCEPTS below.
SCENE_CONCEPTS = ["lane marking", "road curb", "guardrail", "road marking"]

#: What each concept IS, so a count is never read as an object count. Not
#: per-detection — kind is a property of the CLASS, and writing it on every
#: detection would be the same cache-of-a-rule trap as the deleted `live` flag.
CONCEPT_KIND = {c: "thing" for c in AGENT_CONCEPTS}
CONCEPT_KIND.update({
    # painted stuff, returned per dash/segment — instance count is a
    # property of the paint pattern, not of the road
    "lane marking": "stuff_instanced",
    "road marking": "stuff_instanced",
    # one extended structure per side, chopped by occlusion
    "road curb": "stuff_extended",
    "guardrail": "stuff_extended",
    # one region, always; an instance count here is meaningless
    "road": "stuff_region",
    "sky": "stuff_region",
})

# ⭐ THE LIVENESS CONTROL — the fix for retraction class C77.
#
# Every concept in AGENT_CONCEPTS can LEGITIMATELY be 0 on a given frame (an
# empty road has no car and no pedestrian), so a run of all-zero records is
# indistinguishable from a run where the model never produced a single
# detection. That is exactly how 115 backfilled clips of pure
# `RuntimeError: mat1 and mat2 must have the same dtype` passed every
# structural check on 2026-08-16 (RETRACTION_LOG C77).
#
# `road` and `sky` are run once per clip as a POSITIVE CONTROL: a
# forward-facing driving frame cannot return zero for BOTH unless the engine
# itself is producing nothing. They are kept OUT of `per_concept_hits` and out
# of `n_det_total` so the fused record's agent-slot contract is unchanged — the
# control lives in its own `liveness` block.
#
# ⚠️ TWO PROPERTIES MAKE THEM THE RIGHT CONTROL, and both were tested rather
# than assumed:
#   1. they are DISJOINT from the measured vocabulary (a control drawn from the
#      quantity under test is circular);
#   2. they score FAR FROM the processor's 0.5 confidence threshold, so they do
#      not flicker on re-encode noise the way the agent concepts do — MEASURED,
#      `raw/mp4_source_check.json` (C79): a different encode of the same clip
#      moves `traffic light` 0→2 while road/sky hold.
#
# ⛔ AND THIS IS WHY THE PI'S `road` IS **NOT** IN `SCENE_CONCEPTS` (2026-08-16).
# The PI named `road` as a class to extract. Adding the string "road" to the
# measured vocabulary would make the control and the measurement THE SAME
# EVENT: a reader could then certify the scene channel live by pointing at a
# number the scene channel itself produced, which is property (1) above
# destroyed. It is not a naming problem either — spelling the class "road
# surface" to dodge the string collision would be a fig leaf, because it is the
# same object under a synonym and its failures would correlate with the
# control's.
#
# ⇒ THE RESOLUTION, and it costs nothing: `road` is extracted WITH FULL
# GEOMETRY (box + mask + contour) by the control itself — `liveness_probe`
# banks `det` now, not only counts — so the PI gets a segmented road region per
# clip, and it stays OUT of `per_concept_hits`, `per_scene_hits` and
# `n_det_total`. The price, stated because it is real: road geometry is banked
# on the ONE control frame per clip, not on all six run frames. Per-frame road
# segmentation is available for the asking, but it requires re-basing the
# positive control onto a different concept pair first — a decision escalated
# to the PI rather than taken here, because `road` (83/83 clips) and `sky`
# (81/83) are the only two concepts MEASURED reliable enough to be a control
# (`…/2026-08-16-sam3-concept-reliability/SAM3_CONCEPT_RELIABILITY.md` §2).
LIVENESS_CONCEPTS = ["road", "sky"]


def find_bpe() -> str | None:
    """SAM3's text encoder needs the CLIP BPE vocab, and the sam3 wheel does NOT
    ship it — it defaults to `site-packages/assets/bpe_simple_vocab_16e6.txt.gz`
    which does not exist, so the builder dies with a FileNotFound three frames
    deep. ⚠️ It is NOT in the `facebook/sam3` HF repo either: that repo carries
    HF-format tokenizer files (vocab.json + merges.txt), not the CLIP .gz.
    `open_clip` ships the canonical file, so we locate it there."""
    import glob
    # ⛔ THE INTERPRETER'S OWN site-packages FIRST. The absolute roots below are
    # pod4's and cover NOTHING on Colab (`/usr/local/lib/python3.12/
    # dist-packages`), and the `**/` globs are cwd-relative, so on 2026-08-16
    # the headless Colab run had to copy the vocab to `/content` by hand as
    # session bring-up. Asking the interpreter where its packages are removes
    # that bring-up step on every host at once.
    try:
        import site
        roots = list(site.getsitepackages() or [])
        usr = site.getusersitepackages()
        if isinstance(usr, str):
            roots.append(usr)
    except Exception:                                   # frozen / venv-less
        roots = []
    try:
        import open_clip
        roots.insert(0, os.path.dirname(os.path.abspath(open_clip.__file__)))
    except Exception:                                   # not installed
        pass
    for r in roots:
        for cand in (os.path.join(r, "bpe_simple_vocab_16e6.txt.gz"),
                     os.path.join(r, "open_clip",
                                  "bpe_simple_vocab_16e6.txt.gz")):
            if os.path.exists(cand):
                return cand
    for pat in ("/workspace/a2venv/lib/python3.12/site-packages/open_clip/"
                "bpe_simple_vocab_16e6.txt.gz",
                "**/open_clip/bpe_simple_vocab_16e6.txt.gz",
                "**/bpe_simple_vocab_16e6.txt.gz"):
        if os.path.exists(pat):
            return pat
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    for root in ("/workspace/a2venv", "/usr/lib/python3", "/root"):
        for dp, _dn, fn in os.walk(root):
            if "bpe_simple_vocab_16e6.txt.gz" in fn:
                return os.path.join(dp, "bpe_simple_vocab_16e6.txt.gz")
    return None


def install_dtype_agreement() -> dict:
    """⛔ THE C77 FIX. Make SAM3's fused MLP kernel keep the dtype it was given.

    THE DEFECT, MEASURED on a Colab T4 (torch 2.11.0+cu128, sam3 @ HEAD,
    2026-08-16). `sam3/model/vitdet.py:71` is

        x = addmm_act(type(self.act), self.fc1, x)      # Mlp.forward

    and `sam3/perflib/fused.py:15-17` force-casts bias, input and weight to
    **bfloat16** before `torch.ops.aten._addmm_activation`. The very next line,
    `vitdet.py:74`, is a plain `nn.Linear` whose weights are **fp32** ⇒

        RuntimeError: mat1 and mat2 must have the same dtype,
                      but got BFloat16 and Float

    on EVERY concept of EVERY frame — the payload of the 115 empty backfill
    records (RETRACTION_LOG C77).

    ⚠️ WHY IT ONLY BITES THE IMAGE PATH. Every SAM3 *video* entry point enters
    a process-wide bf16 autocast at construction
    (`sam3_multiplex_base.py:170-172`, `sam3_tracking_predictor.py:50`,
    `sam3_multiplex_video_predictor.py:51` — "use bfloat16 inference for Flash
    Attention kernel"), which casts fc2's weight to bf16 too and hides the
    split. `Sam3Processor` — the documented IMAGE path, and the one this engine
    uses — enters no such context. MEASURED: `torch.is_autocast_enabled("cuda")`
    is **False** both after `build_sam3_image_model` and at the failing Linear.

    ⚠️ `USE_PERFLIB=0` DOES NOT HELP — MEASURED, do not reach for it:
    `perflib.is_enabled` goes False but `Mlp.forward` calls `addmm_act`
    **unconditionally**; the flag gates other call sites, not this one.

    ⚠️ AND IT MUST BE PATCHED IN `vitdet`'s NAMESPACE, not in `perflib.fused`:
    `vitdet.py:31` does `from sam3.perflib.fused import addmm_act`, so the name
    is already bound and patching the source module is a no-op.

    THE CHOICE, MEASURED head-to-head on one real clip frame, same processor,
    concepts road/sky/car/truck/bus/pedestrian/cyclist/traffic light/traffic
    sign/tree:

      A  scoped `torch.autocast(cuda, bfloat16)`  10.5 s · peak 14 236 MiB
      B  plain fp32 fc1+act (fused kernel dropped)  4.5 s · peak 14 367 MiB
      C  THIS — fused kernel kept, bf16 casts removed  4.6 s · peak 14 367 MiB

    All three are LIVE (road 2, sky 1, car 8, tree 5 — identical counts on all
    ten concepts). A vs B scores differ by up to 5.9e-3, C vs B by up to
    1.3e-3. ⇒ **C is chosen and NOTHING IS DOWNGRADED: the trunk stays fp32**,
    which (a) is the precision of the pod4 2026-08-12 reference, (b) is
    device-independent — A costs 2.3× wall-clock here only because bf16 is
    EMULATED on this T4 (`is_bf16_supported(including_emulation=False)` is
    False, capability 7.5), so A's numerics would differ between the T4 and an
    A40 and break cross-arm comparability, and (c) is one rebound name rather
    than a replaced vendor method. Repeat passes are bit-identical.

    Idempotent; returns the provenance dict that `build_processor` banks."""
    import torch
    try:
        import sam3.model.vitdet as vitdet
    except Exception as e:                              # sam3 not installed
        return {"applied": False, "reason": f"{type(e).__name__}: {e}"[:80]}
    if getattr(vitdet.addmm_act, "_tanitad_dtype_safe", False):
        return {"applied": True, "reason": "already installed"}

    act_op = torch.ops.aten._addmm_activation
    vendor = vitdet.addmm_act

    def addmm_act_same_dtype(activation, linear, mat1):
        """`perflib/fused.py::addmm_act` verbatim MINUS its three
        `.to(torch.bfloat16)` casts — the fused addmm+activation kernel is
        kept, it just runs in the dtype the caller handed it."""
        bias = linear.bias.detach().to(mat1.dtype)
        w = linear.weight.detach().to(mat1.dtype)
        flat = mat1.reshape(-1, mat1.shape[-1])
        if activation in (torch.nn.functional.relu, torch.nn.ReLU):
            y = act_op(bias, flat, w.t(), beta=1, alpha=1, use_gelu=False)
        elif activation in (torch.nn.functional.gelu, torch.nn.GELU):
            y = act_op(bias, flat, w.t(), beta=1, alpha=1, use_gelu=True)
        else:                                  # vendor raises here too
            raise ValueError(f"Unexpected activation {activation}")
        return y.view(mat1.shape[:-1] + (y.shape[-1],))

    addmm_act_same_dtype._tanitad_dtype_safe = True
    addmm_act_same_dtype._vendor = vendor
    vitdet.addmm_act = addmm_act_same_dtype
    return {"applied": True,
            "target": "sam3.model.vitdet.addmm_act",
            "reason": "perflib fused MLP casts to bf16 while fc2 stays fp32 "
                      "(C77); casts removed, fused kernel kept, trunk fp32"}


def build_processor(bpe_path: str | None = None,
                    conf_threshold: float | None = None):
    """The OFFICIAL image path, verbatim from facebookresearch/sam3's README:

        model = build_sam3_image_model()
        processor = Sam3Processor(model)
        state = processor.set_image(image)
        out = processor.set_text_prompt(state=state, prompt="<TEXT>")
        masks, boxes, scores = out["masks"], out["boxes"], out["scores"]

    ⭐ MEASURED WORKING on pod4 2026-08-12 on a real clip frame:
        sky 0.9861 (n=1) · road 0.9406 (n=1) · tree 0.9389 (n=16)
        traffic sign 0.8301 (n=1) at [263.2, 74.2, 273.2, 84.4]
        car 0 · vehicle 0 · person 0
    ⚠️ **THE ZEROS ARE A CORRECT ABSTENTION, NOT A FAILURE** — that frame is an
    open road with trees, sky and one sign, and no car or person in it. An
    earlier run reported "0 frames" on all 8 clips and I could not tell a broken
    reader from an empty scene; the fix is that `n_det` and the per-concept
    scores are now recorded, so the artifact distinguishes them without a re-run.

    ⛔ WHY NOT THE VIDEO PREDICTOR. `Sam3VideoPredictor` runs and its weights do
    load (`build_sam3_video_model(load_from_HF=True)`), but it costs a full
    199-frame decode per session and returns the same information. The image
    path is the documented one, returns boxes in ORIGINAL frame coordinates, and
    is what the cross-engine check needs."""
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    bpe = bpe_path or find_bpe()
    if bpe is None:
        raise SystemExit("[sam3] CLIP BPE vocab not found — install "
                         "open_clip_torch (--no-deps) or pass --bpe-path")
    # resolved HERE, not as a default arg, so the module-level constant may sit
    # beside its own reasoning below rather than being forced above this function
    if conf_threshold is None:
        conf_threshold = CONF_THRESHOLD_DEFAULT
    dtype_fix = install_dtype_agreement()          # ⛔ BEFORE any forward (C77)
    model = build_sam3_image_model(bpe_path=bpe, load_from_HF=True)
    # ⛔ THE THRESHOLD IS APPLIED, THEN READ BACK OFF THE OBJECT. The kwarg was
    # read from vendor source and never executed, and "the signature says so"
    # is precisely the check that failed on this engine before
    # (`SAM3InteractiveImagePredictor`: signature fine, first run dead). A
    # constructor that quietly ignored an unknown kwarg — or a vendor bump that
    # renamed it — would leave the whole 115-clip corpus detected at 0.5 while
    # every report said 0.25, and NOTHING downstream could tell: the floor is
    # invisible in the output except as an absence. So: try the kwarg, fall
    # back to assignment, and bank what the attribute ACTUALLY reads.
    how = "ctor kwarg"
    try:
        proc = Sam3Processor(model, confidence_threshold=conf_threshold)
    except TypeError:
        proc = Sam3Processor(model)
        proc.confidence_threshold = conf_threshold
        how = "attribute assignment (ctor kwarg rejected)"
    eff = getattr(proc, "confidence_threshold", None)
    if eff is None or abs(float(eff) - float(conf_threshold)) > 1e-9:
        raise SystemExit(
            f"[sam3] confidence_threshold REFUSED: asked {conf_threshold}, "
            f"processor reports {eff!r} via {how}. Refusing to detect — a "
            "corpus built at an unknown floor is unattributable and the floor "
            "cannot be recovered afterwards.")
    return proc, {
        "api": "build_sam3_image_model + Sam3Processor "
               "(facebookresearch/sam3 README)",
        "weights": "facebook/sam3 (load_from_HF=True)",
        "bpe_path": bpe,
        "confidence_threshold": float(eff),
        "confidence_threshold_set_via": how,
        "schema_version": SCHEMA_VERSION,
        "dtype_fix": dtype_fix}


#: ⛔ THE DETECTION FLOOR, AND IT IS DESTRUCTIVE AT WRITE TIME (PI, 2026-08-16).
#:
#: `Sam3Processor` applies `keep = out_probs > confidence_threshold` INSIDE
#: `_forward_grounding` (`sam3/model/sam3_image_processor.py`), so anything below
#: it never reaches our record. MEASURED two ways: that source line, and the
#: **minimum banked score across all 2,496 detections is exactly 0.5000** — the
#: vendor default, printed on every record we own.
#:
#: ⚠️ THE ASYMMETRY IS THE WHOLE POINT. Lowering the floor LATER cannot recover
#: the tail — it requires re-detecting every clip (~26 GPU-hours for the 4,472
#: build). Lowering it BEFORE costs nothing: same forward pass, same wall-clock,
#: only more rows survive. `detect(min_score=...)` sits DOWNSTREAM of this and can
#: always filter back UP for free, so a low floor here is strictly more
#: information and never less.
#:
#: ⇒ 0.25 is a DECISION (PI, 2026-08-16), not a default anyone inherited — which
#: is what 0.5 was. The reliability study that motivated it also found the
#: dominant `traffic sign` failure is sign-SHAPED objects (a pharmacy cross at
#: 0.807, a hoarding, a green light) — those are ABOVE 0.5, so no floor removes
#: them and none of this claims otherwise; that needs a KIND check.
#: Provenance: `…/incoming/2026-08-16-sam3-concept-reliability/`.
CONF_THRESHOLD_DEFAULT = 0.25


def _arr(x):
    import numpy as np
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return np.asarray(x)


def detect(processor, image, prompt: str, *, min_score: float = 0.0,
           **kw) -> list[dict]:
    """One image, one concept -> one record per detection, in FRAME pixels.

    ⚠️ `min_score` defaults to 0.0 — nothing is filtered by default, because a
    threshold chosen before we have seen the score distribution is a decision
    dressed as a default. Filter downstream, on banked numbers.

    ⚠️ This encodes the image. For SEVERAL concepts on the same frame use
    `detect_many`, which encodes once — see its docstring for the 4.4×."""
    return _score(processor, processor.set_image(image), prompt,
                  min_score=min_score, **kw)


def detect_many(processor, image, prompts, *, min_score: float = 0.0,
                **kw) -> list[dict]:
    """Several concepts on ONE image — the image encoded ONCE.

    THE DEFECT. `detect` calls `processor.set_image(image)` on every call, and
    `run_clip_frames` called it once per CONCEPT — so a 7-concept vocabulary
    ran the ViT trunk **7 times on the identical frame**: 44 encodes per clip
    where 7 were needed.

    ⭐ MEASURED EQUIVALENT AND 4.21× (T4, 2026-08-16, clip `0089a096`, all six
    run frames, one session): per-concept encode **89.3 s**, encode-once
    **21.2 s**, and the two agree on **every** per-concept count AND every
    per-frame count. A repeat of the per-concept path in the same session
    reproduces itself exactly. Artifacts:
    `…/2026-08-16-sam3-dtype-fix/raw/{encode_once_equivalence,eq3_whole_clip}.json`.

    ⛔ **AND THE COMPARISON THAT SAID OTHERWISE WAS CONFOUNDED — read this
    before "fixing" a difference you see here (RETRACTION_LOG C79).** Diffing a
    re-run against the record banked by an earlier VM showed 60 vs 64
    detections, which reads as *"the optimisation changed the science"*. It did
    not, and it was not machine nondeterminism either: the fixed pipeline on a
    second VM reproduced the first VM's record **exactly**. The real variable
    was **the video file** — the pipeline re-bridges each clip from its w120
    shard, while the experiment pulled the pre-bridged
    `bridged_w120train_2400/videos/<cid>.mp4`. ⇒ **~7 % of detections sit close
    enough to `Sam3Processor(confidence_threshold=0.5)` to flip on re-encode
    noise alone** (`pedestrian` 4→7, `traffic light` 0→2). Compare counts only
    across identical input BYTES — and draw overlays on the bytes the model saw.

    ⚠️ Errors are recorded PER CONCEPT, and a failure of the shared encode is
    recorded once per concept too — so the C77 error census stays complete
    whichever half broke."""
    try:
        state = processor.set_image(image)
    except Exception as e:                      # the encode itself failed
        err = f"{type(e).__name__}: {e}"[:140]
        return [{"concept": p, "error": err} for p in prompts]
    dets = []
    for p in prompts:
        try:
            dets.extend(_score(processor, state, p, min_score=min_score, **kw))
        except Exception as e:
            dets.append({"concept": p,
                         "error": f"{type(e).__name__}: {e}"[:140]})
    return dets


def _score(processor, state, prompt: str, *, min_score: float = 0.0,
           contours: bool = True,
           contour_tol_px: float = CONTOUR_TOL_PX_DEFAULT,
           contour_max_pts: int = CONTOUR_MAX_PTS_DEFAULT) -> list[dict]:
    """Score ONE concept against an already-encoded image state.

    ⚠️ `contours` is ADDITIVE (schema v2): `rle_rows` and `mask_area_px` are
    written exactly as before and a v1 consumer sees no change. Turning it off
    costs nothing but the polygon — it is a flag so a size-constrained run has
    a lever, and so the CPU cost is separable in the pilot measurement."""
    out = processor.set_text_prompt(state=state, prompt=prompt)
    scores, boxes = _arr(out.get("scores")), _arr(out.get("boxes"))
    masks = _arr(out.get("masks"))
    n = 0 if scores is None else int(scores.reshape(-1).shape[0])
    recs = []
    for i in range(n):
        sc = float(scores.reshape(-1)[i])
        if sc < min_score:
            continue
        r = {"concept": prompt, "score": round(sc, 4)}
        if boxes is not None and i < len(boxes):
            r["box_xyxy"] = [round(float(v), 1) for v in boxes[i]]
        if masks is not None and i < len(masks):
            # ⛔ (1, H, W) IS THE REAL SHAPE THE VENDOR RETURNS — squeeze BEFORE
            # anything reads it as a picture. See `as_2d_mask`.
            m = as_2d_mask(masks[i])
            m = m > 0.0 if m.dtype != bool else m
            r["mask_area_px"] = int(m.sum())
            r["mask_hw"] = [int(m.shape[0]), int(m.shape[1])]
            r["rle_rows"] = _rows_rle(m)
            if contours:
                c = contour_of_mask(m, tol_px=contour_tol_px,
                                    max_pts=contour_max_pts)
                if c:
                    r.update(c)
                    obb = oriented_extent(c["contour_xy"])
                    if obb:
                        r["obb_cxcylwa"] = obb
        recs.append(r)
    return recs


def xyxy_to_xywh(b) -> list[float]:
    """⛔ `add_prompt` takes `bounding_boxes` straight through to `boxes_xywh`.
    Our B3 groundings are xyxy. Converting here rather than at the call site so
    the unit test can pin it — an unconverted box still segments SOMETHING, so
    this bug would never announce itself."""
    x0, y0, x1, y1 = [float(v) for v in b]
    return [x0, y0, x1 - x0, y1 - y0]


def read_outputs(out) -> list[dict]:
    """One propagate step -> one record per DETECTION.

    ⛔ MEASURED 2026-08-12 by dumping the live structure, after a first version
    of this reader guessed `pred_masks` / `masks` and silently returned nothing:
    the real payload is

        {"out_obj_ids":  ndarray [N],
         "out_probs":    ndarray [N],
         "out_boxes_xywh": ndarray [N, 4],
         "out_binary_masks": ndarray [N, H, W],
         "frame_stats":  dict}

    ⚠️ **N is the number of detections and it can legitimately be 0** — an empty
    array is "SAM3 saw nothing", NOT a broken reader. The first run reported
    `0 frames` on every clip and that was ambiguous between the two; recording
    `n_det` explicitly per frame is what makes the difference readable from the
    artifact instead of re-derivable only by another GPU run.

    ⭐ This is richer than a bare mask: SAM3 hands back a BOX, a PROBABILITY and
    a STABLE obj_id per detection, which is exactly the agent-slot payload the
    tactical vocabulary needs (`GAP_TARGET(agent_slot)`, `EVADE(obstacle_slot)`,
    `WAIT_FOR_ONCOMING(oncoming_slot)`, `TRAFFIC_LIGHT_REACT(light_slot)`)."""
    import numpy as np
    if not isinstance(out, dict) or "out_obj_ids" not in out:
        return []

    def _np(k):
        v = out.get(k)
        if v is None:
            return None
        if hasattr(v, "detach"):
            v = v.detach().cpu()
        return np.asarray(v)

    ids, probs = _np("out_obj_ids"), _np("out_probs")
    boxes, masks = _np("out_boxes_xywh"), _np("out_binary_masks")
    n = 0 if ids is None else int(ids.shape[0])
    recs = []
    for i in range(n):
        r = {"obj_id": int(ids[i])}
        if probs is not None and i < len(probs):
            r["prob"] = round(float(probs[i]), 4)
        if boxes is not None and i < len(boxes):
            x, y, w, h = [float(v) for v in boxes[i]]
            r["box_xywh"] = [round(x, 1), round(y, 1), round(w, 1), round(h, 1)]
            r["box_xyxy"] = [round(x, 1), round(y, 1),
                             round(x + w, 1), round(y + h, 1)]
        if masks is not None and i < len(masks):
            m = as_2d_mask(masks[i])          # (1, H, W) is the real shape
            m = m > 0.0 if m.dtype != bool else m
            r["mask_area_px"] = int(m.sum())
            r["mask_hw"] = [int(m.shape[0]), int(m.shape[1])]
            r["rle_rows"] = _rows_rle(m)
        recs.append(r)
    return recs


def as_2d_mask(mask):
    """Squeeze a detection mask to (H, W). ⛔ THIS IS NOT DEFENSIVE PADDING —
    it is the fix for a MEASURED corpus-wide defect.

    MEASURED 2026-08-16 on a live T4: `Sam3Processor.set_text_prompt` returns
    `masks` of shape **[N, 1, H, W]**, not [N, H, W]. Every consumer here
    indexed `masks[i]` and treated the result as 2-D, and only ONE of them
    noticed:

      * `mask_area_px` — `m.sum()` is shape-agnostic, so it was RIGHT;
      * `_rows_rle` — `enumerate()` over a (1, H, W) array yields ONE item
        whose `row` is the whole (H, W) plane, and `np.flatnonzero` then
        returns **FLATTENED** indices. The banked runs are therefore all
        `[0, flat_start, flat_end)`: MEASURED on clip `0089a096`, a car at box
        [54.4, 82.1, 66.0, 94.1] banked `[[0, 36794, 36800], …]` on a 448-wide
        frame. The lengths still sum to 120 = `mask_area_px`, so every
        consistency check passed. ⇒ **the v1 corpus's `rle_rows` cannot redraw
        its own mask**, and `ph0_rich_overlay.draw_masks`, which does
        `over[r, a:b]`, painted row 0 with an out-of-range slice — numpy clips
        it to nothing, so the banked overlay videos show boxes and NO fill;
      * `crack_loops` — refused the 3-D input and returned no contour, which is
        how the defect surfaced at all.

    ⚠️ THE DATA IS RECOVERABLE, THE SCHEMA WAS THE LIE: a v1 run decodes as
    `row = start // W`, `col = start % W` (runs never straddle a row boundary
    because a mask row's run ends at the row's end). It does NOT need re-detecting.

    ⭐ The lesson, which is why this is a named function and not an inline
    `[0]`: the STRICT reader found the sloppy one's bug. `_rows_rle` accepted
    any shape and produced plausible garbage; `crack_loops` asserted `ndim == 2`
    and produced nothing. An assertion that fires is worth more than a
    serialiser that never complains."""
    import numpy as np
    m = np.asarray(mask)
    while m.ndim > 2 and m.shape[0] == 1:
        m = m[0]
    if m.ndim != 2:
        raise ValueError(f"mask is not 2-D after squeeze: shape {m.shape}")
    return m


def _rows_rle(mask) -> list[list[int]]:
    """Compact per-row [row, start, end) runs — small enough to bank in JSON and
    enough to redraw the mask exactly.

    ⛔ `as_2d_mask` FIRST. Without it this function silently emits flattened
    indices on a (1, H, W) input — see that docstring; it is the defect that
    made the whole v1 corpus's RLE undrawable."""
    import numpy as np
    runs = []
    for r, row in enumerate(as_2d_mask(mask)):
        idx = np.flatnonzero(row)
        if idx.size == 0:
            continue
        splits = np.flatnonzero(np.diff(idx) > 1)
        starts = np.r_[idx[0], idx[splits + 1]]
        ends = np.r_[idx[splits], idx[-1]] + 1
        for s, e in zip(starts, ends):
            runs.append([r, int(s), int(e)])
    return runs


# =========================================================================== #
# CONTOURS — schema v2 (PI, 2026-08-16)                                       #
# =========================================================================== #
def crack_loops(mask) -> list[list[tuple[int, int]]]:
    """Every closed boundary loop of `mask`, on the PIXEL-CORNER lattice.

    ⛔ WHY THE CORNER LATTICE AND NOT PIXEL CENTRES — this is the whole reason
    this function exists instead of a four-line neighbour walk. A polygon
    through boundary-pixel CENTRES systematically UNDER-COUNTS area by about
    half the perimeter: a 3×3 blob has 9 mask pixels and a centre-polygon area
    of **4** (−55 %). Our objects are small — the MEASURED median `car` box is
    188 px² and `traffic light` 34 px²
    (`…/2026-08-16-sam3-concept-reliability/` §2) — so that bias would be the
    dominant term in every oriented extent we derived, and it would look like a
    tight fit rather than a defect. Traced on the corner lattice the polygon's
    enclosed area equals the pixel count **exactly**, so the ONLY area error is
    the one the RDP tolerance buys, which is the number the brief asks for and
    the number `contour_of_mask` reports.

    Vectorised: the boundary is where a pixel and its neighbour disagree, which
    numpy answers for the whole mask at once; only the chaining is a Python
    loop, and it runs over the PERIMETER, not the area.

    Orientation is CLOCKWISE in image coordinates (y down), so an outer
    boundary has POSITIVE shoelace and a hole has negative — which is how
    `contour_of_mask` tells the two apart without a separate hole test."""
    import numpy as np
    from collections import defaultdict
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or not m.any():
        return []
    h, w = m.shape
    # horizontal cracks: lattice row Y in [0..h], between pixel rows Y-1 and Y
    up = np.zeros((h + 1, w), bool)
    up[1:] = m
    dn = np.zeros((h + 1, w), bool)
    dn[:h] = m
    # vertical cracks: lattice col X in [0..w], between pixel cols X-1 and X
    lf = np.zeros((h, w + 1), bool)
    lf[:, 1:] = m
    rt = np.zeros((h, w + 1), bool)
    rt[:, :w] = m

    succ: dict = defaultdict(list)
    ys, xs = np.nonzero(dn & ~up)              # top edge of a fg pixel
    for y, x in zip(ys.tolist(), xs.tolist()):
        succ[(x, y)].append((x + 1, y))
    ys, xs = np.nonzero(up & ~dn)              # bottom edge
    for y, x in zip(ys.tolist(), xs.tolist()):
        succ[(x + 1, y)].append((x, y))
    ys, xs = np.nonzero(rt & ~lf)              # left edge
    for y, x in zip(ys.tolist(), xs.tolist()):
        succ[(x, y + 1)].append((x, y))
    ys, xs = np.nonzero(lf & ~rt)              # right edge
    for y, x in zip(ys.tolist(), xs.tolist()):
        succ[(x, y)].append((x, y + 1))

    loops = []
    for s in list(succ.keys()):
        while succ.get(s):
            loop, cur = [s], s
            while True:
                nxts = succ.get(cur)
                if not nxts:
                    break                       # cannot happen on a valid mask
                nxt = nxts.pop()
                if not nxts:
                    succ.pop(cur, None)
                if nxt == s:
                    break
                loop.append(nxt)
                cur = nxt
            if len(loop) >= 4:
                loops.append(loop)
    return loops


def shoelace2(pts) -> float:
    """TWICE the signed area of a closed polygon (the doubling is kept so an
    integer lattice polygon has an integer answer and no float creeps in)."""
    s = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return s


def _rdp(pts, tol: float):
    """Ramer-Douglas-Peucker on an OPEN polyline. Iterative, so a 2 000-point
    road boundary cannot blow the recursion limit inside a GPU run."""
    n = len(pts)
    if n < 3:
        return list(pts)
    keep = [False] * n
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    t2 = tol * tol
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        x0, y0 = pts[i]
        x1, y1 = pts[j]
        dx, dy = x1 - x0, y1 - y0
        den = dx * dx + dy * dy
        best, bi = -1.0, -1
        for k in range(i + 1, j):
            px, py = pts[k]
            if den == 0:
                d2 = (px - x0) ** 2 + (py - y0) ** 2
            else:
                cr = dx * (py - y0) - dy * (px - x0)
                d2 = cr * cr / den
            if d2 > best:
                best, bi = d2, k
        if best > t2:
            keep[bi] = True
            stack.append((i, bi))
            stack.append((bi, j))
    return [pts[i] for i in range(n) if keep[i]]


def simplify_closed(loop, tol: float):
    """RDP for a CLOSED loop. Split at two far-apart anchors first — running
    RDP on a ring from an arbitrary seam collapses the seam's own detail and
    makes the result depend on where the trace happened to start."""
    n = len(loop)
    if n <= 4:
        return list(loop)
    x0, y0 = loop[0]
    i1 = max(range(n), key=lambda k: (loop[k][0] - x0) ** 2
             + (loop[k][1] - y0) ** 2)
    a = _rdp(loop[:i1 + 1], tol)
    b = _rdp(loop[i1:] + [loop[0]], tol)
    out = a[:-1] + b[:-1]
    return out if len(out) >= 3 else list(loop)


def contour_of_mask(mask, *, tol_px: float = CONTOUR_TOL_PX_DEFAULT,
                    max_pts: int = CONTOUR_MAX_PTS_DEFAULT) -> dict:
    """One instance mask -> a COMPACT polygon, banked ALONGSIDE the RLE.

    ⭐ WHY A CONTOUR AT ALL, given the mask is already banked. The agent-slot
    decoder's targets are `(cx, cy, yaw, l, w)` — an ORIENTED extent. An
    axis-aligned box cannot express `yaw` at all: a car at 30° and the same car
    at 0° can share a box. A contour can, and `oriented_extent` reads it
    straight off. That is the payload, and it is why the polygon is not a
    prettier redraw of the mask.

    ⛔ THE CONTOUR DOES NOT REPLACE THE RLE, EVER. It is LOSSY twice over: RDP
    moves the boundary by up to `tol_px`, and only the LARGEST outer loop is
    kept, so holes and detached fragments are dropped. The mask is the
    primitive; this is a derived summary of it. Both are banked, and
    `contour_area_px` vs `mask_area_px` is the audit that says how lossy this
    one was — per detection, without a re-run.

    Returns `{}` for an empty mask (a detection whose mask is empty is a
    result, not an error — same rule as a zero detection count).

    Keys: `contour_xy` FLAT `[x0,y0,x1,y1,…]` on the pixel-CORNER lattice
    (integers, so `x` ranges 0..W inclusive); `contour_tol_px` the tolerance
    ACTUALLY used, which is not `tol_px` when the cap bit; `contour_area_px`
    the polygon's own area; `contour_n_loops` how many loops the mask had, so a
    fragmented detection is visible rather than silently summarised."""
    loops = crack_loops(mask)
    if not loops:
        return {}
    outer = [(lp, shoelace2(lp)) for lp in loops]
    outer = [(lp, s) for lp, s in outer if s > 0]
    if not outer:
        return {}
    loop = max(outer, key=lambda t: t[1])[0]
    tol = float(tol_px)
    pts = simplify_closed(loop, tol)
    guard = 0
    while len(pts) > max_pts and guard < 24:
        # ⚠️ `tol *= 2` alone is a SILENT NO-OP at tol=0 — the loop spins 24
        # times, the cap never binds, and an exact-contour run would bank a
        # 2 000-point polygon while the record claimed `max_pts`. Caught by
        # `test_simplification_is_bounded_by_its_tolerance_and_reports_it`.
        tol = tol * 2.0 if tol > 0 else 0.5
        pts = simplify_closed(loop, tol)
        guard += 1
    return {"contour_xy": [int(v) for p in pts for v in p],
            "contour_tol_px": round(tol, 3),
            "contour_area_px": int(round(abs(shoelace2(pts)) / 2.0)),
            "contour_n_loops": len(loops)}


def _convex_hull(pts):
    """Andrew monotone chain. Returns the hull counter-clockwise in a y-up
    reading, which is all `oriented_extent` needs — it only rotates."""
    p = sorted(set(map(tuple, pts)))
    if len(p) <= 2:
        return p

    def half(seq):
        out = []
        for q in seq:
            while len(out) >= 2:
                (ax, ay), (bx, by) = out[-2], out[-1]
                if (bx - ax) * (q[1] - ay) - (by - ay) * (q[0] - ax) <= 0:
                    out.pop()
                else:
                    break
            out.append(q)
        return out
    return half(p)[:-1] + half(reversed(p))[:-1]


def oriented_extent(contour_xy) -> list[float] | None:
    """Minimum-area rectangle over a contour -> `[cx, cy, l, w, angle_deg]`.

    ⭐ THIS IS THE POINT OF THE CONTOUR. It is the agent-slot decoder's target
    tuple `(cx, cy, yaw, l, w)` in image space, and it is exactly what an
    axis-aligned box cannot give: `box_xyxy` has no angle, so a turning car and
    a straight one are indistinguishable in it.

    `l` is the LONGER side and `angle_deg` is that side's direction in
    [0, 180), measured from +x with y DOWN (image convention) — so a value near
    0 is a horizontal extent and near 90 a vertical one. Reported as one flat
    list rather than five keys because it is banked on every detection and the
    key names would outweigh the numbers.

    ⚠️ IMAGE-SPACE, NOT METRIC, AND NOT A YAW. This is the pixel-space
    orientation of the mask. Turning it into a vehicle heading needs the
    ground-plane homography, and NOTHING here does that — a consumer that reads
    `angle_deg` as a yaw is reading a different quantity. Rotating calipers by
    brute force over hull edges: the hull is ≤ `max_pts` points, so O(n²) is
    microseconds and there is no calipers-invariant to get subtly wrong."""
    import math
    pts = [(contour_xy[i], contour_xy[i + 1])
           for i in range(0, len(contour_xy) - 1, 2)]
    hull = _convex_hull(pts)
    if len(hull) < 2:
        return None
    best = None
    n = len(hull)
    for i in range(n):
        x0, y0 = hull[i]
        x1, y1 = hull[(i + 1) % n]
        ex, ey = x1 - x0, y1 - y0
        ln = math.hypot(ex, ey)
        if ln < 1e-9:
            continue
        ux, uy = ex / ln, ey / ln
        us = [p[0] * ux + p[1] * uy for p in hull]
        vs = [-p[0] * uy + p[1] * ux for p in hull]
        du, dv = max(us) - min(us), max(vs) - min(vs)
        area = du * dv
        if best is None or area < best[0]:
            cu, cv = (max(us) + min(us)) / 2.0, (max(vs) + min(vs)) / 2.0
            cx = cu * ux - cv * uy
            cy = cu * uy + cv * ux
            best = (area, cx, cy, du, dv, math.degrees(math.atan2(uy, ux)))
    if best is None:
        return None
    _a, cx, cy, du, dv, deg = best
    if dv > du:                                  # the long side defines the angle
        du, dv, deg = dv, du, deg + 90.0
    deg %= 180.0
    return [round(cx, 1), round(cy, 1), round(du, 1), round(dv, 1),
            round(deg, 1)]


def box_mask_agreement(mask, box_xyxy) -> dict:
    """The cross-engine grounding signal: how much of the mask sits inside the
    VLM's box, and how much of the box the mask covers.

    ⚠️ Reported as TWO numbers, never averaged into one. A mask that spills far
    outside its prompt means SAM3 latched onto a larger structure than the VLM
    meant (low frac_mask_in_box, high frac_box_covered); a mask that fills a
    corner means the VLM's box was too generous (the reverse). One score would
    hide which of those happened, and they call for opposite fixes."""
    import numpy as np
    m = np.asarray(mask, dtype=bool)
    x0, y0, x1, y1 = [int(round(float(v))) for v in box_xyxy]
    bx = np.zeros_like(m, dtype=bool)
    bx[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
    area, barea = int(m.sum()), int(bx.sum())
    inter = int((m & bx).sum())
    return {"mask_area_px": area, "box_area_px": barea,
            "frac_mask_in_box": round(inter / area, 4) if area else 0.0,
            "frac_box_covered": round(inter / barea, 4) if barea else 0.0}


def is_live(liveness: dict | None) -> bool:
    """⛔ THE ONE DERIVATION — and it is NOT STORED, on purpose.

    A record banks the CONTROL COUNTS (`liveness.n_det`) and nothing else. The
    verdict is computed here, by every consumer, at read time.

    WHY THE BOOLEAN WAS DELETED FROM THE SCHEMA (2026-08-16). It existed for
    half a corpus and its rule changed mid-corpus — `all(...)` → `any(...)`,
    after clip `24b6948f` returned `road 2 · sky 0` under an underpass and was
    flagged dead while carrying 22 real detections. MEASURED consequence: that
    record sat on disk with `live: False` **contradicting its own `n_det`**, so
    any consumer reading the flag — the `aug120_pipeline` batch gate, the
    overlay's liveness row, a future re-fuse, a human six months out — would
    classify a healthy clip as the one failure that blocks a PASS.

    ⇒ **The counts are the primitive; the verdict is a cache; a cache of a rule
    that changed is a trap with a long fuse.** A field that cannot be stale
    beats a field that must be kept in sync, so the field is gone and this
    function is the only place the rule lives.

    ⚠️ Reads defensively: pre-2026-08-16 records may still carry `live` /
    `all_fired`. They are IGNORED — `n_det` is the authority."""
    nd = (liveness or {}).get("n_det") or {}
    return any(int(v) > 0 for v in nd.values())


# =========================================================================== #
# EGO LANE — a DERIVED region, never a prompt                                 #
# =========================================================================== #
#: How far up the image still counts as "near field" for the lane derivation.
#: The bottom quarter: far enough down that perspective has not yet collapsed
#: the lanes together, near enough that it is the ego's own surroundings.
LANE_NEAR_FRAC = 0.75


def _footpoint(det) -> tuple[float, float] | None:
    """Where a detection meets the road: the centre of its LOWEST mask row.

    Not the box centre — a lane marking is a long thin diagonal and its box
    centre floats in the middle of the road, metres from where the paint
    actually is at the ego's own distance."""
    rows = det.get("rle_rows")
    if rows:
        ymax = max(r[0] for r in rows)
        xs = [(a + b) / 2.0 for r, a, b in rows if r == ymax]
        if xs:
            return (sum(xs) / len(xs), float(ymax))
    b = det.get("box_xyxy")
    if b:
        return ((float(b[0]) + float(b[2])) / 2.0, float(b[3]))
    return None


def derive_ego_lane(scene_dets, frame_wh, *, ego_u: float | None = None,
                    boundary_concepts=("lane marking", "road curb"),
                    near_frac: float = LANE_NEAR_FRAC,
                    merge_frac: float = 0.03) -> dict:
    """⛔ "EGO LANE" IS NOT A VISUAL CLASS AND IS NOT PROMPTED FOR. It is
    DERIVED here, geometrically, from things that ARE visible.

    A guardrail, a curb and a painted marking are appearances — SAM3 can be
    asked for them and the answer is checkable against the picture. *"the lane
    the ego is in"* is not an appearance: it is a RELATION between the ego and
    the lane boundaries, and the pixels of the ego's lane look exactly like the
    pixels of the next one. Prompting for it would produce a plausible mask
    that no measurement could falsify — a derived quantity smuggled in as a
    perception, which is the failure the goal-input rule was written for. So it
    is computed, from `lane marking` + `road curb`, and it is labelled DERIVED
    in the record it lands in.

    ⭐ WHAT IT IS FOR — and read the primary source before quoting this, because
    the obvious summary of it is WRONG. `lane_change_requirement()` needs FOUR
    named inputs (`s2_derive.LANE_CONTEXT_INPUTS`), and their status is
    (`…/2026-08-16-s2-v1-labels/review/LANE_CHANGE_DEEP_REVIEW.md` §3):

      n_lanes_same_direction  ⚠️ VLM `lanes_visible` — unreliable
      ego_lane_idx            ⚠️ VLM `lane_ego`      — same instrument, same doubt
      route_lane_idx          ⛔ DOES NOT EXIST — needs a map / lane graph
      lane_continues          ⛔ DOES NOT EXIST

    ⇒ **THIS FUNCTION SUPPLIES THE FIRST TWO, NOT THE LAST TWO.** It is exactly
    the review's own option 2 — *"a vision lane-boundary estimator (lane-marking
    segmentation → lane index + count from the front-wide camera) … the
    vision-only-at-inference path"* — so it replaces an unreliable VLM
    instrument with a vision-only one that also satisfies the vision-only rule.
    ⛔ **It does NOT unblock `PREPARE_LANE_CHANGE`**: `route_lane_idx` and
    `lane_continues` need lane TOPOLOGY, which no camera frame contains and
    which PhysicalAI-AV does not ship. Claiming otherwise would be the
    programme's favourite error — an input renamed into a capability.

    METHOD (and every step of it is a stated assumption, not a fact):
      1. keep boundary detections whose FOOTPOINT is in the near field — the
         bottom `near_frac` of the frame;
      2. cluster the footpoints in x, because SAM3 returns a dashed line as one
         detection PER DASH and three dashes of one lane line are one boundary,
         not three;
      3. the ego sits at `ego_u` (default: image centre column);
      4. `lane_idx_est` = (boundaries RIGHT of the ego) − 1. ⚠️ **From the
         RIGHT, because that is `ego_lane_idx`'s definition** — *"ego's 0-based
         lane index from the right"*. Inventing a second convention for the same
         quantity is how two correct numbers become one wrong one.

    ⚠️ FOUR WAYS THIS IS WRONG, stated because a derived number with no caveat
    is how a heuristic becomes a fact:
      * `ego_u` = image centre assumes the camera is on the vehicle centreline.
        MEASURED false on PhysicalAI: the AV front-wide corpus has TWO rigs
        with principal points ~212 px apart in y, and nothing here has checked
        x. Pass `ego_u` from the calibration when it is available.
      * a boundary the camera cannot see does not exist to this function, so
        `lane_idx_est` counts from the LEFTMOST VISIBLE boundary, not from the
        road edge. It is a lower bound on the true index.
      * a curb and the painted line beside it are two boundaries here and one
        edge in the world; `merge_frac` mitigates and does not solve it.
      * it is a SINGLE-FRAME estimate with no temporal smoothing.
      * ⛔ `n_lanes_est` counts the gaps between ALL perceived boundaries, which
        on an undivided road INCLUDES THE ONCOMING CARRIAGEWAY. It is therefore
        an upper bound on `n_lanes_same_direction`, not that quantity, and it is
        named `_est` for that reason.
    ⇒ Evidence class DERIVED-ESTIMATED. It is an input to a label, never a label.
    """
    w, h = int(frame_wh[0]), int(frame_wh[1])
    u0 = float(w) / 2.0 if ego_u is None else float(ego_u)
    want = set(boundary_concepts)
    y_min = h * float(near_frac)
    fps = []
    for d in scene_dets:
        if d.get("concept") not in want or "error" in d:
            continue
        fp = _footpoint(d)
        if fp is None or fp[1] < y_min:
            continue
        fps.append((fp[0], d.get("concept"), float(d.get("score") or 0.0)))
    out = {"class": "DERIVED-ESTIMATED", "method": "near-field footpoint "
           "clustering of lane markings + curbs; ego at image centre column",
           "derived_from": sorted(want), "ego_u": round(u0, 1),
           "near_frac": near_frac, "n_boundary_det": len(fps),
           "supplies": ["ego_lane_idx", "n_lanes_same_direction"],
           "does_not_supply": ["route_lane_idx", "lane_continues"],
           "index_convention": "lane_idx_est is 0-based FROM THE RIGHT, "
                               "matching s2_derive LANE_CONTEXT_INPUTS"}
    if not fps:
        out.update({"boundaries": [], "n_left": 0, "n_right": 0,
                    "lane_idx_est": None, "lane_width_px": None,
                    "n_lanes_est": None,
                    "reason": "no boundary detection in the near field"})
        return out
    merge = max(3.0, merge_frac * w)
    fps.sort()
    clusters: list[list] = [[fps[0]]]
    for f in fps[1:]:
        if f[0] - clusters[-1][-1][0] <= merge:
            clusters[-1].append(f)
        else:
            clusters.append([f])
    bnds = [{"x": round(sum(c[0] for c in cl) / len(cl), 1),
             "n": len(cl),
             "concepts": sorted({c[1] for c in cl}),
             "score_max": round(max(c[2] for c in cl), 4)} for cl in clusters]
    left = [b for b in bnds if b["x"] < u0]
    right = [b for b in bnds if b["x"] >= u0]
    out.update({"boundaries": bnds, "n_left": len(left), "n_right": len(right),
                "left": left[-1] if left else None,
                "right": right[0] if right else None,
                # ⚠️ UPPER BOUND on n_lanes_same_direction — see the docstring;
                # on an undivided road the oncoming carriageway is in here too.
                "n_lanes_est": max(0, len(bnds) - 1) or None})
    if left and right:
        out["lane_idx_est"] = len(right) - 1        # 0-based FROM THE RIGHT
        out["lane_width_px"] = round(right[0]["x"] - left[-1]["x"], 1)
    else:
        out.update({"lane_idx_est": None, "lane_width_px": None,
                    "reason": "ego lane not bounded on both sides "
                              f"(left {len(left)}, right {len(right)})"})
    return out


def liveness_probe(processor, image, *, concepts=None,
                   min_score: float = 0.0, keep_det: bool = False,
                   **kw) -> dict:
    """⭐ THE POSITIVE CONTROL FOR C77 — is this engine PRODUCING anything?

    Runs `LIVENESS_CONCEPTS` on ONE frame and reports their counts.
    The record holds only `n_det` (and any per-concept `errors`); the verdict
    is `is_live()`, computed at read time and never stored. Not live is an
    ALARM: the engine returned nothing at all on
    concepts that a forward-facing driving frame is full of — the C77 dtype
    crash reads exactly like this. Contrast with `AGENT_CONCEPTS`, every one of
    which may correctly be zero, which is what made 115 empty clips look
    plausible.

    ⚠️ Errors are recorded per concept, exactly like `detect`'s callers do, so
    a crash census is readable from the artifact without a re-run.

    ⭐ `keep_det` (schema v2) also banks the control's OWN detections — box,
    mask, contour. This is where the PI's `road` class is delivered: `road` is
    barred from `SCENE_CONCEPTS` because a control drawn from the measured
    vocabulary is circular (see LIVENESS_CONCEPTS), but nothing stops the
    control from banking the geometry it already computed. The counts remain
    the primitive and stay out of `per_concept_hits` / `per_scene_hits`; the
    detections ride along in the control's own block."""
    cs = list(concepts or LIVENESS_CONCEPTS)
    out = {c: 0 for c in cs}
    err = {}
    dets = detect_many(processor, image, cs, min_score=min_score, **kw)
    for d in dets:
        if "error" in d:
            err[d["concept"]] = d["error"]
        else:
            out[d["concept"]] = out.get(d["concept"], 0) + 1
    # ⛔ NO `live` BOOLEAN IS STORED. See `is_live` — the counts are the
    # primitive, the verdict is derived at read time, and a derived field that
    # is written down is a cache of a rule that has already changed once
    # mid-corpus.
    rec = {"concepts": cs, "n_det": out}
    if err:
        rec["errors"] = err
    if keep_det:
        rec["det"] = [d for d in dets if "error" not in d]
    return rec


def run_clip_frames(processor, frames, concepts, vlm_boxes, *,
                    frame_stride: int = 8, min_score: float = 0.0,
                    liveness: bool = True, scene_concepts=None,
                    scene_min_score: float | None = None,
                    contours: bool = True,
                    contour_tol_px: float = CONTOUR_TOL_PX_DEFAULT,
                    contour_max_pts: int = CONTOUR_MAX_PTS_DEFAULT,
                    lane: bool = True, ego_u: float | None = None) -> dict:
    """One clip -> SAM3's own detections on a strided set of frames, plus the
    cross-engine check against the VLM's B3 sign boxes.

    ⭐ THE CROSS-ENGINE CHECK IS THE POINT. SAM3 finds signs INDEPENDENTLY of
    Qwen. Where both fire, `box_mask_agreement` says whether they mean the same
    object; where only one fires, that disagreement is the finding. A VLM box
    alone is an unverifiable claim, and averaging the two would hide exactly the
    cases worth looking at.

    ⭐ SCHEMA v2 — the scene channel, and the ONE encode it must not cost.
    `scene_concepts` are scored in the SAME `detect_many` call as the agent
    concepts, so the ViT trunk still runs exactly ONCE per frame. Splitting
    them into a second call would have re-created the defect the encode-once
    fix removed (44 encodes where 7 were needed, 4.21× wall-clock) in a new
    place, and it would have looked like the cost of the new classes rather
    than the cost of the mistake. The scene results are separated AFTERWARDS,
    on the returned records, which is free.

    The agent contract is untouched: `frames[*].det`, `per_concept_hits` and
    `n_det_total` still count `concepts` and nothing else. Scene detections
    land in `frames[*].scene` / `per_scene_hits` / `n_scene_det_total`.

    ⚠️ ONE v1 KEY DOES CHANGE MEANING AND IT IS DELIBERATE: `n_err_total` and
    `err_kinds` now count errors from BOTH channels. An error census that
    silently omitted a channel would be C77's own defect rebuilt — the whole
    reason those keys exist is that a failure must be countable next to the
    detections. `n_err_agent` / `n_err_scene` split it back out for anyone who
    needs the v1 quantity."""
    from PIL import Image
    scene = [c for c in (scene_concepts or []) if c not in set(concepts)]
    concept_set, scene_set = set(concepts), set(scene)
    s_floor = min_score if scene_min_score is None else scene_min_score
    fh, fw = int(frames[0].shape[0]), int(frames[0].shape[1])
    ckw = dict(contours=contours, contour_tol_px=contour_tol_px,
               contour_max_pts=contour_max_pts)
    out_frames: dict[str, dict] = {}
    per_concept: dict[str, int] = {c: 0 for c in concepts}
    per_scene: dict[str, int] = {c: 0 for c in scene}
    lane_frames: dict[str, dict] = {}
    # ⛔ THE FIX. Running only strided frames and then snapping each VLM box to
    # the nearest strided frame compared engine B and engine C on frames up to
    # ~3.5 s of driving apart, and the resulting 0/8 "agreement" was a property
    # of the SNAPPING, not of the VLM's boxes. Every frame the VLM actually
    # grounded on is now run EXACTLY, so the cross-check is frame-identical.
    todo = sorted(set(range(0, len(frames), max(1, frame_stride)))
                  | {int(v.get("frame_idx", 0)) for v in vlm_boxes
                     if 0 <= int(v.get("frame_idx", 0)) < len(frames)})
    # ⭐ ONE liveness probe per clip, on the middle frame of the set actually
    # run. One frame (not all) keeps the cost at ~2 extra concept passes per
    # clip; the middle frame rather than the first because clip starts are the
    # most likely to be atypical.
    live = None
    if liveness and todo:
        live = liveness_probe(processor,
                              Image.fromarray(frames[todo[len(todo) // 2]]),
                              min_score=min_score, keep_det=True, **ckw)
        live["frame_idx"] = int(todo[len(todo) // 2])
    for fi in todo:
        img = Image.fromarray(frames[fi])
        # ⛔ ONE ENCODE PER FRAME, not one per concept — see detect_many. The
        # per-concept loop that used to live here re-ran the ViT trunk 7x on
        # the identical frame (MEASURED banked wall_s 97-98 s per 6-frame clip).
        # ⭐ AND the scene concepts ride the SAME encode; this list is why.
        raw = detect_many(processor, img, list(concepts) + scene,
                          min_score=min_score, **ckw)
        dets, scene_d = [], []
        for d in raw:
            c = d.get("concept")
            if c in scene_set:
                if "error" not in d and float(d.get("score") or 0.0) < s_floor:
                    continue
                scene_d.append(d)
                if "score" in d:
                    per_scene[c] = per_scene.get(c, 0) + 1
            else:
                dets.append(d)
                if "score" in d:
                    per_concept[c] = per_concept.get(c, 0) + 1
        # ⚠️ EVERY strided frame is recorded, including the empty ones: a frame
        # dropped for having no detections is indistinguishable downstream from
        # a frame never run, and those are different failures.
        fr = {"n_det": sum(1 for d in dets if "score" in d), "det": dets}
        if scene:
            fr["n_scene"] = sum(1 for d in scene_d if "score" in d)
            fr["scene"] = scene_d
        out_frames[str(fi)] = fr
        if lane and scene:
            lane_frames[str(fi)] = derive_ego_lane(scene_d, (fw, fh),
                                                   ego_u=ego_u)
    agree = []
    for vb in vlm_boxes:
        fi = str(int(vb.get("frame_idx", 0)))       # EXACT frame, never snapped
        best = None
        for d in out_frames.get(fi, {}).get("det", []):
            if d.get("concept") != "traffic sign" or not d.get("box_xyxy"):
                continue
            iou = _box_iou(vb["box_xyxy"], d["box_xyxy"])
            if best is None or iou > best["iou"]:
                best = {"iou": round(iou, 4), "sam3_box": d["box_xyxy"],
                        "sam3_score": d.get("score")}
        agree.append({"vlm_box": vb["box_xyxy"], "vlm_label": vb.get("label"),
                      "frame_idx": vb.get("frame_idx"),
                      "sam3_frame_idx": int(fi),
                      "frame_aligned": True,
                      "n_sam3_signs_on_frame": sum(
                          1 for d in out_frames.get(fi, {}).get("det", [])
                          if d.get("concept") == "traffic sign"),
                      "best_sam3_sign": best,
                      "matched": bool(best and best["iou"] > 0.0)})
    # ⛔ THE ERROR CENSUS IS PART OF THE RECORD, NOT A LOG LINE. C77's 115
    # clips carried their own cause per concept per frame and nobody counted
    # it, because the summary keys only described CONTAINERS. `n_err_total`
    # and `err_kinds` put the failure count next to the detection count, where
    # a completeness check cannot miss it.
    errs: dict[str, int] = {}
    n_err_a = n_err_s = 0
    for f in out_frames.values():
        for key in ("det", "scene"):
            for d in f.get(key) or []:
                if "error" not in d:
                    continue
                if key == "det":
                    n_err_a += 1
                else:
                    n_err_s += 1
                k = str(d["error"]).split(":")[0]
                errs[k] = errs.get(k, 0) + 1
    out = {"frames": out_frames, "per_concept_hits": per_concept,
           "n_frames_run": len(out_frames),
           "n_det_total": sum(f["n_det"] for f in out_frames.values()),
           "n_err_total": n_err_a + n_err_s, "err_kinds": errs,
           "liveness": live,
           "vlm_cross_check": agree}
    if scene:
        out.update({
            "schema_version": SCHEMA_VERSION,
            "concepts_agent": list(concepts), "concepts_scene": scene,
            "concept_kinds": {c: CONCEPT_KIND.get(c, "thing")
                              for c in list(concepts) + scene
                              + LIVENESS_CONCEPTS},
            "per_scene_hits": per_scene,
            "n_scene_det_total": sum(f.get("n_scene", 0)
                                     for f in out_frames.values()),
            "n_err_agent": n_err_a, "n_err_scene": n_err_s})
        if lane:
            out["ego_lane"] = {"frames": lane_frames,
                               "note": "DERIVED, never prompted — see "
                                       "ph0_sam3.derive_ego_lane"}
    if contours:
        out["contour"] = {"tol_px": contour_tol_px, "max_pts": contour_max_pts,
                          "lattice": "pixel-corner (x in 0..W, y in 0..H)",
                          "format": "contour_xy is FLAT [x0,y0,x1,y1,...]",
                          "obb": "obb_cxcylwa = [cx, cy, long, short, deg] in "
                                 "IMAGE space; deg in [0,180) from +x, y down "
                                 "— NOT a vehicle yaw"}
    return out


def _box_iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    ub = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    den = ua + ub - inter
    return inter / den if den > 0 else 0.0


def draw_masks(img, segs: list[dict], labels: list[str] | None = None):
    """Translucent mask fill + CONTOUR + outline + label chip, one colour per
    instance.

    ⚠️ A v1 record draws NOTHING here and that is not this function's bug — see
    `as_2d_mask` (C85): v1's `rle_rows` are flattened, so `over[r, a:b]` gets
    `a` ≈ 36 794 on a 448-wide array and numpy CLIPS the out-of-range slice to
    nothing. The result looked like a sparse scene. The count of runs that fell
    outside the frame is now returned to the caller's eye as a red frame rather
    than being silently dropped."""
    import numpy as np
    from PIL import Image, ImageDraw
    base = np.asarray(img.convert("RGB")).astype(np.float32)
    over = base.copy()
    n_off = 0
    for i, s in enumerate(segs):
        if not s.get("rle_rows"):
            continue
        col = np.array(MASK_COLOURS[i % len(MASK_COLOURS)], np.float32)
        for r, a, b in s["rle_rows"]:
            if not (0 <= r < over.shape[0]) or a >= over.shape[1]:
                n_off += 1                       # a flattened v1 run lands here
                continue
            over[r, a:b] = 0.55 * col + 0.45 * over[r, a:b]
    out = Image.fromarray(over.astype(np.uint8))
    d = ImageDraw.Draw(out)
    for i, s in enumerate(segs):
        col = MASK_COLOURS[i % len(MASK_COLOURS)]
        xy = s.get("contour_xy") or []
        if len(xy) >= 6:
            pts = [(xy[k], xy[k + 1]) for k in range(0, len(xy) - 1, 2)]
            d.line(pts + [pts[0]], fill=col, width=1)
        if s.get("box"):
            x0, y0, x1, y1 = [int(v) for v in s["box"]]
            d.rectangle([x0, y0, x1, y1], outline=col, width=2)
            y_lab = max(0, y0 - 12)
        else:
            y0 = x0 = 4 + 14 * i
            y_lab = y0
        lab = (labels[i] if labels and i < len(labels) else f"sam3[{i}]")
        if s.get("frac_mask_in_box") is not None:
            lab += f"  in-box {s['frac_mask_in_box']:.2f}"
        d.text((x0 + 3, y_lab), lab, fill=col)
    if n_off:
        # ⛔ LOUD, not silent. A run outside the frame means the RLE does not
        # describe THIS image — almost always a v1 flattened record (C85). The
        # old behaviour was to skip it and render a plausible, empty picture.
        d.rectangle([0, 0, out.width - 1, out.height - 1],
                    outline=(255, 0, 0), width=2)
        d.text((4, 4), f"{n_off} RLE runs OUTSIDE the frame — flattened "
                       "record? (C85)", fill=(255, 0, 0))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("ph0_sam3")
    ap.add_argument("--v2-json", required=True,
                    help="ph0_v2.json — supplies the VLM B3 sign boxes to cross-check")
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--concepts", default=",".join(AGENT_CONCEPTS))
    ap.add_argument("--scene-concepts", default=",".join(SCENE_CONCEPTS),
                    help="the STUFF vocabulary (lane marking, curb, "
                         "guardrail…). Kept out of per_concept_hits and "
                         "n_det_total on purpose — pass '' to disable.")
    ap.add_argument("--scene-min-score", type=float, default=None)
    ap.add_argument("--no-contours", action="store_true",
                    help="skip the polygon per detection (the RLE mask is "
                         "banked either way — the contour is the DERIVED "
                         "summary, never the primitive)")
    ap.add_argument("--contour-tol-px", type=float,
                    default=CONTOUR_TOL_PX_DEFAULT)
    ap.add_argument("--contour-max-pts", type=int,
                    default=CONTOUR_MAX_PTS_DEFAULT)
    ap.add_argument("--conf-threshold", type=float,
                    default=CONF_THRESHOLD_DEFAULT,
                    help="the DESTRUCTIVE floor inside Sam3Processor — "
                         "anything below never reaches the record and cannot "
                         "be recovered without re-detecting.")
    ap.add_argument("--frame-stride", type=int, default=8)
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="0.0 by default ON PURPOSE — a threshold picked before "
                         "the score distribution is known is a decision dressed "
                         "as a default. Filter downstream on banked numbers.")
    ap.add_argument("--bpe-path", default=None)
    ap.add_argument("--no-liveness", action="store_true",
                    help="⛔ turn the C77 positive control OFF. Only for a run "
                         "on frames where road/sky genuinely may be absent "
                         "(indoor, night-blind, synthetic). A production "
                         "backfill without it cannot tell an empty scene from "
                         "a dead engine — that is what C77 was.")
    a = ap.parse_args(argv)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ph0_pilot import sample_clip_frames
    from ph0_v2 import norm_to_px

    os.makedirs(a.out, exist_ok=True)
    d = json.load(open(a.v2_json))
    concepts = [c.strip() for c in a.concepts.split(",") if c.strip()]
    scene = [c.strip() for c in a.scene_concepts.split(",") if c.strip()]

    t0 = time.time()
    proc, meta = build_processor(a.bpe_path, conf_threshold=a.conf_threshold)
    print(f"[sam3] processor up in {time.time()-t0:.0f}s · {meta['weights']}",
          flush=True)

    results = []
    for rec in d.get("clips", [])[:a.n]:
        cid = rec.get("clip_id")
        if not cid or rec.get("fatal"):
            continue
        vp = os.path.join(a.video_root, f"{cid}.mp4")
        if not os.path.exists(vp):
            print(f"[sam3] {str(cid)[:8]} NO VIDEO at {vp}", flush=True)
            continue
        frames, _t, _n = sample_clip_frames(vp, t0_s=8.0)
        fh, fw = int(frames[0].shape[0]), int(frames[0].shape[1])

        signs = (rec.get("signs") or {}).get("signs") or []
        vlm_boxes = []
        for i, g in enumerate(rec.get("grounding") or []):
            if not g or not g.get("visible") or not g.get("bbox"):
                continue
            vlm_boxes.append({
                "box_xyxy": norm_to_px(g["bbox"], fw, fh),
                "frame_idx": int(g.get("frame_idx", 0)),
                "label": signs[i].get("kind", "sign") if i < len(signs)
                else "sign"})

        t1 = time.time()
        out = run_clip_frames(proc, frames, concepts, vlm_boxes,
                              frame_stride=a.frame_stride,
                              min_score=a.min_score,
                              liveness=not a.no_liveness,
                              scene_concepts=scene,
                              scene_min_score=a.scene_min_score,
                              contours=not a.no_contours,
                              contour_tol_px=a.contour_tol_px,
                              contour_max_pts=a.contour_max_pts)
        out.update({"clip_id": cid, "frame_wh": [fw, fh],
                    "wall_s": round(time.time() - t1, 1)})
        results.append(out)
        hits = ", ".join(f"{k}:{v}" for k, v in out["per_concept_hits"].items()
                         if v)
        n_match = sum(1 for c in out["vlm_cross_check"] if c["matched"])
        lv = out.get("liveness")
        lvs = (("LIVE " if is_live(lv) else "DEAD ")
               + ",".join(f"{k}:{v}" for k, v in lv["n_det"].items())
               if lv else "liveness OFF")
        shits = ", ".join(f"{k}:{v}"
                          for k, v in (out.get("per_scene_hits") or {}).items()
                          if v)
        print(f"[sam3] {str(cid)[:8]} {out['n_frames_run']}f · "
              f"{out['n_det_total']} det · {out.get('n_scene_det_total', 0)} "
              f"scene · {out['n_err_total']} err · "
              f"[{hits or 'none'}] · [{shits or 'no scene'}] · {lvs} · "
              f"vlm-sign match {n_match}/{len(vlm_boxes)}", flush=True)

    tot = {c: sum(r["per_concept_hits"].get(c, 0) for r in results)
           for c in concepts}
    stot = {c: sum((r.get("per_scene_hits") or {}).get(c, 0) for r in results)
            for c in scene}
    # ⛔ THE COMPLETION CENSUS (C77). A run is complete when DETECTIONS EXIST,
    # never when files exist — so the summary carries the quantity the artifact
    # exists to produce, the error census beside it, and the positive control.
    census = {
        "n_det_total": sum(r["n_det_total"] for r in results),
        "n_scene_det_total": sum(r.get("n_scene_det_total", 0)
                                 for r in results),
        "n_err_total": sum(r.get("n_err_total", 0) for r in results),
        "err_kinds": {},
        "clips_with_zero_det": sum(1 for r in results
                                   if not r["n_det_total"]),
        # recomputed via is_live(); the boolean is not stored (see is_live)
        "clips_not_live": sum(1 for r in results
                              if r.get("liveness")
                              and not is_live(r["liveness"])),
        "liveness_concepts": ([] if a.no_liveness else LIVENESS_CONCEPTS)}
    for r in results:
        for k, v in (r.get("err_kinds") or {}).items():
            census["err_kinds"][k] = census["err_kinds"].get(k, 0) + v
    json.dump({"engine": "C_sam3", "schema_version": SCHEMA_VERSION,
               "api": meta, "concepts": concepts, "scene_concepts": scene,
               "concept_kinds": {c: CONCEPT_KIND.get(c, "thing")
                                 for c in concepts + scene
                                 + LIVENESS_CONCEPTS},
               "frame_stride": a.frame_stride, "min_score": a.min_score,
               "n_clips": len(results), "per_concept_hits_total": tot,
               "per_scene_hits_total": stot,
               "census": census,
               "_note": "SAM3 detects INDEPENDENTLY of the VLM; "
                        "vlm_cross_check is the agreement between engine B's "
                        "B3 sign box and engine C's own 'traffic sign' "
                        "detection. Zero detections for an AGENT concept is a "
                        "valid ABSTENTION — but zero on the LIVENESS concepts "
                        "(road/sky) is an ALARM, and `census` is the thing to "
                        "read before calling a run complete (C77).",
               "clips": results},
              open(os.path.join(a.out, "sam3.json"), "w"), indent=1)
    print(f"[sam3] totals: {tot}", flush=True)
    print(f"[sam3] scene totals: {stot}", flush=True)
    print(f"[sam3] census: {census}", flush=True)
    bad = census["clips_not_live"] or (results and not census["n_det_total"])
    if bad:
        print("SAM3_LIVENESS_ALARM", flush=True)
    print("SAM3_DONE", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
