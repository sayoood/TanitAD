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

# The concept vocabulary for the `text` mode. Each entry is a tactical-vocabulary
# slot that PH0 cannot currently fill — see the module docstring. Deliberately
# CONCRETE nouns: SAM3 grounds objects, not abstractions like "hazard".
AGENT_CONCEPTS = ["car", "truck", "bus", "pedestrian", "cyclist",
                  "traffic light", "traffic sign"]

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
    return Sam3Processor(model, confidence_threshold=conf_threshold), {
        "api": "build_sam3_image_model + Sam3Processor "
               "(facebookresearch/sam3 README)",
        "weights": "facebook/sam3 (load_from_HF=True)",
        "bpe_path": bpe,
        "confidence_threshold": conf_threshold,
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


def detect(processor, image, prompt: str, *, min_score: float = 0.0) -> list[dict]:
    """One image, one concept -> one record per detection, in FRAME pixels.

    ⚠️ `min_score` defaults to 0.0 — nothing is filtered by default, because a
    threshold chosen before we have seen the score distribution is a decision
    dressed as a default. Filter downstream, on banked numbers.

    ⚠️ This encodes the image. For SEVERAL concepts on the same frame use
    `detect_many`, which encodes once — see its docstring for the 4.4×."""
    return _score(processor, processor.set_image(image), prompt,
                  min_score=min_score)


def detect_many(processor, image, prompts, *, min_score: float = 0.0
                ) -> list[dict]:
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
            dets.extend(_score(processor, state, p, min_score=min_score))
        except Exception as e:
            dets.append({"concept": p,
                         "error": f"{type(e).__name__}: {e}"[:140]})
    return dets


def _score(processor, state, prompt: str, *, min_score: float = 0.0
           ) -> list[dict]:
    """Score ONE concept against an already-encoded image state."""
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
            m = masks[i]
            m = m > 0.0 if m.dtype != bool else m
            r["mask_area_px"] = int(m.sum())
            r["rle_rows"] = _rows_rle(m)
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
            m = np.asarray(masks[i])
            m = m > 0.0 if m.dtype != bool else m
            r["mask_area_px"] = int(m.sum())
            r["mask_hw"] = [int(m.shape[0]), int(m.shape[1])]
            r["rle_rows"] = _rows_rle(m)
        recs.append(r)
    return recs


def _rows_rle(mask) -> list[list[int]]:
    """Compact per-row [row, start, end) runs — small enough to bank in JSON and
    enough to redraw the mask exactly."""
    import numpy as np
    runs = []
    for r, row in enumerate(np.asarray(mask)):
        idx = np.flatnonzero(row)
        if idx.size == 0:
            continue
        splits = np.flatnonzero(np.diff(idx) > 1)
        starts = np.r_[idx[0], idx[splits + 1]]
        ends = np.r_[idx[splits], idx[-1]] + 1
        for s, e in zip(starts, ends):
            runs.append([r, int(s), int(e)])
    return runs


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


def liveness_probe(processor, image, *, concepts=None,
                   min_score: float = 0.0) -> dict:
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
    a crash census is readable from the artifact without a re-run."""
    cs = list(concepts or LIVENESS_CONCEPTS)
    out = {c: 0 for c in cs}
    err = {}
    for d in detect_many(processor, image, cs, min_score=min_score):
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
    return rec


def run_clip_frames(processor, frames, concepts, vlm_boxes, *,
                    frame_stride: int = 8, min_score: float = 0.0,
                    liveness: bool = True) -> dict:
    """One clip -> SAM3's own detections on a strided set of frames, plus the
    cross-engine check against the VLM's B3 sign boxes.

    ⭐ THE CROSS-ENGINE CHECK IS THE POINT. SAM3 finds signs INDEPENDENTLY of
    Qwen. Where both fire, `box_mask_agreement` says whether they mean the same
    object; where only one fires, that disagreement is the finding. A VLM box
    alone is an unverifiable claim, and averaging the two would hide exactly the
    cases worth looking at."""
    from PIL import Image
    out_frames: dict[str, dict] = {}
    per_concept: dict[str, int] = {c: 0 for c in concepts}
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
                              min_score=min_score)
        live["frame_idx"] = int(todo[len(todo) // 2])
    for fi in todo:
        img = Image.fromarray(frames[fi])
        # ⛔ ONE ENCODE PER FRAME, not one per concept — see detect_many. The
        # per-concept loop that used to live here re-ran the ViT trunk 7x on
        # the identical frame (MEASURED banked wall_s 97-98 s per 6-frame clip).
        dets = detect_many(processor, img, concepts, min_score=min_score)
        for d in dets:
            if "score" in d:
                per_concept[d["concept"]] = per_concept.get(d["concept"], 0) + 1
        # ⚠️ EVERY strided frame is recorded, including the empty ones: a frame
        # dropped for having no detections is indistinguishable downstream from
        # a frame never run, and those are different failures.
        out_frames[str(fi)] = {"n_det": sum(1 for d in dets if "score" in d),
                               "det": dets}
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
    n_err = 0
    for f in out_frames.values():
        for d in f["det"]:
            if "error" in d:
                n_err += 1
                k = str(d["error"]).split(":")[0]
                errs[k] = errs.get(k, 0) + 1
    return {"frames": out_frames, "per_concept_hits": per_concept,
            "n_frames_run": len(out_frames),
            "n_det_total": sum(f["n_det"] for f in out_frames.values()),
            "n_err_total": n_err, "err_kinds": errs,
            "liveness": live,
            "vlm_cross_check": agree}


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
    """Translucent mask fill + outline + label chip, one colour per instance."""
    import numpy as np
    from PIL import Image, ImageDraw
    base = np.asarray(img.convert("RGB")).astype(np.float32)
    over = base.copy()
    for i, s in enumerate(segs):
        if not s.get("rle_rows"):
            continue
        col = np.array(MASK_COLOURS[i % len(MASK_COLOURS)], np.float32)
        for r, a, b in s["rle_rows"]:
            if 0 <= r < over.shape[0]:
                over[r, a:b] = 0.55 * col + 0.45 * over[r, a:b]
    out = Image.fromarray(over.astype(np.uint8))
    d = ImageDraw.Draw(out)
    for i, s in enumerate(segs):
        col = MASK_COLOURS[i % len(MASK_COLOURS)]
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
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("ph0_sam3")
    ap.add_argument("--v2-json", required=True,
                    help="ph0_v2.json — supplies the VLM B3 sign boxes to cross-check")
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--concepts", default=",".join(AGENT_CONCEPTS))
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

    t0 = time.time()
    proc, meta = build_processor(a.bpe_path)
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
                              liveness=not a.no_liveness)
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
        print(f"[sam3] {str(cid)[:8]} {out['n_frames_run']}f · "
              f"{out['n_det_total']} det · {out['n_err_total']} err · "
              f"[{hits or 'none'}] · {lvs} · "
              f"vlm-sign match {n_match}/{len(vlm_boxes)}", flush=True)

    tot = {c: sum(r["per_concept_hits"].get(c, 0) for r in results)
           for c in concepts}
    # ⛔ THE COMPLETION CENSUS (C77). A run is complete when DETECTIONS EXIST,
    # never when files exist — so the summary carries the quantity the artifact
    # exists to produce, the error census beside it, and the positive control.
    census = {
        "n_det_total": sum(r["n_det_total"] for r in results),
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
    json.dump({"engine": "C_sam3", "api": meta, "concepts": concepts,
               "frame_stride": a.frame_stride, "min_score": a.min_score,
               "n_clips": len(results), "per_concept_hits_total": tot,
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
    print(f"[sam3] census: {census}", flush=True)
    bad = census["clips_not_live"] or (results and not census["n_det_total"])
    if bad:
        print("SAM3_LIVENESS_ALARM", flush=True)
    print("SAM3_DONE", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
