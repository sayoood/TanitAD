"""Engine C — SAM3 segmentation, prompted by the VLM's grounded sign boxes.

The PI's decided stack: engine B = Qwen3.5-9B (symbols + OCR), engine C = SAM3
(pixels), engine A = algorithmic integrated ego path (numbers). This is C.

⭐ WHY BOX-PROMPTED. B3 gives a coarse box in Qwen's normalized 0–1000 space;
SAM3 refines it to a pixel-accurate mask. That makes the two engines
INDEPENDENTLY CHECKABLE against each other: the box↔mask IoU is a measurable
grounding signal, where a VLM box alone is an unverifiable claim. A low IoU
means the VLM pointed at something SAM3 does not see as one object — which is
exactly the disagreement worth surfacing rather than averaging away.

API verified on pod4 2026-08-12, not guessed:
  sam3.build_sam3_image_model(device=..., load_from_HF=True, ...)
  sam3.model_builder.SAM3InteractiveImagePredictor(sam_model, ...)
    .set_image_batch(image_list) · .predict(box=..., multimask_output=...)

⚠️ SAM3 weights come from `facebook/sam3` (HTTP 200 with our token, verified;
`sam3-large`/`sam3-base` are 404 and do not exist). `load_from_HF=True` fetches
them, so HF_HOME must be set and the token present.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

MASK_COLOURS = [(42, 120, 214), (235, 104, 52), (74, 58, 167),
                (26, 148, 106), (196, 62, 140), (176, 132, 20)]


def find_bpe() -> str | None:
    """SAM3's text encoder needs the CLIP BPE vocab, and the sam3 wheel does NOT
    ship it — it defaults to `site-packages/assets/bpe_simple_vocab_16e6.txt.gz`
    which does not exist, so the builder dies with a FileNotFound three frames
    deep. ⚠️ It is NOT in the `facebook/sam3` HF repo either: that repo carries
    HF-format tokenizer files (vocab.json + merges.txt), not the CLIP .gz.
    `open_clip` ships the canonical file, so we locate it there."""
    import glob
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


def build_predictor(device: str = "cuda", bpe_path: str | None = None):
    """Returns (predictor, meta) or raises with a useful message."""
    import sam3
    from sam3.model_builder import SAM3InteractiveImagePredictor
    bpe = bpe_path or find_bpe()
    if bpe is None:
        raise SystemExit("[sam3] CLIP BPE vocab not found — install "
                         "open_clip_torch (--no-deps) or pass --bpe-path")
    model = sam3.build_sam3_image_model(device=device, eval_mode=True,
                                        load_from_HF=True, bpe_path=bpe,
                                        enable_segmentation=True)
    return SAM3InteractiveImagePredictor(model), {
        "builder": "sam3.build_sam3_image_model",
        "predictor": "SAM3InteractiveImagePredictor",
        "weights": "facebook/sam3 (load_from_HF=True)",
        "bpe_path": bpe}


def segment_boxes(predictor, image, boxes_px: list[list[int]]) -> list[dict]:
    """One image, N box prompts -> N masks with their box↔mask agreement.

    ⚠️ Records `box_iou` per mask: the fraction of the mask inside the prompting
    box, and of the box covered. A mask that spills far outside its prompt means
    SAM3 latched onto a larger structure than the VLM meant, and that
    disagreement is the SIGNAL, not noise to be hidden."""
    import numpy as np
    out = []
    if not boxes_px:
        return out
    predictor.set_image_batch([np.asarray(image)])
    for bi, b in enumerate(boxes_px):
        x0, y0, x1, y1 = [int(v) for v in b]
        if x1 <= x0 or y1 <= y0:
            out.append({"box": b, "error": "degenerate box", "mask": None})
            continue
        try:
            masks, scores, _ = predictor.predict(
                box=np.array([x0, y0, x1, y1])[None, :],
                multimask_output=False)
            m = np.asarray(masks).reshape(-1, *np.asarray(masks).shape[-2:])[0]
            m = m > 0.0 if m.dtype != bool else m
            area = int(m.sum())
            bx = np.zeros_like(m, dtype=bool)
            bx[max(0, y0):y1, max(0, x0):x1] = True
            inter = int((m & bx).sum())
            out.append({
                "box": b, "score": float(np.asarray(scores).ravel()[0]),
                "mask_area_px": area,
                "frac_mask_in_box": round(inter / area, 4) if area else 0.0,
                "frac_box_covered": round(inter / int(bx.sum()), 4)
                if bx.sum() else 0.0,
                "rle_rows": _rows_rle(m)})
        except Exception as e:                                # per box
            out.append({"box": b, "error": f"{type(e).__name__}: {e}"[:140],
                        "mask": None})
    return out


def _rows_rle(mask) -> list[list[int]]:
    """Compact per-row [start, end) runs — small enough to bank in JSON and
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
        x0, y0, x1, y1 = [int(v) for v in s["box"]]
        d.rectangle([x0, y0, x1, y1], outline=col, width=2)
        lab = (labels[i] if labels and i < len(labels) else f"sam3[{i}]")
        if s.get("frac_mask_in_box") is not None:
            lab += f"  in-box {s['frac_mask_in_box']:.2f}"
        d.text((x0 + 3, max(0, y0 - 12)), lab, fill=col)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("ph0_sam3")
    ap.add_argument("--v2-json", required=True,
                    help="ph0_v2.json — supplies the VLM sign boxes to prompt with")
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bpe-path", default=None,
                    help="CLIP BPE vocab .gz; auto-located from open_clip")
    a = ap.parse_args(argv)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ph0_pilot import sample_clip_frames
    from ph0_v2 import norm_to_px

    from PIL import Image

    os.makedirs(a.out, exist_ok=True)
    d = json.load(open(a.v2_json))
    t0 = time.time()
    predictor, meta = build_predictor(a.device, a.bpe_path)
    print(f"[sam3] predictor up in {time.time()-t0:.0f}s · {meta['weights']}",
          flush=True)

    results, made = [], []
    for rec in d.get("clips", [])[:a.n]:
        cid = rec.get("clip_id")
        if not cid or rec.get("fatal"):
            continue
        frames, _t, _n = sample_clip_frames(
            os.path.join(a.video_root, f"{cid}.mp4"), t0_s=8.0)
        fh, fw = frames[0].shape[0], frames[0].shape[1]
        signs = (rec.get("signs") or {}).get("signs") or []
        gnd = rec.get("grounding") or []
        # group the VLM's boxes by the frame it reported them in
        per_frame: dict[int, list] = {}
        for i, g in enumerate(gnd):
            if not g or not g.get("visible"):
                continue
            fi = int(g.get("frame_idx", 0))
            if not 0 <= fi < len(frames):
                continue
            px = norm_to_px(g["bbox"], fw, fh)
            lab = signs[i].get("kind", "sign") if i < len(signs) else "sign"
            if i < len(signs) and signs[i].get("text"):
                lab += f' "{signs[i]["text"]}"'
            per_frame.setdefault(fi, []).append((px, lab))

        clip_out = {"clip_id": cid, "frames": {}}
        for fi, items in per_frame.items():
            boxes = [b for b, _ in items]
            labs = [l for _, l in items]
            img = Image.fromarray(frames[fi])
            segs = segment_boxes(predictor, img, boxes)
            clip_out["frames"][str(fi)] = {"labels": labs, "segs": segs}
            png = os.path.join(a.out, f"{str(cid)[:8]}_f{fi:02d}_sam3.png")
            draw_masks(img, segs, labs).save(png)
            made.append(png)
            ok = sum(1 for s in segs if s.get("rle_rows"))
            print(f"[sam3] {str(cid)[:8]} frame {fi}: {ok}/{len(segs)} masks",
                  flush=True)
        results.append(clip_out)

    json.dump({"engine": "C_sam3", "api": meta, "n_clips": len(results),
               "_note": "boxes are the VLM's B3 groundings converted to pixels; "
                        "frac_mask_in_box is the box<->mask agreement and is the "
                        "cross-engine grounding signal",
               "clips": results},
              open(os.path.join(a.out, "sam3.json"), "w"), indent=1)
    print(f"[sam3] {len(made)} overlays · {len(results)} clips", flush=True)
    print("SAM3_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
