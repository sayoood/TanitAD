"""STEP 3 — draw the samples and render BLIND contact sheets, for the two legs
G1 and the reliability study actually disagree about.

⛔ **BLIND BY CONSTRUCTION, unchanged from the study.** Each cell carries **only
an integer index**. Score, clip, frame and box are held in the sample JSON and
joined afterwards by `r5_precision.py` / `r8_g1_verdict_join.py`. A sheet
labelled `0.94` would buy the adjudicator's agreement, and the threshold
relation is derived FROM the score↔correctness join, so letting the score reach
the eye would make it self-fulfilling.

⛔ **NO NEW ADJUDICATION PROTOCOL.** `crop_cell()` is IMPORTED from the study's
`r4_sample_and_render.py`, and the G1-tight path is `r7_g1_reconcile.py`'s code
transcribed exactly (tight box crop from the 448 bridge → 4× LANCZOS →
letterboxed into the same 320-px cell). Only the FRAME SOURCE differs, and that
is PREREG.md §2 delta 1:

  ⚠️ **THE FRAMES COME FROM THE PIPELINE'S OWN BRIDGE.** No `w120val` videos are
  banked anywhere — `aug120` holds only `bridged_w120train_2400`. So each clip's
  `<clip>.v2ep.pt` is pulled from the w120 corpus and bridged with
  `stack/scripts/v2_to_pilot.py`'s own `decode_full_episode` →`stacked_to_rgb` →
  `write_mp4`, then read with `ph0_pilot.sample_clip_frames(t0_s=8.0)` — the
  exact call `ph0_sam3.py:1472` makes to produce the boxes being adjudicated.
  This is STRICTLY STRONGER than the study's pre-bridged videos (its §8.5
  caveat): the frame is the pipeline's by construction, not by measurement.

**Arms** (PREREG.md §1): `A` = w120val, uniform `traffic sign`, n=64, seed
20260817. `B` = pilot-50, G1's max-area-per-clip rule, CENSUS of all 37 clips
that carry a sign, rendered under BOTH protocols.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import truststore

truststore.inject_into_ssl()
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
KEYS = os.path.join(REPO, "Keys.txt")
STUDY = os.path.join(REPO, "TanitAD Research Hub", "Data Engineering",
                     "Implementation", "incoming",
                     "2026-08-16-sam3-concept-reliability", "code")
CORP = "Sayood/tanitad-physicalai-w120-256x640cyl"
VAL = "physicalai-val-0c5f7dac3b11-w120-256x640cyl"
SCR = (r"C:\Users\Admin\AppData\Local\Temp\claude"
       r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
       r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad\w120sign")
CACHE = os.path.join(SCR, "records")
MP4 = os.path.join(SCR, "mp4")
sys.path.insert(0, os.path.join(REPO, "stack"))
sys.path.insert(0, os.path.join(REPO, "stack", "scripts"))
sys.path.insert(0, STUDY)

SAM3_PX = 448.0
COLS, ROWS, CELL, PAD = 4, 4, 320, 26


def token() -> str:
    return re.search(r"hf_[A-Za-z0-9]+",
                     open(KEYS, encoding="utf-8", errors="replace").read()
                     ).group(0)


def leg_records(leg: str) -> dict:
    blob = json.load(open(os.path.join(CACHE, f"{leg}.json"), encoding="utf-8"))
    return {r["clip_id"]: r for r in blob["clips"]}


def sign_population(recs: dict) -> list[dict]:
    pop = []
    for cid, r in sorted(recs.items()):
        for fk in sorted((r.get("frames") or {}), key=lambda k: int(k)):
            for j, d in enumerate(r["frames"][fk].get("det", []) or []):
                if d.get("concept") != "traffic sign":
                    continue
                if "score" not in d or not d.get("box_xyxy"):
                    continue
                pop.append({"clip_id": cid, "frame_idx": int(fk), "det_i": j,
                            "concept": d["concept"], "score": float(d["score"]),
                            "box_xyxy": d["box_xyxy"],
                            "mask_area_px": d.get("mask_area_px")})
    return pop


def maxarea_per_clip(recs: dict) -> list[dict]:
    """G1's own selection rule: the LARGEST-AREA `traffic sign` per clip."""
    picks = []
    for cid, r in sorted(recs.items()):
        best = None
        for fk, f in (r.get("frames") or {}).items():
            for j, d in enumerate(f.get("det", []) or []):
                if d.get("concept") != "traffic sign" or not d.get("box_xyxy"):
                    continue
                b = d["box_xyxy"]
                ar = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
                if best is None or ar > best["area_px"]:
                    best = {"clip_id": cid, "frame_idx": int(fk), "det_i": j,
                            "concept": d["concept"], "score": float(d["score"]),
                            "box_xyxy": b, "area_px": round(ar, 1),
                            "mask_area_px": d.get("mask_area_px")}
        if best:
            picks.append(best)
    return picks


# ---------------------------------------------------------------------------
# frames: the PIPELINE'S OWN BRIDGE, per PREREG delta 1
# ---------------------------------------------------------------------------
def fetch_v2ep(cid: str, tok: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(CORP, f"{VAL}/{cid}.v2ep.pt", repo_type="dataset",
                           token=tok)


def bridge_one(cid: str, src: str) -> str:
    """`<clip>.v2ep.pt` -> mp4, via v2_to_pilot's OWN functions (not a copy)."""
    import v2_to_pilot
    from tanitad.data.v2_dataset import decode_full_episode
    os.makedirs(MP4, exist_ok=True)
    out = os.path.join(MP4, f"{cid}.mp4")
    if os.path.exists(out) and os.path.getsize(out) > 1024:
        return out
    ep = decode_full_episode(src)
    arr = v2_to_pilot.stacked_to_rgb(ep.frames)
    v2_to_pilot.write_mp4(out, arr, 10)             # the pilot bridge's fps
    return out


def frames_for(cids: list[str], tok: str, workers: int = 5) -> dict:
    """Pull + bridge in parallel, then decode at BOTH pixel scales.

    640 = the native long side, for the context render; 448 = the frame SAM3
    actually scored, for the G1-tight render."""
    import ph0_pilot
    paths: dict[str, str] = {}
    t0 = time.time()

    def one(cid):
        try:
            return cid, fetch_v2ep(cid, tok), None
        except Exception as e:                            # noqa: BLE001
            return cid, None, f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, (cid, p, err) in enumerate(ex.map(one, cids)):
            if err:
                print(f"[pull] FAILED {cid}: {err}", flush=True)
                continue
            paths[cid] = p
            if (k + 1) % 10 == 0:
                print(f"[pull] {k+1}/{len(cids)} in {time.time()-t0:.0f}s",
                      flush=True)
    out = {}
    for k, (cid, src) in enumerate(sorted(paths.items())):
        try:
            mp4 = bridge_one(cid, src)
            f640 = ph0_pilot.sample_clip_frames(mp4, t0_s=8.0, px=640)[0]
            f448 = ph0_pilot.sample_clip_frames(mp4, t0_s=8.0)[0]
            out[cid] = (f640, f448)
        except Exception as e:                            # noqa: BLE001
            print(f"[bridge] FAILED {cid}: {type(e).__name__}: {e}", flush=True)
        if (k + 1) % 10 == 0:
            print(f"[bridge] {k+1}/{len(paths)} in {time.time()-t0:.0f}s",
                  flush=True)
    return out


def render(sel, frames, out_dir, tag, style, header):
    """One contact sheet family. `style` in {context, g1tight}."""
    from PIL import Image, ImageDraw
    from r4_sample_and_render import crop_cell            # the STUDY'S code
    os.makedirs(out_dir, exist_ok=True)
    sheets = []
    for s0 in range(0, len(sel), COLS * ROWS):
        chunk = sel[s0:s0 + COLS * ROWS]
        sheet = Image.new("RGB", (COLS * CELL, ROWS * (CELL + PAD) + PAD),
                          (10, 10, 10))
        ImageDraw.Draw(sheet).text((8, 6), header, fill=(235, 235, 235))
        for k, p in enumerate(chunk):
            r_, c_ = divmod(k, COLS)
            fr = frames.get(p["clip_id"])
            if fr is None:
                continue
            f640, f448 = fr
            fi = p["frame_idx"]
            if fi >= len(f640):
                continue
            if style == "context":
                cellimg, bx = crop_cell(f640[fi], p["box_xyxy"],
                                        f640[fi].shape[1] / SAM3_PX)
            else:                                          # r7's tight path
                x0, y0, x1, y1 = [int(round(v)) for v in p["box_xyxy"]]
                src = Image.fromarray(f448[fi]).crop(
                    (max(0, x0), max(0, y0), max(x0 + 1, x1), max(y0 + 1, y1)))
                src = src.resize((src.width * 4, src.height * 4), Image.LANCZOS)
                cellimg = Image.new("RGB", (CELL, CELL), (16, 16, 16))
                sc = min(1.0, CELL / max(src.width, src.height))
                src = src.resize((max(1, int(src.width * sc)),
                                  max(1, int(src.height * sc))), Image.LANCZOS)
                cellimg.paste(src, ((CELL - src.width) // 2,
                                    (CELL - src.height) // 2))
                bx = None
            x, y = c_ * CELL, PAD + r_ * (CELL + PAD)
            sheet.paste(cellimg, (x, y))
            dd = ImageDraw.Draw(sheet)
            if bx:
                dd.rectangle([x + bx[0], y + bx[1], x + bx[2], y + bx[3]],
                             outline=(255, 215, 0), width=2)
            dd.text((x + 5, y + CELL + 5), f"#{p['idx']}", fill=(255, 215, 0))
            dd.rectangle([x, y, x + CELL - 1, y + CELL - 1],
                         outline=(60, 60, 60), width=1)
        name = f"{tag}_{style}_{s0 // (COLS * ROWS) + 1:02d}.png"
        sheet.save(os.path.join(out_dir, name))
        sheets.append(name)
        print(f"[sheet] {name} {len(chunk)} cells", flush=True)
    return sheets


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("w3_sample_and_render")
    ap.add_argument("--arm", required=True, choices=("A", "B"))
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260817)
    a = ap.parse_args(argv)
    tok = token()

    if a.arm == "A":
        leg, base = "w120val600", 0
        recs = leg_records(leg)
        pop = sign_population(recs)
        rng = random.Random(a.seed)
        sel = rng.sample(pop, min(a.n, len(pop)))
        mode = "SAMPLE (uniform at random within `traffic sign`)"
        styles = ["context"]
        hdr = ("CONCEPT UNDER TEST: 'traffic sign'  — corpus w120val "
               "— scores hidden on purpose")
    else:
        leg, base = "pilot50", 2000
        recs = leg_records(leg)
        pop = sign_population(recs)
        sel = maxarea_per_clip(recs)
        mode = "CENSUS (G1's rule: largest-area `traffic sign` per clip)"
        styles = ["g1tight", "context"]
        hdr = ("G1's SELECTION on G1's OWN CLIPS (pilot-50): largest-area "
               "'traffic sign' per clip — scores hidden")

    sel.sort(key=lambda p: (p["clip_id"], p["frame_idx"], p["det_i"]))
    for i, p in enumerate(sel):
        p["idx"] = base + i
        p["_mode"] = "CENSUS" if a.arm == "B" else "SAMPLE"
    need = sorted({p["clip_id"] for p in sel})
    print(f"[arm {a.arm}] leg={leg} population={len(pop)} picked={len(sel)} "
          f"over {len(need)} clips · {mode}", flush=True)

    frames = frames_for(need, tok)
    missing = [c for c in need if c not in frames]
    if missing:
        print(f"[arm {a.arm}] ⚠️ {len(missing)} clips have NO FRAMES: "
              f"{missing[:5]}", flush=True)

    sheets = []
    for st in styles:
        sheets += render(sel, frames, a.out_dir, f"w120sign_{a.arm}", st,
                         hdr + (" [G1 tight 4x LANCZOS]" if st == "g1tight"
                                else " [6x context, native 640]"))

    out = {"arm": a.arm, "leg": leg, "seed": a.seed, "sam3_px": SAM3_PX,
           "selection": mode,
           "frame_source": "PIPELINE'S OWN BRIDGE — v2_to_pilot."
                           "decode_full_episode/stacked_to_rgb/write_mp4 on "
                           f"{CORP}/{VAL}/<clip>.v2ep.pt, then "
                           "ph0_pilot.sample_clip_frames(t0_s=8.0)",
           "n_clips_in_leg": len(recs), "n_sign_population": len(pop),
           "n_picked": len(sel), "n_clips_touched": len(need),
           "n_clips_without_frames": len(missing),
           "clips_without_frames": missing,
           "sheets": sheets, "detections": sel}
    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
    json.dump(out, open(a.out_json, "w", encoding="utf-8"), indent=1)
    print(f"SAMPLE_DONE arm={a.arm} n={len(sel)} sheets={len(sheets)}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
