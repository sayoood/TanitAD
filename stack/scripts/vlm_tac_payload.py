"""Build a self-contained payload for ``vlm_tac_extract.py``.

⭐ THE SPLIT THIS FILE EXISTS TO ENFORCE. The remote side (Colab VM, pod, Thor)
gets frames and prompts ALREADY BUILT and nothing else — no repo checkout, no
dataset pull, and above all **no credentials**. Everything that needs the repo
(the prompt text, the Alpamayo taxonomy, the val40 leak exclusions) happens
here, on a machine that already has them.

⚠️ THE EXCLUSIONS ARE APPLIED HERE, WHICH IS THE ONLY PLACE THEY CAN BE. A clip
that leaks into val40 must never reach the extractor at all — filtering
afterwards would mean the label existed, and a label that exists gets used.

Usage (dev box / Thor / pod — same command, different cache):
    python stack/scripts/vlm_tac_payload.py \
        --cache <dir of {clip_id}.v2ep.pt> --out payload.json [--limit N]
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "stack" / "scripts"))

import vlm_tac_prompts as P  # noqa: E402

HUB = REPO / "TanitAD Research Hub" / "Data Engineering"
TAXONOMY = (HUB / "Implementation/incoming/2026-08-16-tactical-labels/raw"
            / "a1_alpamayo_taxonomy_per_clip.jsonl")
EXCLUSIONS = (HUB / "Research/2026-08-18-alpamayo-screening"
              / "alpamayo_val40_exclusions.json")

#: The review sheet's geometry: one frame BEFORE the window opens (so the model
#: can see what the ego was doing), t0, and the 6 s horizon the tactical and
#: strategic layers are asked about.
FRAME_TIMES_S = (-1.0, 0.0, 2.0, 4.0, 6.0)
FPS = 10.0


def frames_b64(pt: Path, times=FRAME_TIMES_S):
    """-> (list[b64 jpeg], note, v0, v_end).

    ⚠️ v2ep stores PRE-ENCODED images concatenated in ``jpeg_buf`` with per-frame
    lengths in ``jpeg_len`` — and ``codec`` says **png** despite the key names,
    so decode by sniffing (PIL) rather than trusting the field. ``poses`` is
    [T, 4] = (x, y, heading, SPEED), which gives the prompts REAL ego numbers
    instead of invented ones.
    """
    import torch
    from PIL import Image
    o = torch.load(pt, map_location="cpu", weights_only=False)
    buf = o["jpeg_buf"].numpy().tobytes()
    lens = o["jpeg_len"].tolist()
    poses = o["poses"]
    T = len(lens)
    mid = T // 2
    offs = [0]
    for L in lens:
        offs.append(offs[-1] + int(L))
    out, idxs, size = [], [], None
    for t in times:
        i = max(0, min(T - 1, int(round(mid + t * FPS))))
        idxs.append(i)
        im = Image.open(io.BytesIO(buf[offs[i]:offs[i + 1]])).convert("RGB")
        size = im.size
        b = io.BytesIO()
        im.save(b, format="JPEG", quality=85)
        out.append(base64.b64encode(b.getvalue()).decode("ascii"))
    v0 = float(poses[mid, 3])
    v_end = float(poses[max(0, min(T - 1, mid + int(6.0 * FPS))), 3])
    return out, f"T={T} mid={mid} idx={idxs} wh={size}", v0, v_end


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, help="dir of {clip_id}.v2ep.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    alp = {}
    with TAXONOMY.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                alp[r["clip_id"]] = r
    excl = set(json.loads(EXCLUSIONS.read_text(encoding="utf-8"))["excluded_clip_ids"])

    cache = Path(args.cache)
    files = sorted(cache.glob("*.v2ep.pt"))
    payload = {"_what": "self-contained VLM tactical/strategic extraction payload",
               "_prompts_from": "stack/scripts/vlm_tac_prompts.py",
               "frame_times_s": list(FRAME_TIMES_S), "sampling": P.SAMPLING,
               "clips": []}
    n_excl = n_notax = 0
    for pt in files:
        cid = pt.name.replace(".v2ep.pt", "")
        if cid in excl:
            n_excl += 1
            continue
        r = alp.get(cid)
        if r is None:
            n_notax += 1
            continue
        fr, note, v0, v_end = frames_b64(pt)
        ctx = P.ClipContext(v0_ms=v0, v_end_ms=v_end,
                            alpamayo_magnitude=r["longitudinal"],
                            alpamayo_cot=r["cot"])
        calls = {k: {"with_ego": fn(ctx, with_ego=True),
                     "no_ego": fn(ctx, with_ego=False)}
                 for k, fn in (("lon", P.build_lon_prompt),
                               ("lane", P.build_lane_prompt),
                               ("sign", P.build_sign_prompt))}
        payload["clips"].append({
            "clip_id": cid,
            "alpamayo": {"lane": r["lane"], "longitudinal": r["longitudinal"],
                         "lateral": r["lateral"], "cot": r["cot"]},
            "ego": {"v0_ms": v0, "v_end_ms": v_end},
            "frames_b64": fr, "frames_note": note, "calls": calls})
        if args.limit and len(payload["clips"]) >= args.limit:
            break

    out = Path(args.out)
    out.write_text(json.dumps(payload), encoding="utf-8")
    lanes = {}
    for c in payload["clips"]:
        k = c["alpamayo"]["lane"] or "(one-axis)"
        lanes[k] = lanes.get(k, 0) + 1
    print(f"cache files {len(files)} | val40-leak excluded {n_excl} | "
          f"no taxonomy row {n_notax}")
    print(f"payload {len(payload['clips'])} clips -> {out} "
          f"({out.stat().st_size/1e6:.1f} MB)")
    for k, v in sorted(lanes.items(), key=lambda x: -x[1]):
        print(f"   {k:22s} {v:>4}")
    turns = sum(v for k, v in lanes.items() if "turn" in k.lower())
    print(f"generations: {len(payload['clips'])}x2 (lon,lane) + {turns} sign "
          f"= {len(payload['clips'])*2 + turns}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
