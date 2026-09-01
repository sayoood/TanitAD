"""Build a self-contained payload for ``vlm_tac_extract.py``.

⭐ THE SPLIT THIS FILE EXISTS TO ENFORCE. The remote side (Colab VM, pod, Thor)
gets frames and prompts ALREADY BUILT and nothing else — no repo checkout, no
dataset pull, and above all **no credentials**. Everything that needs the repo
(the prompt text, the Alpamayo taxonomy, the parity gate) happens here, on a
machine that already has them.

⚠️ THE PARITY GATE IS APPLIED HERE, WHICH IS THE ONLY PLACE IT CAN BE. A clip
that leaks into the deployed val must never reach the extractor at all —
filtering afterwards would mean the label existed, and a label that exists gets
used. It runs through parity.guard_corpus_build (role="train"), NOT a local
exclusion list: the gate checks BOTH overlaps and emits a record that is written
into the payload manifest as `_parity_gate`.

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
from tanitad.data import parity  # noqa: E402

HUB = REPO / "TanitAD Research Hub" / "Data Engineering"
TAXONOMY = (HUB / "Implementation/incoming/2026-08-16-tactical-labels/raw"
            / "a1_alpamayo_taxonomy_per_clip.jsonl")

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
    cache = Path(args.cache)
    files = sorted(cache.glob("*.v2ep.pt"))
    cand = [f.name.replace(".v2ep.pt", "") for f in files]

    # ⭐ THE INGEST GATE, not a hand-rolled exclusion. These labels become
    # SUPERVISION for v6.1 / REF-A v1 / REF-C v3, so role="train": the deployed
    # val may not be inside. An earlier version of this script filtered against
    # alpamayo_val40_exclusions.json by hand — a second, untested implementation
    # of a question parity.py already answers, and one that checked only ONE of
    # the two overlaps. The test suite caught it
    # (test_build_parity_guard.py::test_every_derived_corpus_writer_is_gated).
    #
    # ⚠️ It runs BEFORE a single frame is decoded: C112's defect died AFTER
    # paying for a 536 MB download, and a gate that runs late costs money to
    # trip. mode="exclude" filters and reports rather than raising, because a
    # 130-clip cache legitimately contains val clips and dropping them is the
    # intended outcome — but the record rides along in the manifest, so the
    # payload can never report a clip count whose selection it cannot name.
    parity.require_ingest_gate("vlm_tac_payload")
    kept, gate = parity.guard_corpus_build(
        cand, label=f"vlm_tac_payload -> {args.out}", role="train",
        mode="exclude")
    keep_ids = set(kept)
    files = [f for f in files if f.name.replace(".v2ep.pt", "") in keep_ids]
    print(f"PARITY GATE kept={len(kept)}/{len(cand)} "
          f"in_deployed_val={gate.get('in_deployed_val')} "
          f"in_parity_train={gate.get('in_parity_train')}")
    payload = {"_parity_gate": gate,
               "_what": "self-contained VLM tactical/strategic extraction payload",
               "_prompts_from": "stack/scripts/vlm_tac_prompts.py",
               "frame_times_s": list(FRAME_TIMES_S), "sampling": P.SAMPLING,
               "clips": []}
    n_notax = 0
    for pt in files:
        cid = pt.name.replace(".v2ep.pt", "")
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
    print(f"gate-kept files {len(files)} | no taxonomy row {n_notax}")
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
