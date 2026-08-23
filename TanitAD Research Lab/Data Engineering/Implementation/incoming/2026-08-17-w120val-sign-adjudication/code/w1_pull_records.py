"""STEP 1 — pull the SAM3 records for the corpora G1 and the 4,472 build argue
about, and INDEX them the way `r1_pull_records.py` indexed `aug120`.

⛔ **THE FIRST THING THIS STEP ESTABLISHES IS WHICH CORPUS G1 ACTUALLY MEASURED**,
because the reliability study
(`…/2026-08-16-sam3-concept-reliability/SAM3_CONCEPT_RELIABILITY.md` §4.1) states
G1 measured `w120val` (600 clips) and the primary sources say otherwise:

  · `Project Steering/G1_SIGN_OCR_GRADING_SHEET.md:5` — *"These are the 31
    non-empty OCR texts from the **50-clip pilot** — the pre-registered sample."*
  · `Project Steering/G1_RESULT.md:4-5` — *"for each of the 31 pre-registered
    pilot OCR texts, the sign was cropped from **the pilot videos** using SAM3's
    boxes"*.

The `4 048` figure in `G1_RESULT.md:34` is a SEPARATE, SCALE-ONLY sentence
(*"Production banked 4 048 'traffic sign' detections on the 600-clip set; **if**
this false-positive character generalises …"*) — G1 scoped its own finding
correctly and hedged it. The transfer to `w120val` happened downstream.

⇒ This step pulls BOTH legs so the question can be settled on the corpus G1
really used AND on the corpus the 4,472 build cares about:

  · `Sayood/tanitad-ph0/ph0_pilot50/sam3/sam3.json` — the **pilot-50** leg (G1's
    own corpus);
  · `Sayood/tanitad-ph0/ph0_prod4/sam3/sam3.json` — the **w120val-600** leg (the
    one the brief names, and the one `fused_w120val/` was built from).

Writes one index per leg. No GPU, no HF write.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

import truststore

truststore.inject_into_ssl()
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
KEYS = os.path.join(REPO, "Keys.txt")
DS = "Sayood/tanitad-ph0"
LEGS = {"w120val600": "ph0_prod4/sam3/sam3.json",
        "pilot50": "ph0_pilot50/sam3/sam3.json"}
CACHE = (r"C:\Users\Admin\AppData\Local\Temp\claude"
         r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
         r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad\w120sign\records")


def token() -> str:
    """⚠️ Read in place, never printed and never passed in argv."""
    return re.search(r"hf_[A-Za-z0-9]+",
                     open(KEYS, encoding="utf-8", errors="replace").read()
                     ).group(0)


def iter_dets(rec: dict):
    """Yield (frame_key, det_index, det) over one clip record.

    ⚠️ Written against the ACTUAL banked shape, which is verified by the caller
    (`--dump-shape`) rather than assumed — the aug120 records use
    `frames{key}.det[]`, and a production record from an older engine may not."""
    for fk in sorted((rec.get("frames") or {}), key=lambda k: int(k)):
        for j, d in enumerate((rec["frames"][fk] or {}).get("det", []) or []):
            yield fk, j, d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("w1_pull_records")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--dump-shape", action="store_true")
    a = ap.parse_args(argv)

    from huggingface_hub import hf_hub_download
    tok = token()
    os.makedirs(a.cache, exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)

    for leg, rf in LEGS.items():
        dst = os.path.join(a.cache, f"{leg}.json")
        if not os.path.exists(dst):
            p = hf_hub_download(DS, rf, repo_type="dataset", token=tok)
            with open(p, "rb") as fh:
                blob = fh.read()
            with open(dst, "wb") as fh:
                fh.write(blob)
        blob = json.load(open(dst, encoding="utf-8"))
        if a.dump_shape:
            print(f"== {leg} top-level: {type(blob).__name__} "
                  f"{list(blob)[:12] if isinstance(blob, dict) else len(blob)}")
            k0 = (list(blob)[0] if isinstance(blob, dict) else 0)
            print(f"   first entry key={k0!r} -> "
                  f"{json.dumps(blob[k0] if isinstance(blob, dict) else blob[0])[:900]}")
            continue

        # -- normalise to {clip_id: record} ---------------------------------
        # ⚠️ The banked prod shape is `{engine, api, concepts, frame_stride,
        # min_score, n_clips, per_concept_hits_total, _note, clips: [record]}`.
        # An earlier version of this function fell through all three branches
        # and produced `clips 0 · detections 0` while exiting 0 — the same
        # "empty result that reads like success" the v2 bridge once had. The
        # assert below is what makes that impossible to repeat.
        key = next((k for k in ("clips", "records", "results")
                    if isinstance(blob, dict) and isinstance(blob.get(k), list)),
                   None)
        if key:
            recs = {r["clip_id"]: r for r in blob[key]}
            meta = {k: v for k, v in blob.items() if k != key}
        elif isinstance(blob, list):
            recs = {r["clip_id"]: r for r in blob}
            meta = {}
        else:                                   # {clip_id: record}
            recs = {k: v for k, v in blob.items() if isinstance(v, dict)
                    and ("frames" in v or "clip_id" in v)}
            meta = {k: v for k, v in blob.items() if k not in recs}
        assert recs, f"{leg}: normalisation produced ZERO records from {rf}"

        idx, per_concept, n_err = [], collections.Counter(), 0
        clips_with = collections.defaultdict(set)
        for cid, r in sorted(recs.items()):
            scores, ce = [], 0
            pc = collections.Counter()
            for _fk, _j, d in iter_dets(r):
                if "error" in d:
                    ce += 1
                elif "score" in d:
                    scores.append(float(d["score"]))
                    pc[d.get("concept")] += 1
                    clips_with[d.get("concept")].add(cid)
            n_err += ce
            per_concept.update(pc)
            idx.append({
                "clip_id": cid,
                "n_frames_run": int(r.get("n_frames_run") or
                                    len(r.get("frames") or {})),
                "n_det_seen": len(scores),
                "n_err": ce,
                "per_concept": dict(pc),
                "min_score": round(min(scores), 4) if scores else None,
                "max_score": round(max(scores), 4) if scores else None,
                "frame_wh": r.get("frame_wh")})

        allsc = [c["min_score"] for c in idx if c["min_score"] is not None]
        out = {
            "leg": leg, "repo": DS, "rfile": rf,
            "evidence_class": "MEASURED (this pull; primary = the banked "
                              "record on HF, not a summary)",
            "n_clips": len(idx),
            "n_detections": int(sum(per_concept.values())),
            "n_errors": n_err,
            "corpus_min_score": round(min(allsc), 4) if allsc else None,
            "per_concept_detections": dict(per_concept.most_common()),
            "per_concept_clips": {k: len(v) for k, v in
                                  sorted(clips_with.items(),
                                         key=lambda kv: -len(kv[1]))},
            "record_meta": meta,
            "clips": idx}
        o = os.path.join(a.out_dir, f"records_index_{leg}.json")
        json.dump(out, open(o, "w", encoding="utf-8"), indent=1)
        print(f"[{leg}] clips {out['n_clips']} · detections "
              f"{out['n_detections']} · errors {n_err} · MIN SCORE "
              f"{out['corpus_min_score']}")
        for c, n in out["per_concept_detections"].items():
            print(f"    {str(c):<16} {n:>6}  clips "
                  f"{out['per_concept_clips'].get(c)}")
    print("PULL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
