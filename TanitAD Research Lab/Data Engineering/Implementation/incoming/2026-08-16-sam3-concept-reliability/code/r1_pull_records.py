"""STEP 1 — pull every banked SAM3 record to a local cache, and answer the
question that decides whether a threshold study needs a GPU at all:

  ⛔ **ARE SUB-THRESHOLD DETECTIONS RETAINED IN THE RECORD, OR WERE THEY
     DISCARDED AT INFERENCE?**

`ph0_sam3.detect/_score` filters with `min_score`, which defaults to **0.0**
("a threshold picked before the score distribution is known is a decision
dressed as a default"). That is OUR filter and it is off. The vendor's filter is
a different object: `Sam3Processor.__init__(..., confidence_threshold=0.5)`,
applied INSIDE `set_text_prompt` before it returns. `build_processor` constructs
`Sam3Processor(model)` and passes nothing, so the vendor default is in force.

⇒ The empirical test is the **minimum banked score**. If no detection anywhere in
2 496 carries a score below 0.5, the sub-threshold tail was never written down
and a sweep DOWNWARD needs re-detection. A sweep UPWARD is pure re-analysis.

Writes `raw/records_index.json` (per-clip summary, no payload) and caches the
full records under the scratchpad so later steps never re-download.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import truststore

truststore.inject_into_ssl()
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
KEYS = os.path.join(REPO, "Keys.txt")
DS = "Sayood/tanitad-ph0-aug120"
PREFIX = "sam3_backfill/"
CACHE = (r"C:\Users\Admin\AppData\Local\Temp\claude"
         r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
         r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad\sam3rel\records")


def token() -> str:
    """⚠️ Read in place, never printed and never passed in argv."""
    return re.search(r"hf_[A-Za-z0-9]+",
                     open(KEYS, encoding="utf-8", errors="replace").read()
                     ).group(0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("r1_pull_records")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=CACHE)
    a = ap.parse_args(argv)

    from huggingface_hub import HfApi, hf_hub_download
    tok = token()
    api = HfApi(token=tok)
    far = {f.rfilename: f.size
           for f in api.dataset_info(DS, files_metadata=True).siblings}
    rfs = sorted(rf for rf in far if rf.startswith(PREFIX)
                 and rf.endswith(".json") and "/_runs/" not in rf)
    print(f"[pull] {len(rfs)} clip records on the far side", flush=True)
    os.makedirs(a.cache, exist_ok=True)

    idx = []
    for i, rf in enumerate(rfs):
        cid = rf[len(PREFIX):-len(".json")]
        dst = os.path.join(a.cache, f"{cid}.json")
        if not os.path.exists(dst):
            p = hf_hub_download(DS, rf, repo_type="dataset", token=tok)
            with open(p, encoding="utf-8") as fh:
                blob = fh.read()
            with open(dst, "w", encoding="utf-8") as fh:
                fh.write(blob)
        rec = json.load(open(dst, encoding="utf-8"))
        assert rec.get("clip_id") == cid, f"{rf} carries {rec.get('clip_id')}"
        n_err = 0
        scores = []
        for f in (rec.get("frames") or {}).values():
            for d in f.get("det", []):
                if "error" in d:
                    n_err += 1
                elif "score" in d:
                    scores.append(float(d["score"]))
        idx.append({
            "clip_id": cid,
            "n_frames_run": int(rec.get("n_frames_run") or 0),
            "n_det_total": int(rec.get("n_det_total") or 0),
            "n_det_seen": len(scores),
            "n_err": n_err,
            "has_liveness": rec.get("liveness") is not None,
            "liveness_n_det": ((rec.get("liveness") or {}).get("n_det") or {}),
            "per_concept_hits": rec.get("per_concept_hits") or {},
            "min_score": round(min(scores), 4) if scores else None,
            "max_score": round(max(scores), 4) if scores else None,
            "frame_wh": rec.get("frame_wh"),
            "size_bytes": far[rf]})
        if (i + 1) % 25 == 0:
            print(f"[pull] {i+1}/{len(rfs)}", flush=True)

    live = [r for r in idx if r["has_liveness"]]
    allsc = [r["min_score"] for r in live if r["min_score"] is not None]
    out = {
        "repo": DS, "prefix": PREFIX,
        "n_records": len(idx),
        "n_complete_fixed_engine": len(live),
        "n_stale_c77": len(idx) - len(live),
        "n_det_total_complete": sum(r["n_det_total"] for r in live),
        "n_err_total_complete": sum(r["n_err"] for r in live),
        "corpus_min_score": round(min(allsc), 4) if allsc else None,
        "clips": idx}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(f"[pull] complete(fixed-engine) {out['n_complete_fixed_engine']} · "
          f"stale(C77) {out['n_stale_c77']} · det "
          f"{out['n_det_total_complete']} · err "
          f"{out['n_err_total_complete']}", flush=True)
    print(f"[pull] ⛔ CORPUS MINIMUM BANKED SCORE = {out['corpus_min_score']}",
          flush=True)
    print("PULL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
