"""FAR-SIDE census of the SAM3 backfill — from the dev box, independent of the
run's own report.

⛔ THIS IS THE C77 COMPLETION CRITERION, and it is deliberately not a file
count. The retracted verification enumerated CONTAINERS (records present,
zero-byte scan, clip_id == filename) and never evaluated the quantity the
artifact exists to produce. This script answers, corpus-wide:

  n_det_total > 0 ?  ·  per-concept totals  ·  ERROR-STRING census  ·
  clips with zero detections  ·  and for each of those, whether the LIVENESS
  control (road/sky) also read zero — a real crash — or only the agent
  concepts did, which is a legitimately empty scene.

Usage:  python hf_census.py [--out census.json]
"""
import argparse
import collections
import json
import os
import re
import sys

import truststore

truststore.inject_into_ssl()
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

KEYS = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\Keys.txt"
REPO = "Sayood/tanitad-ph0-aug120"
PREFIX = "sam3_backfill/"
FIXTURE = (r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\colab"
           r"\fixtures\sam3_backfill_expected.json")


def token() -> str:
    return re.search(r"hf_[A-Za-z0-9]+",
                     open(KEYS, encoding="utf-8", errors="replace").read()
                     ).group(0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("hf_census")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    from huggingface_hub import HfApi, hf_hub_download
    tok = token()
    api = HfApi(token=tok)
    info = api.dataset_info(REPO, files_metadata=True)
    far = {f.rfilename: f.size for f in info.siblings
           if f.rfilename.startswith(PREFIX)}
    runs = sorted(rf for rf in far if "/_runs/" in rf)
    clips_far = sorted(rf for rf in far
                       if rf.endswith(".json") and "/_runs/" not in rf)
    want = set(json.load(open(FIXTURE, encoding="utf-8"))["clips"])
    print(f"[far] files {len(far)} · clip records {len(clips_far)} · "
          f"run manifests {len(runs)} · zero-byte "
          f"{sum(1 for rf in clips_far if far[rf] == 0)}")

    per_concept = collections.Counter()
    errs = collections.Counter()
    live_cnt = collections.Counter()
    n_det = 0
    zero_clips, dead_clips, no_liveness, partial = [], [], [], []
    frames_run = 0
    seen = set()
    for i, rf in enumerate(clips_far):
        cid = rf[len(PREFIX):-len(".json")]
        seen.add(cid)
        rec = json.load(open(hf_hub_download(REPO, rf, repo_type="dataset",
                                             token=tok, force_download=True),
                             encoding="utf-8"))
        assert rec.get("clip_id") == cid, f"{rf} carries {rec.get('clip_id')}"
        nd = int(rec.get("n_det_total") or 0)
        n_det += nd
        frames_run += int(rec.get("n_frames_run") or 0)
        for k, v in (rec.get("per_concept_hits") or {}).items():
            per_concept[k] += int(v)
        # the error census: from the summary if present, else from the payload
        if rec.get("err_kinds"):
            for k, v in rec["err_kinds"].items():
                errs[k] += int(v)
        else:
            for f in (rec.get("frames") or {}).values():
                for d in f.get("det", []):
                    if "error" in d:
                        errs[str(d["error"])[:60]] += 1
        # ⛔ RECOMPUTED FROM `n_det`, never read from the stored `live` flag:
        # that flag is DERIVED and its rule changed once (all -> any, after an
        # underpass returned `road 2 · sky 0` on a working engine). The counts
        # are banked, so the derivation does not have to be trusted.
        lv = rec.get("liveness")
        counts = (lv or {}).get("n_det") or {}
        alive = any(int(v) > 0 for v in counts.values())
        if lv is None:
            no_liveness.append(cid)
        else:
            live_cnt[alive] += 1
            if not alive:
                dead_clips.append((cid, counts))
            elif not all(int(v) > 0 for v in counts.values()):
                partial.append((cid, counts))
        if nd == 0:
            zero_clips.append((cid, alive))
        if (i + 1) % 25 == 0:
            print(f"[far] {i+1}/{len(clips_far)} read", flush=True)

    out = {
        "repo": REPO, "prefix": PREFIX,
        "n_records": len(clips_far), "n_run_manifests": len(runs),
        "fixture_coverage": f"{len(seen & want)}/{len(want)}",
        "missing_vs_fixture": sorted(want - seen),
        "extra_not_in_fixture": sorted(seen - want),
        "n_frames_run_total": frames_run,
        "n_det_total": n_det,
        "per_concept_totals": dict(per_concept.most_common()),
        "error_census": dict(errs.most_common()),
        "clips_with_zero_det": len(zero_clips),
        "zero_det_clips": [{"clip_id": c, "liveness_live": lv}
                           for c, lv in zero_clips],
        "liveness": {"live_any": live_cnt[True],
                     "not_live": live_cnt[False],
                     "partial_one_control_occluded": len(partial),
                     "no_control_in_record": len(no_liveness)},
        "partial_control_clips": [{"clip_id": c, "n_det": n}
                                  for c, n in partial],
        "not_live_clips": [{"clip_id": c, "n_det": n} for c, n in dead_clips],
        "PASS": bool(n_det > 0 and not errs and live_cnt[False] == 0
                     and not no_liveness and len(seen & want) == len(want)),
    }
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("zero_det_clips", "not_live_clips",
                                   "missing_vs_fixture")}, indent=1))
    print("zero-detection clips:", out["clips_with_zero_det"],
          "of which liveness-live (legitimately empty scene):",
          sum(1 for z in out["zero_det_clips"] if z["liveness_live"]))
    print("PASS" if out["PASS"] else "FAIL")
    if a.out:
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
        print("wrote", a.out)
    return 0 if out["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
