"""Far-side census of the schema-v2 corpus, run from the DEV BOX.

⛔ WHY IT EXISTS SEPARATELY FROM THE RUN'S OWN CENSUS. The run agent's report is
not evidence about the run: on 2026-08-16 a run agent said *"the main exec has
completed"* and the corpus it pronounced complete held ZERO detections (C77).
The completion criterion has to be evaluated by something that did not produce
the data, from the far side, by CONTENT.

It checks four things the file listing cannot:
  1. detections exist, per concept, agent AND scene channels;
  2. the error census is empty;
  3. the road/sky positive control fired on every clip — and the zero-detection
     clips are SPLIT into "empty scene, control live" vs "dead control";
  4. ⭐ every record is really schema v2 at `confidence_threshold=0.25`. This is
     the one a v1-era census would miss entirely: a 0.5 record is present,
     non-empty, error-free and live while being the wrong record, because a
     detection floor is invisible in the payload — it shows up only as rows
     that are not there.

It also re-derives the CONTOUR fidelity from banked data (polygon area vs the
RLE mask it summarises), so the lossiness claim in the report is checkable
without a GPU.

usage:  python hf_v2_census.py [--prefix sam3_backfill_v2/] [--out FILE]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import statistics
import sys

REPO_ROOT = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
DS = "Sayood/tanitad-ph0-aug120"


def token() -> str:
    """Read IN PLACE from the git-ignored Keys.txt. Never printed, never argv."""
    with open(os.path.join(REPO_ROOT, "Keys.txt"), encoding="utf-8",
              errors="replace") as fh:
        m = re.findall(r"hf_[A-Za-z0-9]+", fh.read())
    if not m:
        raise SystemExit("no HF token in Keys.txt")
    return max(m, key=len)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("hf_v2_census")
    ap.add_argument("--prefix", default="sam3_backfill_v2/")
    ap.add_argument("--schema", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true",
                    help="list only — file count, no record download")
    a = ap.parse_args(argv)

    try:                                   # the dev box sits behind a TLS proxy
        import truststore
        truststore.inject_into_ssl()
    except Exception:
        pass
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=token())
    info = api.dataset_info(DS, files_metadata=True)
    far = {f.rfilename: f.size for f in info.siblings
           if f.rfilename.startswith(a.prefix)}
    recs = sorted(rf for rf in far
                  if rf.endswith(".json") and "/_runs/" not in rf)
    print(f"[far] {len(recs)} records + "
          f"{len(far) - len(recs)} manifests under {a.prefix}")
    if a.quick:
        return 0

    import collections
    per, sper, errs = (collections.Counter(), collections.Counter(),
                       collections.Counter())
    n_det = n_scene = n_live = n_dead = n_nocontrol = 0
    n_bad_schema = n_bad_conf = n_zero_bytes = 0
    zero_live, zero_dead, cerr, lane_bound, lane_frames = [], [], [], 0, 0
    n_contours = n_multiloop = n_obb = 0
    frames_total = 0
    for i, rf in enumerate(recs):
        if far.get(rf) == 0:
            n_zero_bytes += 1
            continue
        rec = json.load(open(hf_hub_download(DS, rf, repo_type="dataset",
                                             token=token(),
                                             force_download=True),
                             encoding="utf-8"))
        cid = rf[len(a.prefix):-len(".json")]
        if rec.get("clip_id") != cid:
            raise SystemExit(f"{rf} carries clip_id={rec.get('clip_id')!r}")
        nd = int(rec.get("n_det_total") or 0)
        n_det += nd
        n_scene += int(rec.get("n_scene_det_total") or 0)
        frames_total += int(rec.get("n_frames_run") or 0)
        per.update({k: int(v) for k, v in
                    (rec.get("per_concept_hits") or {}).items()})
        sper.update({k: int(v) for k, v in
                     (rec.get("per_scene_hits") or {}).items()})
        errs.update({k: int(v) for k, v in (rec.get("err_kinds") or {}).items()})
        n_bad_schema += int(int(rec.get("schema_version") or 0) < a.schema)
        got = (rec.get("engine") or {}).get("confidence_threshold")
        n_bad_conf += int(got is None or abs(float(got) - a.conf) > 1e-9)
        lv = rec.get("liveness")
        counts = (lv or {}).get("n_det") or {}
        if lv is None:
            n_nocontrol += 1
            alive = False
        else:
            alive = any(int(v) > 0 for v in counts.values())
            n_live += int(alive)
            n_dead += int(not alive)
        if nd == 0:
            (zero_live if alive else zero_dead).append(
                {"clip_id": cid, "liveness_n_det": counts,
                 "n_scene_det_total": rec.get("n_scene_det_total")})
        for f in (rec.get("frames") or {}).values():
            for key in ("det", "scene"):
                for d in f.get(key) or []:
                    ar, ca = d.get("mask_area_px"), d.get("contour_area_px")
                    if ar and ca is not None:
                        n_contours += 1
                        cerr.append((ca - ar) / ar)
                        n_multiloop += int((d.get("contour_n_loops") or 1) > 1)
                    n_obb += int("obb_cxcylwa" in d)
        for v in ((rec.get("ego_lane") or {}).get("frames") or {}).values():
            lane_frames += 1
            lane_bound += int(v.get("lane_idx_est") is not None)
        if (i + 1) % 25 == 0:
            print(f"[far] {i+1}/{len(recs)} read", flush=True)

    def q(v, p):
        return round(sorted(v)[int(p * (len(v) - 1))], 5) if v else None

    out = {
        "class": "MEASURED", "repo": DS, "prefix": a.prefix,
        "n_records": len(recs), "n_zero_byte": n_zero_bytes,
        "n_frames_run_total": frames_total,
        "n_det_total": n_det, "n_scene_det_total": n_scene,
        "per_concept_totals": dict(per.most_common()),
        "per_scene_totals": dict(sper.most_common()),
        "error_census": dict(errs.most_common()),
        "liveness": {"live": n_live, "dead": n_dead,
                     "no_control_in_record": n_nocontrol},
        "zero_split": {"empty_scene_control_live": len(zero_live),
                       "dead_control": len(zero_dead)},
        "zero_live_clips": zero_live, "zero_dead_clips": zero_dead,
        "schema": {"require": a.schema, "wrong": n_bad_schema},
        "conf": {"require": a.conf, "wrong": n_bad_conf},
        "contours": {
            "n": n_contours, "n_with_obb": n_obb, "n_multiloop": n_multiloop,
            "area_err_p10": q(cerr, 0.10), "area_err_med": q(cerr, 0.50),
            "area_err_p90": q(cerr, 0.90),
            "area_err_mean_abs": round(statistics.mean(
                [abs(e) for e in cerr]), 5) if cerr else None,
            "area_err_max_abs": round(max((abs(e) for e in cerr), default=0),
                                      5)},
        "ego_lane": {"frames": lane_frames, "bounded_both_sides": lane_bound},
    }
    # ⛔ THE VERDICT IS COMPUTED, NEVER READ FROM A STORED FLAG.
    out["PASS"] = bool(n_det > 0 and not errs and n_dead == 0
                       and n_nocontrol == 0 and n_zero_bytes == 0
                       and n_bad_schema == 0 and n_bad_conf == 0)
    txt = json.dumps(out, indent=1)
    if a.out:
        with io.open(a.out, "w", encoding="utf-8") as fh:
            fh.write(txt)
    print(txt)
    return 0 if out["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
