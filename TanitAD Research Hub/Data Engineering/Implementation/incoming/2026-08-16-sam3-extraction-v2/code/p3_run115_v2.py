"""STEP 3 — re-run ALL 115 aug120 clips with schema v2 at confidence 0.25.

⛔ ALL 115, NOT THE 32 RESIDUAL. 83 of them are banked under the v1 prefix at
`confidence_threshold=0.5`. A corpus that mixes two detection floors is
unattributable in a way NOTHING downstream can detect — the floor shows up only
as rows that are not there — so every per-concept number, every precision
figure and every "the scene channel found N curbs" would silently span two
populations. Re-detecting all 115 costs ~57 GPU-min (MEASURED, p2); getting the
attribution wrong costs the study that uses it.

⛔ AND IT BANKS TO ITS OWN PREFIX. `sam3_backfill_v2/` — see
`s2_lab_lib.BACKFILL_V2_PREFIX`. Overwriting v1 in place would make the corpus
MIXED for the whole length of the run, and free-Colab reclaimed the T4 three
times during the last pass, so "the whole length of the run" is not a
hypothetical window.

COMPLETION IS BY CONTENT, NEVER FILE COUNT (C77) — and the predicate is
stricter than v1's, because a v1 record is present, non-empty, error-free and
live while still being the WRONG record:
    liveness control present  AND  zero error entries
    AND schema_version >= 2   AND  engine.confidence_threshold == 0.25

Env: S2_N caps clips this invocation (resumable), S2_BATCH shards per pull.
"""
import json
import os
import sys
import time

os.chdir("/content")
for p in ("/content/repo/colab", "/content/repo/stack",
          "/content/repo/stack/scripts"):
    if p not in sys.path:
        sys.path.insert(0, p)
from pathlib import Path                                         # noqa: E402
import s2_lab_lib as L                                           # noqa: E402
import ph0_sam3                                                  # noqa: E402

N_LIMIT = int(os.environ.get("S2_N", "0")) or None
BATCH = int(os.environ.get("S2_BATCH", "12"))
FRAME_STRIDE = 8
CONF = 0.25
ROOT = Path("/content/repo")
WORK = Path("/content/bf2")
WORK.mkdir(parents=True, exist_ok=True)
OUT = Path("/content/out")
OUT.mkdir(exist_ok=True)
REPO, PREFIX = L.DS_LABELS, L.BACKFILL_V2_PREFIX
SCENE = ph0_sam3.SCENE_CONCEPTS
print(f"[cfg] N={N_LIMIT} BATCH={BATCH} stride={FRAME_STRIDE} conf={CONF} "
      f"schema=v{ph0_sam3.SCHEMA_VERSION} scene={SCENE} -> {REPO}/{PREFIX}")

api = L.hf_api()
L.ensure_repo(api, REPO)

gap = L.derive_sam3_gap(api)
L.cross_check_gap(api, gap, partial=False)
L.check_gap_fixture(gap, ROOT)
todo_all = gap["absent"]
print(f"[gap] SAM3-absent clips: {len(todo_all)} of "
      f"{gap['n_records_checked']} records checked")

t_scan = time.time()
cen0 = L.content_census(api, REPO, PREFIX, require_schema=2, require_conf=CONF)
done = set(cen0["complete_clips"])
print(f"[resume] far side {cen0['n_records']} records in {time.time()-t_scan:.0f}s "
      f"· det {cen0['n_det_total']} · scene {cen0['n_scene_det_total']} "
      f"· wrong_schema {cen0['wrong_schema']} · wrong_conf {cen0['wrong_conf']} "
      f"· errors {sum(cen0['error_census'].values())} -> complete {len(done)}")
todo = [c for c in todo_all if c not in done]
if N_LIMIT:
    todo = todo[:N_LIMIT]
print(f"[resume] this run: {len(todo)} clips")

if not todo:
    print("V2_NOTHING_TODO")
else:
    v2_by = L.load_v2_records(api, set(todo))
    loc = L.w120_locations(api)
    miss = [c for c in todo if c not in loc]
    assert not miss, f"{len(miss)} gap clips lack w120 shards: {miss[:3]}"
    REC_PQ = str(WORK / "records.parquet")
    if not Path(REC_PQ).exists():
        import shutil
        shutil.copyfile(L.hf_download(L.DS_ALP, "records.parquet"), REC_PQ)
    print(f"[assets] w120 shards located for all {len(todo)} clips")

    for _n in ("proc", "_proc"):
        if _n in globals():
            L.free_leg(globals().pop(_n))       # one processor per kernel
    proc, meta = ph0_sam3.build_processor(None, conf_threshold=CONF)
    assert meta["dtype_fix"]["applied"], "C77 dtype fix did NOT install"
    assert meta["confidence_threshold"] == CONF, meta
    assert meta["schema_version"] >= 2, meta
    print(f"[sam3] up · conf={meta['confidence_threshold']} via "
          f"{meta['confidence_threshold_set_via']}")
    L.gpu_mem_report("sam3 load")

    import ph0_pilot
    import shutil
    t0, n_banked, n_live, n_det, n_scene, n_bytes = time.time(), 0, 0, 0, 0, 0
    for b0 in range(0, len(todo), BATCH):
        batch = todo[b0:b0 + BATCH]
        bwork = WORK / f"b{b0:05d}"
        L.bridge_batch(batch, loc, REC_PQ, bwork)
        for cid in batch:
            frames = ph0_pilot.sample_clip_frames(
                str(bwork / "videos" / f"{cid}.mp4"), t0_s=8.0)[0]
            rec = L.sam3_leg(proc, frames, v2_by[cid],
                             frame_stride=FRAME_STRIDE, scene_concepts=SCENE,
                             contours=True, meta=meta)
            rec["_n_explicit"] = len(batch)
            # ⚠️ compact: 64 % of an indent=1 v2 record is whitespace (p2)
            sz = L.bank_json(api, REPO, f"{PREFIX}{cid}.json", rec, indent=None)
            n_banked += 1
            n_bytes += sz
            lv = rec.get("liveness") or {}
            n_live += int(ph0_sam3.is_live(lv))
            n_det += int(rec.get("n_det_total") or 0)
            n_scene += int(rec.get("n_scene_det_total") or 0)
            hits = ",".join(f"{k}:{v}"
                            for k, v in rec["per_concept_hits"].items() if v)
            sh = ",".join(f"{k}:{v}"
                          for k, v in rec["per_scene_hits"].items() if v)
            print(f"[bank] {n_banked}/{len(todo)} {cid[:8]} {sz}B "
                  f"det={rec['n_det_total']} scene={rec['n_scene_det_total']} "
                  f"err={rec.get('n_err_total')} "
                  f"live={ph0_sam3.is_live(lv)}{lv.get('n_det')} "
                  f"[{hits or 'none'}] [{sh or 'none'}]", flush=True)
        L.gpu_mem_report(f"after batch b{b0:05d}")
        shutil.rmtree(bwork, ignore_errors=True)
    print(f"BANKED {n_banked} clips in {time.time()-t0:.0f}s | det={n_det} "
          f"scene={n_scene} live={n_live}/{n_banked} bytes={n_bytes}")

cen = L.content_census(api, REPO, PREFIX, want=set(todo_all),
                       require_schema=2, require_conf=CONF)
zero = cen["zero_det_clips"]
cen_small = {k: v for k, v in cen.items()
             if k not in ("zero_det_clips", "complete_clips", "missing",
                          "extra")}
cen_small["zero_split"] = {
    "empty_scene_control_live": sum(1 for z in zero if z["liveness_live"]),
    "dead_control": sum(1 for z in zero if not z["liveness_live"])}
cen_small["n_missing"] = len(cen.get("missing") or [])
print("[census]", json.dumps(cen_small, indent=1))
json.dump({"census": cen, "small": cen_small},
          open(OUT / "p3_census.json", "w"), indent=1)
json.dump({"residual": sorted(cen.get("missing") or []),
           "n": len(cen.get("missing") or []),
           "resume": "colab exec -s tanitad-sam3v2 -f p3_run115_v2.py"},
          open(OUT / "p3_residual.json", "w"), indent=1)
L.run_manifest(api, REPO, PREFIX, "sam3-v2", {
    "schema_version": ph0_sam3.SCHEMA_VERSION, "conf_threshold": CONF,
    "scene_concepts": SCENE, "frame_stride": FRAME_STRIDE,
    "contour_tol_px": ph0_sam3.CONTOUR_TOL_PX_DEFAULT,
    "contour_max_pts": ph0_sam3.CONTOUR_MAX_PTS_DEFAULT,
    "completion_rule": "liveness control present AND zero errors AND "
                       "schema>=2 AND engine.confidence_threshold==0.25 "
                       "— never a file count (C77)",
    "census": cen_small, "evidence_class": "MEASURED"})
print("V2_PASS" if cen["pass_"] else "V2_INCOMPLETE")
print("V2_DONE")
