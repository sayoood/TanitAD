"""STEP 2 - re-run the SAM3 leg for the 115 aug120 gap clips, with the C77 fix.

Same library, same gap derivation, same per-clip far-side-verified banking as
`colab/SAM3_BACKFILL_115.ipynb` (cells 3-9), with TWO deliberate differences,
both forced by C77:

 1. RESUME IS CONTENT-AWARE. `done_set` counts a clip done when a non-empty
    file EXISTS. The far side currently holds 115 such files whose payload is
    an error census, so a stem-based resume would skip the entire job. A clip
    counts done here only when its banked record has `n_det_total > 0` AND a
    live road/sky control.
 2. THE COMPLETION CRITERION IS THE CENSUS, not the file count: detections,
    per-concept totals, error strings, liveness alarms.

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
ROOT = Path("/content/repo")
WORK = Path("/content/backfill")
WORK.mkdir(parents=True, exist_ok=True)
BANK_REPO, BANK_PREFIX = L.DS_LABELS, L.BACKFILL_PREFIX
print(f"[cfg] N_LIMIT={N_LIMIT} BATCH={BATCH} stride={FRAME_STRIDE} "
      f"-> {BANK_REPO}/{BANK_PREFIX}")

api = L.hf_api()
L.ensure_repo(api, BANK_REPO)

# ---- the gap, from the RECORDS (C18), cross-checked + fixture-diffed --------
gap = L.derive_sam3_gap(api)
L.cross_check_gap(api, gap, partial=False)
L.check_gap_fixture(gap, ROOT)
todo_all = gap["absent"]
print(f"[gap] SAM3-absent clips: {len(todo_all)} of "
      f"{gap['n_records_checked']} records checked")


def content_done(api) -> tuple[set, dict]:
    """Clips the FIXED engine has already produced (C77 resume predicate).

    !! NOT presence: on 2026-08-16 all 115 clips had a non-empty file whose
       whole payload was `RuntimeError: mat1 and mat2 ... BFloat16 and Float`,
       so a stem-based resume skips the job forever.
    !! NOT `n_det_total > 0` either: a clip whose scene is LEGITIMATELY EMPTY
       would never be marked done and would be re-run every session. Its zero
       is the right answer.
    A clip is complete when its record carries the road/sky liveness control
    AND holds zero error entries. Presence of the control is exactly what
    distinguishes a repaired record from a stale one.
    """
    far = L.list_far(api, BANK_REPO, BANK_PREFIX)
    stems = [rf for rf in far
             if rf.endswith(".json") and "/_runs/" not in rf]
    done, stat = set(), {"present": len(stems), "complete": 0,
                         "no_control": 0, "with_errors": 0,
                         "complete_but_no_objects": 0, "read_err": 0}
    for rf in stems:
        cid = rf[len(BANK_PREFIX):-len(".json")]
        try:
            rec = json.load(open(L.hf_download(BANK_REPO, rf, force=True)))
        except Exception:
            stat["read_err"] += 1
            continue
        lv = rec.get("liveness")
        nerr = int(rec.get("n_err_total") or 0)
        if not nerr and not rec.get("err_kinds"):
            for f in (rec.get("frames") or {}).values():
                nerr += sum(1 for d in f.get("det", []) if "error" in d)
        if lv is None:
            stat["no_control"] += 1
        elif nerr:
            stat["with_errors"] += 1
        else:
            done.add(cid)
            stat["complete"] += 1
            if not int(rec.get("n_det_total") or 0):
                stat["complete_but_no_objects"] += 1
    return done, stat


t_scan = time.time()
done, scan = content_done(api)
print(f"[resume] far-side scan in {time.time()-t_scan:.0f}s: {scan}")
if os.environ.get("S2_FORCE") == "1":
    # MEASURED 2026-08-16: the SAME code on the SAME clip gives 60 vs 64
    # detections on two different Colab T4 VMs (in-session it is
    # bit-deterministic). A corpus assembled across sessions therefore carries
    # VM-level noise on every per-concept count, so the 115 are re-run in ONE
    # session rather than resumed across two.
    print(f"[resume] S2_FORCE=1 -> re-running ALL {len(todo_all)} in this "
          "session (cross-VM nondeterminism, see eq3.json)")
    done = set()
todo = [c for c in todo_all if c not in done]
if N_LIMIT:
    todo = todo[:N_LIMIT]
print(f"[resume] content-complete {len(done)} -> this run: {len(todo)} clips")

if not todo:
    print("BACKFILL_NOTHING_TODO")
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

    proc, meta = L.load_sam3()
    assert meta["dtype_fix"]["applied"], "C77 dtype fix did NOT install"
    L.gpu_mem_report("sam3 load")

    import shutil
    t_start, n_banked, n_live, n_det_run = time.time(), 0, 0, 0
    for b0 in range(0, len(todo), BATCH):
        batch = todo[b0:b0 + BATCH]
        bwork = WORK / f"b{b0:05d}"
        L.bridge_batch(batch, loc, REC_PQ, bwork)
        import ph0_pilot
        for cid in batch:
            frames = ph0_pilot.sample_clip_frames(
                str(bwork / "videos" / f"{cid}.mp4"), t0_s=8.0)[0]
            rec = L.sam3_leg(proc, frames, v2_by[cid],
                             frame_stride=FRAME_STRIDE)
            rec["_n_explicit"] = len(batch)
            sz = L.bank_json(api, BANK_REPO, f"{BANK_PREFIX}{cid}.json", rec)
            n_banked += 1
            lv = rec.get("liveness") or {}
            n_live += int(ph0_sam3.is_live(lv))   # never the stored flag
            n_det_run += int(rec.get("n_det_total") or 0)
            hits = ",".join(f"{k}:{v}"
                            for k, v in rec["per_concept_hits"].items() if v)
            print(f"[bank] {n_banked}/{len(todo)} {cid[:8]} {sz}B "
                  f"det={rec['n_det_total']} err={rec.get('n_err_total')} "
                  f"live={ph0_sam3.is_live(lv)}{lv.get('n_det')} [{hits or 'none'}]",
                  flush=True)
        L.gpu_mem_report(f"after batch b{b0:05d}")
        shutil.rmtree(bwork, ignore_errors=True)
    print(f"BANKED {n_banked} clips in {time.time()-t_start:.0f}s | "
          f"det={n_det_run} live={n_live}/{n_banked}")

print("BACKFILL_DONE")
