"""STEP 1 (VM) — re-detect one CHUNK of the 86 residual clips at schema v2 / floor 0.25.

⛔ SAME ENGINE, SAME CALL, SAME PARAMETERS AS THE 115. This script exists to move
the 86 onto the v2 leg, not to build a third one. Every knob below is copied from
`…/2026-08-16-sam3-extraction-v2/code/p3_run115_v2.py` and must stay copied:

    FRAME_STRIDE = 8 · CONF = 0.25 · scene_concepts = ph0_sam3.SCENE_CONCEPTS
    contours = True · liveness = True (inside sam3_leg) · bank compact

A second extraction path would re-create exactly the heterogeneity this run is
removing — and it would be INVISIBLE, because two records built by two paths are
structurally identical and differ only in what is absent. That is the same reason
the floor had to be stamped in the first place.

⛔ AND IT DOES NOT PUSH TO HUGGINGFACE. p3 banked per clip to HF, which is both the
durability and the resume mechanism. A push to a public-facing platform needs the
PI's authorisation, which this task does not carry, so the durable bank is the DEV
BOX: this script writes to /content/out86 and the driver pulls the whole directory
after every chunk. HF is READ here (video shards, v2 labels, records.parquet) and
never written.

⚠️ THE VM IS NOT THE RESUME AUTHORITY. Free-Colab reclaims the T4 without warning
and /content goes with it. The driver (`f2_drive86.py`) computes the todo list from
the DEV BOX copy by content and ships it in `/content/todo86.json`; this script runs
what it is told. That way a reclaim costs one chunk, not the run.

⭐ THE HOST IS AN ENV VAR, NOT A HARD-CODED PATH (2026-08-17). Acquisition and
extraction are separable: this file is the EXTRACTION and knows nothing about how
its GPU was obtained. Defaults are the Colab values, so the VM path is unchanged.

    F86_ROOT   cwd + base for the defaults below   (default /content)
               ⚠️ must contain bpe_simple_vocab_16e6.txt.gz — find_bpe() globs cwd
    F86_REPO   holds colab/ + stack/               (default /content/repo)
    F86_WORK   scratch for bridged batches         (default $F86_ROOT/bf86)
    F86_OUT    the record bank                     (default $F86_ROOT/out86)
    F86_TODO   clip list to run                    (default $F86_ROOT/todo86.json)
               or F86_TODO_INLINE='["clip", ...]'  when there is no remote driver
    F86_TAR    tar of the bank for a remote pull   (default $F86_ROOT/out86.tgz)
               set to "" on a local GPU — the bank is already durable there

Reads : $F86_TODO  {"todo": [clip_id, ...], "chunk": int}
Writes: $F86_OUT/<clip_id>.json   (compact, schema v2, conf 0.25)
        $F86_OUT/_run_manifest.json
        $F86_TAR                  (transport only; skipped when F86_TAR is empty)
"""
import json
import os
import shutil
import sys
import tarfile
import time

# ⭐ HOST-AGNOSTIC BY ENV, COLAB BY DEFAULT (2026-08-17). Every path below used to
# be a literal `/content/...`. They are now env-overridable with the Colab values
# as defaults, so behaviour on the VM is unchanged while ANY box with a CUDA GPU
# can run the identical extraction.
#
# ⛔ THIS IS NOT A CONVENIENCE. The one thing that must never happen to this corpus
# is a SECOND EXTRACTION PATH: two runners producing records that are structurally
# identical and differ only in what is absent is precisely the heterogeneity this
# package exists to remove, and it would be invisible. A host that cannot reuse
# this file will write its own loop — so making the file reusable IS the invariant,
# not a nicety.
#
# ⚠️ `F86_ROOT` becomes the cwd because `ph0_sam3.find_bpe()` globs the CLIP BPE
# vocab RELATIVE TO THE CWD. Point it at a directory holding
# `bpe_simple_vocab_16e6.txt.gz`, or find_bpe() returns nothing and the processor
# will not build.
ROOT = os.environ.get("F86_ROOT", "/content")
REPO = os.environ.get("F86_REPO", "/content/repo")
TODO = os.environ.get("F86_TODO", f"{ROOT}/todo86.json")
TAR = os.environ.get("F86_TAR", f"{ROOT}/out86.tgz")   # "" = skip (local runs)

os.chdir(ROOT)
for p in (os.path.join(REPO, "colab"), os.path.join(REPO, "stack"),
          os.path.join(REPO, "stack", "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
from pathlib import Path                                         # noqa: E402
import s2_lab_lib as L                                           # noqa: E402
import ph0_sam3                                                  # noqa: E402
import ph0_pilot                                                 # noqa: E402

FRAME_STRIDE = 8
CONF = 0.25
BATCH = int(os.environ.get("S2_BATCH", "12"))
WORK = Path(os.environ.get("F86_WORK", f"{ROOT}/bf86"))
OUT = Path(os.environ.get("F86_OUT", f"{ROOT}/out86"))
WORK.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

# ⚠️ A LOCAL RUN NEEDS NO TODO FILE. The list exists because the Colab VM is not
# the resume authority — the dev box computes it and ships it. When OUT *is* the
# durable bank, the same predicate answers it in-process, so fall back to that
# rather than making a local caller fabricate a file.
if os.path.exists(TODO):
    spec = json.load(open(TODO))
    todo, chunk_id = list(spec["todo"]), int(spec.get("chunk", 0))
elif os.environ.get("F86_TODO_INLINE"):
    todo = json.loads(os.environ["F86_TODO_INLINE"])
    chunk_id = int(os.environ.get("F86_CHUNK", "0"))
else:
    raise SystemExit(
        f"no todo list: {TODO} absent and F86_TODO_INLINE unset. The residual is "
        "the 86 clips named in raw/floor_homogeneity_manifest.json -> residual")
print(f"[cfg] chunk={chunk_id} n_todo={len(todo)} stride={FRAME_STRIDE} "
      f"conf={CONF} schema=v{ph0_sam3.SCHEMA_VERSION} "
      f"scene={ph0_sam3.SCENE_CONCEPTS}")

# ⛔ THE GATE THAT REFUSES A DRIFTED KERNEL. Same class as the pod-checkout trap:
# the kernel persists across `colab exec`, so a stale module would silently run the
# OLD extraction and bank records that LOOK right. Assert the shipped code's
# identity before any GPU is spent.
assert ph0_sam3.SCHEMA_VERSION >= 2, "kernel holds a pre-v2 ph0_sam3"
assert hasattr(L, "content_census_local"), "kernel holds a pre-split s2_lab_lib"
assert not (set(ph0_sam3.SCENE_CONCEPTS) & set(ph0_sam3.LIVENESS_CONCEPTS))

api = L.hf_api()                     # READ ONLY — no ensure_repo, no bank_json

if not todo:
    print("F1_NOTHING_TODO")
else:
    # ---- assets (cached in /content across execs within a session) ---------- #
    v2_by = L.load_v2_records(api, set(todo))
    loc = L.w120_locations(api)
    miss = [c for c in todo if c not in loc]
    assert not miss, f"{len(miss)} clips lack w120 shards: {miss[:3]}"
    REC_PQ = str(WORK / "records.parquet")
    if not Path(REC_PQ).exists():
        shutil.copyfile(L.hf_download(L.DS_ALP, "records.parquet"), REC_PQ)
    print(f"[assets] w120 shards located for all {len(todo)} clips")

    # ---- processor: ONCE PER KERNEL, cached across chunks ------------------- #
    # p3 rebuilt it every exec because it ran the whole corpus in one call. This
    # run is chunked so the driver can pull between chunks, and a rebuild is ~30 s
    # of T4 time per chunk for no benefit.
    if "_FLOOR86_PROC" in globals():
        proc, meta = globals()["_FLOOR86_PROC"]
        print(f"[sam3] reusing cached processor · conf={meta['confidence_threshold']}")
    else:
        proc, meta = ph0_sam3.build_processor(None, conf_threshold=CONF)
        globals()["_FLOOR86_PROC"] = (proc, meta)
        print(f"[sam3] up · conf={meta['confidence_threshold']} via "
              f"{meta['confidence_threshold_set_via']}")
    assert meta["dtype_fix"]["applied"], "C77 dtype fix did NOT install"
    assert meta["confidence_threshold"] == CONF, meta
    assert meta["schema_version"] >= 2, meta
    L.gpu_mem_report("sam3 ready")

    # ⭐ BATCH BY SEGMENT, NOT BY POSITION. `bridge_batch` copies the side files
    # (_geometry.json, _v2manifest.pt) of `loc[batch[0]]`'s segment ONLY, so a
    # batch spanning two segments bridges the second one against the first one's
    # geometry. p3's todo happened to be segment-ordered; this one is a residual
    # list and is not. Grouping is free and removes the confound.
    by_seg: dict[str, list[str]] = {}
    for cid in todo:
        by_seg.setdefault(loc[cid].split("/")[0], []).append(cid)
    batches = [seg_clips[i:i + BATCH]
               for seg, seg_clips in sorted(by_seg.items())
               for i in range(0, len(seg_clips), BATCH)]
    print(f"[plan] {len(todo)} clips · {len(by_seg)} segments · "
          f"{len(batches)} batches (max {BATCH})")

    t0 = time.time()
    n_banked = n_live = n_det = n_scene = n_err = n_bytes = 0
    for bi, batch in enumerate(batches):
        bwork = WORK / f"b{bi:03d}"
        L.bridge_batch(batch, loc, REC_PQ, bwork)
        for cid in batch:
            frames = ph0_pilot.sample_clip_frames(
                str(bwork / "videos" / f"{cid}.mp4"), t0_s=8.0)[0]
            rec = L.sam3_leg(proc, frames, v2_by[cid],
                             frame_stride=FRAME_STRIDE,
                             scene_concepts=ph0_sam3.SCENE_CONCEPTS,
                             contours=True, meta=meta)
            rec["_n_explicit"] = len(batch)
            payload = json.dumps(rec, separators=(",", ":")).encode("utf-8")
            (OUT / f"{cid}.json").write_bytes(payload)
            n_banked += 1
            n_bytes += len(payload)
            lv = rec.get("liveness") or {}
            n_live += int(ph0_sam3.is_live(lv))
            n_det += int(rec.get("n_det_total") or 0)
            n_scene += int(rec.get("n_scene_det_total") or 0)
            n_err += int(rec.get("n_err_total") or 0)
            hits = ",".join(f"{k}:{v}"
                            for k, v in rec["per_concept_hits"].items() if v)
            sh = ",".join(f"{k}:{v}"
                          for k, v in rec["per_scene_hits"].items() if v)
            print(f"[bank] {n_banked}/{len(todo)} {cid[:8]} {len(payload)}B "
                  f"det={rec['n_det_total']} scene={rec['n_scene_det_total']} "
                  f"err={rec.get('n_err_total')} "
                  f"live={ph0_sam3.is_live(lv)}{lv.get('n_det')} "
                  f"[{hits or 'none'}] [{sh or 'none'}]", flush=True)
        L.gpu_mem_report(f"after batch b{bi:03d}")
        shutil.rmtree(bwork, ignore_errors=True)
    dt = time.time() - t0
    print(f"BANKED {n_banked} clips in {dt:.0f}s ({dt/max(n_banked,1):.2f} "
          f"s/clip) | det={n_det} scene={n_scene} err={n_err} "
          f"live={n_live}/{n_banked} bytes={n_bytes}")

# ---- local run manifest (NOT pushed — the driver pulls it) ------------------ #
json.dump({"class": "MEASURED", "chunk": chunk_id,
           "schema_version": ph0_sam3.SCHEMA_VERSION, "conf_threshold": CONF,
           "scene_concepts": ph0_sam3.SCENE_CONCEPTS,
           "frame_stride": FRAME_STRIDE,
           "contour_tol_px": ph0_sam3.CONTOUR_TOL_PX_DEFAULT,
           "contour_max_pts": ph0_sam3.CONTOUR_MAX_PTS_DEFAULT,
           "agent_concepts": ph0_sam3.AGENT_CONCEPTS,
           "liveness_concepts": ph0_sam3.LIVENESS_CONCEPTS,
           "completion_rule": "liveness control present AND zero errors AND "
                              "schema>=2 AND engine.confidence_threshold==0.25 "
                              "— never a file count (C77)",
           "pushed_to_hf": False,
           "ts_utc": time.strftime("%Y%m%d-%H%M%S")},
          open(OUT / "_run_manifest.json", "w"), indent=1)

# ---- VM-side census (the driver re-runs the SAME predicate on its copy) ------ #
cen = L.content_census_local(OUT, require_schema=2, require_conf=CONF)
small = {k: v for k, v in cen.items()
         if k not in ("zero_det_clips", "complete_clips", "missing", "extra")}
small["zero_split"] = {
    "empty_scene_control_live":
        sum(1 for z in cen["zero_det_clips"] if z["liveness_live"]),
    "dead_control":
        sum(1 for z in cen["zero_det_clips"] if not z["liveness_live"])}
print("[census-vm]", json.dumps(small, indent=1))

# ⚠️ THE TAR IS TRANSPORT, NOT OUTPUT — it exists so a remote driver can pull the
# bank off an ephemeral VM. When OUT already IS the durable bank (a local GPU),
# `F86_TAR=""` skips it and nothing else about the run changes.
if TAR:
    with tarfile.open(TAR, "w:gz") as tf:
        tf.add(str(OUT), arcname="out86")
    print(f"[tar] {TAR} {os.path.getsize(TAR)} B · "
          f"{len(list(OUT.glob('*.json')))} files")
else:
    print(f"[tar] skipped (F86_TAR empty) · {len(list(OUT.glob('*.json')))} "
          f"records in {OUT}")
print("F1_DONE")
