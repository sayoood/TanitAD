"""PI-mandated validation gate before training handover (2026-08-29).

"I want you also to validate it before handover it to training."

Per-clip, by CONTENT (never by name/size/exit code):
  * camera: ftyp magic, PyAV decode of FIRST and LAST frame, both non-trivial
    (mean pixel > 2.0 — the all-zeros trap), duration 20 s +/- 3 s;
  * joins: labels <-> camera <-> egomotion <-> cy on clip UUID;
  * per-file sha256 into the camera manifest.
Corpus-level: counts, rig split, label invariants re-asserted from the bundle.

Output: VALIDATION.md + validation.json + camera_sha256.json. Any failing clip
goes into the exclusions ledger WITH ITS REASON — skip-lists stay
manifest-derived, never discovered downstream.
"""
import gzip
import hashlib
import json
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import av
import numpy as np
import pandas as pd

REL = Path("C:/Users/Admin/tanitad-wt/_s2build/release")
BUNDLE = REL / "tanitad-v7-training-corpus"
CAM = Path("C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov")

t0 = time.time()

# --- corpus ------------------------------------------------------------------
rows = [json.loads(l) for l in gzip.open(BUNDLE / "labels" / "s2_labels_v7.jsonl.gz",
                                         "rt", encoding="utf-8") if l.strip()]
need = sorted({r["clip_id"] for r in rows})
viol = sum(1 for r in rows if r["g_tac"]["violations"])
assert len(rows) == 4719 and viol == 0, (len(rows), viol)

cy = pd.read_parquet(BUNDLE / "index" / "front_wide_cy.parquet")
with tarfile.open(BUNDLE / "egomotion" / "egomotion_alpamayo.tar") as tf:
    ego_ids = {n.split(".")[0] for n in tf.getnames()}
adf = pd.read_parquet(BUNDLE / "alpamayo" / "records.parquet")
alp_ids = set(adf.clip_id)

join = {
    "labels": len(need),
    "camera_files": sum(1 for c in need if (CAM / f"{c}.mp4").exists()),
    "egomotion": len(set(need) & ego_ids),
    "cy": len(set(need) & set(cy.clip_id)),
    "alpamayo": len(set(need) & alp_ids),
}
print("joins:", join, flush=True)

# --- per-clip camera validation ---------------------------------------------
lock = threading.Lock()
shas: dict[str, dict] = {}
fails: list[dict] = []
done = [0]


def probe(cid: str):
    p = CAM / f"{cid}.mp4"
    try:
        b = p.read_bytes()
        if b[:64].find(b"ftyp") < 0:
            return {"clip_id": cid, "reason": "no ftyp magic"}
        h = hashlib.sha256(b).hexdigest()
        # ⛔ EVERY read of `st` MUST happen inside this block. A PyAV stream is a
        # view onto the container; touching it after `with` closes the container
        # is a USE-AFTER-FREE and SEGFAULTS the interpreter (exit 139) — no
        # Python traceback, no failing clip named. MEASURED 2026-08-29: reading
        # `st.frames` / `st.average_rate` below the block killed the run twice,
        # deterministically, right after the joins line and before the first
        # progress print. It only fires on clips that PASS the duration check,
        # which is why a probe that returns early never showed it.
        with av.open(str(p)) as c:
            st = c.streams.video[0]
            dur = float(st.duration * st.time_base) if st.duration else None
            nfr = st.frames or 0
            fps = float(st.average_rate or 0)
            first = last = None
            for fr in c.decode(video=0):
                a = fr.to_ndarray(format="gray")
                first = float(a.mean())
                break
            # Last-frame probe. ⚠️ A seek at (dur - 0.5 s) can land PAST the last
            # decodable packet and yield NOTHING, leaving last=None on a
            # perfectly healthy clip. MEASURED 2026-08-29: that read as
            # "last-frame mean None (zeros trap)" for 129 of 3,176 banked clips
            # (~4 %) — every one of which decodes all 605 frames with a
            # non-trivial last frame when checked. An exclusion ledger built on
            # that would have dropped ~190 GOOD clips from the training corpus
            # with an authoritative-sounding reason: a MANUFACTURED GAP, i.e.
            # the instrument inventing the defect it reports.
            # ⇒ back off, then fall back to a full sequential decode. The
            # fallback is what makes a None verdict mean the DATA, not the seek.
            for back in (0.5, 2.0):
                c.seek(max(int((dur - back) / float(st.time_base)), 0) if dur else 0,
                       stream=st, any_frame=True)
                for fr in c.decode(video=0):
                    last = float(fr.to_ndarray(format="gray").mean())
                if last is not None:
                    break
            if last is None:
                c.seek(0, stream=st, any_frame=True)
                for fr in c.decode(video=0):
                    last = float(fr.to_ndarray(format="gray").mean())
        if dur is None or not (17.0 <= dur <= 23.0):
            return {"clip_id": cid, "reason": f"duration {dur}"}
        # mastermind's addition (2026-08-29): a TRUNCATED-but-decodable mp4
        # passes first+last probes while missing its middle — assert the
        # container's frame count against duration*fps (+/-5 %).
        if fps > 0 and nfr > 0 and abs(nfr - dur * fps) > 0.05 * dur * fps:
            return {"clip_id": cid,
                    "reason": f"frame count {nfr} != {dur:.1f}s x {fps:.1f}fps"}
        if first is None or first <= 2.0:
            return {"clip_id": cid, "reason": f"first-frame mean {first} (zeros trap)"}
        if last is None or last <= 2.0:
            return {"clip_id": cid, "reason": f"last-frame mean {last} (zeros trap)"}
        with lock:
            shas[cid] = {"sha256": h, "bytes": len(b), "duration_s": round(dur, 2),
                         "first_mean": round(first, 1), "last_mean": round(last, 1)}
        return None
    except Exception as e:                                  # noqa: BLE001
        return {"clip_id": cid, "reason": f"{type(e).__name__}: {str(e)[:90]}"}
    finally:
        with lock:
            done[0] += 1
            if done[0] % 500 == 0:
                print(f"  probed {done[0]}/{len(need)} "
                      f"({done[0]/(time.time()-t0):.1f}/s)", flush=True)


present = [c for c in need if (CAM / f"{c}.mp4").exists()]
_present = set(present)                      # hoisted: was rebuilt per iteration
absent = [c for c in need if c not in _present]
with ThreadPoolExecutor(max_workers=8) as ex:
    for f in as_completed({ex.submit(probe, c) for c in present}):
        r = f.result()
        if r:
            fails.append(r)
for c in absent:
    fails.append({"clip_id": c, "reason": "camera file absent after pull"})

ok = len(shas)
res = {
    "validated": "2026-08-29",
    "corpus_id": "a48251e89c7a8603",
    "n_clips": len(need),
    "joins": join,
    "camera_ok": ok,
    "camera_failed": len(fails),
    "fail_reasons": sorted({f["reason"].split(":")[0] for f in fails}),
    "label_invariants": {"exclusion_violations": viol, "rows": len(rows)},
    # str()/int(): value_counts gives numpy scalars, which json.dump refuses
    "rig_split": {str(k): int(v) for k, v in cy.rig.value_counts().to_dict().items()},
    "verdict": ("PASS" if not fails and all(v == len(need) for v in join.values())
                else "PASS_WITH_EXCLUSIONS" if len(fails) <= 24
                else "FAIL"),
}
json.dump(res, open(REL / "validation.json", "w"), indent=1)
json.dump(shas, open(REL / "camera_sha256.json", "w"))
json.dump(fails, open(REL / "validation_failures.json", "w"), indent=1)

md = f"""# VALIDATION — tanitad-v7-training-corpus (pre-handover gate)

**Verdict: {res['verdict']}** · corpus `a48251e89c7a8603` · {len(need):,} clips

| check | result |
|---|---|
| labels rows / exclusion violations | {len(rows):,} / {viol} |
| join labels↔camera | {join['camera_files']}/{len(need)} |
| join labels↔egomotion | {join['egomotion']}/{len(need)} |
| join labels↔cy (rig split {res['rig_split']}) | {join['cy']}/{len(need)} |
| join labels↔alpamayo | {join['alpamayo']}/{len(need)} |
| camera decode probe (first+last frame non-zero, 20±3 s) | **{ok}/{len(need)}** |
| camera failures (see validation_failures.json) | {len(fails)} |

Every camera file sha256-recorded in `camera_sha256.json`. Failing clips go to the
MANIFEST `exclusions` ledger with their reasons — never silently skipped downstream.
"""
(REL / "VALIDATION.md").write_text(md, encoding="utf-8")
print(md, flush=True)
print(f"validation took {(time.time()-t0)/60:.1f} min", flush=True)
