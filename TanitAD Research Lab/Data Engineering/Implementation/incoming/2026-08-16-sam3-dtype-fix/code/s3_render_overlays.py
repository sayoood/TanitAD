"""STEP 3 — overlay videos for the fixed SAM3 backfill, rendered on the dev box.

Per the PI's standing viz standard, each frame carries **camera + metric BEV +
the strategic label together**: camera 2x with SAM3 masks/boxes and their
concept/score, the BEV of the integrated ego path (engine A), and the S2
`g_str`/`a_str` the perception feeds — plus the C77 road/sky liveness control,
so a viewer can tell an empty scene from a dead engine without opening JSON.

Rendering is `stack/scripts/ph0_rich_overlay.py`; this script only selects the
clips, pulls the assets and calls it.

Selection is DETERMINISTIC and spans detection density: the sparsest clip, the
three quartiles, the busiest, plus **every clip that is still empty or whose
liveness control did not fire** (those go first — they are the ones a reviewer
must see).

    python s3_render_overlays.py --out <dir> [--n 8]
"""
import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import sys

import truststore

truststore.inject_into_ssl()
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
KEYS = os.path.join(REPO, "Keys.txt")
DS = "Sayood/tanitad-ph0-aug120"
DSV = "Sayood/tanitad-physicalai-w120-256x640cyl"
DSA = "Sayood/tanitad-alpamayo2-augmentation"
PREFIX = "sam3_backfill/"
EGO = "bridged_w120train_2400/ego/"
S2 = os.path.join(REPO, "TanitAD Research Hub", "Data Engineering",
                  "Implementation", "incoming", "2026-08-16-s2-v1-labels",
                  "labels", "s2_labels_aug120.jsonl")
SP = (r"C:\Users\Admin\AppData\Local\Temp\claude"
      r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
      r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad\sam3fix_assets")
sys.path.insert(0, os.path.join(REPO, "stack"))
sys.path.insert(0, os.path.join(REPO, "stack", "scripts"))

# ⚠️ IMPORTED AT MODULE TOP ON PURPOSE. Importing `v2_to_pilot` (and through it
# torch + torchvision) AFTER a few hundred `hf_hub_download` calls SEGFAULTS
# this interpreter (exit 139, MEASURED 2026-08-16 on the dev box); importing it
# first is stable. Same family as the repo's `git commit -- <pathspec>` crash:
# do not re-derive a theory, just keep the working order.
import v2_to_pilot                                               # noqa: E402
from ph0_sam3 import is_live                                     # noqa: E402
import ph0_rich_overlay                                          # noqa: E402


def token() -> str:
    return re.search(r"hf_[A-Za-z0-9]+",
                     open(KEYS, encoding="utf-8", errors="replace").read()
                     ).group(0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("s3_render_overlays")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--clips", default=None, help="override the selection")
    a = ap.parse_args(argv)

    from huggingface_hub import HfApi, hf_hub_download
    tok = token()
    api = HfApi(token=tok)

    def pull(rf, dst=None):
        p = hf_hub_download(DS, rf, repo_type="dataset", token=tok)
        if dst:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(p, dst)
            return dst
        return p

    far = {f.rfilename for f in api.dataset_info(DS).siblings}
    rfs = sorted(rf for rf in far
                 if rf.startswith(PREFIX) and rf.endswith(".json")
                 and "/_runs/" not in rf)
    recs = {}
    for rf in rfs:
        recs[rf[len(PREFIX):-len(".json")]] = json.load(
            open(pull(rf), encoding="utf-8"))
    print(f"[sel] {len(recs)} banked records")

    if a.clips:
        pick = [c.strip() for c in a.clips.split(",") if c.strip()]
    else:
        # ⛔ The density spread is taken over COMPLETE records only. Mixing in
        # the not-yet-re-run C77 records would put 32 zeros at the bottom of
        # the distribution and make "sparsest" mean "not processed yet".
        done = {c: r for c, r in recs.items() if r.get("liveness")}
        stale = sorted(c for c in recs if c not in done)
        print(f"[sel] complete {len(done)} · still-stale (pre-fix) "
              f"{len(stale)}")
        dens = sorted((int(r.get("n_det_total") or 0), c)
                      for c, r in done.items())
        n = len(dens)
        recs_all, recs = recs, done
        # ⛔ THE ALARM CASES GO FIRST — a reviewer must see the failures, and a
        # dead control is not the same thing as an empty scene.
        dead = sorted(c for c, r in recs.items()
                      if r.get("liveness") and not is_live(r["liveness"]))
        empty = sorted(c for c, r in recs.items()
                       if not int(r.get("n_det_total") or 0))
        # ⭐ RARE-CONCEPT COVERAGE. A selection spread on DENSITY alone shows
        # the corpus at its most flattering: the busiest clips are wall-to-wall
        # cars. The thin tail (`cyclist`, `bus`) is exactly where a false
        # positive would hide, so the clip with the most of each rare concept
        # is pinned into the set regardless of how dense it is.
        RARE = ("cyclist", "bus", "truck", "pedestrian")
        rare_pick = []
        for k in RARE:
            best = max(((r.get("per_concept_hits") or {}).get(k, 0), c)
                       for c, r in recs.items())
            if best[0] > 0:
                rare_pick.append(best[1])
                print(f"[sel] rare '{k}': best clip {best[1][:8]} "
                      f"with {best[0]}")
        idx = [0, n // 2, n - 1, (3 * n) // 4, n // 4]
        pick, seen = [], set()
        # one STILL-STALE clip is included on purpose: its panel prints
        # "liveness control ABSENT" in orange, which is what the C77 state
        # looks like on the figure — the contrast is the point.
        for c in (dead[:2] + empty[:1] + rare_pick
                  + [dens[i][1] for i in idx if 0 <= i < n]
                  + stale[:1]):
            if c not in seen:
                seen.add(c)
                pick.append(c)
        pick = pick[:a.n]
        print(f"[sel] density min {dens[0][0]} · median {dens[n // 2][0]} · "
              f"max {dens[-1][0]} | dead-control {len(dead)} · "
              f"zero-detection {len(empty)}")
    recs = recs_all if "recs_all" in dir() else recs
    for c in pick:
        r = recs[c]
        hits = {k: v for k, v in (r.get("per_concept_hits") or {}).items()
                if v}
        print(f"    {c[:8]} det={r.get('n_det_total')} "
              f"live={is_live(r.get('liveness'))} {hits}")

    # ⛔ THE FRAMES MUST BE THE ONES SAM3 SCORED (RETRACTION_LOG C79). The
    # pipeline re-bridges every clip from its w120 shard; the pre-bridged
    # `bridged_w120train_2400/videos/<cid>.mp4` on HF is a DIFFERENT ENCODE of
    # the same content — MEASURED on clip 0089a096: 2 983 186 B vs 1 373 844 B,
    # decoded frames differing by mean 1.66/255 with ~20 % of pixels off by
    # more than 2, which is enough to move ~7 % of detections across the
    # processor's 0.5 confidence threshold. Drawing banked boxes on that copy
    # would put real detections on frames that did not produce them.
    # The bridge is bit-deterministic (md5 equal across repeat runs), so
    # bridging here reproduces the pipeline's own bytes.
    vdir = os.path.join(SP, "videos")
    os.makedirs(vdir, exist_ok=True)
    todo = [c for c in pick if not os.path.exists(
        os.path.join(vdir, f"{c}.mp4"))]
    if todo:
        vloc = {f.rfilename.split("/")[-1][:-len(".v2ep.pt")]: f.rfilename
                for f in api.dataset_info(DSV).siblings
                if f.rfilename.endswith(".v2ep.pt")}
        pq = hf_hub_download(DSA, "records.parquet", repo_type="dataset",
                             token=tok)
        for c in todo:
            rf = vloc[c]
            cdir = os.path.join(SP, "corpus", c)
            os.makedirs(cdir, exist_ok=True)
            seg = rf.split("/")[0]
            for side in ("_geometry.json", "_v2manifest.pt"):
                try:
                    shutil.copyfile(hf_hub_download(
                        DSV, f"{seg}/{side}", repo_type="dataset", token=tok),
                        os.path.join(cdir, side))
                except Exception as e:
                    print(f"[bridge] side MISS {side} {type(e).__name__}")
            shutil.copyfile(hf_hub_download(DSV, rf, repo_type="dataset",
                                            token=tok),
                            os.path.join(cdir, f"{c}.v2ep.pt"))
            bo = os.path.join(SP, "bridge", c)
            rcb = v2_to_pilot.main(["--corpus", cdir, "--records", pq,
                                    "--out", bo, "--n", "1"])
            assert rcb == 0, f"bridge failed for {c}"
            shutil.copyfile(os.path.join(bo, "videos", f"{c}.mp4"),
                            os.path.join(vdir, f"{c}.mp4"))
            shutil.rmtree(cdir, ignore_errors=True)
            print(f"[bridge] {c[:8]} -> "
                  f"{os.path.getsize(os.path.join(vdir, f'{c}.mp4'))} B "
                  f"md5={hashlib.md5(open(os.path.join(vdir, f'{c}.mp4'), 'rb').read()).hexdigest()}")
    for c in pick:
        pull(f"{EGO}{c}.npz", os.path.join(SP, "ego", f"{c}.npz"))

    # the v2 records (engine B's B3 sign boxes the figure draws dashed)
    v2files = sorted(rf for rf in far
                     if rf.startswith("batch_") and rf.endswith("/v2/ph0_v2.json"))
    want, v2 = set(pick), {}
    for rf in v2files:
        for rec in json.load(open(pull(rf), encoding="utf-8")).get("clips", []):
            if rec.get("clip_id") in want and rec["clip_id"] not in v2:
                v2[rec["clip_id"]] = rec
    miss = want - set(v2)
    assert not miss, f"no v2 record for {sorted(miss)[:3]}"
    os.makedirs(SP, exist_ok=True)
    json.dump({"clips": [v2[c] for c in pick]},
              open(os.path.join(SP, "ph0_v2.json"), "w", encoding="utf-8"))
    json.dump({"engine": "C_sam3", "clips": [recs[c] for c in pick]},
              open(os.path.join(SP, "sam3.json"), "w", encoding="utf-8"))

    os.makedirs(a.out, exist_ok=True)
    rc = ph0_rich_overlay.main([
        "--v2-json", os.path.join(SP, "ph0_v2.json"),
        "--sam3-json", os.path.join(SP, "sam3.json"),
        "--video-root", os.path.join(SP, "videos"),
        "--ego-root", os.path.join(SP, "ego"),
        "--s2-jsonl", S2, "--clips", ",".join(pick),
        "--fps", str(a.fps), "--crf", str(a.crf), "--out", a.out])
    tot = 0
    for f in sorted(os.listdir(a.out)):
        if not f.endswith(".mp4"):
            continue
        b = open(os.path.join(a.out, f), "rb").read()
        tot += len(b)
        print(f"[mp4] {f} {len(b)} B md5={hashlib.md5(b).hexdigest()}")
    print(f"[mp4] total {tot} B ({tot / 2**20:.2f} MiB)")
    per = collections.Counter()
    for c in pick:
        per.update({k: v for k, v in recs[c]["per_concept_hits"].items()})
    json.dump({"picked": pick,
               "n_det": {c: recs[c].get("n_det_total") for c in pick},
               "liveness": {c: is_live(recs[c].get("liveness"))
                            for c in pick},
               "per_concept_over_selection": dict(per.most_common()),
               "fps": a.fps, "crf": a.crf},
              open(os.path.join(a.out, "_selection.json"), "w",
                   encoding="utf-8"), indent=1)
    print("RENDER_DONE rc", rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
