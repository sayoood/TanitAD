"""Rolling, disk-bounded rebuild of a PhysicalAI epcache from the gated HF origin.

Byte-identical to a source pod's epcache BY CONSTRUCTION: identical pipeline code
+ the SAME ordered clip list (proven via the cache key, which is a hash of the
ordered clip_ids + params) + the SAME build params. The train/val split is NOT
recomputed here (that would risk cross-machine RNG / discovery-order drift) — it
is fed in as a precomputed order file (idx<TAB>clip_id<TAB>chunk), derived once on
the source pod where the cache was built and verified via the key.

Why rolling: the full camera (~394 GB transient) and the epcache (~260 GB) do not
both fit under a pod's ~466 GB quota. So we fetch ONE camera chunk at a time from
HF, extract only the needed clips, build their episodes into the cache at their
FIXED indices, then DELETE that chunk's camera before the next. Peak extra disk is
one chunk's camera (~a few GB) + the 2 GB zip, so the cache can grow to full size
under the quota. Idempotent: existing ep_%05d.pt / skip_%05d are never rebuilt, so
a killed run resumes exactly where it stopped.

Determinism guardrails:
  * cache_key(order, params) MUST equal --expect-key or the run aborts (the order
    or params drifted from the source).
  * --skip-idx pre-seeds the source pod's known-corrupt indices as skip markers so
    the rebuilt ep set matches the source's exactly (same indices present/absent).
  * per-clip f-theta intrinsics must resolve (per_clip=True) — asserted on the
    first built clip — else the crop silently reverts to geometric-center (wrong
    pixels). Final proof is an external tensor-content hash vs the source pod.

Usage (pod3):
  TANITAD_PHYSICALAI_ROOT=/workspace/pai_build HF_TOKEN=hf_... \
  python scripts/rebuild_pai_rolling.py \
      --order /workspace/tmp/train_clip_order.tsv \
      --cache-root /workspace/pai_epcache --tag physicalai-train \
      --expect-key e438721ae894 --skip-idx 1798,1835,...,1941
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = os.environ.get("TANITAD_PHYSICALAI_ROOT", "/workspace/pai_build")
REPO = os.environ.get("TANITAD_REPO", "/workspace/TanitAD/stack")
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + "/scripts")

from tanitad.config import base250cam_config                     # noqa: E402
from tanitad.data.epcache import cache_key                       # noqa: E402
from tanitad.data.mixing import save_episode                     # noqa: E402
from tanitad.data.physicalai import (build_episode,              # noqa: E402
                                     discover_r0_clips)
from tanitad.data import physicalai as _pai                      # noqa: E402

HF_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
EGO_TMPL = "labels/egomotion/egomotion.chunk_{c:04d}.zip"
CAM_TMPL = ("camera/camera_front_wide_120fov/"
            "camera_front_wide_120fov.chunk_{c:04d}.zip")


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _hf_download(rel: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HF_REPO, rel, repo_type="dataset", local_dir=ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", required=True,
                    help="TSV: idx<TAB>clip_id<TAB>chunk (precomputed split order)")
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--tag", default="physicalai-train")
    ap.add_argument("--expect-key", required=True)
    ap.add_argument("--skip-idx", default="",
                    help="comma list of source-pod known-corrupt indices to skip")
    ap.add_argument("--only-idx", default="",
                    help="comma list: build ONLY these indices (gate/debug)")
    args = ap.parse_args()

    # TLS + HF token (dev-box proxy helper; harmless on pods, env HF_TOKEN also read)
    try:
        from tanitad.keys import enable_tls, load_keys
        enable_tls()
        load_keys()
    except Exception as e:  # noqa: BLE001
        log(f"keys helper unavailable ({e}); relying on env HF_TOKEN")

    cfg = base250cam_config()
    params = {"size": cfg.encoder.image_size, "n_stack": 3, "hz": 10,
              "calib": "ftheta_v2"}

    order = []
    with open(args.order) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            order.append((int(parts[0]), parts[1], int(parts[2])))
    log(f"order rows: {len(order)}")

    # --- DETERMINISM ORACLE: the key is a hash of the ordered clip_ids + params.
    sources = [{"clip_id": cid} for (_i, cid, _c) in order]
    key = cache_key(sources, params)
    log(f"cache_key={key} expect={args.expect_key} params={params}")
    if key != args.expect_key:
        raise SystemExit(f"ABORT: key mismatch {key} != {args.expect_key} "
                         f"(order/params drift — NOT byte-identical)")

    d = Path(args.cache_root) / f"{args.tag}-{key}"
    d.mkdir(parents=True, exist_ok=True)

    skip_idx = {int(x) for x in args.skip_idx.split(",") if x.strip()}
    only_idx = {int(x) for x in args.only_idx.split(",") if x.strip()}

    # pre-seed known-corrupt skip markers (match source pod's reconcile drop)
    for i in skip_idx:
        sk = d / f"skip_{i:05d}"
        if not (d / f"ep_{i:05d}.pt").exists() and not sk.exists():
            sk.write_text(f"reconcile: source-pod known-corrupt idx {i} "
                          f"(pre-seeded skip; not built)")

    by_chunk: dict[int, list[tuple[int, str]]] = defaultdict(list)
    n_have = 0
    for (i, cid, ch) in order:
        if i in skip_idx:
            continue
        if only_idx and i not in only_idx:
            continue
        if (d / f"ep_{i:05d}.pt").exists():
            n_have += 1
            continue
        if (d / f"skip_{i:05d}").exists():
            continue
        by_chunk[ch].append((i, cid))
    target = len(order) - len(skip_idx) if not only_idx else len(only_idx)
    log(f"already built {n_have}; chunks to fetch {len(by_chunk)}; "
        f"remaining {sum(len(v) for v in by_chunk.values())}; target {target}")

    cam_dir = Path(ROOT) / "r0" / "camera_front_wide"
    cam_dir.mkdir(parents=True, exist_ok=True)

    checked_calib = False
    built = 0
    for ci, ch in enumerate(sorted(by_chunk), 1):
        need = by_chunk[ch]
        want = {cid for (_i, cid) in need}

        ego = Path(ROOT) / EGO_TMPL.format(c=ch)
        if not ego.exists():
            _hf_download(EGO_TMPL.format(c=ch))

        zp = _hf_download(CAM_TMPL.format(c=ch))
        n_ext = 0
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                base = name.rsplit("/", 1)[-1]
                cid = base.split(".")[0]
                if cid in want and (base.endswith(".mp4")
                                    or base.endswith(".timestamps.parquet")):
                    with z.open(name) as src, open(cam_dir / base, "wb") as dst:
                        shutil.copyfileobj(src, dst)          # FLAT extract
                    n_ext += 1
        os.unlink(zp)                                          # 2 GB zip — drop now
        clips = {c["clip_id"]: c for c in discover_r0_clips(ROOT)}

        n_chunk_built = 0
        for (i, cid) in need:
            f = d / f"ep_{i:05d}.pt"
            if f.exists():
                continue
            clip = clips.get(cid)
            if clip is None:
                (d / f"skip_{i:05d}").write_text(
                    f"clip {cid} not discoverable after extract")
                log(f"WARN idx {i} clip {cid}: not found post-extract -> skip")
                continue
            try:
                ep = build_episode(clip, size=cfg.encoder.image_size)
                tmp = str(f) + ".tmp"
                save_episode(ep, tmp)
                os.replace(tmp, f)                            # atomic publish
                del ep
                built += 1
                n_chunk_built += 1
                if not checked_calib:
                    checked_calib = True
                    lc = getattr(_pai._decode_mp4, "last_calib", {})
                    log(f"first-clip calib: {lc}")
                    if not lc.get("per_clip", False):
                        raise SystemExit(
                            "ABORT: per-clip intrinsics did NOT resolve "
                            "(per_clip=False -> geometric crop, wrong pixels). "
                            "Check <root>/calibration + r0_selection.parquet.")
            except SystemExit:
                raise
            except Exception as e:  # noqa: BLE001
                (d / f"skip_{i:05d}").write_text(f"{type(e).__name__}: {e}")
                log(f"BUILD-FAIL idx {i} clip {cid}: {type(e).__name__}: {e}")

        # rolling delete: this chunk's extracted camera (mp4 + timestamps)
        for p in cam_dir.iterdir():
            if p.name.split(".")[0] in want:
                p.unlink()

        n_now = len(list(d.glob("ep_*.pt")))
        log(f"chunk {ch} ({ci}/{len(by_chunk)}): extracted {n_ext}, "
            f"built {n_chunk_built}; cache ep total {n_now}")

    n_final = len(list(d.glob("ep_*.pt")))
    n_skip = len(list(d.glob("skip_*")))
    if not only_idx:
        (d / "DONE").write_text(json.dumps({"episodes": n_final,
                                            "skipped": n_skip}))
    log(f"FINAL ep={n_final} skip={n_skip} built_this_run={built} at {d}")
    log("REBUILD_DONE")


if __name__ == "__main__":
    main()
