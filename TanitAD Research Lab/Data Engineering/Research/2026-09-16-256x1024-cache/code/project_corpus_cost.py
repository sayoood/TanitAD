"""Project the full-corpus cost of the 256x1024 rebuild from MEASURED rates.

Nothing here is a guess: the per-clip download seconds come from the fetch
receipt, the per-clip build seconds and bytes from the cache MANIFEST, and the
corpus's own byte total from the pinned revision's ``camera/camera_sha256.json``
(so the projection is scaled by the REAL size distribution, not by assuming the
139-clip sample is typical -- it is not; it is ~1.12x the corpus mean).
"""
from __future__ import annotations
import argparse, json, sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                                # noqa: BLE001
        pass

#: MEASURED 2026-09-01 by parity.guard_corpus_build on the 4,719 v7 clips:
#: 6 are inside the 40-episode deployed val -> a TRAINING corpus is 4,713.
CORPUS_N_RAW = 4719
CORPUS_N_TRAIN = 4713


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-receipt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--sha-table", required=True)
    ap.add_argument("--ref-cache-bytes", type=float, default=166.0,
                    help="GB of the deployed 256x640 cache (du on Thor)")
    ap.add_argument("--ref-cache-clips", type=int, default=4715)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    f = json.load(open(a.fetch_receipt))
    m = json.load(open(a.manifest))
    tab = json.load(open(a.sha_table))

    corpus_bytes = sum(v["bytes"] for v in tab.values())
    corpus_mean_mb = corpus_bytes / len(tab) / 1e6
    sample_mean_mb = f["total_bytes"] / f["n_clips"] / 1e6
    size_ratio = sample_mean_mb / corpus_mean_mb

    dl_rate = f["total_bytes"] / f["dl_seconds_total"] / 1e6            # MB/s
    train_bytes = corpus_bytes * CORPUS_N_TRAIN / CORPUS_N_RAW
    dl_h = train_bytes / 1e6 / dl_rate / 3600

    built = [c for c in m["clips"] if c.get("build_s", 0) > 0]
    cache_mb = m["total_bytes"] / m["n_clips"] / 1e6
    # wall rate of the real parallel run, not the sum of per-clip seconds
    eps_per_h = m["n_clips"] / (m["build_seconds_this_run"] / 3600) \
        if m["build_seconds_this_run"] else 0
    build_h = CORPUS_N_TRAIN / eps_per_h if eps_per_h else float("nan")
    ref_mb = a.ref_cache_bytes * 1024 / a.ref_cache_clips              # GB->MB? no:
    ref_mb = a.ref_cache_bytes * 1e3 / a.ref_cache_clips               # GB*1e3 = MB

    out = {
        "measured": {
            "fetch": {"n_clips": f["n_clips"],
                      "total_gb": round(f["total_bytes"] / 1e9, 3),
                      "mb_per_clip": round(sample_mean_mb, 2),
                      "dl_seconds_total_serial": f["dl_seconds_total"],
                      "dl_s_per_clip_mean": f["dl_s_per_clip_mean"],
                      "dl_mb_per_s_serial": round(dl_rate, 3)},
            "build": {"n_clips": m["n_clips"], "workers": m["workers"],
                      "wall_seconds": m["build_seconds_this_run"],
                      "wall_s_per_clip": round(m["build_seconds_this_run"]
                                               / m["n_clips"], 3),
                      "single_clip_cpu_s_mean": m["build_s_per_clip_mean"],
                      "clips_per_hour_wall": round(eps_per_h, 1),
                      "cache_mb_per_clip": round(cache_mb, 2),
                      "cache_total_gb": round(m["total_bytes"] / 1e9, 3)},
            "corpus_source": {"n_clips": len(tab),
                              "total_gb": round(corpus_bytes / 1e9, 2),
                              "mb_per_clip": round(corpus_mean_mb, 2),
                              "sample_is_x_corpus_mean": round(size_ratio, 3)},
            "reference_640_cache": {"clips": a.ref_cache_clips,
                                    "total_gb": a.ref_cache_bytes,
                                    "mb_per_clip": round(ref_mb, 2)},
        },
        "projected_full_corpus": {
            "n_clips": CORPUS_N_TRAIN,
            "note": f"{CORPUS_N_RAW} in production_order; {CORPUS_N_RAW - CORPUS_N_TRAIN}"
                    f" are inside the deployed val40 and are dropped from a TRAINING"
                    f" corpus by the parity ingest gate",
            "download_gb": round(train_bytes / 1e9, 1),
            "download_hours_serial": round(dl_h, 2),
            "download_hours_4x_parallel_est": round(dl_h / 4, 2),
            "build_hours_at_measured_rate": round(build_h, 2),
            "cache_gb_256x1024": round(CORPUS_N_TRAIN * cache_mb / 1e3, 1),
            "cache_gb_256x640_deployed": round(CORPUS_N_TRAIN * ref_mb / 1e3, 1),
            "cache_growth_x": round(cache_mb / ref_mb, 3),
            "disk_needed_gb_source_plus_cache": round(
                train_bytes / 1e9 + CORPUS_N_TRAIN * cache_mb / 1e3, 1),
            "wall_hours_if_pipelined": round(max(dl_h, build_h), 2),
            "wall_hours_if_serial": round(dl_h + build_h, 2),
        },
    }
    json.dump(out, open(a.out, "w"), indent=1)
    p = out["projected_full_corpus"]
    print(json.dumps(out["measured"], indent=1))
    print(f"\nFULL CORPUS ({p['n_clips']} clips): download {p['download_gb']} GB "
          f"in {p['download_hours_serial']} h serial; build "
          f"{p['build_hours_at_measured_rate']} h; cache "
          f"{p['cache_gb_256x1024']} GB "
          f"({p['cache_growth_x']}x the {p['cache_gb_256x640_deployed']} GB 640 "
          f"cache); disk {p['disk_needed_gb_source_plus_cache']} GB; "
          f"wall {p['wall_hours_if_pipelined']} h pipelined / "
          f"{p['wall_hours_if_serial']} h serial -> {a.out}")


if __name__ == "__main__":
    main()
