"""Price the three ways the 416x1024 refcv6 data can reach a pod (E20 input), and size what
the pod needs FIRST. Pure arithmetic over this package's measured inputs plus the INHERITED
constants below, each with its source. Nothing is transferred or provisioned.

    python price_paths.py <inventory.json> <size_model.json> <uplink_probe.json> <out.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

GB = 1e9
# ── INHERITED constants (not re-measured here), with their sources ────────────────────────
HF_TO_POD_MBPS = 118.0          # CLAUDE.md "HF push/pull (~118 MB/s)"; memory pod-ckpt-transfer-hf-relay (RunPod, 2026-07)
RELAY_2026_07_MBPS = 3.26e3 / (47 * 60)   # the same memory: 3.26 GB in 47 min through the dev box
BUILD_WORKER_S_PER_CLIP = 6 * 1125 / 138  # 2026-09-16 package: 138 clips in 1,125 s at 6 workers, 408x1024
D_READ_MBPS = 18.7              # MEASURED 2026-09-19, …/2026-09-19-local-mirror-staging/raw/d_drive_io.json
HF_PRIVATE_INCLUDED_GB = 1000.0 # PUBLISHED, HF storage-limits (fetched 2026-09-19 in the C4 package)
HF_USD_PER_TB_MONTH = 18.0      # PUBLISHED, same page: pay-as-you-go base above the included TB
SAM3_CLIPS_PER_H = 25.0         # INHERITED, memory sam3-corpus-production-2026-09 ("~24-25 clips/h")
N_SHARDS = 8


def hours(nbytes: float, mbps: float) -> float:
    return nbytes / (mbps * 1e6) / 3600


def main(inv_p, size_p, up_p, out_p) -> int:
    inv = json.loads(Path(inv_p).read_text(encoding="utf-8"))
    sm = json.loads(Path(size_p).read_text(encoding="utf-8"))
    up = json.loads(Path(up_p).read_text(encoding="utf-8"))["summary"]
    m2 = sm["models"]["M2 linear in mp4 bytes"]
    a, b = sm["coefficients"]["a"], sm["coefficients"]["b"]
    aux, hf = inv["aux_local"], inv["hf_corpus"]
    uplink = [up["single_stream_25MB_MBps_range"][0], up["parallel_4_streams_aggregate_MBps"]]

    # ── bytes ───────────────────────────────────────────────────────────────────────────────
    train_gb = m2["train_total_GB"]
    clean124_gb = (aux["clean124_halfA"]["bytes"] + aux["clean124_halfB"]["bytes"]) / GB
    eval139_gb = inv["eval139_416"]["dir_bytes"] / GB
    eval_aux_gb = sum(aux[k]["bytes"] for k in ("sam3_maps_eval", "agent_join_3d_eval",
                                                 "extrinsics141", "v8_labels")) / GB
    cache_to_pod_gb = train_gb + clean124_gb
    f = hf["folders"]
    maps = hf["maps"]["gt"]
    map_mb_per_clip = maps["bytes"] / maps["clips"] / 1e6
    n_train = inv["train_covariates"]["n"]
    train_maps_gb = n_train * map_mb_per_clip / 1e3
    eval_rows = inv["eval139_416"]["rows"]
    sources_gb = {
        "train mp4 (4,572, MEASURED)": inv["train_covariates"]["mp4_bytes"] / GB,
        "eval139 mp4 (MEASURED)": sum(r["mp4_mb"] for r in eval_rows) * 1e6 / GB,
        "egomotion tar": f["egomotion"]["bytes"] / GB,
        "timestamps tar": f["timestamps"]["bytes"] / GB,
        "calibration (intrinsics + extrinsics parquet)": f["calibration"]["bytes"] / GB,
        "agents/obstacle_offline (for the train agent join)": f["agents/obstacle_offline"]["bytes"] / GB,
        "labels": f["labels"]["bytes"] / GB,
    }
    src_gb = sum(sources_gb.values())

    # ── HF private storage (the account as measured today) ────────────────────────────────────
    priv_gb = inv["hf_account"]["pools_bytes"]["private"] / GB
    big_public = next(r for r in inv["hf_account"]["largest"] if not r["private"])
    sam3_remaining_clips = 4719 - maps["clips"] - hf["maps"]["gt_flagged"]["clips"]
    sam3_growth_gb = sam3_remaining_clips * (f["semantic_maps/gt"]["bytes"] + f["semantic_maps/worldmap"]["bytes"]) \
        / (maps["clips"] + hf["maps"]["gt_flagged"]["clips"]) / GB
    base_priv = priv_gb + sam3_growth_gb

    def bill(total_gb):
        over = max(0.0, total_gb - HF_PRIVATE_INCLUDED_GB)
        return {"private_after_GB": round(total_gb, 1), "over_1TB_GB": round(over, 1),
                "usd_per_month_at_base_rate": round(over / 1000 * HF_USD_PER_TB_MONTH, 2)}

    # ── the three routes ──────────────────────────────────────────────────────────────────────
    build_h_dev6 = n_train * BUILD_WORKER_S_PER_CLIP / 6 / 3600
    build_h_pod = {w: (n_train + len(eval_rows)) * BUILD_WORKER_S_PER_CLIP / w / 3600 for w in (8, 16, 32)}
    routes = {
        "a_hf_private_push": {
            "bytes_GB": round(cache_to_pod_gb, 1),
            "upload_from_devbox_h": [round(hours(cache_to_pod_gb * GB, r), 1) for r in reversed(uplink)],
            "upload_MBps": uplink, "upload_class": "MEASURED today (random bytes, neutral endpoint)",
            "pod_pull_h": round(hours(cache_to_pod_gb * GB, HF_TO_POD_MBPS), 2),
            "pod_pull_class": "ESTIMATED from the INHERITED 118 MB/s",
            "devbox_build_h_6_workers": round(build_h_dev6, 1),
            "devbox_build_class": "ESTIMATED from the INHERITED 442 clips/h (408x1024)",
            "hf_today": bill(base_priv + cache_to_pod_gb),
            "hf_if_largest_public_repo_goes_private": bill(base_priv + cache_to_pod_gb + big_public["bytes"] / GB),
        },
        "b_direct_upload": {
            "bytes_GB": round(cache_to_pod_gb, 1),
            "upload_h": [round(hours(cache_to_pod_gb * GB, r), 1) for r in reversed(uplink)],
            "upload_MBps": uplink, "hf_usd": 0.0,
            "second_probe_2026_07_relay_MBps": round(RELAY_2026_07_MBPS, 2),
            "note": "a pod (or its volume behind a CPU pod) must be online for the whole upload",
        },
        "c_build_on_pod": {
            "download_GB": round(src_gb + train_maps_gb + eval_aux_gb, 1),
            "download_breakdown_GB": {**{k: round(v, 3) for k, v in sources_gb.items()},
                                      "SAM3 map GT, train, projected": round(train_maps_gb, 2),
                                      "eval aux (maps, 3-D join, extrinsics, labels)": round(eval_aux_gb, 3)},
            "download_h_at_118MBps": round(hours((src_gb + train_maps_gb) * GB, HF_TO_POD_MBPS), 2),
            "download_h_at_30MBps": round(hours((src_gb + train_maps_gb) * GB, 30.0), 2),
            "eval_aux_from_devbox_min": round(hours(eval_aux_gb * GB, uplink[0]) * 60, 1),
            "build_h_by_workers": {str(w): round(h, 1) for w, h in build_h_pod.items()},
            "build_class": "ESTIMATED: the dev box's 48.9 worker-s/clip, scaled linearly",
            "hf_usd": 0.0,
            "pod_disk_peak_GB": round(cache_to_pod_gb + (eval139_gb - clean124_gb) + src_gb + train_maps_gb, 1),
            "pod_disk_steady_GB": round(cache_to_pod_gb + train_maps_gb + eval_aux_gb, 1),
        },
    }

    # ── what the pod needs FIRST ──────────────────────────────────────────────────────────────
    per = sorted(inv["train_covariates"]["per_clip"])           # sorted by sha12: a hash order
    k = len(per) // N_SHARDS
    shards = []
    for i in range(N_SHARDS):
        chunk = per[i * k:(i + 1) * k] if i < N_SHARDS - 1 else per[i * k:]
        gb = sum(a + b * m * 1e6 for _, m, _ in chunk) / GB
        shards.append({"shard": i, "clips": len(chunk), "cache_GB": round(gb, 2),
                       "maps_GB": round(len(chunk) * map_mb_per_clip / 1e3, 2)})
    first = {
        "tier0_clean124_plus_eval_aux_GB": round(clean124_gb + eval_aux_gb, 2),
        "tier0_upload_from_devbox_h": round(hours((clean124_gb + eval_aux_gb) * GB, uplink[0]), 1),
        "tier0_build_on_pod_min_16_workers": round(len(eval_rows) * BUILD_WORKER_S_PER_CLIP / 16 / 60, 1),
        "shards": shards,
        "first_two_shards_GB": round(sum(s["cache_GB"] + s["maps_GB"] for s in shards[:2]), 1),
        "first_two_shards_build_min_16_workers": round(sum(s["clips"] for s in shards[:2]) * BUILD_WORKER_S_PER_CLIP / 16 / 60, 1),
        "per_step_reads_GB_batch20": round(20 * train_gb / n_train * (1 - 64 / n_train), 2),
    }
    out = {"inputs": {"uplink_MBps": uplink, "hf_to_pod_MBps_inherited": HF_TO_POD_MBPS,
                      "build_worker_s_per_clip_inherited": round(BUILD_WORKER_S_PER_CLIP, 1),
                      "d_read_MBps_measured": D_READ_MBPS, "private_used_GB": round(priv_gb, 2),
                      "sam3_growth_to_completion_GB": round(sam3_growth_gb, 1),
                      "sam3_maps_remaining_clips": sam3_remaining_clips,
                      "sam3_eta_h": round(sam3_remaining_clips / SAM3_CLIPS_PER_H, 0),
                      "largest_public_repo": big_public},
           "sizes_GB": {"train_cache_M2": round(train_gb, 1), "train_cache_M2_ci95": [round(x, 1) for x in m2["ci95_GB"]],
                        "clean124": round(clean124_gb, 2), "eval139": round(eval139_gb, 2),
                        "cache_to_pod": round(cache_to_pod_gb, 1), "train_maps_projected": round(train_maps_gb, 2),
                        "map_MB_per_clip": round(map_mb_per_clip, 3)},
           "routes": routes, "first_subset": first}
    Path(out_p).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:5]))
