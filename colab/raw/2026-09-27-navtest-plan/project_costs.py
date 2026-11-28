"""Wall-clock and compute-unit projections for REFe's navtest eval on Colab -- every input named with its class.

MEASURED inputs are read from the JSON artifacts in this directory (or quoted with their source file); ESTIMATED ones
are the constants in ASSUMED below, each with its basis. Re-run after the parity run (P0) replaces the ESTIMATED
per-token GPU times, S3 and relay rates with measured ones. Writes projections.json and prints the plan's table.

DATA ROUTES (coordinator, 2026-09-27: the HF token NEVER goes to a VM; private files travel dev box -> private relay
`Sayood/tanitad-colab-relay` (hf_relay_push.py) -> signed links minted on the dev box (hf_signed_links.py) -> the VM
(signed_pull.py); public inputs come straight from their public sources):
  "public+relay"    private/derived files (checkpoint, sub200 frames + sub200 metric-cache pickles, W3 export) via the
                    relay; everything else public (S3 nuPlan DBs + maps, HF navsim_logs + trunk; for the full navtest
                    the 32 public camera shards are streamed and the metric cache is rebuilt on the VM)
  "relay-bundle"    as above, plus a ONE-TIME dev-box push of the full-navtest frames (10.05 GB) and the full metric
                    cache (3.16 GB) so the final pulls them instead of streaming 128 GB / rebuilding (full only)
Pushes happen BEFORE the VM exists (dev-box uplink, not billed); only the VM's time is billed.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda n: json.load(open(os.path.join(HERE, n), encoding="utf-8"))  # noqa: E731
RELAY = "D:/Projects/TanitAD/colab/raw/2026-09-27-hf-relay"

ins = J("input_sizes.json")
cpu_side = J("seam_cpu_side_timing.json")
run1 = J("smoke/vm_results.json")
run2 = J("smoke_run2/vm_results_run2.json")
proof = json.load(open(f"{RELAY}/signed-proof.json", encoding="utf-8-sig"))

# ---------------------------------------------------------------------------------------------------------------
# MEASURED (sources in comments)
RATE = {"G4": 8.9, "L4": 1.54, "CPU": 0.08}          # CU/h: COLAB_PRO.md table (G4, L4); smoke/smoke_record.json (CPU)
CORES = {"G4": 48, "L4": 12, "CPU": 2}                # COLAB_PRO.md; smoke vm_facts
RAM_GB = {"G4": 189.9, "L4": 56.9, "CPU": 12}
FP32_TFLOPS = {"G4": 73.7, "L4": 11.0}                # COLAB_PRO.md (8192^2 matmul)
HF_PUBLIC_MBPS = 110.0                                # COLAB_PRO.md: one urllib stream on G4 108.7 / L4 109.9
N_SUB, N_FULL = 200, 12146
SEAM_CPU_S_SUBLIKE = cpu_side["regimes"]["sub200-like"]["s_per_token"]   # dev-box CPU, model stubbed
SEAM_CPU_S_FULLLIKE = cpu_side["regimes"]["full-like"]["s_per_token"]
HARNESS_FULL_SEQ_S = 1220.2                           # W3 raw/*_navtest.counts.json: 1050.1-1220.2 s (take the max)
HARNESS_SUB_S = 59.2                                  # W3 raw/A1sub200_navtest: 59.2 s for 200 tokens / 93 logs
E6_RUN_S = 49.5                                       # proptable/sub200_ep016/logs: median per-run wall (n=64)
MCACHE_S_PER_SCENE = 1.47                             # W3 RESULT.md: 1.47 s/scene, one worker
VENV_NAVSIM_S = run1["steps"]["navsim_venv"]["s"]     # smoke run 1 (CPU runtime, 2 vCPU)
S3_MBPS = run2["s3_speed_4streams"]["MBps"]           # smoke run 2: 4 parallel 12.5 MiB range reads (a floor: ramp-up)
RELAY_MBPS_CPU_VM = proof["pull"]["files"]["snap_epoch016.pt"]["mb_s"]   # coordinator's signed-link pull, CPU VM
PUSH_MBPS = 76052289 / 77.4 / 1e6                     # hf-relay/push_snap016.txt: dev box -> relay
SNAP_B = ins["checkpoint"]["snapshot_bytes"]
S = {k: v for k, v in (("db_sub_c", ins["nuplan_test_dbs"]["sub200"]["zip_compressed_bytes"]),
                       ("db_full_c", ins["nuplan_test_dbs"]["full"]["zip_compressed_bytes"]),
                       ("maps_zip", ins["nuplan_maps"]["zip_bytes"]), ("logs_tgz", ins["navsim_logs_test"]["tgz_bytes"]),
                       ("backbone", ins["backbone"]["model_safetensors_bytes"]),
                       ("frames_sub", ins["frames"]["sub200"]["jpg_bytes"]), ("frames_full", ins["frames"]["full"]["jpg_bytes"]),
                       ("cam_shards", ins["openscene_sensor_test_camera_local"]["bytes"]),
                       ("mc_sub", ins["metric_cache_navtest"]["sub200"]["bytes"]),
                       ("mc_full", ins["metric_cache_navtest"]["full"]["bytes"]),
                       ("export", ins["w3_export"]["bytes"]))}
MODEL_FINAL_B = 1.3e9                                 # ESTIMATED (ckpt_io.py docstring; not written yet)

# ---------------------------------------------------------------------------------------------------------------
# ESTIMATED (basis stated; P0 replaces the GPU time, the S3 rate and the GPU-VM relay rate with measurements)
ASSUMED = {
    "fwd_flops_T": 6.4,       # ViT-L/16 trunk 304 M params x 4 cams x 1,925 tokens (x2 FLOP/MAC) + attention; head ignored
    "gemm_efficiency": 0.75,  # fraction of the 8192^2 matmul rate a batch-1 ViT forward sustains (fp32, no TF32 GEMM)
    "vm_cpu_factor": 2.0,     # Colab VM core vs the dev box i9-12900F on the seam's single-threaded CPU phases; run 2:
                              # JPEG decode+resize 0.1254 s (2 vCPU Xeon 2.2 GHz) vs 0.0411 s (dev box, 24 threads)
    "model_load_s": 45.0,     # per seam process: trunk 1.21 GB + snapshot + build (run 3: ViT-L build 4.9 s on CPU)
    "driverl_venv_s": 90.0,   # run 2/3 MEASURED 25-31 s with torch 2.7.1+cpu; + ~1 GB cu128 wheel + ~3 GB nvidia-* wheels
    "extract_overhead": 1.25, # inflate / untar on top of pure transfer
    "teardown_s": 60.0,       # download outputs + stop + server-side check
    "new_s": 15.0,            # COLAB_PRO.md / smoke: 14.6-15.7 s to READY
    "relay_mbps_gpu_vm": RELAY_MBPS_CPU_VM,   # conservative: the CPU-VM measurement; GPU VMs pull public HF at ~110
}


def gpu_s_per_token(g):
    return ASSUMED["fwd_flops_T"] / (FP32_TFLOPS[g] * ASSUMED["gemm_efficiency"])


def mb(b):
    return b / 1e6


def project(split, gpu, route, relay_mbps=None):
    rl = relay_mbps or ASSUMED["relay_mbps_gpu_vm"]
    n = N_SUB if split == "sub200" else N_FULL
    f = ASSUMED["vm_cpu_factor"]
    ex = ASSUMED["extract_overhead"]
    cpu_tok = (SEAM_CPU_S_SUBLIKE if split == "sub200" else SEAM_CPU_S_FULLLIKE) * f
    g = gpu_s_per_token(gpu)
    shards = max(1, min(CORES[gpu] // 4, 8 if gpu == "G4" else 3))
    seam = max(n * g, n * cpu_tok / shards) + ASSUMED["model_load_s"]
    setup = ASSUMED["new_s"] + VENV_NAVSIM_S + ASSUMED["driverl_venv_s"]
    # independent sources run CONCURRENTLY; the data phase is the slowest chain
    s3_chain = (mb(S["db_sub_c" if split == "sub200" else "db_full_c"]) + mb(S["maps_zip"])) / S3_MBPS * ex
    if split == "sub200":
        relay_b = SNAP_B + S["frames_sub"] + S["mc_sub"] + S["export"]
        hf_chain = (mb(S["logs_tgz"]) + mb(S["backbone"])) / HF_PUBLIC_MBPS * ex
        cpu_chain = 0.0
    else:
        relay_b = MODEL_FINAL_B + S["export"]
        hf_chain = mb(S["logs_tgz"]) / HF_PUBLIC_MBPS * ex
        rebuild = N_FULL * MCACHE_S_PER_SCENE * f / max(CORES[gpu] - 2, 1)
        if route == "relay-bundle":
            relay_b += S["frames_full"] + S["mc_full"]
            cpu_chain = 0.0
        else:
            hf_chain += mb(S["cam_shards"]) / HF_PUBLIC_MBPS * 1.1       # stream 32 shards, keep 48,584 JPEGs
            cpu_chain = (mb(S["maps_zip"]) / S3_MBPS + mb(S["logs_tgz"]) / HF_PUBLIC_MBPS) * ex + rebuild
    relay_chain = mb(relay_b) / rl * ex
    data = max(s3_chain, hf_chain, cpu_chain, relay_chain)
    harness = HARNESS_SUB_S * f if split == "sub200" else HARNESS_FULL_SEQ_S * f / max(min(CORES[gpu] // 2, 16), 1) + 60
    par = max(1, min(CORES[gpu] - 2, int(RAM_GB[gpu] // 1.0)))
    e6 = (-(-64 // par)) * E6_RUN_S * f if split == "sub200" else 0.0
    total = setup + data + seam + harness + e6 + ASSUMED["teardown_s"]
    crit = max((("S3 DBs+maps", s3_chain), ("HF public", hf_chain), ("cache rebuild", cpu_chain),
                ("relay pull", relay_chain)), key=lambda kv: kv[1])[0]
    return {"split": split, "gpu": gpu, "route": route, "relay_mbps": rl, "seam_s": round(seam),
            "gpu_s_per_token_est": round(g, 3), "seam_shards": shards, "harness_s": round(harness), "e6_s": round(e6),
            "setup_s": round(setup), "data_s": round(data), "data_critical": crit, "total_min": round(total / 60, 1),
            "CU": round(RATE[gpu] * total / 3600, 2)}


def main() -> int:
    rows = [project(sp, g, "public+relay") for sp in ("sub200", "full") for g in ("L4", "G4")]
    rows += [project("full", g, "relay-bundle") for g in ("L4", "G4")]
    rows += [project("full", "G4", "relay-bundle", relay_mbps=HF_PUBLIC_MBPS)]      # if GPU VMs pull the relay at ~110
    pushes = {"snapshot_s": round(mb(SNAP_B) / PUSH_MBPS), "sub200_frames_cache_export_s":
              round(mb(S["frames_sub"] + S["mc_sub"] + S["export"]) / PUSH_MBPS),
              "model_final_s": round(mb(MODEL_FINAL_B) / PUSH_MBPS),
              "full_frames_s": round(mb(S["frames_full"]) / PUSH_MBPS), "full_cache_s": round(mb(S["mc_full"]) / PUSH_MBPS)}
    out = {"assumed": ASSUMED,
           "measured": {"rate": RATE, "cores": CORES, "fp32_tflops": FP32_TFLOPS, "hf_public_mbps": HF_PUBLIC_MBPS,
                        "s3_mbps_4streams": S3_MBPS, "relay_pull_mbps_cpu_vm": RELAY_MBPS_CPU_VM,
                        "relay_push_mbps_devbox": round(PUSH_MBPS, 3), "seam_cpu_s_sub_like": SEAM_CPU_S_SUBLIKE,
                        "seam_cpu_s_full_like": SEAM_CPU_S_FULLLIKE, "harness_full_seq_s": HARNESS_FULL_SEQ_S,
                        "harness_sub_s": HARNESS_SUB_S, "e6_run_s": E6_RUN_S, "mcache_s_per_scene": MCACHE_S_PER_SCENE,
                        "venv_navsim_s": VENV_NAVSIM_S, "sizes_bytes": S, "snapshot_bytes": SNAP_B},
           "devbox_push_seconds_unbilled": pushes, "rows": rows}
    json.dump(out, open(os.path.join(HERE, "projections.json"), "w", encoding="utf-8"), indent=1)
    print(f"{'split':7s} {'gpu':3s} {'route':13s} {'relay':>5s} {'setup':>6s} {'data':>6s} {'critical':>14s} "
          f"{'seam':>6s} {'harn':>5s} {'E-6':>5s} {'min':>6s} {'CU':>5s}")
    for r in rows:
        print(f"{r['split']:7s} {r['gpu']:3s} {r['route']:13s} {r['relay_mbps']:5.1f} {r['setup_s']:6d} {r['data_s']:6d} "
              f"{r['data_critical']:>14s} {r['seam_s']:6d} {r['harness_s']:5d} {r['e6_s']:5d} {r['total_min']:6.1f} {r['CU']:5.2f}")
    print("dev-box pushes (s, before the VM exists):", pushes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
