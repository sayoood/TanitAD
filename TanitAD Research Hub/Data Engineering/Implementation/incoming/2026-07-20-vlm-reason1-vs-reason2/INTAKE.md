# INTAKE — Cosmos-Reason1-7B vs Cosmos-Reason2-8B route-labeling head-to-head

**Date:** 2026-07-20 · **Pod:** `tanitad-eval` (A40 46 GB) · **Write-up:**
[`../../../Research/2026-07-20-cosmos-reason1-vs-reason2-headtohead.md`](../../../Research/2026-07-20-cosmos-reason1-vs-reason2-headtohead.md)

Raw per-window records and the scored result for the controlled head-to-head that decides which VLM labels
TanitDataset v1. Both arms ran **on the same pod, over the same 200 windows, with the same prompt version
(`vlmroute-2026-07-20-a`), back to back** — pod3's 400 banked Reason2 records were produced on a *different*
val build and are used only as a consistency check, never pooled.

## Files

| file | what it is |
|---|---|
| `windows.json` | the **shared** window manifest — 200 windows over 40 episodes (`physicalai-val-0c5f7dac3b11`, `stride 40`, `t0 0`), each carrying its kinematic **v2.1** ground truth (`refb_labels.route_from_future_v21`). Built ONCE and read back by every arm; this is what makes the comparison paired. |
| `reason1.jsonl` | 200 Pass-A records, `nvidia/Cosmos-Reason1-7B` (`qwen2_5_vl`) — one JSON object per line, including the **raw model reply**, generated token count, per-window seconds and the parse/enum outcome. |
| `reason2.jsonl` | 200 Pass-A records, `nvidia/Cosmos-Reason2-8B` (`qwen3_vl`), same schema. |
| `run_reason1.json`, `run_reason2.json` | per-arm run summary: load time, weights resident, peak VRAM, wall clock, s/window. |
| `reason1_B.jsonl`, `reason2_B.jsonl` | 50 **Pass B** records per arm (full v3 vocabulary), a deterministic `--window-stride 4` sub-sample of the same manifest. **Schema adherence only** — Pass B is shown the numeric future ego track, so no accuracy or agreement number may be derived from it. |
| `run_reason1_B.json`, `run_reason2_B.json` | Pass B run summaries. |
| `results_passB.json` | per-slot in-vocab / violation / unknown / missing rates over the 32 categorical vocabulary slots, both arms. |
| `results.json` | the full scored comparison — per-arm metrics, confusion matrices, failure taxonomy, slot fill, throughput, inter-arm Cohen's κ, and the paired tests (McNemar exact + paired episode-cluster bootstrap). |

## Re-score without the pod

```bash
# Pass A — the head-to-head accuracy statistics
python stack/scripts/vlm_compare_score.py --out <this dir> --arms reason1,reason2 --json results.json

# Pass B — SCHEMA ADHERENCE ONLY (emits no accuracy or agreement numbers)
python stack/scripts/vlm_compare_score.py --out <this dir> --arms reason1_B,reason2_B     --passb-slots --json results_passB.json
```

Verified to reproduce `results.json` **byte-identically** from the `.jsonl` form (the scorer accepts either the
pod's per-window directories or these consolidated files). Requires only `numpy` + `taniteval/taniteval/ci.py`.

## Provenance and admissibility

- **Pass A only.** Pass A is not shown the numeric future ego track, so its ROUTE is evidence independent of our
  kinematics. Pass B ROUTE is downstream of them by construction and is refused entry to every accuracy and
  agreement number (`vlm_compare_score.py --pass` accepts `A` only).
- **Intervals are the episode-cluster bootstrap** (`taniteval/taniteval/ci.py`, 2000 draws, clustered on
  EPISODE), paired where two arms are compared. The retired `overlapping_holdout_se` appears nowhere, and no two
  independent intervals were combined in quadrature.
- **Ground truth is kinematic v2.1**, stamped into each record on the pod so scoring needs neither the GPU nor
  the 4.4 GB val build.
- Generating harness: `stack/scripts/vlm_model_compare.py`. Scorer: `stack/scripts/vlm_compare_score.py`.
  Both are in the repo — nothing here lives only on a pod.
