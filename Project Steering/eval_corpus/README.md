# v1arch clean-val corpus — the 19 canonical val episodes NOT in its training pool

**MEASURED 2026-08-05.** `v1arch_clean_val_19.txt` (md5 `92ce42e4cfe25a8a375c681ef5eb70da`,
verified byte-identical on pod4) is the admissible val split for `flagship-v1arch-v2bal-30k`.

## Why 19 and not 40

The canonical val is 40 episodes. Against v1arch's 9000-clip training pool
(`epcache-physicalai-v2bal-4b7eeeac222d`):

| | |
|---|---|
| canonical val episodes **in** the training pool (trained on) | **21** |
| canonical val episodes **NOT** in it (clean) | **19** |

⚠️ **The 21 are the only ones available in v1arch's 256×256 geometry — because being in the pool is
exactly what makes them available.** The 19 clean ones exist on the pods only at **256×640
cylindrical**, which a 256×256 checkpoint cannot consume. Hence the build.

## Why the eval could not simply be run

`eval_flagship_v4.py` ran to completion on the 21 contaminated episodes (3937 windows, ckpt step
29999) and **refused to certify its own output**:

```
"canary_ade_2s_MEASURED": 0.6838   vs v1's known full-set 0.4271, tolerance 0.05
"HARNESS_VALIDATED": false
"verdict": "HARNESS NOT VALIDATED — DO NOT proceed to score any v4 checkpoint …"
```

⭐ That is the parity guard working. The canary exists to reproduce v1's canonical-val number; fed a
non-canonical split it stops instead of emitting a plausible figure.

## Build plan (remaining work)

1. Map each of the 19 clip UUIDs to its shard via `metadata/` in
   `nvidia/PhysicalAI-Autonomous-Vehicles` (70 775 files, organised by MODALITY — camera 22 022,
   calibration 18 876, radar 17 285, labels 9 438, lidar 3 146 — **clip UUIDs are not in filenames**).
2. Download only the needed shards (camera + calibration + labels; radar/lidar are not ingested —
   `physicalai_r0.py` reads 4 features, plus `camera_intrinsics` / `sensor_extrinsics`).
3. Build v2 caches at **256×256** to match v1arch (`image_size 256, image_h 256, image_w 256,
   n_stack 3`, manifest version 3).
4. Re-run `eval_flagship_v4.py --v2-val-cache <clean19> --require-parity` and confirm
   `HARNESS_VALIDATED: true` before quoting anything.

⚠️ **n = 19 halves the episode count.** The episode-cluster bootstrap resamples EPISODES, so CIs
will be materially wider than the 40-episode arms. Admissible and comparable episode-for-episode,
but less powerful — state it beside every number.

## Environment state

* pod4: GPU idle, training finished (step 29999). `huggingface_hub` installed, HF auth OK
  (user `Sayood`, token in `/root/.cache/huggingface/token`, mode 600).
* ⚠️ `df` on `/workspace` reports the **965 TB cluster**, not the pod quota — use a real `dd` write
  test before committing to a large download (CLAUDE.md).
* ⛔ `Keys.txt` holds **four** live credentials (OpenRouter, HuggingFace, Google AI Studio, NGC).
  All four were exposed to a session transcript on 2026-08-05 and **should be rotated**.

---

## ⭐ SUPERSEDED — the corpus DOES ship an official split, and it is CLEAN

**Corrected 2026-08-05 after the PI challenged the "no split" claim. The challenge was right and my
claim was wrong** — I had searched FILENAMES (70 775 paths, 0 hits for split/train/val/test) and not
the dataset card. Root-cause class: *absence found at ONE location is not absence*, which CLAUDE.md
already carries and which I re-committed anyway.

`reasoning/ood_reasoning.parquet` — 1 740 rows, columns `[feature, event_cluster, events, split]`,
indexed by `clip_id`:

| split | clips |
|---|---|
| train | 1 450 |
| val | **290** |
| test | held out (OOD Benchmark Challenge) |

⚠️ **Scope: 1 740 of 306 152 clips (0.57 %), and it is the OOD *reasoning* benchmark split**
(Chain-of-Causation labels), not a trajectory train/val split of the full corpus. So
`physicalai-train-e438721ae894` / `physicalai-val-0c5f7dac3b11` remain the PROGRAMME's own splits,
and the v1arch leak finding stands unchanged.

### MEASURED: the official val is disjoint from v1arch's training pool

```
official val clips           290
v1arch v2bal pool          9 000
official_val ∩ pool            0     ← zero contamination
official_val clean           290
```

⇒ **This is the better eval corpus**, and it supersedes the 19-episode plan:

| | n=19 (ours) | n=290 (official) |
|---|---|---|
| clean vs v1arch | yes | yes |
| episode count for the cluster bootstrap | 19 | **290** |
| who defined it | us, post hoc | **NVIDIA, externally** |

⚠️ **Two caveats that must travel with any number from it.**
1. These clips were selected for **OOD reasoning** — an out-of-distribution event set, so it is a
   *harder and different* distribution from a random val. Not a flaw; it must be labelled.
2. It is **not** `physicalai-val-0c5f7dac3b11`, so numbers are **not** directly comparable to arms
   scored on the canonical val. To restore comparability, score the other arms on it too — which is
   cheap, since it is disjoint from every pool by construction of being held out.

`v1arch_clean_val_19.txt` is retained as the canonical-subset fallback for when comparability with
the existing 40-episode arms matters more than power.
