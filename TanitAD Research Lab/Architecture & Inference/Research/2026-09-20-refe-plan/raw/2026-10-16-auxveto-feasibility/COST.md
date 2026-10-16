# COST: CPU time to obtain held-out predicted rasters (C3, and the full navtest if features exist)

**2026-10-16.** Evidence class on every row: M = MEASURED (artifact named), I = INHERITED (another package's measurement, not re-run), E = ESTIMATED (arithmetic from M/I rows).
Dev box, 16 threads shared with other sessions' jobs, so every CPU figure is an upper-ish bound. The dev-box GPU was held by another session's job (`devbox_gpu.lock`) and was not touched.

## 1. The 1,058 C3 tokens: the predictions already exist
| what | cost | class | where |
|---|---|---|---|
| held-out predicted rasters, A2 (V, seed 0, 10,000 updates, IoU 0.675), all 5 folds, all 1,058 tokens | **0** | M | `scene_tap_probe_a2/units/V_s0_f*.npz` (INVENTORY section 5) |
| the same for 2,000-update V (3 seeds), S (3 seeds), C (shuffle control), and 400-update V | **0** | M | `scene_tap_probe/units/`, `scene_tap_probe_a3r/units/` |
| S1 heads (400 updates, navtrain-trained, IoU 0.50) re-run from the cached fp32 ctx | **252 to 470 s CPU per arm per 1,058 tokens = 0.24 to 0.44 s / token** | M | `raw/2026-10-16-aux-shift-locate/logs/s1_regen_navtest_preds.log` (470 s, 252 s, 262 s) |

## 2. If a NEW head must be trained on C3 features (the probe family has NO saved weights)
| what | cost | class |
|---|---|---|
| one fold unit, 10,000 updates, batch 8, GPU RTX 4060 (shared) | 1,190 s; **5 units = 5,858 s = 1.63 h GPU** | M (`A2_RESULT.json` wall time; unit `meta_json.info.wall_s = 1190.1`) |
| one update on CPU (`scene_tap_probe.train_unit(device="cpu")`, real shapes `[8,7680,256]`, 16 threads) | **5.4 s / update** | M (`CPU_TIMING.json`, this package; synthetic tensors, timing only) |
| one fold unit on CPU, 10,000 updates | 54,000 s = **15.0 h** | E (5.4 s x 10,000) |
| the 5 fold units on CPU | **75 h = 3.1 days** | E |
| CPU prediction of held-out tokens with a trained head | **0.178 s / token** (188 s for 1,058) | M (`CPU_TIMING.json`) |

⇒ A new head on C3 is a **GPU job (1.6 h per configuration)**. On CPU it is not a same-day job.
The 2,000-update configuration costs one fifth of that (CPU 15 h for 5 units, E).

## 3. The full navtest (12,146 tokens, 136 logs)
| ingredient | status | cost | class |
|---|---|---|---|
| GT rasters, 12,146 keys | **exist** | 0 | M (`label_store_navtest`, store_sha256 `985a3bc7…`) |
| u400 `visual_ctx` / `scene_ctx` | **do NOT exist beyond the 1,058 C3 tokens** | GPU: 2.15 s / token (`dump_seconds 2275` for 1,058, includes scenario build) x 12,146 = **7.3 h**; warm / cold smoke figures 3.5 / 12.5 s per token give 11.8 to 42 h | M for the unit costs, E for the scaling |
| the same on CPU | not measured at scale; the C3 2-token CPU smoke took 57 s total, so up to ~28 s / token (includes model load, an upper bound) | **up to ~95 h** | I (`c3-selection-headroom/HANDOFF.md`), E |
| ctx storage | `visual_ctx` 7.86 MB / token x 12,146 = **95.5 GB** vs **16 GB free on D:** | must stream (trunk -> head -> discard) | M (file size), E |
| | `scene_ctx` 65.5 KB / token x 12,146 = 0.8 GB | fits | M, E |
| head forward from cached ctx | CPU 0.18 to 0.44 s / token | **36 to 89 min** for 12,146 | M rows of sections 1 and 2, E |
| a navtest cross-fold head (the probe recipe on all folds) | needs ctx for ALL 12,146 tokens first | GPU 0.1135 s / update (M, PREREG probe) x updates x 5 folds; CPU 5.4 s / update | M, E |
| S1 heads (navtrain-trained) on the full navtest | held-out by construction (token and log overlap 0), IoU 0.50 on C3 | head forward only, once the ctx exists | M |

## 4. The NAVSIM side of a veto read (not asked, but it is the other half of the bill)
| what | cost | class |
|---|---|---|
| DAC of all 64 proposals, re-derived from NAVSIM's own code (`tracked_dump.py`), 1,058 tokens | **330 to 347 s wall per shard x 4 shards in parallel** (264 to 265 tokens each, under contention) | M (`logs/tracked_*.log`) |
| the same for 12,146 tokens | **~63 min on 4 workers** (scaled; ~0.5 to 1.3 s per token per worker) | E |
| the full 6-sub-score harness for all 64 proposals, 1,058 tokens | 1,852 s on 4 workers | M (`c3-selection-headroom/STATUS.json` `score_seconds`) |
| the same for 12,146 tokens | **~5.9 h on 4 workers**, plus the proposal dump (GPU, 7.3 h as above) | E |
| P2's full-navtest table | in progress (990 of 12,146 tokens dumped, no `P2FULL_DONE`) | M (`p2_full_table/dumps/dump_main_c00.npz`) |

## 5. Reading
* **C3 costs nothing more** for the held-out predicted rasters the veto read needs. Its limit is a pre-registration, not compute.
* **The full navtest is blocked on features, not on the head**: no ctx exists for 11,088 of the 12,146 tokens, `visual_ctx` cannot be cached on D: (95.5 GB), and the probe-family heads have no weights. The cheapest route to a full-navtest read is: stream u400 trunk -> S1 head (navtrain-trained, held-out) -> raster, GPU ~7.3 h (E) plus ~1 h CPU; or `scene_ctx` caching (0.8 GB) with a navtest cross-fold S-arm head (IoU 0.609 at 2k updates).
* **RL-val has no cheap route**: it needs navtrain `visual_ctx`, of which the dev box holds 6,044 of 412,148 images (I, `S1_DATA_AUDIT.md`).
