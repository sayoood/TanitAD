# PREFLIGHT_L7 -- is the H-L7-1 PILOT ready on the REAL Thor capture? (zero optimizer steps)

2026-10-22, dev box, CPU only (`CUDA_VISIBLE_DEVICES=""`, 4 threads), every process < 4 GB (peak working set 2.02 GB max). Banks: `D:/tanitad-archive/l7cap_thor_1022/{train139_s0,train139_s1,eval139_s0,eval139_s1}/dump`
(copy complete, `ZZL7ARCHDONEZZ` 12:33:05). Package: `.../2026-10-21-refc-l7-scorer` at 701922c. Evidence class of everything below: **MEASURED today**, artifacts `preflight_l7.json`
(`smoke/preflight_l7_real_banks.json`), `targets/*.json`, `dryrun/*/dry_run_log.json`, unless marked INHERITED / ESTIMATED.

## Verdict

**The PILOT can start the moment the PI says go -- with the changed code in the code-hold, not with the code as sealed at 701922c.** The file formats MATCH (no key, shape or dtype the code reads is
missing or different). What did NOT match was behaviour: eight deviations of the first-version code from the sealed PREREG / the capture's actual layout (M1, S1-S4, T1-T3, below); four of them (S1, S2, T1, T2) would have
produced a wrong or invalid PILOT *silently*, and only M1 and T3 would have raised. Preflight: **34 gates, 33 blocking all PASS, 1 non-blocking FINDING** (a property of the DAC label, section 4).

## 1. What matched (all four banks)

| item | read | result |
|---|---|---|
| `fan_extras.npz` | keys `clip_index, ws, t, inference_seed, anchor_traj[W,149,8,2], anchor_logits, refined_logits[W,149], sel_score, sel_score_v3, traj, traj_base, goal_dist, r8_sub_logits[W,149,4], reach_keep, ceil_keep, r8_emit_keep, e9_keep, e9_argmax, sel_idx, sel_idx_base, r8_cand_src, v0, is_alloc[149]` | every key `train_l7` / the builder read is present; 0 non-finite in `anchor_traj, sel_score, refined_logits, v0`; `sel_idx` in `reach_keep` on 100 % of rows |
| `ep*.npz` | `g[n,4,2], v0, os[n,4,2], ws, eid[1], clip_index[1]` (139 files/bank) | (clip_index, ws) keys equal the fan's exactly (0 missing either way); `v0` equals the fan's `v0` (max abs diff 0.0) |
| `l7cap_labels.npz` | `r7_agent_fut[W,8,397,5], _valid[W,8,397], _label[W,8], map_drivable_u8[W,120,64] (u8 fraction 0..255), map_seen, map_label[W]` | key names equal the builder's `--agents-npz` contract |
| windows / seeds | train139 s0 == s1 on `(clip_index, ws)` and clip order; eval139 s0 == s1 | 4,855 / 4,754 windows; 139 clips each; train and eval clip sets (sha12 from the manifests) **intersect in 0** (K8) |
| capture identity | Thor `K3_eval139_s1_vs_fanceil_s1.json` (INHERITED, read via ssh cat) | `anchor_traj, sel_score, refined_logits, reach_keep, sel_idx, traj` **bit-exact** vs fanceil s1 |
| eval targets vs banked FANCEIL (INHERITED refs `FANCEIL_R8_s{0,1}.json`) | own-pick / oracle-in-fan / random-expectation S2 ADE | s0 0.29727 / 0.0847 / 0.83181 vs 0.29727 / 0.0847 / 0.8318; s1 0.29903 / 0.08406 / 0.8318 vs 0.299033 / 0.0841 / 0.8318 (|diff| <= 5e-5) |
| e9 mask | `train_l7.effective_keep` (our restatement of `refc_v3.e9_rank`) vs the capture's own dumped `e9_keep` | **0 mismatching cells** in 9,710 train + 9,508 eval rows (an independent derivation) |
| controls on real masks | K1 const scorer, K2 oracle scorer, K5 sign-flipped | lowest-emittable index on 512/512; masked argmin to 0.0 m; flipped 11.48 m vs random 3.83 m |

`eid` in `ep*.npz` is the run-local episode index (== `clip_index`, 0..138), so it is **not** comparable across banks: K8 must use the manifest sha12 (it does).

## 2. What did NOT match -- and the fix (all in the code-hold)

| # | where | mismatch | measured evidence | fix |
|---|---|---|---|---|
| **M1** | `train_l7.py` | takes ONE `--fan-npz` / `--targets-npz`; the PILOT is TWO inference seeds | old script: `train_l7.py: error: unrecognized arguments: b` (measured) | `nargs="+"`, banks concatenated, validation split on the episode id so both seeds of an episode stay together (val = 14 of 139 episodes, 980 windows; train 8,730) |
| **S1** | `train_l7.py` masks | loss, validation and early stopping used `reach_keep` ALONE; PREREG sec. 1 / DESIGN sec. 3.3: masks are `e9_rank` (reach AND ceiling AND emit, with fallbacks) | `reach_keep` keeps **149.00 / 149** candidates; the e9 keep **133.86**; the ceiling removes 147,052 of 1,446,790 train cells and acts on **66 %** of rows; the shipped pick is always inside it. The scorer would be trained to rank, and early-stopped on picking, candidates the shipped selection can never emit | `effective_keep` = `RS.e9_masked` on a zero score for the loss/validation mask; validation pick uses `RS.select(score, reach, ceil, emit)`. Mode-F FEATURES deliberately still use `reach_keep` (that is what `integration.diff.txt:29-30` passes at inference); a test pins that the loss mask does not leak into the features |
| **S2** | `train_l7.py` guard | the eval-bank refusal reads `windows_json` of `<dump>/../fan_roll_s1.json`; the capture wrapper's roll json has **no such key** (it has `mode` and `child_argv --windows-json`), the archive copy carries **no roll json at all** (only `dump/`), and a bank with no provenance passed silently | roll json on Thor (read): `mode: "eval139"`, `child_argv ... --windows-json .../windows_s1.json`; the old guard returns "do not refuse" on it (test, deliberate-regression arm) | provenance = roll json (`mode`, `--windows-json`) AND `manifest.json` (`corpus.labels_path` `..._eval...`, `corpus.episodes` `...eval139`), fail-CLOSED on unknown; 2 subprocess refusals in the preflight |
| **S3** | `train_l7.py` stop rule | patience 3 on the validation *pick cost*; PREREG sec. 1 says "validation loss" | code read (PREREG sec. 1 vs `train_l7.py` main loop) | default `--early-stop-on loss`; `pick` restores it; both logged every epoch; **this is a semantics choice for the Master Mind to confirm** |
| **S4** | `train_l7.py` K7 | no per-head held-out AUC (PREREG sec. 7 K7: `comf` AUC >= 0.95; sec. 4: report every head's AUC) | code read | tie-aware AUC for `comf, nc, ttc, dac` logged per epoch (`val.head_auc`) |
| **T1** | `l7_targets_pai.py` NC/TTC | with `--agents-npz` the oracle returns `nc_mask`/`ttc_mask` True for EVERY window (agents passed), so a window whose agent labels are incomplete scored a near-1 NC / TTC (unlabelled slots add no hit) **with the mask on** -- the silent "no collision" its own docstring forbids. The stack's rule is `r7_agent_fut_label.all()` (`refc_v3_train.py:6617`) | real eval139 s1: **280 of 4,754 windows (5.9 %)** have incomplete labels on their own grid; OLD builder, 40 of them: mask on in **40/40**, mean NC **0.971** / TTC 0.950 (fabricated; labelled windows average NC 0.767). Train: 4.6 % of windows. NEW: mask IN iff the labels cover the window's grid (95.4 % of train windows, 94.1 % of eval) | override the masks with the label rule; control `C_NC_unlabelled_windows_with_a_mask_true == 0`; refuse a labels file that has agents but no `_label` |
| **T2** | `l7_targets_pai.py` DAC | `dac` abstained on every window although the capture carries the SAM3 raster in the SAME labels file | OLD `dac_mask` any: 0/24; NEW 24/24 on the same windows; coverage now 100 % train / 98.57 % eval (`map_label`) | drivable = u8/255, `seen`, AND `map_label` (`refc_v3_train.py:6648-6650`). Without it the `dac` head gets **no gradient** and "every parameter group gets a gradient" is false |
| **T3** | `l7_targets_pai.py` scale | one oracle call for the whole bank: NC/TTC distances are ~34 MB/window (B x 149 x 397 x 8 x 18 floats), i.e. **~165 GB** for 4,855 windows (arithmetic, ESTIMATED); and `np.load` was re-decompressing 300 MB per access | old builder measured 0.238 s/window at 16 windows | chunks of 8 windows, exact pruning of agent columns invalid everywhere in the chunk, labels read once. Chunk sizes 1/3/8/64 give bit-identical targets (test); **83-86 s and 1.75-2.02 GB per bank** (0.0162 s/window) |

Not a code change but required: **`build_gt6.py` must run per bank** (decisions present) or every window gets the 2 s grid (`grid = 4`) and `cost` is a 2 s ADE, contradicting PREREG sec. 2; with it
3,749 / 4,855 train and 3,668 / 4,754 eval windows have the 8-slot GT (the rest are grid 4). Commands in `PILOT_COMMANDS.md`.

## 3. Preflight numbers (train139 s0+s1: 9,710 windows; `preflight_l7.json`, wall 9 s, peak working set 1.43 GB, commit 1.94 GB)

| | L7-E | L7-F |
|---|---|---|
| parameters / parameter tensors | 154,504 / 34 | 155,400 / 34 |
| forward + `pai_loss` finite on 6 real batches of 32 | yes (batch 1: imit 4.82, reg 2.99, total 7.09) | yes (imit 4.72, reg 1.79, total 6.35) |
| gradient coverage (union over 6 batches) | **34/34 tensors, all 12 groups (embed, trunk.0, trunk.1, out_norm, 8 heads) finite non-zero** | **34/34, 12/12** |
| supervised cells per head over the 6 batches (of 32 x 134 kept) | imit/along/cross/comf/ep 25,255; nc/ttc 24,571; dac 24,947 | 25,215; 23,673; 24,923 |
| optimizer steps / state change | 0 (no optimizer object exists in the preflight) / state hash unchanged | same |
| fwd+bwd per batch-32 (median) | 0.0365 s | 0.0353 s (+ 0.0058 s features) |
| untrained `evaluate()` on 256 val windows | loss 7.09, pick cost 7.40 m, own pick 1.005, oracle 0.380 (val slice, 8-slot cost) | loss 6.49, pick 1.96 m |
| estimated s/epoch (ESTIMATED, x1.15) | 12.3 | 14.0 |

Target coverage (window fraction with any mask-in cell), train / eval: comf 1.00/1.00, ep 1.00/1.00, nc and ttc 0.954/0.941, dac 1.00/0.986. Positive rates where masked (train s0): comf 0.935, nc 0.781, ttc 0.657, dac 0.427.
Human path scored as a candidate (320 train windows): comf 0.975, nc 0.938, ttc 0.753.

## 4. FINDING (label quality, not plumbing): the SAM3 DAC target calls the human "off drivable" 30 % of the time

Orientation is right: on turning windows the human path scores DAC 0.729 against 0.120 for its y-mirror, and 0.699 against 0.069 when shifted 6 m left (blocking gate, PASS). But the human's own path has
DAC **0.699** over 8 slots and **0.867 already at 0.5 s** (my pre-set bar 0.80, not moved; non-blocking gate `human_path_dac_rate_ge_0.80` FAILS as a finding). With a looser cell threshold the rate rises
(0.20 -> 0.87 in a 600-window side probe, `dac_diag.py`), so the "drivable" class under-covers real road. Consequence: the `dac` head is trained against a pessimistic label and the secondary rule R-SAFE
(`log sigma(dac)`) inherits it; the primary R-IMIT does not read `dac`. Not fixed here: changing the DAC definition is a PREREG decision.

## 5. For the Master Mind

1. Land the code-hold; confirm the semantics choices S1 (e9 masks in the loss), S3 (stop on validation loss), T1 (NC/TTC mask = labels cover the window's grid, not all 8 slots, for grid-4 windows).
2. The PILOT's one eval139 read goes through the existing picker/battery lane (outside this package); no eval number of any scorer was produced here.
3. Unchanged, noted: builder control `C_GT` computes `norm(gz - gz)` (a tautology -- the class "a check that shares the defect it checks"); `refs/__init__.py` is emptied in the dev-box stack slice (the real one imports `refa`).
4. `fan_roll_s*.json` and the K3 json exist only on Thor; copies are in the hold under `capture/thor_run/`.
