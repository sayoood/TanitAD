# Reachability of `--w-field` from P7's trainer to `.backward()` (CPU, real data, P7's model_final) — 2026-11-17

Verdict: **the flag IS wired in the trainer P7 was trained with. Nothing to patch.** No patch file is needed or written.
Evidence class: MEASURED (ours, CPU, `CUDA_VISIBLE_DEVICES=-1`) unless marked. Trainer of P7: `refe/train.py` at commit `181c7a3b86d4` (sha256 `eafd03e576ba…`; launch record `yawrate_wp/YP7/LAUNCH_RECORD_181c7a3b.md`).
P7's own `config.json` is on the pod / HF only (not read: no pod use). The argv is therefore taken from `refe/p7_pod.sh:346` at `181c7a3b` (INHERITED, `--w-field 0.0` is NAMED there).
`PK/` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. Field-related lines of `181c7a3b`'s `train.py` and of the tip `6b789b7e`'s are identical (diff over every `w_field|l_field|field_` line is empty, MEASURED); the tip adds only the P5-C traj-label hook.

## The trace (line numbers are `PK/refe/train.py` at 181c7a3b)
| hop | where | what |
|---|---|---|
| argparse | `:2227` `--w-field` (default **0.0**), `:2230-2240` `--field-margin 0.25` / `--field-interp 5hz` / `--field-labels` / `--field-shuffle-seed` | declared; the default disables the term, so an arm must NAME it |
| namespace | `:2252` `run(build_parser().parse_args())`, `run(a)` at `:1199` | `a.w_field` |
| validation | `:1235-1239` | `--w-field` finite and >= 0; `> 0` needs `--field-labels` (refuses otherwise) |
| labels to the loader | `:1415`, `:1674` `TargetBank(field_labels=a.field_labels, …)`; `:663` `AuxLabelSource(…, pinned_sha256=PINNED_STORE_SHA256 (the FULL store 4a17db67), pinned_keys=103139)`; item tail `(sdf int8 [160,128], valid)` | `:1724` `field_on`, `:1735` `sdf_t, valid_t` |
| the term | `:1766-1768` `if a.w_field > 0.0: l_field, fparts = field_loss(traj, sdf_t, margin=…, valid=…, interp=…, return_parts=True)` (`traj` is the FP32 proposal tensor `[B,64,20,3]`) | computed ONLY when w > 0 |
| into the loss, **online labeller (P7's path)** | `:1781` `"l_field": l_field` rides in the deferred dict `P`; `:1907-1908` `if a.w_field > 0.0: tot_ = tot_ + a.w_field * P["l_field"]`; `:1911` `(tot_ / a.accum).backward()` | the line P7's argv reaches |
| into the loss, bank labeller | `:1925-1926` `loss = loss + a.w_field * l_field`; `:1930` `loss.backward()` | the line the CPU probe exercises |
| telemetry | `:1957-1959` `field_L`, `field_w`, `field_viol`, `field_inext` per logged row; `:1712` the `field` event | what G-LIVE / K6 read |

## The probe (`code/fieldp7_reach.py`, `code/fieldp7_reach_compare.py`, `code/run_reach_cpu.sh`)
One REAL `train.run(a)` per arm, stopped at the first optimiser step (batch 1, accum 1); the gradient is captured at `clip_grad_norm_` (pre-clip, what the optimiser would see). ViT-L blocks use activation checkpointing (memory only; the LoRA gradient is still computed).
Data: one real navtrain row (`2021.10.05.07.38.12_veh-50_01085_01463` / `22668531ae67547f`, bank row 934 of the local rank-0 bank `E:/Projects/TanitAD/data/refe_m5/bank_train`), its real SDF from the pinned store (`store_sha256 4a17db67…`, 103,139 keys), P7's `model_final.pt` (`2613fc17…`, strict warm start). L_field on this row, w = 1: **0.0723**.
**Row choice:** the first attempt used a row whose fan was already clean (L_field exactly 0.0: the draft's own clean-fan identity), so no gradient could reach; it is kept as `raw/reach_attempt1_cleanfan_row/` (F reads RED there, honestly). The second row was chosen by `L_field > 0` only (`code/fieldp7_pickrow.py`, `raw/pickrow.json`), never by any metric.
Arms: **F** `--w-field 2.0 --field-labels`; **C** `--w-field 0.0 --field-labels`; **N** no field flag at all; **M** = F on a copy of the trainer with line `:1926` replaced by `pass` (the term is computed but never added).

| check | result (`raw/reach/COMPARE.out`, `COMPARE.json`) |
|---|---|
| R1 term finite and non-zero in F (1 call), absent in C | PASS (0.0723; C has 0 calls) |
| R2 contribution `g_F − g_C` non-zero on every parameter of `traj_head` 4/4, `dec` 72/72, `queries` 1/1, `scene_proj` 2/2, `reg_compress` 14/14, backbone LoRA 96/96 | **GREEN** (abs sums 2717 / 4152 / 15.6 / 892 / 4262 / 173) |
| R3 contribution EXACTLY 0.0 on `score_q_mlp`, `score_dec`, `score_head` | **GREEN** (0 of 78 non-zero; note both arms ran the zero-placeholder scorer, so this reads the contribution `F−C`, not the totals) |
| zero weight: `g_C == g_N` on every parameter, `max abs diff = 0.0`, sha256 of all gradients equal; C never calls `field_loss` | **GREEN** (bit-identical) |
| mutation M (adding line deleted): same check on `M` vs `C` | **RED**: `R2 contribution is zero` on all six proposal-side groups (the term is computed, 0.0723, and never reaches a parameter) |

## What this does NOT show (stated, not hidden)
* The **online-labeller site** (`:1907-1908`, P7's real path) is traced by reading and by the 2026-10-13 tiny-model proof (`PK/raw/2026-10-13-field-loss-wiring/RESULT.md`, mutations on both sites, MEASURED there) but was **not executed here**: `--labeller online` needs the NAVSIM pdm workers + metric cache (pod). UNVERIFIED end to end on P7 scale; the launch gate's G-DVB multiplier-at-the-graph row exercises it on the pod.
* One row, one step: reach, not magnitude. Magnitude is the weight rule (`WEIGHT_FIELDP7.md`).
* cv2 is absent from the dev box's `tanitad` venv and `driverl-eval`'s torch segfaults on this model; image decode/resize used a PIL shim (`code/cv2_shim_pil.py`), bilinear instead of cv2's: sub-LSB pixel differences, irrelevant to reach.
* The scorer was the zero placeholder (no scorer targets locally); the proposal-side gradient does not depend on it.
