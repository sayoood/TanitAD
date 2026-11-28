# S0-L read package for STOPPERC (NOT RUN: L_s0 is not banked)

Card: `theory/stopperc/PREREG_V8F_STOPPERC.md` §4, S0-L line (`:48`): *"STOPCUE on L vs B-prime s0, both dumped T1-kin through their own caches. A gain counts only if it exceeds the AUROC floor of 0.037."* L-DAG is "conditional on S0-L" (`:64`); PERC-NULL's next step is "L-DAG if S0-L passed, otherwise the 416x1024 source (Q5)" (`:83`).
Written 2026-11-13 on the dev box, CPU only, nothing contacted off-box, nothing landed. Evidence classes: **M** MEASURED here (artifact named), **I** INHERITED (another file, not re-verified), **E** ESTIMATED, **U** UNVERIFIED.
Tier of everything below: **T1-kin**, first tick, eval-141; a decision-head diagnostic, never driving performance.

## 1. Find-first answers (file:line)

**1a. STOPCUE.** `theory/stopcue/stopcue_read.py`.
* Statistic: `read_arm` (`:69-97`): windows = `has` True (`:75`); score = `goal_prob[:, 6]`, p(STOP_POINT) (`STOP_I = 6`, `:19`; `:77`); label = `stop_hit` (`:76`); AUROC = Mann-Whitney, ties one half (`auroc`, `:25-41`). Beside it: lon-STOP score vs regime stop `v_end_gt < 0.5` (`:78, :87`), AP, recall at 5 % FPR.
* Input file: the per-window STOPDEC npz (`stopdec_<tag>.npz`, keys `has, stop_hit, goal_prob[N,22], lon_prob[N,7], eid, v_end_gt, v2_gt, ws, clip_index, …`), written by `stopdec_read.py` from an arm's T1-kin dump folder. For the STOPBAL / STOPPERC arms the driver is `theory/stopbal/read/stopdec_arm.py`.
* Estimator: `stat_cluster_bootstrap` (`:90`, B 2,000, seed 0, by episode). For two arms: `taniteval.ci.paired_stat_cluster_bootstrap` (`ci.py:360`), the same draw for both arms; the STOPPERC reads use it (`stopperc_s1_read.py:auc_cell`, `_auc_stat` `:91`, `auroc_fast` `:78`, `cue_arrays` `:113`).
* Windows **(M)**: counting `stopdec_PARITY_DAGOACC_s0.npz` and `stopdec_BAL-NULL_s0.npz` gives **2,365 windows / 131 positives / 139 episodes**, equal to the card (`:42`) and to `README_READ_S1.md:41`. Every model-free array (`clip_index, ws, eid, has, stop_hit, v_end_gt, v2_gt`) is identical between those two files, so arms from different dumps share the windows.
* Floor 0.037 **(I)**: the card's figure; `stopbal/read/VERDICT_STOPBAL.md:35` calls it "about 0.037" (a replicate gap). Not re-measured here. Amendment 2 (`AMENDMENT_2_S0P.md:7`) adds "CI lower bound > 0" for HEAD-UNDERUSES (S0-P) only. The card and Amendments 1-4 are silent on the CI for S0-L, so the MM froze **`AMENDMENT_5_S0L.md`** on 2026-11-13, before any L data: **GAIN iff the paired dAUC point exceeds 0.037 AND the paired CI lower bound is above 0** (the HEAD-UNDERUSES form as Amendment 2 made it paired). `decide()` implements it, and the point-only reading is printed beside it (`point_only_beside`).

**1b. Where B-prime s0's dumps are.** **(M) None in T1-kin form.** Searched: `lorapod_eval/` (all of it), `F:/TanitAD-scratch/` (`otband_read/t1` holds only `argv_INIT_M2b_s{0,1}_*.txt`, no dump folder), every `stopdec` out folder, `find -iname "*m2bp*"` under the artifacts root and `C:/Users/Admin`. What exists for B-prime s0:
* checkpoint `lorapod_eval/bprime/M2b_s0/ckpt_last.pt` (+ `argv.json`, `summary.json`: 19,164 steps, argv `a2b68127`; sha `b9aaf33e…` per LOOP_STATE:544 **I**);
* pod-encoded eval-141 cache `lorapod_eval/eval/` (141 clips, 2.8 GB, `meta.json` pipeline `ra1a-pooled-v1`, RTX 4000 Ada, bf16) and `lorapod_eval/native_tracks.pt`;
* **mini-eval outputs** `lorapod_eval/out/eval_M2bp_s{0,1}.npz` (v8f_mini_eval, 2,399 windows, keys include `goal_prob[2399,22]`, `lon_logits`, `eid`). These carry p(STOP_POINT) but are **not T1-kin dumps** and lack `has` / `stop_hit`; their `eid` equals the STOPDEC npz's (M), so labels can be borrowed for a control only.
* **A T1-kin dump of a B-prime (M2b-recipe) checkpoint has not been made.** `v8f_t1_eval.py` was run on M2b-family inits (argv files `argv_INIT_M2b_s*` exist), but whether its STOPDEC step accepts a pod-trained M2b checkpoint is **U**.

**1c. How L's dumps will be produced, what exists, what is missing.**

| piece | exists? | where / note |
|---|---|---|
| L_s0 checkpoint | training | pod `/root/runs/lora/L_s0`, 19,164 steps, step 11,200 at 26.8 s/step in LOOP_STATE:1294 (**I**; ~59 h left then, **E**); `bank_ckpts.py` copies `ckpt_last.pt` to PRIVATE HF `Sayood/tanitad-lorapod-ckpts` every 1,000 steps and at the final step (`HANDOFF_LORAPOD3.md:12-13`; `HF_HUB_DISABLE_XET=1`) |
| L blob format check | yes | `t2v8f/code_lc/check_l_ckpt.py` (format, B not all zero, fold digest). A REAL L blob was never read (`HANDOFF_LC.md` UNVERIFIED 1) |
| LoRA fold (q, v of 24 blocks, `bf16(fp32(W)+fp32(B)@fp32(A)*alpha/r)`) | yes | `t2v8f/code_lc/t2v8f_lora_merge.py` (`fold_lora`); equal to the training side's `merge()` on CPU, 18 tests green (`HANDOFF_LC.md`). Cross-CPU / GPU digest equality **U** |
| L-encoded eval-141 cache (pooled `pool4x10`, `pool8x20`, `mean`, fp8) | **missing** | needs the encoder route `pull_encode_pooled.py` (`--source local --src-geom 256`, bf16; tip a9da6e8f) run with the merged DINOv3-L weights on the eval frames. `PREREG_V8F_LORA.md:115` (tip copy `scratch_lc_t2v8f/PREREG_V8F_LORA_tip.md:115`): "re-encoded on box 2's GPU at the merged weights". No script found that does fold + re-encode in one step: work item |
| eval frames for the re-encode | **U** | local `refcv5cmp/data/eval/*.v2ep.pt` (E:); the pod's eval frame root and whether it matches the pod cache build are **U** |
| T1-kin dump driver for L | **missing** | `run_dumps_stopbal.py:37` pins `ARMS` to the 7 BAL tags and refuses a tag without a BAL `BANK_MANIFEST.json` / levers (`:63-90`). It needs a sibling for tags `BPRIME_s0`, `L_s0`. `v8f_t1_eval.py` and `v8f_mini_eval.py` have **no LoRA path** (0 hits, card `:28`), so L is read from the merged cache through the head-only checkpoint. Whether `v8f_t1_eval`'s loader tolerates the extra `lora` / `encoder_levers` blob keys (the T2 policy ignores them, `HANDOFF_LC.md`) is **U** |
| STOPDEC step | yes, needs adaptation | `stopbal/read/stopdec_arm.py` (CPU; takes `--tags`, `STOPBAL_TREE`, `STOPBAL_DUMPS`); its forward pass must run from a tree whose `V8FMini` config accepts the checkpoint |
| the read | **built here** | `s0l_read.py` |

**Compute (CPU unless stated).**
* Fold of L: CPU, ~30 s **E** (`HANDOFF_LC.md`).
* L eval-cache re-encode: **GPU**, 141 clips. DINOv3 at ~2.9 s per ~100-frame episode (`T2_SIZING_V8F.md:58`, **I**) ⇒ well under 1 h **E**, plus model load. Venue: the pod (same device and route as B-prime's pod eval cache, but its RTX 4000 Ada is training L_s0 / L_s1, so it must be free) or box 2's GPU (the card's route, `PREREG_V8F_LORA.md:115`). Both need a PI / MM go.
* B-prime s0 T1-kin dump through the pod cache: CPU, 2 threads; a BAL-arm dump took 276 s and its STOPDEC pass ~74 s per arm (**M**, `DUMPS_20261111_180433.json`, `stopdec_run.log`); expect the same order for B-prime and L **E**.
* The read: 44 s for the full control suite at B 2,000 (**M**).

## 2. Steps from "L_s0 banked" to RESULT_S0L.json

Owner of every item is unknown (assign at launch). Items marked **WORK ITEM** do not exist yet.

1. **Confirm the bank.** L_s0 `done` (19,164 steps) and the final `ckpt_last.pt` in the HF ledger; sha256 recorded. Needs: HF token read in place. (Mid-run checkpoints are not S0-L's L_s0.)
2. **Fetch and check the blob.** `python -I t2v8f/code_lc/check_l_ckpt.py <ckpt> --expect-sha256 <ledger>`; rc 0 required. Needs: ~1 GB disk, CPU.
3. **WORK ITEM: L-merged eval cache.** Fold, then re-encode eval-141 with the B-prime cache's own pipeline (`ra1a-pooled-v1`, bf16, `local256`) at the merged weights. Write `meta.json` stating the merged-weights digest. Check: LoRA = 0 (B = 0) re-encode reproduces `lorapod_eval/eval` to rel-L2 ≤ 1e-3 on pooled fp8 if on the pod, per `PREREG_V8F_LORA.md` τ_route (the card's τ for another device is also 1e-3). Needs: GPU, frames, a go.
4. **WORK ITEM: T1-kin dump driver for `BPRIME_s0` and `L_s0`.** Clone the argv of `v8f_t1_eval.py` of the DAG-OACC baseline (as `run_dumps_stopbal.py` does), replacing `--ckpt`, `--tag`, `--out-dir`, with **`--eval-cache` = the arm's own cache root** (B-prime: `lorapod_eval`; L: the step-3 cache). Same tree, same environment, same thread count for both arms (the OTBAND amendment-1 trap: dumps of every arm in ONE environment). `CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=2`.
5. **STOPDEC** on both tags with `stopdec_arm.py --tags BPRIME_s0 L_s0 --outdir <dir>` ⇒ `stopdec_BPRIME_s0.npz`, `stopdec_L_s0.npz`. Gate: the model-free arrays of the two files equal each other and equal `stopdec_PARITY_DAGOACC_s0.npz`'s (`s0l_read.py` refuses otherwise).
6. **Identity control (card requirement for any new dump route).** The dumps' own-plan-vs-dump identity checks that `stopdec_arm` prints (`own_all_vs_ident_max_abs`, `top1_vs_dump_ol_max_abs_m`) must read 0.0 for both arms.
7. **Read.**
   ```
   set OMP_NUM_THREADS=2
   C:/Users/Admin/venvs/tanitad/Scripts/python.exe s0l_read.py read --L stopdec_L_s0.npz --B stopdec_BPRIME_s0.npz --out RESULT_S0L.json
   ```
   Output: `RESULT_S0L.json` with tier stamp, estimator, `counts` (must equal 2,365 / 131 / 139, else rc 2), AUROC of each arm, dAUC with the paired CI, `gain` (Amendment 5), `point_only_beside`, lon-STOP beside.
8. **Re-run the controls in the same environment** (`s0l_read.py controls`) and attach `CONTROLS_S0L.json`.
9. **Register.** `GOALS_AND_CLAIMS.md` row and the STOPPERC verdict in the same turn (MM). S0-L is one seed (L_s0 vs B-prime s0): the interval is "another draw of episodes" only; a gain beyond 0.037 is one seed pair, and "another training run" is what the 0.037 floor stands in for.

What S0-L licenses is the card's: a gain lets "L-DAG" proceed (`:64`); no gain does not. It never licenses a driving-quality claim.

## 3. Controls (M, `CONTROLS_S0L.json`, B 2,000 seed 0, 44 s)

| control | result |
|---|---|
| **B-prime s0 against itself**, on the real B-prime mini-eval scores (labels borrowed from PARITY; eid equal) | dAUC **0.0 exactly**, CI **[0, 0]**, no gain; AUROC 0.8748 (**tier: mini-eval, not T1-kin; control use only**) |
| BAL-NULL_s0 against itself, real T1-kin STOPDEC npz | dAUC 0.0, CI [0, 0], no gain; AUROC 0.8294 |
| planted gain above the floor | dAUC +0.0921 ⇒ counts as a gain; planted +0.0128 ⇒ does not |
| label shuffle | AUROC CI covers 0.5: [0.434, 0.535] (B-prime scores), [0.427, 0.519] (BAL-NULL) |
| counts | 2,365 / 131 / 139 equal STOPCUE's |
| alignment | one moved label in one arm is refused |
| **deliberate-regression arms (all RED, as required)** | inverted second copy: dAUC +0.659, gain ⇒ self control RED; real labels instead of shuffled: AUROC CI [0.777, 0.881] excludes 0.5 ⇒ RED; one positive dropped: 2,364 / 130 ⇒ counts RED; floor set to 0: the small plant counts as a gain ⇒ RED |

`test_s0l_read.py`: 12 passed in 4.4 s (CPU).
A smoke of the `read` path on two real aligned arms (BAL-NULL_s0 vs PARITY_DAGOACC_s0, B 500) ran end to end: dAUC −0.0063, CI [−0.0162, +0.0071], no gain. It is a path check, not an S0-L result.

Beside, descriptive only (M, mini-eval tier, point values): B-prime s0 AUROC 0.8748, B-prime s1 0.8727. Not T1-kin and not the card's floor.

## 4. Metric families this read covers

| family | S0-L |
|---|---|
| TACTICAL (decision quality: stop / not-stop) | **covered**: AUROC of p(STOP_POINT) vs `stop_hit`, one seed pair, paired episode-cluster bootstrap |
| LONGITUDINAL | **not covered**: S0-L reads one tactical decision score; no speed, acceleration or distance-keeping cell is computed. A lon-STOP AUROC is printed beside (a decision score, not a speed metric) |
| LATERAL | **not covered**: no heading, curvature, yaw-rate or cross-track cell is computed |
| STRATEGIC | **PAUSED** (PI item 25), not a pass, not omitted |

## 5. Open items and caveats
* The CI rule for S0-L is `AMENDMENT_5_S0L.md` (frozen 2026-11-13 before any L data; MM). Under it a GAIN licenses L-DAG, the next experiment, and never a lever claim from one seed per arm.
* One seed per arm: B-prime s0 vs L_s0. The interval cannot speak for training variance (H-ESTIM-SEED-1).
* Step-3 route: B-prime's cache was encoded on the pod GPU, so an L cache built on another device adds a device term to dAUC. Decide the venue before step 3.
* Disk at write time: E: 5.3 GB, C: 6.8 GB, F: 21 GB free; the dumps need ~F: scratch (a BAL dump set is small, the L cache ~2.8 GB).
