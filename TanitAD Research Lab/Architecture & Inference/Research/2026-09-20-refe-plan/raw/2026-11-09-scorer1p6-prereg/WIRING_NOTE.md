# WIRING NOTE: SCORER1 on the P6 checkpoint (2026-11-09; read-only; nothing run, no GPU, no pod)
Paths: V3 = `E:/Projects/TanitAD-artifacts/scorer1_impl_20261108/v3/` (files/ = the v3 bytes; the E: tree has no scorer1_chain.sh, so v3 is the only copy read). PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. Classes: M = MEASURED, I = INHERITED, E = ESTIMATED.
**Main finding: P6 needs almost no trainer code. It needs a new label set, a new profile, a new argv pin, and a ruling on EP.**

## 1 Labels
- No P6 label set exists (M). P6 labelled online, lag 0, and kept nothing: `PK/refe/online_labeller.py:1-14` hands each micro-batch's proposals to a pool and discards them (no persistence code found there). Its bank is in `PK/SPEC_P6_PAPER_FAITHFUL.md:63` (189,888 tuples / 103,037 scenes) plus the pod `metric_cache`. u400's labels (`train_pdm` 174,912 rows, `heldout_pdm` 3,137) are for u400's proposals and do not apply: labels are keyed to the proposals' own `ckpt_step` (`PK/refe/onpolicy_relabel_pdm.py:5-12`).
- Job A, GPU fan dump of P6 (`PK/refe/rl_fan_dump.py:4-5`, takes `--ckpt model_final.pt`, builds the config via `config_for_checkpoint` at :253-254, so nav_mode=command comes from meta). Changes: `--targets /workspace/data/refe_navtrain/train_grow_cmd_3c` (the command rows P6 trained on; `train_grow` falls back to the goal rule, 0.90 agreement, `SPEC_P6:282`), and check :561-563 (a second `REFeConfig.for_backbone` that bypasses `config_for_checkpoint`) before running. Sets: (i) the 24 held-out logs (`PK/splits/sft_heldout_logs.txt`), 3,137 sets; (ii) a train draw from the other logs.
- Job B, CPU PDM labelling: `onpolicy_relabel_pdm.py --sets --out --workers`, or the cache route `online_labeller._job` / `label_worker.py`; output `pdm_targets` side files in the `train_pdm/` and `heldout_pdm/` layout; then `pdm_label_complete.py --expect N`.
- Size: an arm reads only 400 x 64 = 25,600 sets (`scorer1_pod.sh` COMMON: `--max-updates 400`, batch 8 x accum 8); C0r uses another order. So label **about 60,000 train sets + 3,137 held-out**, not 189,888. The registered counts 174,912 / 3,137 (`V3/files/launch_gate.py:224`) become new numbers.
- Cost (E): dump 63k sets x 0.066 s = **1.2 L40-h** (P6 trains 256 samples in 50.5 s, `scorer_gym_20261107/PREREG_SCORER1_AMENDMENT_3.md:24`; forward-only taken as 1/3 of that, an assumption). Labels 63k / 4.49 sets/s (`SPEC_P6:187`) = **3.9 h CPU** on 6 workers, parallel with the dump. The full 189,888 would be 11.7 h. Carry-over of that rate to this job is not measured.
- Also needed: an E2 truth table for P6's own 64 proposals on 2,000 navtest tokens (u400's table does not apply, the generator differs): about 2.5 h CPU (E, scaled from 1.3-1.8 h per 1,058 tokens, `refe_behaviour_20261107/PREREG_P6_SCORER_SPEED.md:58`). E1 reuses P6's existing C3 table under `F:/TanitAD_p6read_work/score/`.

## 2 Trainer (`V3/files/scorer_finetune.py`)
- Already generic for a 6-row head, no edit: guards at :1593 (`p.shape[-1]==7`), :1596-1598, :1796, :1814, :2150, :2326. The `native7` refusal :1381-1383 passes for a 6-row head; `extend_lane_head` runs only with `--lane-head` (:1387). `--lane-head` with 6 columns would REFUSE (:1411-1412).
- Checkpoint (M, mmap read of `F:/TanitAD_p6read_work/ckpt/run/model_final.pt`): format `refe-full-v1`, 643 tensors, `score_head` (6,256), meta `nav_mode=command`, `per_sample_calib=True`, 4 cameras, no `n_score_components`. `PK/refe/ckpt_io.py` `load_for_inference` (:138-170) loads FULL strictly; `config_for_checkpoint` (:116-135) sets nav_mode. `save_full` (:1934-1951) copies the input meta, so nav_mode survives into the output and the planner builds a command-mode model.
- `scorer1_pod.sh` COMMON: drop `--lane-head --lane-weight --teacher-lane-train --teacher-lane-heldout --lane-labels-heldout`; keep `--pdm-labels-train/-heldout --require-pdm`; new `--ckpt`, `--targets .../train_grow_cmd_3c`, new `--onpolicy` / `--heldout` dirs.
- `scorer1_chain.sh`: u400 sha (:24-25, :138), path checks (:134-144 incl. teacher_lane dirs), row counts 174,912 / 3,137 / 2,109 (:141-144, :241), the lane manifest block (s1l overlay), human gates STAGE0_READING / P6_SPEED_READING (:104-106; see 7).
- UNVERIFIED: the v3 preflight (G1-G16) has not run with a 6-row head. Cheap check: a CPU test with a tiny 6-row model under `CUDA_VISIBLE_DEVICES=-1`.
- Docstring :35 and the PREREG text say 7-output; cosmetic.

## 3 Loss: EP
- Add `"EP": 2` to `PAIR_HEADS` (:149; column order NC DAC EP TTC C DDC) and the help text :1326. `pair_refusals` (:220-223) then accepts it. The loss is sign-agnostic: `select()` orders by the head's own label (`y[:,None] > y[None,:]`, :178), so EP needs no sign flip.
- **The balance step is the real problem.** `select()` keeps min(|F|,|S|) pairs per side (:188-194). For TTC and NC the truth says fast is worse, so balancing removes "slow is good". For EP the truth says fast is better (true rho +0.443 on P6 cruise, `F:/TanitAD_p6read_work/run/P6_SPEED_RESULT.md`; pred +0.301), so balancing deletes the signal the head lacks, and a set with no slower-better pair yields no EP pair. Options for the MM: (a) EP unbalanced, natural pairs, plus a new guard on overspeed share (P6 already overshoots: 0.535, plan v4 minus GT +0.84 m/s, `P6_RESULT.md:37`; a "fast is good" term could worsen it); (b) EP only on pairs whose NC, DAC and TTC labels are equal (progress-only pairs). Either needs a per-head `balance` field in `PairLoss`.
- Knock-on: G16 requires `kept_F == kept_S` for every head (:2298 `fs_ok`) and `gate_checks.py:1407` asserts `pair.balanced`; both become per-head. The NOBALANCE mutant (:157) must still turn RED for the balanced heads. EP labels are graded, so ties are rare; `bank_share` for EP must be > 0 (:2300).
- EP gap evidence: u400 Stage 0 -0.476 (`scorer1_impl_20261108/stage0_result/STAGE0_RESULT.md`); P6 navtest cruise -0.139 [-0.204, -0.087], n=153 (M, `P6_SPEED_RESULT.md`).

## 4 Launch gate (`V3/files/launch_gate.py`, `gate_checks.py`)
- New profile `refe_scorer1p6`, built as a NEW dict from `PROFILES["refe_scorer1"]` (:501), not an edit of it. Per `V3/LAND_PLAN.md:8`, adding a profile moved 0 of 16 fingerprints (M, `fp_base.json` vs `fp_v3.json`); `refe_p6` stays `80fa4d7e...` (pinned at `test_p7_head_wiring.py:64`, M, tip grep). **Do not edit shared constants** (`SFT4R_RECIPE :221`, `SFT4R_DATA_EXPECT :224`, `RLS1_DATA`, `_BASE`): that moves every profile built on them (memory note: 6 fingerprints pinned in 3 tests; I found the full digest in one test only).
- In the new dict: `required_flags` without `--lane-head`, `--teacher-lane-*`, `--lane-labels-heldout` (inherited at :331 / :442); forbid them; `ckpt_sha256` = `968a469d...`; `data_expect` with the new counts; `forbidden_flags` as SCORER1 (`--pair-calibrate`, `--dump-scores`).
- Code edits (not fingerprinted, but they change the code-tree sha): `scorer1_argv` (:473) is u400-specific (SFT4R_ARGV with the ckpt text replaced) -> a `scorer1p6_argv`; `scorer1_refusals` (:1085-1104) hard-codes `SCORER1_PAIR_HEADS`, `SCORER1_ARM_SEED` and the u400 sha -> read them from the profile; the `scorer1_rules` branch at :639.
- G-DVB: the `gate_checks.py:873` table already lists `pair_*` as "built"; EP needs the `pair.heads` row (:1406) to carry it and a per-head balanced row (:1407). Add a built `nav_mode` fact to the data event (only for a command checkpoint, so other profiles' events stay byte-identical) with a G-DVB row; today a checkpoint whose meta lost nav_mode would load as goal mode.
- Tokens: `gate_token.TREE_ROOTS=(refe,code,splits)` (LAND_PLAN:9): re-shipping refe/ over a pod dir holding a token voids it, so SCORER1 gets its own dir. `refe/p7_ship_list.json` goes stale (9 files differ already; re-derive ~45 min).
- Land order: v3 (waits on P7 start), the S1L overlay (`scorer_gym_20261107/s1l/HANDOFF.md`) and the `a2/wiring_v1` overlay all touch `scorer_finetune.py` and `launch_gate.py`; the P6 change is a fourth. Rebase onto the landed tip and run all launch_gate-reading tests (about 69).

## 5 Stage 0 equivalent
- `H-P6-SPEED-1` (REFUTED; the file is `F:/TanitAD_p6read_work/run/P6_SPEED_RESULT.md`; I did not find `p6_speed_result.json` at that path) shows the gap exists on navtest cruise for P6: TTC -0.190, NC -0.157, DAC -0.154, EP -0.139, CIs exclude 0 (M, n 66/49/138/153). It does **not** answer Stage 0's question (navtrain held-out vs navtest shift), and the Z bars do not transfer: P6's mean(TTC,NC) is -0.174, not <= -0.20, so the u400 rule reads INCONCLUSIVE and Stage 1 would run anyway.
- Caution: all 24 held-out logs are in `navtrain.yaml` (M, checked) and `train.py` has no exclusion (grep 0 hits), so P6 very likely trained on them with online labels (UNVERIFIED on the pod bank). A Stage 0 there is in-sample for P6's scorer, understates the gap, and a Z2 would be uninterpretable.
- Recommendation: no blocking Stage 0. The P6 held-out set is a by-product of Jobs A+B; read the statistic as DESCRIPTIVE, and re-register M1's bars at P6's size (|delta rho| about 0.17, so S1-C0 >= +0.087).

## 6 Smallest arm set (needs a new PREREG + amendment, bars written before any read)
| arm | difference | why |
|---|---|---|
| P6-C0 | scorer-only PDM fine-tune, 400 updates, seed 0 | same-code control |
| P6-C0r | seed 1 | training-variance floor (`H-ESTIM-SEED-1`) |
| P6-S1E | C0 + L_pair on NC,DAC,TTC (balanced) + EP (rule from section 3) | the lever |
- 3 arms x 3.0 L40-h (I, Amendment 3: 9.1 for 3 arms) + dump 1.2 + held-out 0.2 = **about 10.5 L40-h (E)**; plus CPU 3.9 h labels and 2.5 h E2 table; calibrate, preflight and the two mutants are not measured and not in the figure.
- Conditional 4th arm P6-S1 (no EP), +3.0 L40-h, only if S1E reads harm (O3) or overspeed moves: it isolates the EP term. The full 4-arm set is about 13.5 L40-h (E).
- Keep guards G1-G3 and add an overspeed-share guard (no worse than P6's 0.535 by more than the C0r floor).

## 7 Blockers and decisions for the Master Mind
1. PI: pod L40 time (~10.5 h) and a new PREREG for a P6 base; the frozen PREREG_SCORER1 names u400 as the only base and calls P6 a reference, not a gate.
2. MM ruling on EP's pair rule (3a or 3b) and the overspeed guard; EP is a lever the Stage 0 note says needs its own pre-registration.
3. Held-out set: the 24 logs are probably in P6's training bank (section 5); report with that disclosure (no other navtrain carve exists, P6 used all of navtrain).
4. Human gate `P6_SPEED_READING` (REFUTED) was defined as "launch on u400"; for a P6 base it must be redefined, or replaced by a new gate in the new chain.
5. Jobs A/B run on the pod (P7 queue, GPU lock, 6 CPU workers); pod state for P6's metric cache and bank is unverified. Do not copy landed refe/ into the p6/p7 dirs.
6. The untested 6-row preflight (section 2) and the 4-way land conflict (section 4).
