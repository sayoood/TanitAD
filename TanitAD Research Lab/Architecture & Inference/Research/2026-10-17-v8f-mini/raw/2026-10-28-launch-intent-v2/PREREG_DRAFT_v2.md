# PREREG_V8F_LAUNCH_INTENT (DRAFT v2): O's launch-sample along-track weight on DAG-OACC's recipe, aimed at the crawl rows

**Status: DRAFT v2 for the Master Mind's promotion.** v1 was written 2026-10-28, **before any LAUNCH-INTENT gate, run, rollout or row**: no LAUNCH-INTENT checkpoint exists, and none exists at v2. Proposed claim row `H-V8F-LAUNCH-INTENT-1` (the Master Mind registers it; this draft edits no register).
**v2 (2026-10-28)** is v1 edited to the Master Mind's redesign `MM_REDESIGN_2026-10-28.md`: two same-seed one-lever pairs at each seed, the bar `2 × max(F, RF0)` with RF reported beside, and per-seed predicates that hold only when they hold on both pairs of the seed. The disclosure is below (verbatim from `MM_REDESIGN_2026-10-28.md` §5) and in §0 item 7; the line diff against v1 is `PREREG_V2_vs_V1.diff`.
**Disclosure.** *"The bar and arms were changed by the Master Mind before promotion, after DAG-ORC's floors and the REP s0 diagnosis were known, and before any LAUNCH-INTENT number existed. The change follows from arithmetic on banked numbers (section 1). It is not a reaction to a LAUNCH-INTENT outcome, which does not exist."*
It is the `LAUNCH-INTENT` branch of `PREREG_V8F_DAG_ORC` §8 (NEXT-CRAWL), re-based on DAG-OACC as `PREREG_V8F_DAG_ORC` Amendment 1 §3 directs (*"its branch is LAUNCH-INTENT, with DAG-OACC's recipe in place of 'DAG init'"*).
Registered spec, verbatim from §8: *"O's along-track loss over the executed horizon (knots 1–12 at m = 10), up-weighted on GT-launch TRAIN samples (v0 < 6 m/s and GT v(6 s) ≥ v0 + 1 m/s). One flag with its G-DVB entry, DAG init, 3,000 steps, under its own PREREG. The launch subset is label-side (future GT speed), which is admissible for labels; nothing new enters at inference."*

**Disclosure of what was read before writing** (all MEASURED, from the git mirror at `d7208328` unless stated):
* `PREREG_V8F_DAG_ORC.md` §0–§10, Amendment 1, Amendment 2; `PREREG_V8F_HACC_MC.md` §2 and §5; `hmc_read.py` (the launch-subset code).
* `stack/scripts/v8f_mini_train.py`, `stack/tanitad/models/v8f_mini.py`, `stack/tanitad/data/v8f_mini_data.py` at `d7208328`.
* DAG-OACC's banked `argv.json` and `summary.json` (`E:/Projects/TanitAD-artifacts/v8f-dagoacc/M2b_s{0,1}_dagoacc/`); DAG-ORC r2's REP gate pins (`E:/Projects/TanitAD-artifacts/v8f-dag-orc-oacc/M2b_s{0,1}_dagoaccorcrep/gate/pin.txt`).
* The train manifest and the native pose tracks, to count launch windows (§3).
* **Not read:** any LAUNCH-INTENT row (none exists). Every DAG-OACC / DAG-ORC / DAG row quoted below is INHERITED from `PREREG_V8F_DAG_ORC`'s text, not re-verified here, except what v2 states as MEASURED in §6 and §14: the bars, RF0, RF and the CTRL − P10 gaps, read from the banked `P0_RF0.json` (sha256 `1608d70b…`), `RF_REP.json` (`64c9444d…`) and `DAGORC_READ.json` (`70a7a3a4…`).
* **v2 reads, in addition:** `MM_REDESIGN_2026-10-28.md` (the spec); the DAG-ORC reader `dag_orc_read.py` (sha256 `2e5f9f17…`) and the SPEEDSUB-T1 reader (the pattern for extending it); `rep_s0_diag/rep_s0_speedcells.py` (the slow-draw method). All from the git mirror at `266b470e` unless a path says otherwise.

## 0. Items the Master Mind fixes before promotion

| # | item | state, and why it is open |
|---|---|---|
| 1 | **The registered object does not exist at the tip.** There is no standalone "O's along-track loss". O's position loss is `T["ade"]` / `T["fde"]`: the 2-D L2 distance to the GT, averaged over all 30 knots (`v8f_mini.py:870`, `traj_terms` `:365`), plus `w_acc`'s second-difference term (`:871`, `:888`). | **OPEN.** This draft builds the registered object as an **additive term** (§2). Confirm the reading, or name another. |
| 2 | **Arm B's share: FIXED by the Master Mind at one quarter (W_B = 0.726175).** One sixth was a near no-op at the measured f (f = 0.1619, so the unweighted launch share is already 0.1619; one sixth meant W_B = 0.0357, a 3.6 % weight increase). One quarter sits between the unweighted share and A's one third, so B is a real lower dose. | **RESOLVED** before promotion (§3, §4, §9). At v2, B follows the pair design: 4 runs (B_s and B_rep,s). |
| 3 | **REP reuse** (saves ≈ 1.4 h of box 2: REP's two runs are not trained) is licensed by an argv comparison made today (§4). It needs DAG-ORC r2's REP runs to finish and pass their run record. | **RESOLVED in design, conditional in fact.** At v2, REP_s is the control of the replicate pair, so the licence carries a decisive read and not only a floor. §4 lists the conditions and the fallback. |
| 4 | **NEXT-LAUNCH (§11) and its arm LAUNCH-INTENT-HI are drafted here.** `PREREG_V8F_DAG_ORC` registers nothing after LAUNCH-INTENT. | **OPEN.** Rule Zero: an outcome that ends at "not a lever" must register what follows. Edit freely. |
| 5 | **Not in this deliverable:** the box-2 chain (the four arms, its one-lever asserter, the K-SHARE run-record check) and the lander. `launch_gate.py` needs no change (§2). | **PARTLY RESOLVED.** The reader `launch_intent_read.py` is built with its tests, mutation pass and dry run in the v2 package (§13). The chain and the lander remain integration steps. |
| 6 | **Entry conditions.** This draft is written for DAG-OACC states. If a DAG-ORC read adopts another base first, this PREREG is amended to that base with its banked run as CTRL before any run (as `PREREG_V8F_DAG_ORC` launch precondition 4). | **OPEN** (§13). |
| 7 | **The bar and the arms were redesigned by the Master Mind before promotion.** Disclosure, verbatim from `MM_REDESIGN_2026-10-28.md` §5: *"The bar and arms were changed by the Master Mind before promotion, after DAG-ORC's floors and the REP s0 diagnosis were known, and before any LAUNCH-INTENT number existed. The change follows from arithmetic on banked numbers (section 1). It is not a reaction to a LAUNCH-INTENT outcome, which does not exist."* (The spec's section 1 is the arithmetic reproduced in §6 below.) Seen before deciding: `RESULT_V8F_DAG_ORC` (HARMS; the REP pair sets every crawl bar); the REP s0 diagnosis (a slow-T training draw: its T's five speed cells read 0.73–1.17 m/s slow); the `RESULT` of SPEEDSUB-T1 (NULL; speed cells do not bind on healthy DAG-OACC draws). None of these is a LAUNCH-INTENT number. | **DISCLOSED.** |
| 8 | **Readings this edit chose where the redesign is silent** (each is flagged where it is used; confirm or name another): (a) the replicate pair's domain D, for R-NEW and R-NOBASE, is every row not separated-worse for REP − P10 (the pair's own control in place of DAG-OACC), while the crawl set C_s stays DAG-OACC's for both pairs (§6, §7); (b) M-LAUNCH is "moved" only when it moves in both pairs at both seeds (§11); (c) LAUNCH-INTENT-HI follows the same pair design as B (§13); (d) the slow-draw label needs all five T0 speed cells separated from the pair's control, each shift larger than 0.5 m/s in magnitude (§7); (e) the v1 K-SHARE clause "`levers.launch_intent_w == 0.0`" for CTRL and REP is read as "absent, or exactly 0.0", because both were trained before the flag existed and carry no such key (MEASURED: neither `summary.json` has a `launch` key) (§9). | **OPEN.** |

## 1. Why it runs

* **The rows.** DAG-OACC − P10 is separated-worse at both seeds on yaw-rate MAE, curvature MAE, L2 and M1 (INHERITED, `PREREG_V8F_DAG_ORC` A1). Those are the crawl rows C.
* **What the data rank** (INHERITED, `PREREG_V8F_DAG_ORC` §8): the executed-horizon along-track profile is the only thing that has moved the crawl rows. HACC's zero-training post-map cut M1 by 49 % / 36 %, L2 by 61 % / 54 %, curvature by 72 % / 58 %, and failed KEEP at s0 (+0.1264 m). The state-channel levers have not moved them: DAgger iteration 2 failed R-MOVE at both seeds. N0 read NOT NAVMOVE and COMPOSE is done (DAG-OACC − DAG is not separated-better on C), so this is the branch left.
* **The mechanism it targets** (INHERITED, `DIAGNOSIS.md` via §8): O's own slow launch intent, v0 −1.78 / −0.93 m/s at tick 1 in the launch windows.
* **Why position-level pressure on the launch windows.** HYPOTHESIS: the 2-D ADE averages 30 knots over every window. The GT-launch windows are 16.2 % of the training windows (MEASURED, §3), so their along-track error is diluted about 5:1. DAG-OACC's acceleration term shapes the profile on every window; it does not raise the launch level.
* **The known risk** (HYPOTHESIS, not measured): the weight teaches "accelerate" in low-speed states that look like launches. The states where the human does not launch (a stop, a queue behind a lead) keep weight 1, so the bias is bounded by the base rate. The distance-keeping rows (headway, time gap, TTC, DK-VIOL), R-ADE and R-NEW are the guards that would see it.

## 2. The lever, exactly (built and tested, not run)

**The term.** For a GT-valid sample b with launch mask L_b ∈ {0, 1}:

```
extra  = ( sum_b gok_b · L_b · (1/30) · sum_{k=1..12} sqrt( (traj_x[b,k] − gt_x[b,k])² + 1e-9 ) ) / sum_b gok_b        (launch_along_term, v8f_mini.py:409)
total  = (the tip's total) + w_traj · W · extra / 3                                                                    (v8f_mini.py:926, ONE line)
```

* It is **W × the along-axis (x of the ego frame at NOW), knots 1–12 slice of ADE's own per-knot distance**, with ADE's own smoothing, 1/30 and denominator, restricted to launch samples. In the total it carries ADE's own scale, `w_traj / 3`.
* So on a launch sample the along-axis slice of O's position loss carries weight **1 + W** exactly where the cross error is 0; elsewhere the baseline slice's along-gradient is Δx/d ≤ 1, so 1 + W is the nominal weight. A non-launch sample's loss is the tip's, bit for bit.
* Knots 1–12 are τ = 0.2–2.4 s: the executed horizon at m = 10 (`PREREG_V8F_HACC_MC` §2: every replan period ≤ 2.2 s executes, or reads into the state, only knots ≤ 12). Knots 13–30 and the cross axis get a gradient of exactly 0.0 from the term (test `…EXACTLY_zero_everywhere_else`).
* **W = 0.0 (default) ⇒ the term is not computed and not in the loss**: no new tensor op, no new `T` key, no new RNG draw. The flag-off run is the tip's, bit for bit (§9, K-BUILD).

**The mask** (`launch_mask`, `v8f_mini.py:401`): `gok ∧ v0 < 6.0 ∧ ‖gt[29] − gt[28]‖ / 0.2 ≥ v0 + 1.0`.
* This is `hmc_read.py`'s launch-subset definition (*"v0 (ego speed at t0) < 6 m/s and the GT chord speed over the last 0.2 s of the 6 s horizon ≥ v0 + 1 m/s"*): chord over knots 29 → 30, both axes, `/ DT` with DT = 0.2.
* **Labels only.** v0 is the LOGGED `kin3[:, 0]` and the chord is the GT future. The loop takes the mask **before** the ego stages (`v8f_mini_train.py:1946`) and attaches it **after** them (`:1965`), because a swapped sample's `kin3` is the model's own state, not the label. Test `…LOGGED_state_not_the_swapped_own_state` recomputes it from the clip store on a swapped run and requires the mask to differ from a post-swap mask somewhere (non-vacuous).
* **Inference is untouched.** The mask is a training-loss weight. No input to the model changes, so inference stays vision plus measured ego history. This is not a goal input and cannot carry the situation classifier (`CLAUDE.md`, 2026-08-03): the mask is never an input.

**Reachability, hop by hop** (the 2026-09-10/11 rule; every hop has `file:line` in the staged tree):

| hop | where |
|---|---|
| argparse (default 0.0) | `v8f_mini_train.py:128` `--launch-intent-w` |
| G-DVB entry (a parser dest without one is refused by `declared_vs_built` itself) | `:84` `FLAG_CHECKS["launch_intent_w"] = "cfg:launch_intent_w"` |
| namespace → config | `:263` `make_config(... launch_intent_w=a.launch_intent_w)` → `v8f_mini.py:189` field, `:212` finite ≥ 0 |
| declared == built (G-DVB) | `:694` lever read-back, `:711` cfg read-back, `:696` w_traj > 0 clause ("declared and does nothing") |
| G-LIVE | `:1853` `launch_intent_probe` on the loop's own mask call `launch_stage` (`:1032`): the total moves by exactly w_traj·W·term/3 on the rows the mask names; O's head gets a gradient; an all-False mask reads **exactly 0.0**; the prior's and the WM's gradient differences read exactly 0.0; the probe batch carries the mask (`:1814`) |
| loop: mask before the stages | `:1946` `lmask = launch_stage(b)`; attach after them `:1965` `b["launch"] = lmask` |
| loss term | `v8f_mini.py:908` `T["launch_along"]`; total `:926`; the batch the loss receives carries `launch` (test spy) |
| backward | `v8f_mini_train.py` `T["total"].backward()` (unchanged line, `:1844` at the tip) |
| records | `levers.launch_intent_w`, `levers.launch_intent`, `levers.launch_n` / `launch_eligible` (probe batch) in `selfcheck.json` and `summary.json`; run totals `summary.launch_n` / `launch_eligible` / `launch_rows` / `launch_weight_share` and `train_log.jsonl` (lever-ON runs only); `ckpt["launch_totals"]` |

**`launch_gate.py` needs no change** (MEASURED: `grep -n 'ego_perturb\|w_acc\|launch_intent' stack/scripts/launch_gate.py` finds nothing; the v8f-mini gate is the trainer's own `--selfcheck-only`, as DAG-ORC's token records). `e7ea20d1` added `--ego-perturb-with-dagger` the same way: trainer, test, golden. The chain's one-lever asserter must allow `--launch-intent-w` as the only extra flag (§13).

## 3. The dose, by a rule fixed before any run

**f** = the fraction of TRAIN windows (the DAG-OACC recipe's window list) that are GT-launch, among windows with a valid along-track label. **MEASURED** (`tools/measure_f.py`, CPU, 5.7 s; `f_measure.json`, sha256 `b5be3192…`):

| quantity | value |
|---|---|
| window list | 4,512 clips, 306,566 windows; **sha256 `e144adcd…8168` equals `train_window_list_sha256` of both banked DAG-OACC runs**, and `n_windows` equals theirs |
| windows with a valid along-track label | 306,566 of 306,566; 0 with a non-finite v0 or GT |
| **GT-launch windows** | **49,618 → f = 0.161851** (1,891 of 4,512 clips hold at least one) |
| launch windows' median v0 / median GT chord speed | 2.85 / 6.85 m/s |
| the trainer's `launch_mask` against an independent numpy spelling | 0 disagreements on all 306,566 windows |

**The rule.** Launch samples hold a share s of the along-track weight: f·w / (f·w + 1 − f) = s, with w = 1 + W. So **1 + W = s (1 − f) / ((1 − s) f)**, rounded to 6 significant digits like OACC's `w_acc`.
* **Arm A, s = 1/3: 1 + W = (1 − f) / (2 f) ⇒ W_A = 1.58926.** The argv carries `--launch-intent-w 1.58926`.
* **Arm B, s = 1/4: 1 + W = (1 − f) / (3 f) ⇒ W_B = 0.726175** (fixed by the Master Mind before promotion; the drafted one sixth gave W = 0.0357048, a 3.6 % weight increase over the unweighted share f = 0.1619, below any replicate floor by construction).
* **Arm HI (NEXT-LAUNCH only), s = 1/2: 1 + W = (1 − f) / f ⇒ W_HI = 4.17852.**

**The realised share is a control (K-SHARE, §9).** ESTIMATED by simulating the trainer's own `ChunkedSampler` (3,000 steps, batch 64, `tools/measure_f.py`) with the measured flags:

| trainer seed | rows | launch rows | f̂ | A's share (target 1/3) |
|---|---|---|---|---|
| 0 | 191,203 | 30,727 | 0.1607 | 0.3315 |
| 1 | 191,396 | 30,674 | 0.1603 | 0.3307 |
| 10 | 191,333 | 30,068 | 0.1572 | 0.3256 |
| 11 | 191,391 | 30,367 | 0.1587 | 0.3281 |

All four lie inside 1/3 ± 0.02. Seed 0's row count equals DAG-OACC's banked `dagger_rows` (191,203), so the simulated sampler stream is the run's.

The same simulation gives the other doses' realised shares (ESTIMATED, from the same sampler rows and launch counts): **arm B** (W_B 0.726175, target 1/4) 0.2484 / 0.2478 / 0.2435 / 0.2456 and **arm HI** (W_HI 4.17852, target 1/2) 0.4979 / 0.4971 / 0.4912 / 0.4941 at trainer seeds 0 / 1 / 10 / 11, all inside target ± 0.02. Seeds 0 and 1 are the A_s arms' trainer seeds; seeds 10 and 11 are the A_rep,s arms' (§4).

## 4. The arms (box 2; each through the trainer's own launch gate; both seeds)

At each seed s ∈ {0, 1} there are two same-seed one-lever pairs (`MM_REDESIGN_2026-10-28.md` §3 item 1):
* **the main pair:** A_s − CTRL_s, with CTRL_s = DAG-OACC_s, banked;
* **the replicate pair:** A_rep,s − REP_s, with REP_s banked from DAG-ORC r2.

Each pair differs in the lever alone: same init, seed, bank and argv, apart from `--out` and the lever.

| arm | what | role |
|---|---|---|
| **A_s** (LAUNCH-INTENT) | DAG-OACC_s's training argv **verbatim** (`v8f-dagoacc/M2b_s{s}_dagoacc/argv.json`: init M2b_s, 3,000 steps, `--dagger-p 0.1` on the iteration-1 bank, `--w-acc 49.6969`, `--seed s`), plus `--launch-intent-w 1.58926`, with its own `--out` | **decisive**, main pair |
| **CTRL_s** = DAG-OACC_s | **banked, reused.** Checkpoints `67265ec1…` (s0) / `90b01474…` (s1); T1-kin dumps `v8f-dagoacc-t1/` | the main pair's control: A_s minus the lever |
| **A_rep,s** | REP_s's argv **verbatim** (DAG-OACC_s's argv at `--seed 10 + s`, that is 10 / 11), plus `--launch-intent-w 1.58926`, with its own `--out` | **decisive**, replicate pair |
| **REP_s** | DAG-OACC_s's argv with `--seed 10` / `--seed 11`. **Banked from DAG-ORC r2** (`M2b_s{s}_dagoaccorcrep`, checkpoints `b03fd6ef…` / `77853e56…`) under the licence below; trained new otherwise | the replicate pair's control: A_rep,s minus the lever |
| **B_s, B_rep,s** | A_s and A_rep,s with `--launch-intent-w 0.726175` (share 1/4), the same pair design | the dose fallback. 4 runs, trained **only after A's HARMS or NOT ADOPTED, with its own PI go** |

**New box-2 runs: A_0, A_1, A_rep,0, A_rep,1.** ESTIMATED ≈ 2.8 h at ≈ 40 min each (DAG-ORC's measured run times, §13).

**The REP reuse licence.** MEASURED today (`tools/compare_rep_argv.py`, `compare_rep_argv.out`, exit 0; the v1 package's file, read unchanged):
* the argv list DAG-ORC r2's gate pinned for REP (`gate/pin.txt`) has 58 tokens and equals DAG-OACC's banked `argv.json` (58 tokens) token for token after removing `--out` and `--seed`, at both seeds. The seeds are 0 / 10 and 1 / 11;
* A_rep,s's argv is REP_s's argv plus the lever by definition, and REP_s is DAG-OACC_s's argv at `--seed 10` / `11`. So **argv(REP_orc) = argv(REP_li) apart from `--out`**, and it carries `--ego-perturb-p 0`, `--dagger-seed s` and no lever flag.

The licence holds for a REP run only if all of these hold, else REP is trained anew (2 runs, ≈ 1.4 h) before any LAUNCH-INTENT row is read:
1. DAG-ORC r2's REP run finished and passed its run record (swap share 0.1 ± 0.02, ego exactly 0, composition flag absent, `levers.ego_perturb_stream` = 0);
2. its launch-gate token has the same `init_ckpt_sha256` (`601cb435…` / `6104fb15…`), `bank_sha256` (`197a68bb…` / `36c7aea9…`), `w_acc` 49.6969, `excluded_sha256` and `train_window_list_sha256` as DAG-OACC's;
3. its T1-kin dumps were made with the settings of §5, and the reader's pins match; its box-2 standard T0 eval (eval-141, ages cycle) exists, because R-T0 in the replicate pair reads it;
4. **the build term.** REP was trained at the DAG-ORC r2 commit (token binding `8c24823f`); A_s and A_rep,s are trained at the launch commit. This PREREG's changes sit behind a default-off flag and the flag-off golden is GREEN on the launch commit (K-BUILD, §9). The golden is a 4-step CPU run on the synthetic world; the GPU run on box 2 is not covered. ESTIMATED, not measured.

**Disclosed coupling.** DAG-ORC and LAUNCH-INTENT share the REP_s runs. In DAG-ORC they set the floor RF; here they are the control of the replicate pair. RF is not a bar term in this PREREG (§6), so an unusually large or small REP draw cannot loosen or block this read's bar. REP enters through the contrast A_rep,s − REP_s, in which the two arms share their seed (10 / 11), init, bank and argv. Neither read's result changes the other's bar.

## 5. Evaluation and tiers

`PREREG_V8F_DAG_ORC` §2 applies verbatim, with X = A_s or A_rep,s and the pair's control Y = CTRL_s (DAG-OACC) or REP_s.

| where | what | tier |
|---|---|---|
| dev-box CPU (decisive) | `v8f_t1_eval run --replan-steps 10 --limit causal --dump-ticks --dump-heads` on eval-141, plain (no post-map), DAG's decisive settings (dead-reckoned nav, logged scene, periodic, executor `plan`). **A_s and A_rep,s at both seeds (4 rollouts)**; CTRL, REP, DAG and P10 banked | **T1-kin**: self-action OPEN loop. ⛔ Never "closed loop" |
| dev-box CPU (K4 only) | one `--limit logged` run per new arm and seed (4), and one fresh-age T0 eval each (4) | control |
| box 2 | the chain's standard T0 eval (eval-141, ages cycle) of every checkpoint | **T0**: R-T0 and the T0 four families. ⛔ Never "driving performance" |

**Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0, 141 episodes / 2,399 windows, per seed and per pair, paired, same windows and matched cells. **Three variances, named:** another draw of EPISODES (the bootstrap answers it); another TRAINING RUN (the replicate pair at each seed answers it: a lever effect is credited at a seed only when both pairs show it, and the two seeds add two further draws); another INFERENCE RUN (none arises: the planner is deterministic).

## 6. The bar (every Read-A row, defined now), and RF beside

`PREREG_V8F_DAG_ORC` §3 applies with its Amendment 1 substitutions and **one change of the bar** (`MM_REDESIGN_2026-10-28.md` §3 item 2). Read-A rows as there; d_r,s(X − Y) is Read A's paired point of X − Y on row r at seed s.

| symbol | definition |
|---|---|
| **F_r** | ADE6: **0.1207 m** (`MODEL_REGISTRY.md` §17, `H-V8F-SEED-1`). Every other row: 0 |
| **RF0_r** | the frozen same-seed cross-build floor `P0_RF0.json` (`raw/2026-10-27-dag-orc-read/P0_RF0.json`, sha256 `1608d70b…`), unchanged |
| **bar_r** | **2 × max(F_r, RF0_r)**, one value per row, applied at both seeds and in both pairs, inclusive. A row whose bar is 0 clears on separation alone |
| **RF_r, beside** | max over s of \|d_r,s(REP − DAG-OACC)\|, recomputed by the reader; DAG-ORC banked it as `RF_REP.json` (sha256 `64c9444d…`). **Not a bar term**: reported beside every row, decides nothing |

**What a row "clears", and what "fires"** (X is A_s or A_rep,s; Y is the pair's control, CTRL_s = DAG-OACC_s or REP_s):
* **clears (R-MOVE, per row):** X − Y is separated-better AND the improvement is ≥ bar_r.
* **fires (R-NEW, per row):** all three of:
  1. X − P10 is separated-worse;
  2. Y − P10 is not separated-worse there, i.e. the row is in the pair's domain D (§7);
  3. X − Y is separated-worse with the worsening ≥ bar_r.

**Why RF left the bar** (MEASURED arithmetic on `DAGORC_READ.json`, sha256 `70a7a3a4…`, and `RF_REP.json`). In v1 the bar was 2 × max(F, RF0, RF), with RF the maximum of two REP draws, and REP s0 set RF on 17 of 19 rows. R-MOVE needs an improvement of at least the bar on every crawl row. Against the whole gap DAG-OACC leaves to P10:

| crawl row | v1 bar | CTRL − P10, s0 / s1 | v1 bar ÷ gap | **v2 bar** | v2 bar ÷ gap |
|---|---|---|---|---|---|
| yaw-rate MAE | 31.7106 | 3.4726 / 8.6844 | 9.1 / 3.7 | **1.1070** | 0.32 / 0.13 |
| curvature MAE | 0.2648 | 0.0854 / 0.1352 | 3.1 / 2.0 | **0.0526** | 0.62 / 0.39 |
| L2 curvature MAE (GT ≥ 5.2) | 0.1740 | 0.0443 / 0.0529 | 3.9 / 3.3 | **0.0136** | 0.31 / 0.26 |
| M1 crawl share | 0.0776 | 0.0568 / 0.0604 | 1.4 / 1.3 | **0.0096** | 0.17 / 0.16 |

A lever that removed the crawl completely, bringing the model to P10 parity on all four rows, would have failed v1's R-MOVE at both seeds: the decisive condition was unreachable by construction, whatever the data say. That is a design defect, not a strict bar, and it is fixed before any LAUNCH-INTENT data exists. RF answers a real question (how far apart two training runs of one recipe land, `H-ESTIM-SEED-1`) with n = 2 draws, and the maximum of two draws is set by the worse one (here a slow-T draw). Used as a *level* bar on a cross-run contrast it asks two questions at once:
* *would another training run say this?* That is answered by **replication across runs**, which the per-seed predicates of §10 now require;
* *is the effect large enough to matter?* That is answered by a **size bar**, which must be reachable. Each v2 crawl bar is below the CTRL − P10 gap at both seeds.

⚠ **Stricter in one way:** every guard must hold in both pairs of the seed. ⚠ **Looser in one way:** the size bar drops from 2 × RF to 2 × RF0, which is why replication in both pairs is required.

**The bars, all 19 rows** (v2 bar = 2 × max(F, RF0); RF and the v1 bar are beside; units are the row's own):

| row | F | RF0 | **v2 bar** | RF (beside) | v1 bar |
|---|---|---|---|---|---|
| ADE6 | 0.1207 | 0.0138 | **0.2414** | 0.4025 | 0.8050 |
| speed MAE | 0.0000 | 0.0796 | **0.1592** | 0.3373 | 0.6746 |
| along-track MAE | 0.0000 | 0.0201 | **0.0402** | 0.4596 | 0.9192 |
| accel MAE | 0.0000 | 0.1323 | **0.2646** | 0.5603 | 1.1206 |
| DK-VIOL | 0.0000 | 0.0050 | **0.0100** | 0.0017 | 0.0100 |
| DK-HW | 0.0000 | 0.1308 | **0.2616** | 0.2540 | 0.5080 |
| DK-TG | 0.0000 | 0.0041 | **0.0082** | 0.1042 | 0.2084 |
| heading MAE | 0.0000 | 0.3801 | **0.7602** | 2.5074 | 5.0148 |
| yaw-rate MAE | 0.0000 | 0.5535 | **1.1070** | 15.8553 | 31.7106 |
| curvature MAE | 0.0000 | 0.0263 | **0.0526** | 0.1324 | 0.2648 |
| cross-track MAE | 0.0000 | 0.0320 | **0.0640** | 0.0224 | 0.0640 |
| L1 lateral-accel MAE | 0.0000 | 0.3985 | **0.7970** | 1.2242 | 2.4484 |
| L2 curvature MAE (GT>=5.2) | 0.0000 | 0.0068 | **0.0136** | 0.0870 | 0.1740 |
| TAC-LAT | 0.0000 | 0.0183 | **0.0366** | 0.0421 | 0.0842 |
| TAC-LON | 0.0000 | 0.0125 | **0.0250** | 0.0950 | 0.1900 |
| S-POS@30 | 0.0000 | 0.1877 | **0.3754** | 0.2592 | 0.5184 |
| S-POS@50 | 0.0000 | 0.4449 | **0.8898** | 0.3912 | 0.8898 |
| S-POS@80 | 0.0000 | 1.0082 | **2.0164** | 0.5170 | 2.0164 |
| M1 crawl share | 0.0000 | 0.0048 | **0.0096** | 0.0388 | 0.0776 |

## 7. The rows

`PREREG_V8F_DAG_ORC` §4 with the Amendment 1 substitutions applies ("DAG" and "CTRL" both mean DAG-OACC; no exempt rows), **evaluated in each of the seed's two pairs**. In a pair, X is A_s (main) or A_rep,s (replicate) and Y is the pair's control: CTRL_s = DAG-OACC_s (main) or REP_s (replicate). The seed's predicate holds only when it holds in both pairs (§10).

| row | contrast, in each pair | role |
|---|---|---|
| **R-CRAWL** | every row of C_s is not separated-worse for X − P10 (C = yaw-rate MAE, curvature MAE, L2, M1) | decisive |
| **R-MOVE** | every row of C_s **clears** for X − Y | decisive |
| **R-TLAT / R-TLON** | the T head's lat / lon macro-F1 at ticks ≥ 1, X − Y: not separated-worse | guard |
| **R-ADE** | T1-kin plain ADE6, X − Y: not separated-worse | guard |
| **R-T0** | T0 own-code ADE6, X − Y: point ≤ +F/2 = **0.0604 m** | guard |
| **R-NEW** | no row of the pair's domain D fires | guard |
| **R-KEEP** | T1-kin ADE6, X − Y: point ≤ +0.0604 m | adoption |
| **R-NOBASE** | no row of D has X − P10 separated-worse AND X − Y separated-worse with the worsening ≥ bar_r | adoption |

R-ADE and R-KEEP share one contrast (X − Y) in each pair, as do R-NEW and R-NOBASE; both coincidences are disclosed and no row is dropped. **The crawl set C_s** is fixed by DAG-OACC − P10 (`PREREG_V8F_DAG_ORC` Amendment 1) and is the same for both pairs: yaw-rate MAE, curvature MAE, L2 and M1 at both seeds. **The domain D** of a pair is every Read-A row not separated-worse for Y − P10 at s, with no exempt rows: DAG-OACC − P10 for the main pair, REP − P10 for the replicate pair (§0 item 8 (a)).

**Beside (never decisive), and the lever's aimed rows:**
* **RF**, per row, from the REP pair (§6), and each row's bar next to it.
* **The slow-draw flag**, for every trained arm. The reader reports the arm's T0 speed-cell shift against its pair's control, on the five cells and with the method of `rep_s0_diag/rep_s0_speedcells.py` (`lon_v_target_ms`, `v_a_ms`, `v_end_ms`, `v_lo_ms`, `v_hi_ms` of box 2's standard T0 eval, per-window paired difference, episode-cluster bootstrap B = 2,000, seed 0). An arm whose five cells are all separated by more than 0.5 m/s is labelled a **slow-T draw** in the RESULT (all five CIs exclude 0 and each shift exceeds 0.5 m/s in magnitude, downward). The flag decides nothing.
* **M-LAUNCH, the mechanism row.** X − Y of the mean (v_own − v_GT) at ticks 1–2 in the GT-launch windows, from the tick dumps, in each pair. The lever predicts it moves toward 0. It routes NEXT-LAUNCH (§11) and decides no outcome.
* along-track MAE, speed MAE and L2 on the GT-launch subset (the reader's definition, `hmc_read.py`, which `launch_mask` agrees with on all 306,566 training windows); M2.
* X − DAG-OACC for the replicate arms, and B − Y on every Read-A row; the anti-echo block per run; the lever counts `launch_n`, `launch_eligible`, `launch_rows`, `launch_weight_share` per run.

## 8. The four metric families (all reported, per seed, per family, never pooled)

`PREREG_V8F_DAG_ORC` §5 applies verbatim, for each of the seed's two pairs (the reader reports X − Y, X − P10 and Y − P10 per pair). **STRATEGIC is not paused for v8F** (PI ruling 2026-10-17). T1-kin carries the Read-A rows per family (LONGITUDINAL: speed, along-track, accel MAE, DK-VIOL / DK-HW / DK-TG; LATERAL: heading, yaw-rate, curvature, cross-track, L1 lateral-accel, L2; TACTICAL: TAC-LAT / TAC-LON and R-TLAT / R-TLON; STRATEGIC: S-POS@30 / 50 / 80 against CTRV within a run). T0 carries the box-2 families via `orc_families.py` and `v8f_paired_families.py`. Where a family cannot be computed it is listed per family with the reason, as DAG-ORC §5 lists them. A missing family is a work item, never a silent omission. **The lever acts on LONGITUDINAL (along-track) pressure, so the LONGITUDINAL family and the distance-keeping rows are read first.**

## 9. Controls (each must pass, or the read is REFUSED)

**Differences from `PREREG_V8F_DAG_ORC` §6 + Amendment 1 (every one):**

| DAG-ORC control | here |
|---|---|
| K-GATE | the same, on `--launch-intent-w`, for every trained arm (A_s, A_rep,s; B, HI when trained): G-DVB (FLAG_CHECKS, declared == built, w_traj > 0) and G-LIVE with `launch_intent_probe`. No `--ego-perturb-*` flag is named |
| K-PAIR | **one K-PAIR for each pair** (argv equal apart from `--out` and the lever): argv(A_s) equals argv(CTRL_s) = argv(DAG-OACC_s) except `--out` and `--launch-intent-w 1.58926`; argv(A_rep,s) equals argv(REP_s) except `--out` and `--launch-intent-w 1.58926`. argv(REP_s) equals argv(DAG-OACC_s) except `--out` and `--seed` (10 / 11) — MEASURED (§4). argv(B_s) equals argv(A_s) and argv(B_rep,s) equals argv(A_rep,s), except `--out` and the W of B |
| K-INIT | the same, on all four new runs: `built_levers.init_ckpt.sha256` equals M2b_s's (`601cb435…` / `6104fb15…`) |
| **K-REACH** | **replaced by K-SHARE, per trained arm** (A_s and A_rep,s; B, HI likewise): from the arm's `summary.json`, `launch_weight_share` recomputed as (1 + W)·n / ((1 + W)·n + (eligible − n)) from `launch_n`, `launch_eligible` and `levers.launch_intent_w` lies in **1/3 ± 0.02** (B: 1/4 ± 0.02; HI: 1/2 ± 0.02), and `launch_rows == launch_eligible`. CTRL's and REP's `summary.json` carry no `launch_*` key, and `levers.launch_intent_w` is absent or exactly 0.0 (both were trained before the flag existed; §0 item 8 (e)) |
| (new) K-MASK | per trained arm: `train_window_list_sha256` equals DAG-OACC's (`e144adcd…8168`); `launch_n / launch_eligible` lies within ±0.02 of f = 0.1619 (ESTIMATED f̂ 0.1572–0.1607 at the A_s and A_rep,s seeds) |
| K-BUILD | the launch commit's `v8f_mini_train.py`, `v8f_mini_data.py` and `v8f_mini.py` satisfy, against `aeb54468` (the commit DAG-OACC trained at): byte-identical, or every changed hunk sits behind a default-off flag whose bit-identity test passes on the launch commit. **This build changes `v8f_mini_train.py` and `v8f_mini.py` only; `v8f_mini_data.py` is byte-identical.** The flag's bit-identity test is `test_flag_off_reproduces_the_unmodified_tip_bit_for_bit` (golden written by `d7208328`); the diff is `stage/diff.patch` |
| K-P0 | unchanged: RF0's inputs match `PINS.txt` and predate every A_s and A_rep,s rollout |
| K-RC | the reader's Read-A table for DAG-OACC − P10 equals `DAGOACC_READ.json`'s with 0 differences; **and**, with the dag-orc dumps standing in for the pairs (CTRL = DAG-OACC, REP = REP), the reader reproduces DAG-ORC's banked `A − CTRL`, `A − P10`, `DAG-OACC − P10` and `REP − CTRL` tables and `RF_REP.json` with 0 differences (MEASURED, `evidence/dry_run.json`: 8 table x seed blocks, 152 rows, 0 differences, max |delta| over delta / lo / hi 0.0; RF equals `RF_REP.json`; v1's bars and the banked per-seed composition of the main pair are reproduced; every (X, X) contrast reads exactly 0.0; 204 controls, 0 failed) |
| K-ZERO | unchanged (REP − REP reads exactly 0.0, CI [0, 0]; the RF of REP fed in as its own control is exactly 0; a reader with the bar's RF0 term deleted fails its self-test), extended to X − X for every trained arm, **and the lever's own structural zero**: an all-False mask reproduces the unweighted run bit for bit (test `…all_false_mask_trains_bit_for_bit…`) |
| K-REP-LICENCE | the conditions of §4 hold, or REP is trained anew |
| (new) slow-draw flag | beside, decides nothing (§7): the T0 speed-cell shift of every trained arm against its pair's control, labelled in the RESULT |
| the class of Amendment 2 | does not arise: the lever draws nothing. The mask is a deterministic function of the labels, so two generators cannot share a stream |
| Read A's and DAGGER2's reader controls | unchanged and applied to all eight runs: K4 against the fresh-age T0 eval, K7 track rebuild, PREREG-TEXT, window and floor-planner equality (C3, C4) |

## 10. Outcomes, committed now (arm X = A; X = B only after A's HARMS or NOT ADOPTED)

**The per-seed predicates, defined on both pairs of the seed** (`MM_REDESIGN_2026-10-28.md` §3 item 3). Each condition the table states "at seed s" holds only when it holds on **both pairs of that seed**: the main pair (A_s − CTRL_s) and the replicate pair (A_rep,s − REP_s). Clears, fires and the other row predicates are those of §6 and §7, with the §6 bar and each pair's own control in place of DAG-OACC:

| per-seed predicate | holds at seed s iff |
|---|---|
| R-MOVE | every crawl row clears in A_s − CTRL_s **and** in A_rep,s − REP_s |
| R-CRAWL | A_s − P10 **and** A_rep,s − P10 are each not separated-worse on every crawl row |
| R-TLAT, R-TLON, R-ADE | not separated-worse in either pair |
| R-T0 | the T0 ADE6 point is ≤ +F/2 = 0.0604 m in both pairs |
| R-NEW | no row fires in either pair (fire as in §6: each pair's own control in place of DAG-OACC, and the §6 bar) |
| R-KEEP | the T1-kin ADE6 point is ≤ +F/2 in both pairs |
| R-NOBASE | holds in both pairs |

The outcome table below, its precedence and its enumeration over the per-seed booleans are unchanged. A lever effect is credited at a seed only when two independent training draws show it. Across both seeds that is four draws, the same replication standard as `PREREG_V8F_SPEEDSUB_T1`.

The conditions are `PREREG_V8F_DAG_ORC` §7's, in the same order, re-based by its Amendment 1. Only the committed next changes. A guard "fails at ≥ 1 seed" when it fails at that seed in either pair.

| outcome | condition | committed next |
|---|---|---|
| **REFUSED** | any control fails | fix and re-read; no row is reported |
| **HARMS** | controls pass; a guard (R-TLAT, R-TLON, R-ADE, R-T0, R-NEW) fails at ≥ 1 seed | **A:** not adopted; read **B** under this table (train B first, with its own PI go). **B:** not adopted; the launch weight cannot be dosed under the guards; **NEXT-LAUNCH** (§11) |
| **FIXED / ADOPTED** | guards held; R-MOVE and R-CRAWL at both seeds; R-KEEP and R-NOBASE at both seeds | adopt X as the base. Re-run Read A on X: DOMINANT there is the dominance claim |
| **FIXED / NOT ADOPTED** | guards held; R-MOVE and R-CRAWL at both seeds; R-KEEP or R-NOBASE fails at ≥ 1 seed | the lever is attributed but X cannot replace DAG-OACC. **A:** B (the lower dose) under this table, with its own PI go. **B:** NEXT-LAUNCH |
| **PARTIAL / ADOPTED** | guards held; R-MOVE at both seeds; R-CRAWL fails at ≥ 1 seed; R-KEEP and R-NOBASE at both seeds | adopt X. The crawl rows that still fail go to **NEXT-LAUNCH on X's states** |
| **PARTIAL / NOT ADOPTED** | guards held; R-MOVE at both seeds; R-CRAWL fails at ≥ 1 seed; R-KEEP or R-NOBASE fails at ≥ 1 seed | as FIXED / NOT ADOPTED |
| **PARTIAL / ONE SEED** | guards held; R-MOVE at exactly one seed | not adopted, and never called an effect of the lever (`H-ESTIM-SEED-1`). **NEXT-LAUNCH on DAG-OACC states** |
| **NO EFFECT** | guards held; R-MOVE at neither seed | the along-track weight on launch samples is not the crawl's lever at this dose. **NEXT-LAUNCH on DAG-OACC states** |

**The table is complete and exclusive (MEASURED, `tools/enumerate_cells_li.py`, exit 0; `enumerate_cells_li.out`).** It is DAG-ORC's `enumerate_cells.py` with `outcome_rows` unchanged and the committed-next texts replaced.
* 524,288 cells; each matches exactly one row: REFUSED 262,144 · HARMS 261,888 · FIXED / ADOPTED 1 · FIXED / NOT ADOPTED 15 · PARTIAL / ADOPTED 3 · PARTIAL / NOT ADOPTED 45 · PARTIAL / ONE SEED 128 · NO EFFECT 64.
* **The mapping is DAG-ORC's:** these per-row counts equal the counts in DAG-ORC's own `enumerate_cells.out`, and the script fails if they differ.
* With adoption projected away, the 16 (R-CRAWL, R-MOVE) cells map as `PREREG_V8F_DAGGER2` Amendment 1 (FIXED 1 / PARTIAL 11 / NO EFFECT 4). Adoption is reachable only with R-MOVE at both seeds.
* **The per-seed composition (MEASURED, `enumerate_cells_li.py` section 10):** a seed's predicate is the AND of its main-pair and replicate-pair values (all four combinations checked for every predicate and guard; both per-seed values reachable); 200,000 seeded random pair-level assignments (9 predicates x 2 seeds x 2 pairs, controls on or off) each matched exactly one of the 8 rows and all 8 rows were reached; a miss in either pair is a miss at the seed (three hand-made cases).
* The A → B cascade is total over its 128 combinations. NEXT-LAUNCH is total over its 8 cells, and the launch routing over its 4. Mutation proof: `enumerate_cells_li_mutation.txt` (every mutant exits 1).

**Precedence.** REFUSED, then HARMS. With the guards held, R-MOVE partitions the rest (both seeds, one seed, neither). R-CRAWL splits FIXED from PARTIAL, and adoption is decided last. B is a second shot taken after A, labelled with its dose and carrying its own seeds; it is never pooled with A, and if A is not sent to B it is reported beside as a dose-response row that decides nothing.

## 11. NEXT-LAUNCH: what follows if the crawl does not move, registered now (Rule Zero)

LAUNCH-INTENT is the last lever `PREREG_V8F_DAG_ORC` §8 ranks. The routing variable is M-LAUNCH (§7), read on A (or on B when B is the arm that sent the programme here): **moved** = separated toward 0 in both pairs at both seeds (the replication standard of §10; §0 item 8 (b)).

| M-LAUNCH | what it says | committed next |
|---|---|---|
| **moved** | the plan's launch speed was repaired and the crawl rows did not follow: the crawl is not in O's plan profile | **LOOP-STATE** (zero training, dev-box CPU): decompose the crawl by plan index in X's tick dumps, as `DIAGNOSIS.md` did for HACC (share of the excess in plan 2, curvature error at equal own speed, the replan-state term). Its read names the loop term; the next PREREG intervenes on it |
| **not moved** | the lever did not reach O's launch behaviour at this dose | **LAUNCH-INTENT-HI**: the same lever at s = 1/2 (W_HI = 4.17852), under §10's table, with its own PI go. If it also leaves M-LAUNCH in place, label weighting is not a lever on O's launch |

**If both branches are exhausted**, the programme has no registered lever for the crawl rows on DAG-OACC. The blockers, by name:
* **route extent / nav timing:** eliminated as the lever by N0 (NOT NAVMOVE, INHERITED);
* **state-channel levers:** DAgger iteration 2 and (when read) the perturbation;
* **zero-training post-map (HACC family):** moves the crawl but failed KEEP at s0 (+0.1264 m);
* **what would unblock:** a new diagnosis of the loop term (LOOP-STATE), or a PI decision on an architecture change that gives O a launch-intent input. Neither is a lever to launch without that read or that decision.

## 12. What this read cannot claim

* **It is not closed loop.** T1-kin is a self-action open loop.
* **A one-seed R-MOVE is not an effect** (`H-ESTIM-SEED-1`): only R-MOVE at both seeds, each in both pairs (four training draws), attributes a crawl change to the lever. The replicate pair is the training-run test; the bar is a size, and it is reachable (§6).
* **It attributes to a weighting, not to a mechanism.** The lever is one label-defined weight on one term. A separated effect is the weight's; the read does not say which of the 12 knots or which launch windows carried it.
* **The mask uses the future GT speed.** That is admissible for labels. A result here is a statement about training with a label-defined weight; it says nothing about a deployable launch detector.
* **f and the dose are measured on the TRAIN window list.** The eval launch subset is a different, smaller set (327 windows / 49 episodes for the HACC-MC definition, INHERITED); the share 1/3 is a training-weight fact, not an eval fact.
* ⛔ No bar, row, floor formula, dose or routing changes after any LAUNCH-INTENT row is seen.

## 13. Queue, cost, and launch preconditions

**Box 2** (needs the **PI's go for box-2 time**). Chain order A_0, A_1, A_rep,0, A_rep,1; each run through its gate.

| item | MEASURED basis | ESTIMATED |
|---|---|---|
| gate | DAGGER2's four gates ran 12:02–12:06 (`chain_v8f_dag2.out`) | ≈ 1 min per run |
| train + standard T0 eval | DAG-OACC's two runs took 41 and 42 min with their T0 evals (`chain_v8f_dagoacc_r2.out`); the lever adds one 12-knot slice to a 30-knot loss | ≈ 42 min per run |
| **A (4 runs: A_0, A_1, A_rep,0, A_rep,1)** | | **≈ 2.8 h** (v1: 2 runs, ≈ 1.4 h; the redesign adds 2 runs, ≈ 1.4 h) |
| REP | reused (§4) | 0 (fallback +1.4 h) |
| B (4 runs, only after HARMS or NOT ADOPTED) | | + ≈ 2.8 h, own PI go |
| HI (4 runs, only from NEXT-LAUNCH; the same pair design, §0 item 8 (c)) | | + ≈ 2.8 h, own PI go |
| no bank stage | the iteration-1 bank is reused at DAG-OACC's sha256 | 0 |

**Dev-box CPU:** the T1-kin rollouts of A_s and A_rep,s at both seeds double against v1: 4 rollouts, ≈ 140–305 s each (MEASURED in the SPEEDSUB-T1 and DAG-OACC logs), about 37 min with the K4 lane (ESTIMATED from DAG-ORC §10's 37 min for A + REP); about 1.3 GB of dumps to E: (v1: 0.65 GB). The reader's heavy layer reads eight runs, CPU only.

**Launch preconditions (committed now).**
1. **The trainer is landed**: `stage/stack/scripts/v8f_mini_train.py`, `stage/stack/tanitad/models/v8f_mini.py`, `stage/stack/tests/test_v8f_launch_intent.py`, `stage/stack/tests/launch_intent_golden.json` (the lander, not the author). The launch commit is the one that carries them.
2. **The box-2 chain** for the four arms exists (the arms in the queue, a one-lever asserter that allows `--launch-intent-w` and nothing else beyond `--out` (and `--seed` for REP's reuse check), the K-SHARE and K-MASK run-record checks, the pulls). It is **not** part of this deliverable.
3. **The reader** `launch_intent_read.py` is built from `dag_orc_read.py` (imported unchanged for the Read-A rig), with K-SHARE / K-MASK in place of K-REACH, K-RC green on the dry run and a mutation proof (v2 package, `NOTES.md`). It is landed with this PREREG.
4. **Entry conditions.** LAUNCH-INTENT starts when a DAG-ORC read is NO EFFECT, PARTIAL / ONE SEED, or HARMS of B (`PREREG_V8F_DAG_ORC` §8). If the DAG-ORC read is PARTIAL / ADOPTED with crawl rows left, the base is DAG-ORC's X, and this PREREG is amended to that base with its banked run as CTRL before any run. If a read adopts another base before A's gate, the same.
5. **REP**: the §4 licence is checked on DAG-ORC r2's REP run record before any row is read.

## 14. Evidence register (every number's class and path)

| number | class | path |
|---|---|---|
| f = 49,618 / 306,566 = 0.161851; W_A 1.58926, W_B 0.726175 (share 1/4), W_HI 4.17852; window-list sha256 `e144adcd…` | MEASURED | `f_measure.json`, `tools/measure_f.py` |
| simulated realised shares 0.3315 / 0.3307 / 0.3256 / 0.3281 | ESTIMATED (the sampler stream, not a training run) | `f_measure.json` |
| REP argv = DAG-OACC argv apart from `--out`, `--seed` | MEASURED | `compare_rep_argv.out` |
| the table's cell counts | MEASURED | `enumerate_cells_li.out` |
| flag-off bit-identity, mutation counts, suite counts | MEASURED | `RESULTS.md`, `mutation.txt`, `logs/` |
| the crawl rows, HACC's reductions, O's v0 deficit, F = 0.1207, DAG-OACC's read | INHERITED (`PREREG_V8F_DAG_ORC` text; not re-verified here) | — |
| "the weight dilutes launch windows 5:1", "the bias is bounded by the base rate" | HYPOTHESIS | — |
| the 19 bars `2 × max(F, RF0)`; the crawl bars 1.1070 / 0.0526 / 0.0136 / 0.0096; each below the CTRL − P10 gap at both seeds | MEASURED (arithmetic on the banked files) | `P0_RF0.json` sha256 `1608d70b…`; `DAGORC_READ.json` sha256 `70a7a3a4…`; `enumerate_cells_li.out`; reader test `test_the_crawl_bars_are_reachable_...` |
| RF per row, the v1 bar, the REP s0 slow-T draw | MEASURED / INHERITED (`RF_REP.json` sha256 `64c9444d…`; the REP s0 diagnosis is the Master Mind's, `rep_s0_diag/`) | `DAGORC_READ.json`, `RF_REP.json` |
| realised shares of B (0.2484 / 0.2478 / 0.2435 / 0.2456) and HI (0.4979 / 0.4971 / 0.4912 / 0.4941) | ESTIMATED (the sampler stream, not a training run) | `enumerate_cells_li.out` |
| the reader's tests, mutation pass and dry run (counts in `NOTES.md`) | MEASURED | `read/test_launch_intent_read.py`, `read/mutation_launch_intent_reader.py` (`evidence/mutation_reader.json`), `evidence/dry_run.json`, `evidence/pytest_final.txt`, `evidence/static_check.txt` |
| the redesign itself | INHERITED (the Master Mind's decision, `MM_REDESIGN_2026-10-28.md`) | — |
