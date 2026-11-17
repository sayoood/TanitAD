# PREREG_V8F_NAVRB2 (written 2026-11-17 before any arm could train): a ROUNDABOUT state channel in the route-text nav

Status: drafted by the Architecture agent; to be FROZEN by the MM before any arm trains. No arm is minted, launched or trained by this card's author. PI decision, verbatim (2026-11-17): **"add the channel for the roadabout"** (Option B of `BUILD_NAVRB.md`, e7342444).
Evidence classes: MEASURED (M, artifact named), PUBLISHED (P), INHERITED (I), ESTIMATED (E), HYPOTHESIS (H), UNVERIFIED (U).

## 0. Decisions for the MM / PI (escalated)

1. **Code base `4dedac45`** (branch `agent/v8f-strat1-s1d-20261117`), not the brief's `agent/arch-inf-20260803` tip `5e7083d8`: that tip holds neither `strat1_check.py` nor `build_navtext_nav.py`, and its `v8f_mini_train.py` / `v8f_mini.py` / `v8f_mini_data.py` blobs differ from every STRAT1 training commit (M, `git rev-parse <c>:<path>`). `4dedac45` descends from `a4357e79` (N1/S1/C0 s10/s11, C0REP, S1d) and changes only `strat1_check.py` + a test. The banked s0/s1 arms trained at `22999538`; its model and data blobs equal `a4357e79`'s (`a9f46a37`, `b59b91b7`) and its trainer differs by 5 lines in two gate probes (`bad_ctx`, `nav_source_probe`), not on the training path (M, `git diff 22999538 a4357e79`).
2. **Cost**: base 4 arms, about 0.63 L40-h (E, 0.19 s/step x 3,000 steps, the rate INHERITED from `launch/strat1_chain2.sh:2`), plus mint and gate. The PI's go covers the channel; this card reads it as covering the 4 base arms. Every conditional arm (section 3.2) goes back to the PI with its cost before it runs.
3. **New lever `--nav-rb-lr-mult 10`** for the fresh channel weight, the programme's convention for a fresh parameter in a fine-tune (`--ctx-lr-mult 10`, `--s-far-lr-mult 10`). Ruling needed: accept, or 1.

## 1. Question

Does a ROUNDABOUT state, read from the route text and fed as one extra nav channel, close the text-nav arms' tactical gap on roundabout windows (NAVRB: the router text names no side for a roundabout exit, RB-STATE-ONLY), without a cost on the four families over all windows?

## 2. Encoding (chosen: one extra channel, not a token in the existing vocabulary)

* **Definition.** `nav_rb[j] = 1` iff the navtok row at cache step `j` (the latest router row with `ts <= now_s x 1e6 - 195,000 us`, `navtok/build_table.py:42-71`, I) has `flag == 0` and a roundabout class (`RB_ENTER`, `RB_EXIT`, `ROTARY_EXIT`, `RB_TAKE_EXIT`, `RB_NEXT_EXIT`; the 44-class vocab of `navtok.py`); else 0. No distance, no exit number, no ego quantity. It is exactly the state NAVRB measured (`navrb.py is_state`).
* **Model.** `e_nav_rb = Linear(1, d, bias=False)`, weight initialised to 0.0 with no RNG draw, ADDED to `e_nav(nav)` wherever the 5-channel nav enters (S, T and O tokens). This is the 5-channel `e_nav` widened to 6 channels with a zero-initialised sixth column, kept as a separate key and module.
* **Why not a token in the existing vocabulary.** Channels 0-2 are a one-hot over FOLLOW / TURN_L / TURN_R (`build_navtext_nav.py:65-106`, the v9 layout); a roundabout row already maps to FOLLOW `[1,0,0,0,0]`. A fourth token value has no slot: it would either reuse a code the banked arms learned with another meaning, or change the shape of `e_nav`. Both break parity with the banked N1/S1 and C0.
* **Why a separate key and module.** (a) Parity: with `--nav-rb off` nothing is built; with it on, the step-0 model is the flag-off model plus an exact 0.0 term (state dict of every other key, RNG stream and outputs bit-identical, pinned by a snapshot test). An M2b init loads with exactly `e_nav_rb.*` fresh. (b) DAgger swap rule: `swap_rows` (`v8f_dagger.py:540-583`) rebuilds only nav channel 3 (distance) and holds token and validity as logged. `nav_rb` has no distance part, so it is held as logged on swapped rows, the same rule as the token channels; the swap code is not changed and the probe checks it is the same object. (c) The evaluators' dead-reckoned nav (`v8f_t1_eval.py`, sha-pinned): ticks after `t0` rebuild channel 3 from the own arc and hold the token from `t0`. `nav_rb` is held from `t0` by the same token rule, in a study wrapper (`v8f_t1_eval_rb.py`); the pinned tools are not edited.

## 3. Admissibility (binding)

* **Computed from the route text only, at training and inference.** The builder (`build_navtext_nav_rb.py`) reads `{split}_cls_id`, `{split}_flag`, `vocab` of the pinned `navtok_table.npz` (sha256 recorded in the table's report) and nothing else. It reads no track, no ego speed, no join label, no v9 nav. The 5-channel text nav reads ego speed and the logged arc; this channel reads neither. Nav is an input at training and inference (PI).
* **Not the situation classifier's output.** The builder and the model path import no situation module (`stack/tanitad/data/situations.py` is not imported; AST guard in the test). The class comes from the frozen navtok grammar over NVIDIA's router strings.
* **Not the ego's future path (vision-only rule).** Future-read probe: table rows after step `j` replaced by garbage leave `nav_rb[j]` unchanged on every step; a mutant that reads row `j + 5` goes RED (test). The model reads `nav_rb` frames up to NOW only (the window's `idc` rows).
* **Optimistic by construction (limit, stated).** On PhysicalAI the router text is GPS matched to OSM routing along the drive's own trace (I, NAVTOK_REPORT headline 1). A roundabout row exists because the ego drove through a roundabout. Every number from this card is worded "with the route supplied along the driven path".
* Parity: corpus `train4572` with `--exclude 540:600`, eval-141, unchanged. Train steps with the state on: 29,836 of 461,518 in 433 of 4,572 clips; eval: 1,334 of 14,237 steps in 17 of 141 clips (M, navtok_table.npz, this agent).

## 4. Arms (LoRA pod, `flock /root/gpu_queue.lock`, one job at a time, 3,000 steps, fine-tune from M2b, the STRAT1 recipe)

### 4.1 Base (4 arms, priority order, not a dependency chain)

| priority | arm | = banked arm + | paired with (banked, same seed) |
|---|---|---|---|
| 1 | **N1R_s0** | N1_s0 argv + `--nav-rb on --nav-rb-npz <t> --nav-rb-sha256 <s> --nav-rb-variant real --nav-rb-lr-mult 10` | N1_s0 (`22999538`) |
| 2 | **S1R_s0** | S1_s0 argv + the same five flags | S1_s0 (`22999538`) |
| 3 | **N1R_s1** | N1_s1 argv + the five flags | N1_s1 |
| 4 | **S1R_s1** | S1_s1 argv + the five flags | S1_s1 |

* Every arm names all five NAVRB2 flags, the defaults included. The banked arms' argvs predate the flags; their argv sha256 and tokens are unchanged (the checker's render of C0 / N1 / S1 / S1d is not touched). With `--nav-rb off` the trainer is bit-identical to `4dedac45` (snapshot test), which is training-identical to `22999538`.
* N1R - N1 is the channel's effect on the nav-only arm; S1R - S1 on the route-text arm of record.

### 4.2 Conditional (pre-registered now; each needs the PI's go on its cost)

* **E4, the seed extension**: N1R and S1R at s10 (init 0) and s11 (init 1), paired with the banked N1_s10/s11 and S1_s10/s11 at `a4357e79`; 4 arms, about 0.63 L40-h (E). Runs when, at s0/s1, the deciding cell (5.1) or a driving cell is a gain or a cost at exactly one seed in a pair (RB-SEEDS-DISAGREE). Then the card's COST4 / GAIN4 count rule applies: a cell is a cost (gain) at >= 3 of the 4 seeds with at least one per init, read under F2 (5.3).
* **E4-S1d**: S1dR at s0, s1, s10, s11 (4 arms, about 0.63 L40-h, E) only if the S1D / C0REP read has made S1d the route-text arm.
* **N1Z_s0, the zeroed-channel arm**: N1R_s0's argv with `--nav-rb-variant zero` (the module built, the channel all 0.0, so its weight gets exactly 0.0 gradient and stays 0.0: a same-seed N1 replicate with the new code). Runs only if the read lands on RB-COSTS or on a deciding gain within F2, where the training-run noise of the N1 family is the open question. 1 arm, about 0.16 L40-h (E).

## 5. Read

**Tier.** LONGITUDINAL, LATERAL, TACTICAL: **T1-kin** (self-action open loop, m = 10, causal bucket, dead-reckoned nav with `nav_rb` held from `t0`, logged scene), eval-141, 141 episodes / 2,399 windows per arm, PRIMARY. STRATEGIC route cells: T0 at the window's first tick with the K9b identity, as STRAT1. Detected information (S1R): T0, as STRAT1.
**Estimator.** `taniteval.ci.paired_episode_cluster_bootstrap`, B 2,000, seed 0, over the episodes of the stratum read. It answers "another draw of EPISODES" only. "Another TRAINING RUN" is the floor (5.3). "Another INFERENCE RUN" does not apply (deterministic planner, one environment and one thread count for all dumps).

### 5.1 The deciding stratum: roundabout windows
* **RB**: eval-141 windows whose navtok row at NOW has the state on: **235 windows / 16 episodes** (M, `RESULT_NAVRB.md` section 1; the builder report re-counts it and refuses on a mismatch).
* **RB-TURN**: the RB windows labelled TURN_L / TURN_R: **43 windows / 6 episodes** (episodes 31, 52, 69, 102, 126, 135; M, `RESULT_NAVRB.json`). Six episodes: intervals on RB-TURN are reported and are descriptive; they decide nothing alone.
* **D1 (deciding cell)**: TAC-LAT accuracy on RB, the 24-cell reader's per-window TAC-LAT indicator (`strat1_read.load_arm_cells`, unchanged) restricted to RB. Delta = R arm - banked arm, paired, per seed.
* **D2 (beside, the targeted gap)**: TURN recall on RB-TURN (share of the 43 where TAC-LAT is correct), and the false-turn share on RB minus RB-TURN (192 windows).
* All 24 driving cells are also printed on RB (beside, no bar).

### 5.2 The four families on all windows (the no-cost guard)
* The STRAT1 24 cells per seed (LONGITUDINAL, LATERAL, TACTICAL, ADE/FDE) for N1R - N1 and S1R - S1, with the STRAT1 cost / gain rule.
* **STRATEGIC, in full** (unpaused, PI 2026-11-16): route setting (S's checkpoint position / heading error at 30 / 50 / 80 m vs RC-A, 6 cells per seed, ROUTE-COST rule); route following S_CKPT at 80 m under real, donor and zero nav (`nav_rb` follows the nav variant), skill = real - donor; S_CKPT on RB windows beside; detected information for S1R (5 variables, STRAT1 rules and K4 / K5 controls). **Weather follows the PI ruling "Drop fog for now" (2026-11-17, `whead/nofog/AMEND_WH_NOFOG.md`), in the read only:** weather is scored on clear / rain / snow; fog-labelled windows are unscored for every arm and control, with their n windows / episodes printed; the fog logit is masked before the argmax. S1R trains on the banked label table, unchanged, for parity with S1. Derived goals and the 6-30 s band: not in this arm (reported as such). Each STRATEGIC cell also prints S1R - S1.

### 5.3 Floor
* F(cell) = |C0_rep - C0_s0| (the STRAT1 card's floor), computed the same way on RB for D1 (F_rb).
* F2(cell) = max(F, F_rep) with F_rep from C0REP (`PREREG_ADDENDUM_C0REP.md`) when both replicate pairs are banked; else F, and the read says which.
* A gain on a cell: separated on the better side and |Delta| > floor. A cost: separated on the worse side and |Delta| > floor.

## 6. Controls (each with the value it must read)

| id | control | must read |
|---|---|---|
| Z0 | flag-off build vs `4dedac45` (snapshot of every state-dict key, value, loss term and output) | bit-identical |
| Z1 | flag-on build at step 0 (zero weight) vs flag-off, same init and batch | every output and loss term bit-identical; RNG stream identical |
| Z2 | gradient on `e_nav_rb.weight`, real channel with at least one 1 in the batch / channel all zero / flag off | > 0 / exactly 0.0 / no parameter |
| Z-EVAL | the R arm's checkpoint dumped with `nav_rb` zeroed (zero-training): real - zero on the windows whose `nav_rb` frames are all 0 | exactly 0.0 on every cell |
| Z-USE | the same, real - zero on RB (beside) | reported; the model reads the channel iff separated |
| K1 | reader self-pair (N1_s0 vs N1_s0) | exactly 0.0 on every cell |
| K-RE | every recomputed banked pair (N1 - C0, S1 - C0 at s0/s1) equals `RESULT_STRAT1.json` | exact |
| K9b | T1 tick-0 S output vs T0, same arm, full grid | equal at the STRAT1 tolerance |
| K-MUT | delete the `e_nav_rb` term in the forward; delete the `nav_rb` join line in main; drop `nav_rb` from the window item | each named test RED |
| G-CLOCK-RB | builder: per-step state equals the NAVRB rule on both splits, every used row has age >= 0, eval reproduces 1,334 steps and 235 / 16 at NOW; regression arm (state read 5 steps = 1 s ahead) | PASS; the regression arm FAILS |

## 7. Outcomes, committed before any number (each names its next lever, RULE ZERO)

Read per pair (N1R - N1, S1R - S1) at s0 and s1. "Guard" = no cost on any of the 24 all-window cells at both seeds and no ROUTE-COST.

| outcome | condition | triggers |
|---|---|---|
| **RB-CLOSES** | D1 a gain at both seeds in S1R - S1, the guard holds, and Z-USE separated | the channel enters the route-text recipe (S1R replaces S1 as the route-text arm); next lever on the remaining RB gap vs C0 (v9 recall): the RB-DIST card (exit number + countdown distance channels) |
| **RB-NAV-ONLY** | D1 a gain at both seeds in N1R - N1 but not in S1R - S1 | S1R's heads compete with the channel through the shared adapter: S1dR at s0 / s1 (detached heads, 2 arms, PI go on cost) |
| **RB-SEEDS-DISAGREE** | D1 a gain (or any guard cell a cost) at exactly one seed in a pair | E4 (section 4.2), COST4 / GAIN4 under F2 |
| **RB-READ-NO-GAIN** | Z-USE separated at both seeds, D1 not a gain at either seed | the state is read but carries no side: RB-DIST card (exit number, where exit 1 -> TURN_R on 20 of 20, and the countdown distance), zero-training first: the navecho2 nav-only classifier with the state + exit channels |
| **RB-UNREAD** | Z-USE not separated at both seeds | dose: the same arms with `--nav-rb-lr-mult` 30 at s0 (2 arms, PI go on cost); first, zero-training, the Adam-state reachability gate on `e_nav_rb.weight` (`zerograd_gate.py`) and its final norm |
| **RB-COSTS** | a guard cell a cost at both seeds in a pair, or D1 a cost at both seeds | not adopted; N1Z_s0 (section 4.2) separates the channel from training noise before any other arm |

* Labels are per pair; both pairs are printed. When two labels hold, both are printed.
* Bars, controls, arms and outcomes do not change after any number is seen. A bar that fails is reported as written, then the triggered arm is built (RULE ZERO).

## 8. Known risks
* 3,000 steps at lr 1e-4 may under-train a zero-initialised weight; the lr-mult and RB-UNREAD cover it.
* The state is on for 192 RB windows that are not turns: a channel read as "turn right" would raise false turns there; D2's false-turn share reports it.
* The DAgger banks were rolled out by M2b under the v9 nav; equal for the R arm and its pair, so no confound.
* RB-TURN rests on 6 episodes; D1 on 16. A non-separated D1 may be thinness; the label then names n.
* Pod preconditions (cache, inputs, memory cap) are UNVERIFIED until the chain's checks run.

## Log
* 2026-11-17, Architecture agent: drafted before any arm could train.
