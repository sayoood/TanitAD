# PREREG_V8F_ROLLHIST (FROZEN 2026-11-20, before any bank, pilot, run or number): train the planner and the strategic head on their own rolled history at the replan tick

Status: FROZEN by the MM at the sha256 beside this file (`PREREG_V8F_ROLLHIST.sha256`), before any bank, pilot, run or number of this card. Drafted by a design agent; the body below is the draft unchanged, and the FREEZE RECORD binds where it is more specific.

## FREEZE RECORD (MM, 2026-11-20 ~19:56Z)
* **PI answers, first-hand:** size "Both seeds (5.2 h)" (~17:35Z); readout and queue "C*, run it next" (~19:55Z).
  * Readout of record: C*, `RH_READOUT=causal_g1x` (`--readout causal_g --acc-gain 1 --path-exec`; argv `--dagger-bank-readout causal_g1x --dagger-bank-kappa 0.2`).
  * Queue: lever C runs next on the LoRA pod, before STAGEB's UE_s0.B, under `/root/gpu_queue.lock`; the four arms RH_s0, B0_s0, RH_s1, B0_s1 in one chain.
  * Consequence, committed now: STAGEB has not run, so the beside contrast RH - UE.B (section 2) and UE.B's term in the floor F_c are printed NOT-COMPUTABLE (reason: STAGEB not run) unless STAGEB has run before the read. The deciding contrast P = RH - B0 is unchanged.
  * Cost line: about 5.2 LoRA pod-h (ESTIMATED upper). A projection above 6.2 pod-h (+20 %) goes back to the PI.
* **Section 6:** item 1 (size), item 2 (queue) and item 3 (readout) are answered as above. MM items: the G-DVB entries live in `v8f_mini_train.py` FLAG_CHECKS (built at 2bda9e1a); RH and B0 run from the one commit 2bda9e1a.
* **Section 4, launch gate (MM ruling):**
  * Every arm trains only through a PASS token from the study-side checker `dscale10k_check_rh.py` (sha256 ccc0904d; kinds RH and B0 at stage B; 37 tests PASS, the mutation without the generator-record compare RED), minted by `chk_arm_rh.sh mint` and enforced by `chk_arm_rh.sh run`. The token binds the commit, the argv sha256 and the data-manifest sha256s as the stage-B token does.
  * FRESH at launch: G-SUITE on 2bda9e1a; each arm's trainer `--selfcheck-only` on its own argv and real bank (G-DVB, `dagger_probe` = G-LIVE); the pair check; the bank generator record.
  * INHERITED from UE_sK stage A and labelled INHERITED in the token's evidence: G-HYG, reachability, G-CLOCK, G-EVAL, G-CKPT. Lever C changes the bank only, not the model, loss, label or clock; this is the pattern that gated STAGEB.
  * The chain-local record `rh_gate_record.py` stays as a second check and is not a token.
* **Not of record:** the `causal_g1` branch of `render_argv_rh.py`, which patches the pair checker in process. At `causal_g1x` it is unreachable (`render_argv_rh.py:66-67` returns the stock checker's result).
* **Section 9 binds as written,** including the settled speed bucket: the bank keeps the clip's logged `limit`, so it carries two of the three members of the coordinator's speed group and the 0.84 share is an upper bound. The s1 pair stays on the LoRA pod; the coordinator's A40 offer is not taken.
* **Pod paths** are syntax-checked only (HANDOFF_B2 item 6). A first-run path fix before any bank or pilot number is allowed and logged. No lever, argv, bar, cell or outcome changes after this freeze.

Authority: decision card D16, recommendation C, approved by the PI on 2026-11-20 ("Approve all four"; `E:/Projects/TanitAD-artifacts/DECISION_CARD_V8F_HANDOVER_2026-11-20.md`, EVAL_DOCTRINE.md amendment 2026-11-20 quotes the sha256 prefix e40c02ca88334fab). The approval names no GPU hours, so the size of the first arm is the PI's decision (section 6).
Code read at mirror tip `fb686f4a` (branch `agent/arch-inf-20260803`, `git ls-remote` 2026-11-20). Every `file:line` below is at that tip.
Evidence classes: MEASURED (ours, artifact named), INHERITED (another agent's measurement, not re-verified here), ESTIMATED (arithmetic on the former), HYPOTHESIS. Tier stamp on every rollout number: `T1-kin/old` (before 2026-11-20) or `T1-kin/hb0.2` (the executor of record from 2026-11-20). Never driving performance.

## 0. Why (the measured case, quoted, not re-measured)

* Row `H-V8F-PROOF-119` (card ADD63, `T1-kin/old`, UE_s0 and UE_s1 stage-A checkpoints, 141 episodes, 2,399 windows, paired episode-cluster bootstrap pooled over the two model seeds, INHERITED): with the share g of the executed acceleration the planner reads at the replan tick dosed at 0, 0.5, 1, both steps read WORSE on all three families (`ACCGAIN10K-{LONGITUDINAL,LATERAL,TACTICAL}-RISES-WITH-OWN-ACCELERATION`); the step D2 = `R0C10 - R0G50` reads speed MAE +1.3832 m/s [+1.143, +1.6598], yaw-rate MAE +25.50 deg/s [+21.19, +30.45], tactical ADE +1.2784 m [+1.0526, +1.5237] (`integ10k/read_accgain/pod/READ_ACCGAIN10K.json`, read here). Readout K (`R0G0X`: no own acceleration, path from executed knots) reads better than pinned on all ten family cells (D16 section 2a).
* Row `H-V8F-PROOF-120` (card ADD66, same models, `T1-kin/old`, INHERITED): the strategic head's route-checkpoint error rises over its first-tick level by +8.38 / +14.77 / +22.95 m (30 / 50 / 80 m) for an output computed 3.6 s into the rollout; 60 to 73 % of it is in the head's own output (descriptive split, note N1); readout K is better (-3.42 / -7.05 / -11.75 m), the causal readout worse (+3.21 / +6.25 / +9.28 m).
* D16's reading: "The models were trained on logged histories; on their own rollout they continue the acceleration they read." The repair at the source is to train on the history the model will read on its own rollout.

## 1. The lever, at file:line level

### 1.1 What already exists: the v8f DAgger own-state swap IS "own rolled history at the tick" (MEASURED from source)

The v8f trainer already trains O, T and S on the model's own T1-kin state. The hops, flag to loss:

| hop | where | what |
|---|---|---|
| argv | `stack/scripts/v8f_mini_train.py:183-194` | `--dagger-p` (default 0.0 = off, `DG.DAGGER_DEFAULTS`), `--dagger-bank`, `--dagger-bank-sha256`, `--dagger-seed` |
| G-DVB table | `stack/scripts/v8f_mini_train.py:91-94` (`FLAG_CHECKS`, declared at :68; the rule "every argparse flag has an entry" at :7) | `"dagger_p": "dagger:dagger_p"`, `dagger_bank`, `dagger_bank_sha256`, `dagger_seed` |
| bank load + controls | `v8f_mini_train.py:438-465` (`load_dagger_bank`) | manifest re-hash, K-B3 TRAIN-only + window-list sha256 + disjoint from eval-141, K-B2 generating checkpoint == `--init-ckpt`, COVERAGE >= 0.5 else BLOCKED |
| build | `v8f_mini_train.py:467-479` (`make_dagger`) -> `stack/tanitad/data/v8f_dagger.py:598` (`DaggerSwap`) | declared-vs-built in `bad_dagger`, `v8f_mini_train.py:482-553` |
| loop | `v8f_mini_train.py:2703` | `b = dag.apply(b, step)`: with probability p per sample, an eligible row is swapped |
| swap | `stack/tanitad/data/v8f_dagger.py:540-582` (`swap_rows`) | REBUILT from the bank record: `p10f` (whole history window), `p10_t`, `kin3`, `base` (P10 prior input), `pose`, `pose_t`, `pose_s`, nav channel 3 (own arc); labels stay the human's at the matched route point r'; `lab_s.rc` re-expressed in the own frame (`rc_frame_change`, :579) |
| strategic head reads it | `stack/tanitad/models/v8f_mini.py:799-807` (`run_s`) | S's conditioning token is `self._ego(self._at(b["p10f"], i))` at its own tick frame `i = NOW - a_s`, plus nav |
| tactical / planner read it | `v8f_mini.py:810-823` (`run_t`, p10f at `NOW - a_t`), `:825-850` (`run_o`, p10f at NOW and `base` through the P10 prior) | |
| loss | `v8f_mini.py:992-994` (`T["ckpt"]`, S's own loss against the re-expressed `lab_s.rc`), `:1048-1057` (total incl. `w_acc`) | `--detach-upper` (`v8f_mini.py:884-885`, `:901-902`, the `.detach()` lines) detaches S's checkpoints and T's code INTO the lower layers only; S still trains on its own `T["ckpt"]` |
| G-LIVE | `v8f_mini_train.py:1970-2090` (`dagger_probe`), called at `:2564` | p = 1 swapped batch: every loss term finite, every trainable group gets a gradient, sum abs(grad swapped - grad logged) > 0; p = 0 control EXACTLY 0.0; frozen P10 prior and WM auxiliary EXACTLY 0.0 |

The bank (what "own rolled history" currently is): `stack/scripts/v8f_dagger_bank.py` runs `v8f_t1_eval.rollout` itself (`v8f_dagger_bank.py:10-12`, flags of `v8f_dagger.py:45`: `--replan-steps 10 --limit causal --nav deadreckon --scene logged --schedule periodic`, executor `plan`) and stores, per TRAIN window, the own state at ticks 1 and 2 (`TICKS = (1, 2)`, `v8f_dagger.py:41`: j = t0 + 10 and t0 + 20, i.e. 2.0 s and 4.0 s into the rollout) with the own history rows j - hmax + 1 .. j (`v8f_dagger_bank.py:58-72`).
The own state's kinematic rows come from the pinned tool: `WindowState.commit_plan` (`stack/scripts/v8f_t1_eval.py:181-192`) fills the executed frames with the module-level `hermite_frames` (central tangents, so the last executed knot's tangent uses the next, discarded plan knot), and `_finish` (`v8f_t1_eval.py:199-219`) derives `kin3`, `p10f`, `base`, `pose` from that track. **So the existing bank is the PINNED readout under the OLD executor (no heading bound).**

The recipe of record already uses it: union stage B = `U_s0.B.tmpl` (`raw/2026-11-15-v8f-datascale/gate10k/pod_out/argv/U_s0.B.tmpl.json` in the mirror, read here): `--arm M2b --kin-mode yaw --batch 64 --init-ckpt @INIT@ --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p 0 --dagger-p 0.1 --dagger-bank @BANK@ --dagger-bank-sha256 @BANKSHA@ --dagger-seed 0 --w-acc 49.6969 --detach-upper --fit-join` plus the union data flags. **The two models D16 measured, UE_s0 and UE_s1, are STAGE-A checkpoints: no stage B (and so no DAgger) has run for UE** (`stageb/FACTS_STAGEB.md` A6, INHERITED; the read dumps are named `DS-UE_s1-A`; stage B for UE is pre-registered, not run: `stageb/PREREG_V8F_DSCALE_STAGEB.md`, register row H-V8F-DSCALE-STAGEB-1).

### 1.2 What this card changes: the bank's history is the one the deployed model will read, under the executor of record

Lever ROLLHIST = the DAgger stage B of the recipe of record, with its bank generated so that the own rolled history at the tick is
1. **under the T1-kin executor of record**: `v8f_t1_eval_hb.installed(T1, 0.2)` (`stack/scripts/v8f_t1_eval_hb.py:94-103`, the bound at `:51-80`), so the own yaw history does not carry the 52.9 deg standstill heading artefact (row 122, INHERITED); and
2. **read the way a car measures it (readout C\*)**: `causal_g` with `--acc-gain 1 --path-exec`: speed, one-frame acceleration and path from the EXECUTED knots only (formulas at `integ10k/v8f_t1_eval_reader_rule_rzg.py:251-296`: `backward_frames`, `causal_frames(..., "step")`; gain 1 routes through the `causal` formula, `:284-285`). Composition: `hermite_frames := bounded(readout(orig))`, the heading bound applied to the readout's frames. MEASURED from source: card ADD67's composed wrapper does exactly this (`integ10k/v8f_t1_eval_reader_rule_rzgfh.py:363-380`, `counted_frames`: the readout's `causal_g_frames(..., gain, path_exec)` first, then `bound_heading(xy, psi, v, knots[0], psi0, kappa)` on its heading column, positions and speeds checked unchanged); with `--path-exec` the unbounded pinned frames function runs on the executed knots and the bound acts after it. Card ADD68 runs this same readout as `R0C10X` / `R0C10XHb20` ("causal, all of the own acceleration, path from executed steps only (`--readout causal_g --acc-gain 1 --path-exec`)", `PREREG_V8F_PROOF_ADD68.md:29-30`), so C\* is already a named readout of the programme.

Everything else is the recipe of record, unchanged: p = 0.1 per sample, ticks 1 and 2, m = 10 (2.0 s cycle), 3,000 stage-B steps, batch 64, `--w-acc 49.6969`, `--detach-upper`, the matcher and its PREREG constants (`v8f_dagger.py:36-50`), labels at r'. Per arm the swap touches about 3,000 x 64 x 0.1 = 19,200 sample draws times the eligible fraction (ESTIMATED).

### 1.3 Flag or new code? BOTH: a flag on the existing trainer, fed by NEW bank-generator code, plus two NEW declaration flags

* Trainer: no new mechanism. The arm names `--dagger-p 0.1 --dagger-bank <RH bank> --dagger-bank-sha256 <sha>` exactly as `U_s0.B.tmpl`.
* NEW code (none written by this agent; each item needs its own build agent, tests and landing):
  * N1 `stack/scripts/v8f_t1_readout.py` (new, name proposed): `backward_frames`, `causal_frames`, `causal_g_frames` promoted from the integ10k study wrapper `v8f_t1_eval_reader_rule_rzg.py:251-296`, arithmetic unchanged, with ADD63's identity controls as unit tests (gain 1 equals `causal`, gain 0 equals `causal_z`, array for array; `integ10k/check_acc_gain.py`). A training chain must not import a Research-Lab study file by path.
  * N2 `stack/scripts/v8f_dagger_bank.py`: flags `--readout {pinned,causal_g}`, `--acc-gain G`, `--path-exec`, `--kappa-max {none,0.2}` (defaults `pinned` / `none` reproduce today's generator bit for bit); install `hb.bounded_frames(readout_frames(orig), kappa)` for the run and restore it; add `v8f_t1_readout.py` and `v8f_t1_eval_hb.py` to `TOOL_FILES` (`v8f_dagger_bank.py:49`); write a `generator` record (readout, acc_gain, path_exec, kappa_max) into every shard provenance and into `bank.json`.
  * N3 `stack/tanitad/data/v8f_dagger.py`: `DaggerSwap.built_levers()` (`:618-622`) adds `dagger_bank_readout` and `dagger_bank_kappa` READ FROM THE LOADED BANK HEADER; a header without the field reads `pinned` / `none`, so every existing bank and argv is unchanged. `constants()` (`:59-62`) is NOT changed (the matching constants are the PREREG's).
  * N4 `stack/scripts/v8f_mini_train.py`: declaration flags `--dagger-bank-readout` (default `pinned`) and `--dagger-bank-kappa` (default `none`), refused when they differ from the loaded bank's header, and refused when non-default at `--dagger-p 0` (a lever declared that does nothing). Reason: the bank sha256 binds the bank, but the argv would not NAME what the history is; CLAUDE.md "a flag whose default disables it must be named in the arm's argv".

## 2. Arms

All on the LoRA pod, one job at a time under the GPU queue lock, from ONE new commit that carries N1-N4 (so arm and control differ in one lever, not in a commit).

| id | what | changed against `U_s0.B.tmpl` (UE's stage-B recipe) | init |
|---|---|---|---|
| **RH_s0** (the lever) | stage B on own rolled history, readout C\*, executor hb0.2 | `--dagger-bank <RH bank of UE_s0.A>` `--dagger-bank-sha256 <its sha>` `--dagger-bank-readout causal_g1x` `--dagger-bank-kappa 0.2`; the union data flags as UE; `--out` | UE_s0.A, HF `Sayood/tanitad-v8f-dscale10k-runs` `UE_s0/A/ckpt_last.pt`, sha256 `2fd4b747...` (`read_b3/banker/UE_s0.A.json`) |
| **B0_s0** (matched control) | the same stage B with no own history | `--dagger-p 0`; `--dagger-bank`, `--dagger-bank-sha256`, `--dagger-seed` removed (the DAG-0 rule, `v8f_mini_train.py:182`); `--out` | UE_s0.A |
| RH_s1, B0_s1 (replicate) | as above at seed 1 | `--seed 1 --dagger-seed 1`, bank generated by UE_s1.A | UE_s1.A, sha256 `f13df8ce...` (FACTS A5) |
| beside, no new cost if STAGEB runs | UE_s0.B / UE_s1.B of `PREREG_V8F_DSCALE_STAGEB` (pinned bank, old executor, tree `cdbff971`) | | |

* The bank of RH is generated by the arm's own init checkpoint (K-B2 refuses anything else, `v8f_mini_train.py:455-457`), over the union TRAIN windows (985,048, FACTS A1).
* Contrasts: **P = RH - B0** (the lever as a whole: own rolled history, as a car reads it, on or off; same commit, same steps, same `--w-acc`, same `--detach-upper`). Beside: RH - UE.B (the bank's readout and executor together; "lever plus commit", two changes, attributes neither alone). Beside: RH and B0 against UE.A (stage A) for the level.
* Replicate plan. A separated interval from one seed is necessary, not sufficient (`H-ESTIM-SEED-1`). This device is not bit-reproducible (FACTS E8: same code, 45 differing cells), so the replicate is a different stage-A seed (s1), never a same-seed rerun. Floor per cell F_c = max(abs(RH_s1 - RH_s0), abs(B0_s1 - B0_s0)) under the same readout and executor; if STAGEB's UE_s0.B / UE_s1.B exist, abs(UE_s1.B - UE_s0.B) joins the max (it can only raise the bar). A one-seed read uses the stage-A floor F_U,c = abs(UE_s1.A - UE_s0.A) under the same readout and executor (from the ADD67 / ADD68 runs at both model seeds, no new rollout) and is worded PROVISIONAL.
* Inference variance: the rollout does not sample (D16 header), so no inference seed is needed; the read prints the K-B1-style re-run check of one chunk to confirm it.

## 3. Reads and bars (all at `T1-kin/hb0.2`; committed before any bank, run or number)

Executor of record: `python stack/scripts/v8f_t1_eval_hb.py --kappa-max 0.2 --inner <readout wrapper> <argv>` (EVAL_DOCTRINE.md amendment 2026-11-20, items 1, 3). Windows: eval-141, 2,399 windows, 141 episodes; the headway reader's lead windows (642 in ADD63 control a6; D16 quotes 562 scored for the unsafe rate). Fog windows unscored (PI 2026-11-17). Estimator: paired episode-cluster bootstrap, B 2,000 (`taniteval/ci.py`); it answers "another draw of EPISODES" only; the seed replicate answers "another TRAINING RUN"; inference is deterministic.

### 3.1 Runs per checkpoint (RH_sk, B0_sk), names as card ADD68 (suffix `Hb20` = bound 0.2)

`R0Hb20` (pinned, the convention), `R0C10Hb20` (causal, all own acceleration, pinned path), `R0G50Hb20` (half), `R0G0Hb20` (none, pinned path), `R0KHb20` (readout K), `R0C10XHb20` (readout C\*: all own acceleration, executed path). Six rollouts per checkpoint. STRATEGIC along the rollout from the same dumps by `integ10k/strat_roll_read.py` (CPU).
The stage-A reference costs nothing new: card ADD67 banked `R0Hb20` and `R0KHb20`, and card ADD68 (frozen, `raw/2026-11-12-v8f-proof/PREREG_V8F_PROOF_ADD68.md:26-37`) runs `R0C10Hb20` and `R0C10XHb20` at both UE.A model seeds. Control c-EXEC (known value): ADD68 runs through the study wrapper `rzgfh` (sha256 373ffe72f2655c35) while this card's reads run through the tier of record `v8f_t1_eval_hb.py --kappa-max 0.2 --inner v8f_t1_eval_reader_rule_rzg`; one run of UE_s0.A under both forms must agree array for array (141 files, 0 differing), else the stage-A reference is rerun under the tier-of-record form.

### 3.2 Cells, per family, never pooled across families

| family | named cells (decide) | reported per family, no bar |
|---|---|---|
| LONGITUDINAL | `speed_mae_mps`, `along_mae_m` (ADD63 names); distance-keeping: headway reader U (unsafe rate) | headway C (contact), F (falling behind) |
| LATERAL | `heading_mae_deg`, `yaw_rate_mae_degps` | `curvature_mae_1pm`, `cross_mae_m` |
| TACTICAL | `ade_tac_m` | `lon_err_rate`, `lat_err_rate`, `joint_err_rate` |
| STRATEGIC (in full since the PI's 2026-11-16 ruling) | ADD66 `pos` at 30 / 50 / 80 m at the third tick (ROLL2), and the ROLL name | ROLL1 (second tick), `head`, `lat`, `abs_along`; first-tick (T0) checkpoint error; strategic decision between routes NOT COMPUTED (n = 0, no instrument, as row 120) |

Cell verdict (ADD63's family reader, `read_accgain10k.py`): BETTER / WORSE when the paired interval excludes 0 on that side, both seed means agree and abs(D) > F_c; family BETTER if every named cell is BETTER, WORSE if every named cell is WORSE, NOT-SEPARATED if none, else MIXED. Guard G-T0: RH - B0 T0 ADE6 (policy-rolled, `v8f_dscale_read.py`) upper interval bound <= +F_T0 = +0.0705 m (abs(UE_s1.A - UE_s0.A) = abs(2.0376 - 1.9671), FACTS E3, INHERITED; one seed pair, a weak floor, so the full F and not F/2).

### 3.3 The four reads

* **P1 (lever)**: RH - B0 under readout C\* (`R0C10XHb20`), all four families.
* **P2 (D16 after-check 1, ADD63 on the bounded executor)**: on RH, ADD68's pair ACC = `R0C10XHb20 - R0KHb20` (all of the own acceleration against none, both on the executed path) reads NOT WORSE on every family, and so do ADD63's dose steps D1 = `R0G50Hb20 - R0G0Hb20` and D2 = `R0C10Hb20 - R0G50Hb20` ("the causal readout stops being the worst"). Reference: the same pairs on UE.A from ADD68 (expected WORSE, as row 119 at `T1-kin/old`). Attribution control: on B0 the ACC pair still reads WORSE on at least two of the three families (if B0 loses it too, stage B's other terms, e.g. `--w-acc`, did it, not the history).
* **P3 (D16 after-check 2, ADD66)**: the ROLL name of RH under `R0Hb20` (the convention) and under `R0C10XHb20` is HOLDS (neither ROLL1 nor ROLL2 reads WORSE at any arc, `STRATROLL10K_DEFS.md`). Attribution control: B0 under `R0Hb20` reads DEGRADES (as UE.A did under the old executor).
* **P4 (K unneeded)**: RH under C\* (`R0C10XHb20`) minus B0 under K (`R0KHb20`): no family WORSE. The training repair at the source matches the inference workaround.

### 3.4 Outcomes, committed in advance, mutually exclusive, two seeds (a one-seed read prints the same name prefixed PROVISIONAL)

| name | condition | reading | next lever |
|---|---|---|---|
| **ROLLHIST-REPAIRS-K-UNNEEDED** | P1: at least two families BETTER at both seeds, none WORSE at either; P2 and P3 at both seeds; P4 at both seeds; G-T0 holds | training on the history a car measures removes the handover defect at the source | to the coordinator and the PI: the integrated candidate is specified with the causal handover and without readout K (recommendation A revisited); register the base as `UE.A + ROLLHIST` |
| **ROLLHIST-REPAIRS** | as above but P4 fails | the source is repaired; K still adds on top | the candidate keeps K beside; next: DAgger iteration 2 from RH's own states (one bank, ~2.3 pod-h) only if P4's gap is above floor on two families |
| **ROLLHIST-PARTIAL** | P1 at least one family BETTER at both seeds, none WORSE; but P2 or P3 fails at one seed or more | the dose shrinks but the model still continues its own acceleration, or S still degrades | **share p = 0.3 on the SAME bank** (stage B only, K-B2 holds, about 0.27 pod-h per seed, the cheapest next arm; p = 0.3 is the DAgger-1 mapping not yet tried on this line) |
| **ROLLHIST-NO-EFFECT** | no family BETTER at both seeds and none WORSE | the swap reached the loss (G-LIVE) but did not move the read | p = 0.3 on the same bank, then a 9,000-step stage B (~0.8 pod-h, ESTIMATED from 0.282 s/step) |
| **ROLLHIST-HARMS** | a family WORSE at both seeds, or G-T0 fails at both | own rolled history at this readout costs | localise with the beside UE.B (pinned bank): if UE.B - B0 is not WORSE, the causal readout's states are the harm; next: a bank under readout K (train on what K hands) or the recovery label for off-route states (`H-ST-RECOVERYLABEL-1`) |
| **ROLLHIST-SEED-DISAGREE** | a family BETTER at one seed and WORSE at the other, or the decisive cells straddle the floor | not decidable on two seeds | third seed pair: needs a UE_s2.A stage A (3.3-3.9 pod-h) plus RH_s2 and B0_s2: back to the PI |
| **ROLLHIST-BLOCKED** | COVERAGE < 0.5 in the pilot or the full bank, or any gate check fails | the arm does not launch; no number | if coverage: bank of tick 1 only (no PREREG constant moves) or the recovery label; if gate: fix and re-gate |

Precedence: BLOCKED, then HARMS, then SEED-DISAGREE, then the rest in the order of the table.

## 4. Launch gate and reachability

### 4.1 Where the G-DVB entries go (MEASURED)

`stack/scripts/launch_gate.py` at the tip has **0** lines matching `dagger` and **0** matching `v8f_mini_train` (11,661 lines; same-breath control: `v8f` matches 20+ lines, the refcv8-derived v8F rung profile at `:979`). The v8f-mini G-DVB table is the trainer's own `FLAG_CHECKS` (`v8f_mini_train.py:68`), checked by `--selfcheck-only`, and the stage-A/B tokens are minted by the chain checker (`gate_block3/block3_files/block3_prep.sh:41`, `chk_arm3.sh mint`; `study/dscale10k_check.py`). **So the entries this card needs go into `v8f_mini_train.py` FLAG_CHECKS, not into `launch_gate.py`** (the brief named launch_gate.py; flagged for the MM).

To add (by the build agent, not here):
* `FLAG_CHECKS`: `"dagger_bank_readout": "dagger:dagger_bank_readout"`, `"dagger_bank_kappa": "dagger:dagger_bank_kappa"` beside `v8f_mini_train.py:94`; their `bad_dagger` clauses (declared vs the loaded header; non-default at p = 0 refused).
* The chain checker: a C-check that RH's argv names both declaration flags and the bank sha, and that B0's argv differs from RH's in exactly the four DAgger flags, the two declaration flags and `--out` (G-PAIR form of STAGEB section 8).
* G-LIVE: `dagger_probe` runs unchanged on a swapped batch of the RH bank (each of its clauses must PASS).

### 4.2 Reachability proof plan (each with a control that reads EXACTLY 0.0 and a mutation that must turn it RED)

| id | proves | non-zero reading | exact-zero control | mutation |
|---|---|---|---|---|
| R1 | the swap reaches the loss | `dagger_probe` gradient clause: sum abs(grad swapped - grad logged) > 0 on the RH bank | p = 0: the same batch object, loss and gradient difference 0.0; frozen P10 prior 0.0; WM auxiliary 0.0 | delete `out.update(upd)` (`v8f_dagger.py:685`) -> RED (existing) |
| R2 | the STRATEGIC head trains on the own rolled history | sum abs over `read_s`, `head_s` parameters of grad(`T["ckpt"]` swapped) - grad(`T["ckpt"]` logged) > 0 | the same at p = 0: 0.0; and the S-parameter gradient of `T["traj"]` alone under `--detach-upper`: 0.0 (S is reached by its own loss, not through O) | replace `b["p10f"]` by the logged tensor in `run_s` only -> the S clause reads 0.0 -> RED |
| R3 | the planner (O) trains on it | the same over `head_o` / `read_o` with `T["traj"]` | p = 0: 0.0 | as R2 in `run_o` |
| R4 | the bank holds readout C\* under hb0.2 | per pilot record: the one-frame acceleration at the last executed knot minus 1.0 x the executed one <= 1e-6 (ADD63 control a3, ported); every heading step <= 0.2 x the step distance + 1e-12 (the hb contract, `stack/tests/test_v8f_t1_eval_hb.py`) | `--readout pinned --kappa-max none` reproduces today's generator's pilot shard bit for bit (max abs difference 0.0) | delete the line that installs the readout (or the bound) -> the RH pilot shard equals the pinned one -> R4 RED |
| R5 | the argv declares what the bank is | — | — | `--dagger-bank-readout pinned` with the RH bank -> refused; the RH bank at `--dagger-p 0` with a declaration -> refused |

## 5. Cost (LoRA pod; all walls on that pod)

⚠ The LoRA pod's GPU is an **RTX 4000 Ada**, not an L40-class GPU (`stageb/FACTS_STAGEB.md` A8, nvidia-smi name, INHERITED; also the programme memory note). Quoted as pod-h.

MEASURED basis (all INHERITED from the DSCALE10K chain logs, not re-read on the pod):
* Stage A UE_s0, 61,568 steps: trainer time 13,804 s (`runs/UE_s0/A/summary.json` `seconds`, quoted in `reps1/build/HANDOFF_REPS1.md:21`) = **0.2242 s/step**; chain wall 3.91 h (FACTS B1, `block3.log:10-11`); UE_s1 3.29 h (B3).
* Union stage B incl. bank, U_s1 (same 985,048 TRAIN windows and the same `U_s*.B.tmpl` recipe): A done 17:56:47 to B done 20:03:11 = **7,584 s = 2.11 h**: bank (4 shards) **6,603 s**; selfcheck + mint + launch 135 s (19:46:50 to 19:49:05); stage B 3,000 steps **846 s = 0.282 s/step** (FACTS B5, `block2.log:21-27`). U_s0: bank 6,567 s, stage B 820 s (B6). Bank pilot, 200 clips, 4 shards: 111 s (`reps1/build/pod_copy/assemble_evidence_reps1_ue.py:40`).

Extra cost of ROLLHIST per step: **none in training** (the swap, `--w-acc` and p = 0.1 are already in the 846 s). The only extra is in the bank: readout C\* and the heading bound are numpy on the 2 committed plans per window (20 frames each). ESTIMATED overhead factor f in [1.00, 1.10] on the bank wall; the pilot measures f before the full bank.

Arithmetic (ESTIMATED from the MEASURED walls above):
* Pilot (RH config + pinned identity config, 200 clips each): 2 x 111 s = 222 s = 0.06 h.
* RH_s0 = bank 6,603 x f + 135 + 846 = 7,584 to 8,244 s, + pilot 222 s = **2.17 to 2.35 pod-h**.
* B0_s0 = 135 + 846 = 981 s = **0.27 pod-h** (no bank; upper bound, a p = 0 step is not slower).
* **First arm (RH_s0 + B0_s0) = 2.44 to 2.62 pod-h, quoted as about 2.6 pod-h (upper).** Below the 6 pod-h line, so no separate size flag; the size is still the PI's (section 6).
* Replicate (RH_s1 + B0_s1, no pilot) = 7,584 to 8,244 + 981 s = **2.38 to 2.56 pod-h**.
* **Both seeds: 4.82 to 5.18 pod-h, about 5.2 pod-h (upper).**
* Cross-check: 0.282 s/step at stage B against 0.2242 s/step at stage A (+26 %, the swap and the extra loss terms); the bank, not the training, is 87 % of an RH arm (6,603 / 7,584).
* After-check reads (not LoRA-pod GPU if run on the eval pod): 6 rollouts per checkpoint x 4 to 5 min (D16 section 4, on the A40, INHERITED) = 24 to 30 min; 2 checkpoints per seed = 0.8 to 1.0 h per seed; plus control c-EXEC, one rollout (~5 min). The UE.A references and the one-seed floor come from cards ADD67 / ADD68 (no new rollout). Readers on CPU, minutes.
* Disk: the size of a union bank is not known to this agent (Q4); the stage-B gate wants 60 GB free on `/root` (FACTS A8).

## 6. Decisions for the PI

1. **Size.** First arm RH_s0 + B0_s0, about 2.6 pod-h (ESTIMATED upper); with the seed-1 replicate RH_s1 + B0_s1, about 5.2 pod-h for both. Recommendation: both seeds in one go: D16's basis is two seeds, and one seed can only give a PROVISIONAL name. Eval reads beside: about 1 to 1.2 eval-pod-h per seed.
2. **Queue order on the LoRA pod** against the frozen STAGEB arms (UE_s0.B first, ~10 pod-h for the deciding set) and the other queued jobs. If UE_s0.B runs before RH, the beside contrast RH - UE.B (does the readout of the bank matter) costs nothing extra.
3. **The readout of the training history**: C\* (all own acceleration AND the executed path: what a car measures; recommended) or ADD63's `causal` (all own acceleration, pinned path, D16's wording "what a car would measure"). It fixes which handover the trained model is deployed with, so it touches recommendation A (the candidate's specification).

Not the PI's, for the MM: the G-T0 guard value; where the G-DVB entries live (section 4.1); whether RH and B0 run from one new commit (recommended) while the STAGEB arms stay on `cdbff971`.

## 7. Open questions (not settled from the code)

* Q1 SETTLED from source (`rzgfh.py:363-380`, section 1.2). Left open: whether the training bank should import the bound from `v8f_t1_eval_hb.py` (tier of record) or from the readout module; the c-EXEC control (section 3.1) and R4 cover either.
* Q2 COVERAGE under readout C\*: a TRAIN window (ci, t) is eligible when at least one own state of any tick has its matched route point r' == t and passed every cut (`v8f_dagger.py:312-318`, `:645-651`). The stage-A model drifts more under the causal readout (D2 above), so more off-route cuts (lateral > 3.0 m, heading > 30 deg) are likely (the M2b banks had 53.9 % usable tick-2 states, row `H-ST-RECOVERYLABEL-1`). Not computable from the code; the pilot (a PARTIAL bank, which K-B3 refuses for training, `v8f_dagger.py:328-330`, so it can only measure) gives it before the full bank.
* Q3 Whether `--dagger-p 0` with the union data flags gives a B0 that is otherwise the identical computation (same RNG stream for batches): read `v8f_mini_train.py` loop around `:2690-2710` and `DaggerSwap._seed_of`.
* Q4 Size on disk of a union bank: `ls -la` of the U_s1 bank directory on the pod (`/root/dscale10k/...`), read-only.
* Q5 The ADD66 reader pools two seeds per drawn episode; a one-seed read needs a per-seed mode: read `integ10k/strat_roll_read.py` arguments.
* Q6 SETTLED: ADD68 runs `R0C10Hb20` and `R0C10XHb20` at both model seeds and reuses ADD67's `R0Hb20`, `R0KHb20` (`PREREG_V8F_PROOF_ADD68.md:26-37`). Left open: whether ADD68 has RUN (its result row) before this card's read; if not, its four stage-A runs per seed are a prerequisite, not a new cost of this card.
* Q7 A bank over a sample of windows would cut most of the 1.8 h (about 1 % of the bank's records are drawn in 3,000 steps at p = 0.1), but COVERAGE and K-B3 (window-list sha256) refuse a partial bank; changing that is a PREREG amendment of `PREREG_V8F_DAGGER`, not this card.

## 8. Register row (proposal, not written)

`H-V8F-ROLLHIST-1` (DRAFT 2026-11-20): stage B of the 10k-union recipe with its DAgger bank generated under the T1-kin executor of record (hb0.2) and readout C\* repairs the handover defect of rows 119 and 120 at the source: RH - B0 at `T1-kin/hb0.2` per section 3, two seeds, outcomes of section 3.4. Status: DRAFTED, NOT FROZEN, NOT BUILT, NOT RUN. Size back to the PI.

## 9. Inputs to fold in at freeze (MM, 2026-11-20 ~13:00Z; the card stays a DRAFT until the PI answers on size)

* **Code of record (MEASURED):** commit `2bda9e1a382986cfdce3bb455822678b42a816e2` on branch `agent/v8f-rollhist-20261120`, a child of `26bdcd3d` (stack/ and taniteval/ byte-identical to the build base `7b26a8ac`).
  * The RH argv spelling is `--dagger-bank-readout causal_g1x --dagger-bank-kappa 0.2`. `readout_token` in `tanitad/data/v8f_dagger.py` is the one spelling, and section 2's arm table must use it.
  * A default bank carries no `generator` record, and its npz and `bank.json` are byte-identical to the unpatched generator's.
* **Coordinator rows 129-131 (INHERITED; T1-kin/hb0.2, two line-model seeds, no training):**
  * At 6 s of own driving the tactical track error splits into a pose part, 7.29 m from the carried pose, and a state part, 7.33 m from the own state. Speed, +1.54 m/s, comes from the state alone. Training on own rolled history reaches the state part only.
  * The state part grows about 1.14x from 6 s to 12 s, so C5 at 6 s also covers 12 s.
  * The C1 reference at age is track +23.9 m (the own-acceleration term against readout K).
* **Coordinator rows 132-133 (INHERITED; same tier and seeds):** the state part enters at the single hand-over tick.
  * Handing over the logged speed group (speed, acceleration, speed bucket) removes 0.84 of the state part of the tactical track error at 6 s (6.284 of 7.468 m) and 0.89 at 12 s (8.690 of 9.812 m).
  * The path group (pose history, yaw rate) removes 0.74 of the heading part at both ages, but only 0.11 / 0.09 of the track part.
  * In T1-kin the frames are logged, so the tokens show the logged car while the state says the own motion.
  * ⇒ The own speed group at the hand-over is the first thing the bank's histories must carry; pose history and yaw rate come second.
  * Proposed for section 3: a NON-deciding diagnostic, the RH - B0 change attributed to the speed group against the path group with the coordinator's hand-over swap tool.
* **Check before freeze (MEASURED from source, 2bda9e1a):**
  * The swap REBUILDS `p10f, base, kin3, p10_t, pose, pose_t, pose_s, nav` and KEEPS `feats, frame_ok, limit, limit_ok, a_t, a_s, pose_now` (`v8f_dagger.py:10-11`). Speed and acceleration (`kin3`, `p10f`) and `base = kdx_base(kin3)` (`:560`) are rebuilt.
  * Settle whether the coordinator's "speed bucket" is `base` (rebuilt) or the KEPT `limit` (the clip's logged bucket). If it is `limit`, the bank cannot carry that third member, and the card must say so.
* **Reader of record for pre-rolled tracks:** `integ10k/read_longroll10k_n4.py`. T1 readers prepend the origin, so add a planted-offset check that ROLLHIST tracks start at the frame origin before any read.
* **Compute:** the coordinator offers its A40 for the s1 pair (RH_s1 and B0_s1 on one device) under our gate token. Taking it changes no cost line on the LoRA pod.
* **SETTLED (MM, 2026-11-20 ~20:15Z, MEASURED from source): the coordinator's "speed bucket" is the bucket's INPUT, not `base`.**
  * `integ10k/make_handover_src.py` at d1dece12, docstring of `speed_logged`: "the speed group from the logged cache (v, a = dv/dt, the speed-bucket input d["v"], pose column 3)". At T1-kin the bucket is computed causally from `d["v"]` (`ROLLOUT["limit"] = "causal"`, `stack/tanitad/data/v8f_dagger.py:46` at 2bda9e1a).
  * The bank's swap KEEPS `limit` ("the clip's logged bucket", `v8f_dagger.py:10`, `EGO_PERTURB_KEPT`) and rebuilds `kin3`, `p10f` and `base`.
  * ⇒ An RH swapped sample carries the own speed and acceleration, but the clip's logged speed bucket, not the bucket its own history would give. **The bank carries two of the three members of the coordinator's speed group.** The 0.84 share (rows 132-133) is for all three together, so it is an upper bound on what RH can reach through the speed group.
  * Section 3's diagnostic must print this, and section 1.2 must name the KEPT bucket as a known limit of the lever. Rebuilding `limit` from the own `d["v"]` in the bank would be a second lever, a new card, not this one.
