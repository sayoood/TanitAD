# PREREG_V8F_DSCALE10K (DRAFT 2026-11-16, NOT FROZEN, NOTHING RUN): the v8f-mini recipe of record on 4,512 + 10,000 clips

Status: drafted by an Architecture agent (Opus) for the MM. Not frozen. No arm trained, no label file merged, nothing launched or landed. The card is committed BEFORE any training data of these arms exists; the MM freezes it (sha256) after the section 0 rulings.
Evidence classes: MEASURED (M, artifact named), PUBLISHED (P), INHERITED (I, another agent or doc, not re-verified here), ESTIMATED (E), HYPOTHESIS (H), UNVERIFIED (U).
Repo reads: mirror `C:/Users/Admin/tanitad-push`, commit `9ef0c72a` unless a commit is named. `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini`; `DS` = `PK/raw/2026-11-15-v8f-datascale`.

## 0. Escalations for the MM (found while drafting; each needs a ruling before freeze)

1. **The baseline is 4,512 clips, not "v9 4,369 clips".** The recipe of record's base run M2b trained on `--v9-train-md5 83e30bd3...` (the 4,572-clip release) with `--exclude 540:600`. Its summary reads `n_clips 4512`, `n_windows 306566`, `S 4791`, `steps 19164`, `train_sha256 37f4f9ee...` (M, `PK/raw/full/runs/M2b_s0/argv.json`, `summary.json`). The OBD-CTL fine-tune uses the same data flags (M, `PK/raw/2026-11-10-v8f-obd/argvs_obd.txt` line 3). The 4,369-clip release `f63ece41` is only the trainer's argparse default. This card uses 4,512.
2. **`ACCEPTED_TRAIN_MD5` has no reader.** `git grep ACCEPTED_TRAIN_MD5` over `stack/` finds only its definition at `v8f_mini_data.py:67`. This holds at `9ef0c72a`, at `cdbff971` and at `1027e0d7` (M). The second probe, the count of `ACCEPTED` in `v8f_mini_train.py`, reads 0. The control, `v9_train_md5` in the same file, reads 4 (M).
   * The v9 guard that does run is the argv's own `--v9-train-md5`. It is checked through `load_v9_release(expect_md5=...)` (`v9_labels.py`) and `declared_vs_built` (`v8f_mini_train.py:1049` at `9ef0c72a`).
   * Only the v10 dict is enforced (`v8f_mini_data.py:84` at `cdbff971`).
   * So `GEO10K/LABEL_INPUTS.md` section 4, *"the trainer refuses it until the Master Mind registers that md5"*, does not describe the code. Adding the md5 to the dict alone would be a registration nothing reads, which is the "built, tested, unreachable" class.
   * Section 5 proposes the dict addition together with a checker-side read.
3. **`--exclude 540:600` is positional.** It drops clips 540..599 of the TRAIN manifest's sha12 order (`v8f_mini_train.py` `--exclude` help). On a 14,572-clip manifest the same flag would drop 60 different clips. The union arms therefore drop the 60 RA-1 holdout clips **by id** when the union manifest is built, and run without `--exclude`. The checker refuses `--exclude` on any union argv (deliberate-regression arm G-R3, section 5).
4. **Base commit `cdbff971`** (OBD-CTL's code of record, as STRAT1 decision 2). The card needs **no trainer, model or library change**. Its new code is a study checker and a label-merge tool, both in the study folder.
5. **The sampler chunk is a derived constant.** `lru = max(8, int(1.5e9 // (n_max * 40 * 1024 * 2)))` and `chunk = max(4, min(48, lru - 2))` (`v8f_mini_train.py:2402`, `:2513` at `9ef0c72a`). A longer clip in the 10k set could shrink the chunk and change batch composition: a second lever riding on the first. The checker asserts `chunk == 48` on every arm.
6. **Gate route**: a study checker `dscale10k_check.py` that mints `dscale10k-gate-token-v1`, in the pattern of `strat1_check.py` / `obd_check.py`. `launch_gate.py` has no v8f-mini profile: its `v8f` profile is refcv8 rung alpha (M, `launch_gate.py:881-895`; I, STRAT1 `BUILD_SPEC.md` section 8).

## 1. Hypothesis and register row

**Question.** v8f-mini has been read as **not data-limited** in the 0.5x to 1x range. That read is one seed, on T's latcons RMSE: r +6.25 %, not separated (I, register `H-V8F-DSCALE-1`, `PK/raw/2026-10-25-dscale/run/dscale_read.json`). Does adding 10,000 new NVIDIA train-split clips, which are stratified over 25 countries, improve the recipe of record's driving on eval-141 beyond the training-run floor? The arms move one lever only: unique clips, at fixed steps.

**Prediction, written so it can be wrong (H).**
* T's latcons RMSE falls by 4 to 10 %.
* Most headline driving cells stay inside the floor (DATA-NEUTRAL or DATA-HELPS-PARTIAL).
* Reason: WC's dose arithmetic needs T to move about 25 % toward the GT code to buy about 0.19 m of ADE6 (I, `PREREG_V8F_DSCALE.md`).
* The log-linear reading of the half-data point gives about −6 % per doubling. That point is admissible only up to 2x data. The 3.2x step here is beyond the 2x extrapolation limit, so the −10 % end is HYPOTHESIS, not ESTIMATED.

**Register row (proposed text for `GOALS_AND_CLAIMS.md`):**

| id | claim | status | next |
|---|---|---|---|
| **H-V8F-DSCALE10K-1** (2026-11-16, DRAFT) | v8f-mini is DATA-LIMITED above its corpus. The arm trains the recipe of record (M2b base, 19,164 steps, then the OBD-CTL fine-tune, 3,000 steps) on the union of its 4,512 clips and the 10,000 train-split clips of `physicalai-datascale-v1-n10000-s20261115` (14,512 clips). ONE lever: unique clips at matched steps. The new clips carry geometry labels; their VLM goal bits are masked. Claim: T1-kin driving on eval-141 improves beyond the training-run floor F in at least 2 of the 4 metric families at both seeds, with no family costing. Mechanism read beside: T's latcons RMSE r ≤ −5 %, separated. This is the 3.2x v8f-mini instance of `H-DATA-SCALE-1`. | **PRE-REGISTERED (DRAFT), NOT RUN.** Card `DS/card/CARD_DATASCALE_TRAIN_DRAFT.md`. Inputs MEASURED: 10k cache banked (`DS/run10k/out/CHECK_10K.json`, I); GEO10K labels built (I); v9 md5 `395221440e3a3c20f17dd381297af4f0`. Predicted (H): T −4 to −10 %, driving mostly inside F. | By outcome, section 6. A NULL moves the lever to the encoder/adapter and to tactical label coverage. It never ends the stream. |

## 2. Arms

### 2.1 Recipe of record, two stages (both run per arm)
* **Stage A (base):** M2b from scratch, the `M2b_s0` argv mapped to pod paths. `--arm M2b --grid pool4x10 --kin-mode yaw --batch 64 --steps-mult 4`, P10 `p10_prior.pt` frozen, the same for every arm (sha256 bound).
* **Stage B (fine-tune):** OBD-CTL (`argvs_obd.txt` line 3): DAG-OACC = `--init-ckpt <that arm's stage-A ckpt_last.pt> --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p 0 --dagger-p 0.1 --w-acc 49.6969 --detach-upper`.
  * Its DAgger bank is built by `v8f_dagger_bank.py` from that arm's own stage-A checkpoint, over that arm's own train windows. The trainer's K-B2 (on-policy against `--init-ckpt`) and K-B3 (TRAIN-only, window-list sha256) require this (`v8f_mini_train.py:213-218` at `cdbff971`, M).
  * COVERAGE must be ≥ 0.5, or the arm is BLOCKED (I, `PREREG_V8F_DAGGER.md` section 6).
* Stage B is where the deployable model comes from, so it carries the deciding read. Stage A is where the clip count enters the step budget, so it carries the mechanism read.

### 2.2 Clip sets
* **B set** = train4572 minus the 60 RA-1 holdout clips (`--exclude 540:600`, the recipe's own flag) = 4,512 clips, `train_sha256 37f4f9ee...` (M, M2b summary).
* **U set** = B set ∪ the 10,000 data-scale clips = **14,512 clips**.
  * It is built as a union view `/root/data/cache_union_dscale/train/`: `clips/` holds symlinks into `/root/data/cache_pod/train/clips/` and the 42 shard folders of `/root/data/cache_10k/` (shard layout U).
  * `manifest.jsonl` lists the 14,512 clips. The 60 holdout clips are absent by id.
  * `meta.json` is equal to the corpus cache's (both `ra1a-pooled-v1`, checked).
  * Overlap between the 10k set and the 4,572 corpus is 0 (I, GEO10K register row). The 10k list is `clips_10k.txt`, sha256 prefix `31aca42a` (I, `HANDOFF_DATASCALE10K.md`).
* **Labels for U.** Every per-clip input below is a merged file: the corpus file and the 10k file concatenated by the merge tool `dscale10k_merge.py` (section 5). The corpus side keeps the original bytes; control C-MERGE checks it.

| input | corpus file (pod copy, re-hashed) | 10k file (HF `labels/`, sha256 verified, I) |
|---|---|---|
| v9 release | `v9_labels_train.npz` md5 `83e30bd3...` | `v9_labels_datascale10k.npz` md5 `395221440e3a3c20f17dd381297af4f0` (MM, M), sha256 `74648b91...` |
| lead | `lead_train4572.npz` | `lead_datascale10k.npz` `5c13b204...` |
| speed-limit sidecar (an INPUT) | `refcv6_speed_max_v8_train.jsonl` | `speed_sidecar_datascale10k.jsonl` `2489c03d...` |
| native tracks | `native_tracks.pt` | `native_tracks_datascale10k.pt` `7eb96925...` |
| join | `join_train.npz` md5 `9ebc7a48...` (M via STRAT1) | built by the trainer's `--fit-join` from the merged release |

### 2.3 Arms, seeds, priority (LoRA pod, one job at a time)

Priority is an order, not a dependency chain. A chain stopped after arm 4 still yields a seed-0 read with its floor and its reach control.

| prio | arm | clips | stage A steps | stage B | seed (`--seed`; B: `--dagger-seed`) | role |
|---|---|---|---|---|---|---|
| 1 | **B_s0** | B set, 4,512 | 19,164 (`--steps-mult 4`) | yes | 0 / 0 | control, same venue |
| 2 | **U_s0** | U set, 14,512 | **19,164** (`--steps 19164`) | yes | 0 / 0 | the lever, matched steps |
| 3 | **B_rep** | B set | 19,164 | yes | **2** / 0 | training-run floor F (both stages) |
| 4 | **R_s0** | U set, 10k labels deranged (2.4) | 19,164 | no | 0 | deliberate regression: the new clips' labels must reach the model |
| 5 | **B_s1** | B set | 19,164 | yes | 1 / 1 | second seed |
| 6 | **U_s1** | U set | 19,164 | yes | 1 / 1 | second seed |
| 7 | **UE_s0** | U set | 4 union epochs = 4 x S_U ≈ 61,600 (E; `--steps-mult 4`, S_U MEASURED at the gate) | no | 0 | data plus compute |
| 8 | **BL_s0** | B set | the same total as UE_s0 (`--steps`) | no | 0 | compute at fixed data (`H-DATA-SCALE-1`'s other half) |
| cond. | X_s0 | the 10k set only | 19,164 | yes | 0 | runs only on DATA-HURTS or MIXED (section 6) |
| cond. | B_s3, U_s3, B_s4, U_s4 | | 19,164 | yes | 3, 4 | runs only on SEEDS-DISAGREE |

* **Replicate seed.** B_rep uses seed 2, not seed 0. A same-seed replicate can be bit-identical, which would read the floor as 0.0 (memory note: wm-only box-2 training is deterministic at a fixed seed; a DAgger run is not). B_rep runs both stages, so F covers the whole recipe.
* **Cross-venue arms, beside only.** The banked box-2 `M2b_s{0,1}` and `OBD-CTL_s{0,1}` ran on a different GPU. They are read beside B_s0 / B_s1 as cross-venue replicates and are never the control (STRAT1 convention).

**Matched steps (primary) vs matched epochs (beside): the choice.**
* U_s0 vs B_s0 at 19,164 steps moves ONE lever, unique clips, with compute, LR schedule and `p_gt` anneal identical.
  * It is `H-DATA-SCALE-1`'s own definition ("unique data at fixed steps"), and `PREREG_V8F_DSCALE` used the same form.
  * U sees each window about 1.24 times (19,164 x 64 / ≈ 985,600 windows, E) against B's 4.0.
* At matched epochs, U would get about 3.2x the compute. A gain there could not be attributed to data.
* The matched-epoch question is still answered, without the confound: UE_s0 (4 union epochs) is read against **BL_s0** (B set at the same step count). UE − BL is data at the larger fixed compute, and BL − B is compute at fixed data.
* Stage B is 3,000 steps for every arm (matched steps, under one epoch for every set).

**Init control (C-INIT).** The checker records the step-0 state-dict sha256 of B_s0 and U_s0. Same seed and same config predict the same init (U, since data building could consume the global RNG first). If the two differ, Δ is read as data plus init and the report says so.

### 2.4 Masks for labels missing on new clips (no imputation), and the reach of every loss term

**What the 10k clips lack (I, GEO10K `LABEL_INPUTS.md` sections 2 and 4; MEASURED there on 20 control clips):**

| label / input | 10k status | treatment in U | head(s) affected |
|---|---|---|---|
| VLM-CoT goal bits (TRAFFIC_LIGHT_*, stop sign, pedestrian; provenance `vlm-cot`) | 0 of 10,000 clips | **masked**: `goal_w` bit 0 on every 10k row, so the bit carries weight 0 in the goal loss (`build_label_join`, `v8f_mini_data.py` goal_y / goal_w, M at `cdbff971`). The geometry goal bits of the 10k rows train. | T goal set: VLM classes learn from the 4,512 corpus clips only |
| lane-change crossing cells `lc_t_cross_s`, `lc_d_cross_m` | builder's no-crossing value NaN (`g3d_fill.py`) | NaN = undefined, so the cell is masked in the constraint loss (`CONS_FIELDS`, "NaN = undefined") | T constraints: lc cells learn from the corpus clips only. The corpus release also holds the no-crossing value on 99.07 % of rows (I). |
| lane-change classes | never positive in both releases (I) | unchanged | none |
| nav (the v9 ego-future token, the recipe's INPUT) | built from egomotion geometry for every v9 row (I, GEO10K) | rows with no v9 row get nav zero and labels IGNORE, the corpus rule (`build_label_join`: `nav[~inside] = 0`, labels `IGNORE`) | all; C-NAVCOV reports the has-row fraction per set |
| NVIDIA nav text | 9,980 / 10,000 clips (I, NAVPROBE) | **not an input** in this card. It is used only to score S_CKPT on eval-141 (eval side, 140 / 141, I). | none in training |
| lead rows | 441 / 10,000 clips without lead rows (I) | the builder's own semantics, the same as for corpus clips without an obstacle join: `lead_gap` NaN, `lead_valid` from the release | lon classes and lead constraint cells. C-LEADCOV reports both fractions; a recipe change for both sets would be a second lever and is not made here. |
| image-derived context (weather, surface, ...) | not built | not read by the recipe of record (no ctx head) | none |

* The NVIDIA `ood_reasoning` event text that exists for some pool clips is NOT used. It is a different label provenance, so using it would add a second lever.

**C-MASK (data check, before the token).** The rule is that a VLM-class bit with `goal_w = 1` and `goal_y = 0` on a 10k row would be an **imputed negative**.
* Expected: the count of 10k rows with a non-zero `goal_w` on any VLM-class bit is **exactly 0**.
* Control, which must read non-zero: the corpus rows with VLM-class `goal_w` set. It reaches at least the release's TRAFFIC_LIGHT positives (376 + 363 + 22 + 18 clips, I, CLAUDE.md v7.2 census).
* A non-zero 10k count is a launch blocker.
* UNVERIFIED today. LABEL_INPUTS states it, but no count of it is banked.

**Loss-term reach table (G-LIVE extension; committed expected values).** Batches are built from the union join three ways: corpus-only, 10k-only and mixed (the sampler's natural mix, about 31 / 69 %). Every declared term of the stage-A and stage-B argv must be present and finite on each batch kind. Per term on a **10k-only** batch:

| term (weight in argv) | 10k-only batch | how it is proven |
|---|---|---|
| `ckpt` (S vs RC-A, `w_ckpt 1.0`) | live, grad > 0 on S | RC-A is geometry |
| `tac` lat / lon classes (`w_tac 1.0`) | live | geometry classes |
| goal bits, geometry classes | live, grad > 0 on their logits | |
| goal bits, VLM classes | **grad exactly 0.0** on their logits; the control (a corpus batch holding a TL-positive row) reads > 0 | mask |
| `cons`, lat and lon cells (`w_cons 0.5`) | live | |
| `cons`, lc cells | exactly 0.0 contribution (NaN-masked) | mask |
| `traj` (O), `pen`, `pen_self`, `wm` | live | cache kinematics and tracks |
| `acc` (stage B, `w_acc 49.6969`) and the DAgger swap (stage B) | live; DAgger eligible fraction reported per set | |

* **Deliberate-regression arm of this check:** a release copy with `goal_w` forced to 0 on every row must make the goal term read dead (exactly 0) and FAIL the check.
* A term that is live on mixed batches and dead on 10k-only batches, outside the two masked rows above, is a launch blocker.

**R_s0, the training-level regression arm.**
* U set. The 10k clips' v9 **label** columns are deranged at clip level: every 10k clip receives another 10k clip's rows, within groups of equal (`nrows`, `k0`), with seed `20261117`.
* The nav INPUT columns (`nav_token`, `nav_args_valid`, `nav_d_next_m`) stay the clip's own. The supervision then contradicts images, kinematics and nav, and cannot be read back through a nav echo.
* The 4,512 corpus rows are untouched (C-MERGE applies).
* Expected: T's latcons RMSE(R) > RMSE(U) with r ≥ +5 %, separated, at stage A.
* If R is not worse, the 10k labels do not reach T (or T ignores 69 % of its windows). U's read is then uninterpretable and the outcome is NO-REACH (section 6).

## 3. Parity and leakage

* **Eval grid untouched**: eval-141, 141 episodes / 2,399 windows (I, STRAT1 section 4).
  * Eval inputs (`join_eval141.npz`, eval labels, tracks, P10) must be byte-equal to those of the OBD-CTL read. The checker re-hashes them.
  * No eval path reads a train-side file or flag (static check, as DSCALE's "eval is untouched").
* **Corpora untouched**: `physicalai-train-e438721ae894` and the v8f-mini corpus are not re-selected or written into. The 10k set is a prefix of the frozen order (seed `20261115`, `SCOPE_DATASCALE.md` sha256 `b558ee81...`).
* **C-DISJ:** `assert_disjoint(U set, eval141)` by clip id AND by stable sid. **Expected 0 and 0.** The 10k set was built excluding `clips141.txt`, the v72 eval 147 digests, the deployed val40 digests, challenging_scenarios and ood_reasoning val (I, SCOPE section 2).
  * sha12 check against `stack/tanitad/data/*_clip_digests.json`: expected 0.
  * `guard_corpus_build(role="train", mode="refuse")` must keep all 14,512.
* **C-HOLD:** the 60 RA-1 holdout ids in the U set. Expected **0**.
* **C-SESS (U):** whether 10k clips share a recording session with eval-141 clips.
  * Eval-141 holds 92 NVIDIA default-train clips (I, SCOPE section 1), so this is possible in principle.
  * If `clip_index.parquet` or the clip metadata carries a session or drive key, count the 10k clips sharing one with an eval-141 clip. Expected 0. Any hit is dropped by id before freeze and listed.
  * If no such key exists, the card says "not computable" and keeps the residual risk in words.
* **C-CACHE:** every 10k `.pt` sha256 equals `corpus_manifest_10k.jsonl` (the MM MEASURED this on the pod on 2026-11-16; re-run by the checker). Every corpus `.pt` keeps the sha256 the B set was trained on.

## 4. Eval (every arm; reads on the dev-box CPU, dumps in one environment and one thread count)

**Tiers.**
* LONGITUDINAL, LATERAL and TACTICAL: **T1-kin** (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene), eval-141, per seed. This is the OBD / STRAT1 read (I, STRAT1 section 4.1). T1-kin is the PRIMARY tier.
* STRATEGIC route cells: **T0 at the window's first tick**, with the T1 tick-0 identity asserted (STRAT1 section 4).
* Beside, at **T0**: the GT-oracle and absmask evals and T's present-slot latcons RMSE (the DSCALE instrument, `v8f_dscale_read.py`). These are world-model and tactical diagnostics, never driving performance.

**Estimator and the variance question each number answers.**

| interval / quantity | answers |
|---|---|
| paired episode-cluster bootstrap (`taniteval.ci.paired_episode_cluster_bootstrap`, B 2,000, seed 0, 141 episodes), U − B per seed | another draw of EPISODES |
| F_cell = abs(B_rep − B_s0) per cell, per stage | another TRAINING RUN (one replicate pair: a weak floor; SEEDS-DISAGREE adds pairs) |
| sign agreement s0 / s1 | whether the effect survives a second init and data order |
| cross-venue: abs(B_s0 − box-2 OBD-CTL_s0) | venue plus nondeterminism, beside only |
| another INFERENCE RUN | not applicable: the planner is deterministic. Thread drift is held fixed by one dump environment (memory note) |

**Families (every cell, per band: operative knots 1-10 = 0-2 s, tactical knots 11-30 = 2-6 s; per seed; never pooled).** ADE and FDE are reported beside and decide nothing alone.

| family | cells reported | **headline cells (decide the label)** |
|---|---|---|
| LONGITUDINAL | speed MAE, acceleration MAE, along-track MAE, headway abs error, time-gap abs error, DK-VIOL | L1 op speed MAE; L2 op headway abs error (distance keeping) |
| LATERAL | heading, curvature, yaw-rate, cross-track MAE | A1 op yaw-rate MAE; A2 op curvature MAE |
| TACTICAL | TAC-LAT and TAC-LON accuracy; beside: T-LAT-F1, T-LON-F1, goal macro-F1 split into VLM-class and geometry-class bits, nav-echo split (STRAT1 4.3) | T1 TAC-LAT acc; T2 TAC-LON acc |
| STRATEGIC | S checkpoint position and heading error vs RC-A at 30/50/80 m; S_CKPT (route following vs the NVIDIA text) at 80 m under real / donor / zero nav, skill = real − donor, paired; best constant 0.794 at 80 m (I) | S1 checkpoint position error at 80 m; S2 S_CKPT skill at 80 m (n 214 windows / 24 episodes, I) |

* **STRATEGIC members not in this card, reported per member with the reason:**
  * detected information: **n/a**, the recipe of record has no ctx head ("no output");
  * derived goals: not in this arm;
  * strategic intent 6-30 s: **not computable**, no output beyond 80 m of arc.
* The C0-type v9 nav shares its source with RC-A, so the S1 cell is echo-prone for every arm alike. Δ (U − B) is read, never the level.

**Cell rule (OBD / STRAT1 form), per seed:**
* **gain**: Δ = U − B is separated on the better side and abs(Δ) > F_cell;
* **cost**: separated on the worse side and abs(Δ) > F_cell.
* **Family gain** at a seed: at least one headline gain in that family and no headline cost there.
* **Family cost** at a seed: at least one headline cost.

**Mechanism read (stage A, T0).** r = RMSE_latcons(U) / RMSE_latcons(B) − 1, with the paired CI on the per-window MSE and DSCALE Amendment 1's sign rule (a separated MSE Δ whose sign contradicts r counts as not separated).
* T-DATA-LIMITED: r ≤ −5 %, separated, and abs(Δ) > the stage-A T floor.
* Why 5 %: about 20x the measured T seed floor of 0.2 % (I, DSCALE).

**`H-DATA-SCALE-1` sub-read (stage A, block 3).** Δ_data = U_s0 − B_s0 (fixed steps), Δ_ep = BL_s0 − B_s0 (fixed data, about 3.2x steps), and UE_s0 − BL_s0.
* Read on ADE6 against 2F_A (F_A = the stage-A ADE6 floor from B_rep), with T's latcons beside.
* Labels: DATA > EPOCHS (U − BL better by more than 2F_A); EPOCHS > DATA (the reverse); NEITHER.

## 5. Launch-gate plan

**Gate route.** `dscale10k_check.py` (study folder) mints `dscale10k-gate-token-v1`. The token binds:
* the commit (`cdbff971`, or its child if the MM adds the checker there), the argv sha256 and the tree manifest;
* the trainer and selfcheck sha256;
* every data sha256 / md5: both v9 releases, the merged release, lead, speed, tracks, join, P10, the union manifest, the R release, and per stage-B arm its DAgger bank `bank.json`.

The runner exits 2 without a matching token. All seven G-checks apply:

| check | in this card |
|---|---|
| G-HYG | no new config field; `V8FMiniConfig(**cfg)` refuses unknown keys. The checker asserts every arm's built cfg and trainable parameter count equal B_s0's (M2b: 7,214,495, M summary). |
| G-DVB | the trainer's `declared_vs_built` (`data_frac -> data:n_clips`, `v9_train_md5 -> data:v9_md5`, `steps -> run`). The checker adds: built `n_clips` (4,512 / 14,512 / 10,000), `train_sha256` (B: `37f4f9ee...`; U: MEASURED by a dry `build_data` before freeze, as DSCALE Amendment 1), `n_windows`, total steps, `chunk == 48`, and that `--v9-train-md5` is in `ACCEPTED_TRAIN_MD5`. **Argv refusal (rc 2):** U's argv may differ from B's only in `--cache-dir --v9-train --v9-train-md5 --lead-train --speed-train --tracks --join-train/--fit-join --out`, the removal of `--exclude`, and the addition of `--steps 19164`. Any other difference is refused. |
| G-LIVE | real-data smoke on the union (200 steps): every declared term finite on all three batch kinds, every trainable group with a gradient, the reach table of 2.4 with its zero controls and its regression arm. Stage B adds the inherited `acc_probe` and `dagger_probe`. |
| G-CLOCK | label time. The 10k clip clock comes from `build_clip_clock_sidecar.fit_clock` (19 / 20 controls bitwise, 364 clips on the builder's fallback, I). The checker compares the per-clip residual of v9 `now_s` against the cache frame time for the 10k set and the corpus set. Bar: the 10k p99 is within the corpus p99. Regression arm: a +0.2 s shift of the 10k clock must FAIL. |
| G-EVAL | the eval loader builds the identical model (strict load, cfg equal); eval inputs byte-equal to the OBD-CTL read's; 2,399 windows, 141 episodes |
| G-CKPT | save / load bit-equal state dict and cfg; the resume test (`V8F_TRAIN_CRASH_AT` hook) on a U-set smoke |
| G-SUITE | `pytest stack/tests` on a clean export of the build commit, in the pod environment, **0 regressions** against `cdbff971` run the same way. The checker's own tests run beside: each data check below with a mutation that must turn it RED. |

**Data checks (each with the value it must read, and its deliberate-regression arm):**

| check | must read | regression arm (must FAIL) |
|---|---|---|
| C-MERGE | corpus rows of the merged release bitwise equal to `83e30bd3`'s; 10k rows bitwise equal to `39522144`'s; sids disjoint; the same for lead, speed and tracks | flip one corpus row byte |
| C-JOIN-EQ | every array of the union join restricted to the 4,512 corpus clips equals `join_train.npz` (md5 `9ebc7a48...`) restricted to them: 0 differing elements | perturb one corpus `nav` row in the merged release |
| C-DISJ, C-HOLD, C-SESS, C-CACHE | 0, 0, 0 (or "not computable"), all equal | put one eval-141 clip id into a test manifest (G-R1) |
| C-MASK | 0 VLM-class `goal_w` bits on 10k rows; corpus control > 0 | set one such bit (G-R2) |
| C-EXCL | no `--exclude` on a union argv | add `--exclude 540:600` to U's argv (G-R3) |
| C-NAVCOV, C-LEADCOV | reported per set (corpus has-row fraction 456,946 / 461,518 steps, I, NAVTOK) | none (report) |
| C-INIT | B_s0 and U_s0 step-0 sha256 recorded (expected equal, U) | none (report) |
| C-STEPS | stage A total 19,164 (U, B, R); UE = BL = 4 x S_U | `--steps 19163` |

**Proposed change to `stack/tanitad/data/v8f_mini_data.py` (NOT applied; the MM lands it with the checker):**

```diff
 ACCEPTED_TRAIN_MD5 = {"f63ece410b725febb8a5242cf2b01d3c": "v9 train, 4,369 clips",
-                      "83e30bd3791f0767707eade5f4671759": "v9 train, 4,572 clips (superset; 4,369 bitwise identical)"}
+                      "83e30bd3791f0767707eade5f4671759": "v9 train, 4,572 clips (superset; 4,369 bitwise identical)",
+                      "395221440e3a3c20f17dd381297af4f0": "v9 train, datascale-10k (GEO10K 2026-11-16): geometry only, VLM goal bits goal_w 0, lc cells no-crossing",
+                      "<UNION_MD5>": "v9 train, union 83e30bd3 + 39522144 (dscale10k_merge.py), 14,572 clips",
+                      "<R_MD5>": "v9 train, union with the 10k label columns deranged (seed 20261117): DELIBERATE-REGRESSION ARM ONLY"}
```

* `<UNION_MD5>` and `<R_MD5>` are written by the merge tool. The MM fills them in at freeze, so the card binds them before any arm.
* **Reachability.** The dict has no reader (escalation 2), so the checker reads it: an argv whose `--v9-train-md5` is not a key is refused.
  * Mutation: delete the union entry and the U argv must be refused.
  * A trainer-side read, mirroring the v10 helper at `v8f_mini_data.py:84`, would touch every test that builds a synthetic release. It is left to a backlog item with its own G-SUITE run.

**New trainer flags: none.** No G-DVB entry is added. U uses existing flags only (`--steps`, the data paths, `--fit-join`).

**Pilot (before the chain).**
* 600 steps of B_s0 and U_s0 on the pod. Each spans more than 10 sampler chunks of 48 clips.
* Measured: steady s/step (`pilot_summary.py` rule: cumulative `sec`, difference over steps), peak GPU, pod RSS against `memory.max`, and the read throughput of `/root/data`.
* One 200-clip DAgger-bank build from B_s0's pilot checkpoint, timed. This is the one term the cost table cannot estimate.
* **Stop rule:** if the projected block 1 exceeds 10 pod-h, STOP and go back to the MM / PI (STRAT1's 10 h rule).

**Post-training gate (each arm).**
* Every registered parameter has Adam state, read from the checkpoint's optimizer (the `probe_zerograd3` count). The count must equal B_s0's.
* Every non-frozen group moved.

## 6. Outcomes, committed before any number (each names what runs next)

Precondition, read first: **R_s0** against U_s0 at stage A.

| label | when | what runs next |
|---|---|---|
| **NO-REACH** | R_s0 not worse than U_s0 on T latcons by r ≥ +5 %, separated | U's read is not interpretable, and no driving label is given. Next: the G-LIVE reach table re-run on the trained U checkpoint, then a count of 10k windows per batch from the logged sampler, a fix, and a re-run of R and U. |
| **DATA-HELPS** | family gain in ≥ 2 families at BOTH seeds, no family cost at either seed | 1. The PI's rebase ruling: U as the base of record for STRAT1 / ACTRSP descendants (Q3). 2. The next rung, 2x the union (about 29k clips: +14.5k from the frozen order, inside the 2x extrapolation limit), encode plus storage go (Q4). 3. The VLM-CoT pass on the new clips as the tactical-coverage lever. |
| **DATA-HELPS-PARTIAL** | family gain in exactly 1 family at both seeds, no family cost | Item 2 of DATA-HELPS for that family. For each family without a gain, its named lever: LONGITUDINAL → LON-PLAN / HWM-PLAN (`AMEND_HWMPLAN_3`); LATERAL → adapter training (LAT-TOK's FAIL branch); TACTICAL → the VLM-CoT pass on the 10k; STRATEGIC → STRAT1's nav text on the U base. |
| **DATA-NEUTRAL** | no family gain at both seeds, no family cost, reach passed | Corpus size is not the binding term at 3.2x for this recipe. Next: if UE − BL is better by more than 2F_A, the lever is compute on the union (UE through stage B, both seeds); else LoRA on DINOv3 on the U set (`PREREG_V8F_LORA.md`, the committed next lever of DSCALE), plus the VLM-CoT pass. |
| **DATA-HURTS** | family cost in ≥ 1 family at both seeds | Diagnose dilution. Run X_s0 (10k only) and the per-goal-class F1 (VLM vs geometry bits). Read eval-141 per country and platform. A mixture arm (corpus clips drawn at B's rate) needs a sampler-weight flag, so it goes in an amendment with its G-DVB entry. |
| **MIXED** | a family gain and a family cost, both at both seeds | The trade-off is reported per family, never pooled. Next: the mixture arm (amendment) and X_s0. |
| **SEEDS-DISAGREE** | a family gain or cost at one seed only, not covered above | Seeds 3 and 4 for B and U (about 2.6 pod-h, E). Re-read with F from the replicate pairs; the label is set then. |
| **BLOCKED** | a gate FAIL, COVERAGE < 0.5 for a bank, or the pilot's projection over the stop | Back to the MM with the failing check. The arm is not launched. |

* **Mechanism label (stage A, beside, steers the next lever):** T-DATA-LIMITED (r ≤ −5 %, separated) or not.
  * T-DATA-LIMITED together with DATA-NEUTRAL driving says the T gain does not reach the planner. The next lever is then the T → O interface (WC dose arithmetic), not more data.
* **`H-DATA-SCALE-1` label** (block 3): DATA > EPOCHS / EPOCHS > DATA / NEITHER. It updates that register row. A label set on one seed is reported as one seed.

## 7. Cost (pod = LoRA pod)

**Venue facts.**
* GPU: "NVIDIA RTX 4000 Ada Generation", 20,475 MiB, idle (M, the MM's `nvidia-smi` on the pod, 2026-11-16 about 19:20Z).
* Disk: 500 GB overlay, 146 GB free (M, MM, same probe).
* Cache: the 10k cache is 195 GB on the pod (M, MM). HF reports 208.8 GB of cache tars (I, register row DATASCALE-10K). The two agree if one count is GiB and the other GB (E: 208.8e9 bytes = 194.5 GiB).

**Rates.**
* Fine-tune: 0.19 s/step, peak GPU 1.4 GB (M by the MM on the STRAT1 pilots, 2026-11-16; not banked at `9ef0c72a`, so I for this card).
* From-scratch stage A: ESTIMATED **0.21 s/step**, which is 0.19 x 1.1. The 1.1 is box 2's ratio of M2b from scratch, 0.821 s/step (15,741 s / 19,164 steps, M, M2b_s0 `summary.json`), to its policy fine-tune, 0.722-0.769 s/step (I, STRAT1 section 7).

| block | arms | pod GPU-h at I/O factor 1.0 (E) | worst case, I/O factor 2.5 on union arms (E) |
|---|---|---|---|
| 0 pilot + gate | 2 x 600 steps, G-LIVE smokes, 200-clip bank timing | ≈ 0.5 | ≈ 0.8 |
| 1 core | B_s0, U_s0, B_rep (A + B), R_s0 (A) | 4 x 1.12 + 3 x 0.16 = **4.96** + 3 bank builds (U) | ≈ 8.5 + banks |
| 2 seed 1 | B_s1, U_s1 (A + B) | 2.56 + 2 bank builds (U) | ≈ 4.5 + banks |
| 3 H-DATA-SCALE-1 | UE_s0, BL_s0 (A only, ≈ 61,600 steps each) | 7.2 | ≈ 12.6 |
| **total** | 8 arms | **≈ 15.2 + 5 DAgger-bank builds (U)** | ≈ 26 + banks |

* **Bank builds** are timed in the pilot. A union bank covers about 3.2x the windows of a B bank (E).
* **Reads** run on the dev-box CPU (no GPU), at OBD's per-arm pull bound of 1.5 GB or less, with F: at 25 GB or more (I, STRAT1 section 11). Read time is U.
* **I/O risk (flagged).**
  * `ClipStore.get` does a full `torch.load` of each clip file, pool8x20 included (`v8f_mini_data.py` ClipStore docstring, M).
  * Mean file size 20.88 MB (M, `PILOT_REPORT.json`), chunks of 48 clips, about 67.9 windows per clip (M: 306,566 / 4,512). That gives about 51 steps per chunk, about 19.7 MB read per step, and about **94 MB/s sustained at 0.21 s/step** (E).
  * The corpus cache (92 GB, I, register) may sit in page cache. The union (about 287 GB) cannot, since pod RAM and `memory.max` are U (the sister pod's cap is 50 GB). The volume read rate is U.
  * The pilot measures it. **Mitigation, used only if the U pilot is above 1.5x B's s/step:** a pool4x10-only copy of the 10k cache (about 4.1 MB per clip, about 41 GB, E). Its control: `ClipStore.get` arrays bitwise equal on every 10k clip. That cuts the reads about 5x.
* **Disk.** Merged labels about 1 GB (v9 10k file 562 MB, M via GEO10K `geo10k_files.json`). Union view: symlinks only. Run directories and banks U, recorded by the checker per arm. **Guard:** the chain pauses below 60 GB free.
* **Deleted afterwards** (literal paths, each logged with its size; after the read is banked by sha256):
  * per-arm intermediate checkpoints under `/root/dscale10k/runs/<arm>/` (keep `ckpt_last.pt` and the read checkpoint, banked);
  * the DAgger banks (keep each `bank.json` and its sha256);
  * the R release, re-creatable by the merge tool and seed;
  * the pool4x10 copy, if built;
  * the union view `/root/data/cache_union_dscale/`.
* **Kept:** the 10k cache, for the follow-on arms. Its source on HF is sha256-verified, so it is re-creatable and deletable later under the PI rule. The merged label files are banked to HF `labels/` with their manifests in the repo.
* **Not touched:** `/root/data/cache_pod`, `/root/data/v2ep`, `/root/q5_416`, and STRAT1's runs.
* **Scheduling** (the MM's): STRAT1's 7-arm chain (10 h or less) runs on the same GPU. One job at a time.

## 8. Known risks
* **One replicate pair** makes F weak. SEEDS-DISAGREE is the remedy, and the label waits for it.
* **Country and platform shift.** The 10k set is stratified over 25 countries (I, SCOPE), while eval-141 is drawn from the corpus' own mix. A union gain or cost may come from distribution shift rather than quantity. The per-country eval breakdown is reported beside.
* **VLM-class dilution.** VLM goal bits train on 31 % of the U windows against 100 % of B's. The goal macro-F1 split by bit provenance is reported beside.
* **Lead semantics.** The 10k has 4.41 % of clips without lead rows. The corpus fraction is measured by C-LEADCOV. The lead file has no known-value control (I, GEO10K).
* **Stage B bank coverage** on the union init is unknown. A BLOCKED arm is reported, never dropped.

## Log
* 2026-11-16: draft written (Architecture agent). Not frozen.
* **2026-11-16 ~19:35Z, Master Mind: FROZEN.** The text above is the agent's draft, unchanged, sha256 `6494dd8f…`. No arm of this card has trained, and no merged label file or union view exists. Rulings on section 0:
  1. **Baseline = 4,512 clips: ACCEPTED.** Re-verified by the MM in `PK/raw/full/runs/M2b_s0/`: `argv.json` carries `--v9-train-md5 83e30bd3…` and `--exclude 540:600`; `summary.json` reads n_clips 4512, n_windows 306566, steps 19164, train_sha256 `37f4f9ee…`.
  2. **`ACCEPTED_TRAIN_MD5` has no reader: CONFIRMED** by the MM (`git grep` over `stack/` at `9ef0c72a` finds only `v8f_mini_data.py:67`). **Ruling: no `stack/` change for this card.** Section 5's dict addition is replaced by a study-local accepted set.
     * `dscale10k_check.py` holds that set in the study folder, bound by this card's sha256: the corpus release `83e30bd3…`, the 10k release `395221440e3a3c20f17dd381297af4f0`, and the union release md5.
     * The union md5 is accepted only when C-MERGE passes: the merged file must equal `dscale10k_merge.py`'s output from the two pinned inputs, and the corpus side must keep its original bytes. The checker records the union md5 at the gate.
     * An argv whose `--v9-train-md5` is not in the set is refused, with a deliberate-regression arm.
     * Why: the union arms train on the MERGED file, whose md5 is neither of the registered ones. A `stack/` edit would also move the code-of-record pins of base `cdbff971`.
     * The unread dict is a register defect (`D-ACCEPTED-MD5-UNREAD`), fixed outside this card. GEO10K `LABEL_INPUTS.md` section 4 states a refusal the trainer does not make; that sentence is corrected with that fix.
  3. **`--exclude` by id on union arms, refused by the checker (G-R3): ACCEPTED.** Confirmed in the trainer: the `--exclude` help says it drops clips A..B of the TRAIN manifest's sha12 order.
  4. **Base commit `cdbff971`: ACCEPTED.**
  5. **`chunk == 48` asserted on every arm: ACCEPTED.**
  6. **Gate route = the study checker minting `dscale10k-gate-token-v1`: ACCEPTED.**
  * **U's `train_sha256`** is MEASURED by a dry `build_data` at block 0 and recorded in the gate token before any training step. It is a data identity, not an outcome, so measuring it after freeze changes no bar.
  * **Scheduling:** STRAT1's 7-arm chain finished on this pod at ~16:33Z (`ZZ STRAT1 CHAIN DONE ZZ`), so the GPU is free. The pod GPU is the RTX 4000 Ada, idle at ~19:20Z (MEASURED).
  * **Launch needs the PI's go** (`QUESTIONS_DATASCALE_TRAIN.md` Q1). Q2-Q4 are asked with it. The MM's recommendation is the questions file's recommended answers. Until the go, the build is limited to the checker, the merge tool, the union view and their RED arms, with no training step.
