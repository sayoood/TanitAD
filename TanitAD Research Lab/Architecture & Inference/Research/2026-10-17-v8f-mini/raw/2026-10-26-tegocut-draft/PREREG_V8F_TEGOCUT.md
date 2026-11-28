# PREREG_V8F_TEGOCUT: should the T head read its own ego state at all? T's ego token replaced by a constant, at training AND inference

**DRAFT, written 2026-10-26, before any DAgger bank, run or number exists and before any T-EGOCUT code exists.** Claim row `H-V8F-TEGOCUT-1`.
No DAgger or DAG-0 output was read to write it: none exists. Every number below is quoted from a primary file that predates the DAgger arm.

**What commits this arm.** `PREREG_V8F_DAGGER.md` §7, two outcome rows, verbatim:
* **NO EFFECT:** *"Next: **T-EGOCUT**: T's ego token (`v8f_mini.py:686`) replaced by a constant at training AND inference, M2b init, 3,000 steps,
  both seeds, under its own PREREG. It decides whether T should read its own ego state at all"*.
* **HARMS:** *"If it ran at p = 0.1: next is NO EFFECT's next (T-EGOCUT)"*. DAgger runs at p = 0.1 (`PREREG_V8F_DAGGER` Amendment 1 §0, because
  O-RECOVER read HARMS).

⇒ **This PREREG runs if and only if DAgger (p = 0.1) reads NO EFFECT or HARMS.** Under DAgger ROBUST or PARTIAL it is parked, not run, and the
register row says so. Under DAgger BLOCKED it is not run (DAgger's own BLOCKED next is the rejoin-target form). Under DAgger REFUSED it waits for
DAgger's re-run. The mapping is fixed here, before any DAgger number exists, so no DAgger number can choose this arm's design.

Nothing in `PREREG_V8F_DAGGER.md` or `PREREG_V8F_ORECOVER.md` (with their amendments) is changed by this file. Their rows, bars, controls and
fine-tune mechanics are reused as written.

## Why (MEASURED, quoted from the primary files)

* `RESULT_V8F_T1_HEADS.md` (T1-kin, m = 10, HACC, causal bucket; read `T1HD_READ.json`): on its own executed state, **T's decisions DEGRADE**.
  * Lateral macro-F1 0.848 → 0.614 (s0) and 0.858 → 0.632 (s1); longitudinal 0.461 → 0.336 / 0.487 → 0.357. Separated at both seeds.
  * The loss is in the turns: TURN_L / TURN_R F1 0.81 / 0.75 → 0.38 / 0.50 at s0; LANE_KEEP 0.98 → 0.96.
  * At zero row offset (the label row is the row whose cameras T saw), lateral F1 is still 0.606 / 0.647 against 0.828 / 0.857 at tick 0, and
    longitudinal 0.350 / 0.364 against 0.461 / 0.467 (`PROBE_LAG.json`, descriptive, decides nothing).
  * ⇒ ESTIMATED by setting the probe's drop beside the decisive one (two F1 definitions, so only a rough share): about 0.9 of the lateral drop and
    0.8 of the longitudinal drop survive at zero row offset. **The own-state inputs carry most of it; the camera-row mismatch at most ~0.1 / ~0.2.**
* `RESULT_V8F_ORECOVER.md` (read `ORC_READ.json`, sha256 `a50dbba9…`): **HARMS.** Training on perturbed ego values moved R-TLAT by +0.0331 / +0.0351
  (not separated) and R-TLON by −0.0214 / +0.0312, i.e. at most ~15 % of the ~0.23 lateral drop (ESTIMATED arithmetic), and cost T0 ADE6
  +0.1103 / +0.0849 m against ORC-0.
* If DAgger also reads NO EFFECT or HARMS, then, in DAgger §7's own words, *"training-distribution levers on T's ego channel are ELIMINATED"*:
  wrong values (ORC) and on-policy values (DAgger) both fail.
* ⇒ **The one lever left on that channel is to remove it.** If T decides as well or better without its ego token, the token is a liability on
  own states. If T loses, T needs it, and the drop enters elsewhere.

## 1. The cut: exactly what changes

**What T reads** (`stack/tanitad/models/v8f_mini.py`, `run_t`, :683–:696): four condition tokens, then the scene.

| T's token | code | at T1-kin, is it the model's own state? | this arm |
|---|---|---|---|
| **ego** | `self._ego(self._at(b["p10f"], i))`, i = `last_t` (:686) | yes: the 8 P10 features of the executed motion at T's tick frame | **replaced by the constant c** |
| nav | `self.e_nav(self._at(b["nav"], i))` (:686) | partly: channel 3 is route progress, dead-reckoned from the own arc (`v8f_t1_eval.py:216`–:220) | kept |
| limit | `self.e_lim([limit, limit_ok])` (:687) | yes under `--limit causal`: max own measured speed over the last 4 s, bucketed (`v8f_t1_eval.py:250`, :261) | kept |
| checkpoints | `self._ck_cond(ck_t, ck_ok)` (:688); `ck_t = transform_ckpt(ck_src, pose_s, pose_t)` (:745) | yes: S's output (S reads its own ego token, :675) moved by the own odometry | kept |
| scene | `self._scene(adapt, frame_ok, i)` (:689) | no: the LOGGED cameras at the clock row (`v8f_t1_eval.py`, module docstring) | kept |

**The tensor that feeds the cut token:** `p10f[:, last_t]` (8 features) → `P10Prior.standardise` = (f8 − xm) / xs (:264) → `e_ego`, ONE
`nn.Linear(8, d)` (:617) shared by S (:675), T (:686) and O (:702).

**The constant.** c := the init checkpoint's tensor `model["e_ego.bias"]` ([d] = [256] float32), read from the init checkpoint FILE.
* It is stored as a non-trainable buffer `t_ego_c`, frozen for all 3,000 steps, saved in the checkpoint, and read by `run_t` at training and at
  inference alike.
* **Identity:** c = `_ego(xm)` exactly, because `standardise(xm)` = 0. c is the token M2b itself emits for the P10 fit's mean kinematic state
  (xm = `Xtr.mean(0)`, `taniteval/taniteval/floors_kin.py:274`).
* It is constant across samples and across steps, so at inference T's ego token carries **zero** information about the sample.

**Why this constant, and the four rejected:**
* **zero vector:** a token `read_t` never saw in M2b's 19,164 steps (`source_step`, ORC-0 gate record). The fine-tune would start from a shock, and
  a NO EFFECT or HARMS could then mean "T could not re-adapt in 3,000 steps" rather than "T does not need ego".
* **a TRAIN-mean token computed for this arm:** by linearity it equals `e_ego(standardise(μ_TRAIN))`, and it differs from c only by
  W·(μ_TRAIN − xm)/xs. It has the same zero information, and it costs a data pass and a new statistic. c is already frozen in the checkpoint.
* **a learned constant (a new Parameter):** it adds a T-specific trained bias, which is a second lever. It changes the optimiser's parameter list,
  which breaks the pairing in §3. And it has the exact shape of the zero-gradient defect class (CLAUDE.md, "built, tested, unreachable"): a learned
  constant that never trains emits M2b's bias forever and still looks correct.
* **the live, detached `e_ego.bias`:** constant across samples but drifting with S's and O's training of `e_ego`. "Constant at training" would
  then hold one step at a time only.

**Untouched:** S's ego token (:675), O's ego token (:702), the frozen P10 prior (reads `p10f[last_now]` and `base`, :710), `e_ego` itself, nav,
limit, checkpoints, scene, ages, every loss term and weight, and the teacher forcing (p_gt 0.2).
* **S stays** because S works on own motion: S ADDS ROUTE at all three arcs at both seeds (`RESULT_V8F_T1_HEADS.md`).
* **O and the prior stay** because the plan needs ego: the P10 prior from the measured ego history is the deployed floor, and HACC's T1-kin margin
  over it holds on a past-only bucket (`RESULT_V8F_T1KIN_CAUSALBUCKET.md`: ADE6 −0.768 / −0.734 m).
* ⇒ The question is T's alone.

**Disclosed consequences:**
* `e_ego` loses T's direct gradient. It is still trained by S's and O's losses, and by T's loss through S's checkpoints, because `detach_upper` is
  False in M2b's argv.
* T's own query offset (`Reader.forward`, `q + cond.mean(1)`, :555) now carries c/4 from the ego slot: a constant.

## 2. Admissibility, and the back-door audit

**Inference stays vision plus measured ego motion** (PI ruling: measured ego motion and its history, including yaw rate, are admissible; no steering
angle). The cut REMOVES one input.
* It adds no batch key, no label and no classifier output.
* `forward_infer`'s read set (`_KeyGuard`, :939) is unchanged, because `p10f` is still read by S, O and the prior. The gate's
  `assert_no_situation_classifier_input` runs on the probe batch as for every v8F arm.
* **The goal input never carries the situation classifier:** this arm adds no goal input, changes none, and no classifier output exists anywhere in
  the model (the module's forbidden-token scan, `scan_source_for_forbidden`).

**The back-door audit: after the cut, can T still read its own ego state?** Yes, through four channels. This arm leaves them open by design: the
committed arm cuts the token only.

| # | channel into T | carries own ego at T1-kin? | size (evidence) | status |
|---|---|---|---|---|
| 1 | shared weights `e_ego` | **no**: under the cut T does not pass through `e_ego` | exact (G-LIVE (b), (c) below must read 0.0) | closed |
| 2 | checkpoint token: S's output (S reads its own ego token at `last_s`, :675) | **yes**: S's ego reading, 0–0.8 s older than T's tick (the same frame when a_s = a_t; `ChunkedSampler`, `v8f_mini_data.py:854`), compressed to 9 numbers | unmeasured; G-LIVE (f) records it | open |
| 3 | the odometry `pose_s → pose_t` in that token (:745) | **yes**: own displacement over a_s − a_t ≤ 4 steps | unmeasured; G-LIVE (f) records it | open |
| 4 | limit token | **yes** under `--limit causal` (own past 4 s max speed). At training it is the clip's logged bucket, the human's realised max speed, one value per clip (`v8f_mini_data.py:511`) | MEASURED on M2b, tick 0: causal vs corpus bucket moves T's lon F1 by 0.045 and lat by 0.0075 (ORC A2, `out/smoke_real.out`) | open |
| 5 | nav token, channel 3 | progress only (one frame value, not a speed) | unmeasured | open |
| 6 | scene | no, it is vision: admissible. ⚠ But at T1-kin the cameras are the HUMAN's, replayed at the clock row | see the rig caveat | open (admissible) |

⚠ **The rig caveat, stated before any data.** With the token cut, T's remaining motion cues at T1-kin include the human's camera motion. A T1-kin
gain for T-EGOCUT can therefore be partly T reading the human's motion from the replayed cameras, which no deployment provides. T2 (re-perception)
is not provisioned. The beside row **R-FROZEN** (§5) bounds this, and §7's ROBUST row says what follows from it.

⇒ **This arm answers "should T read its own ego TOKEN", not "should T be ego-disjoint".** The second question is the NO EFFECT row's next (§7).

## 3. The arms, and why ORC-0 can be the control

| arm | what | role |
|---|---|---|
| **EGOCUT** | M2b_s{seed} fine-tuned **3,000 steps** with `--t-ego const`, both seeds | **decisive** |
| **ORC-0** (exists) | M2b_s{seed} fine-tuned 3,000 steps with `--ego-perturb-p 0`, trained on box 2 at `853f2fc3` (MEASURED, read with 104 controls PASS) | **the control**: separates the cut from "3,000 more steps" |
| DAG-0 (exists whenever this arm runs) | ORC-0's argv + `--dagger-p 0`, trained by the DAgger build | **the replicate floor RF** (§5), at no extra compute |

**EGOCUT's argv = ORC-0's banked argv** (`E:/Projects/TanitAD-artifacts/v8f-orecover/M2b_s{seed}_orc0/argv.json`) **plus `--t-ego const`, with
`--out` changed.**
* The fine-tune mechanics are `PREREG_V8F_ORECOVER` Amendment 1, unchanged: `--init-ckpt <M2b_s{seed}/ckpt_last.pt> --steps 3000 --lr 1e-4
  --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --seed <seed>`, plus M2b's argv.
* `--ego-perturb-p 0` stays named, exactly as in ORC-0's argv. No DAgger flag is named.
* Init sha256: `601cb435…` (s0) / `6104fb15…` (s1).

**Settled by reading the trainer: EGOCUT and ORC-0 see the same stream, so ORC-0 is a valid paired control.**
1. **Windows and ages** come from `ChunkedSampler.epoch`, on `np.random.default_rng((seed, ep))` (`v8f_mini_data.py:863`–:877). That is a
   function of seed, epoch, the window list and `chunk` alone, and `chunk` comes from `--lru` (`v8f_mini_train.py:1181`). No model value or
   model flag enters it.
2. **The forward's random draws** are reseeded before every step: `torch.manual_seed(a.seed * 1_000_003 + step)` (`v8f_mini_train.py:1289`).
   The only draws are the two teacher-forcing `torch.rand(B)` (`v8f_mini.py:741`, :754; `code_mix whole`). The Reader's dropout is 0.0 (:544).
   The number and order of draws do not depend on any value, and `--t-ego const` adds none.
3. **The perturber at p = 0** returns the same batch with no draw (`EgoPerturb.apply`, `v8f_mini_data.py:669`–:678).
4. **c is a buffer, not a Parameter.** The optimiser's parameter list (`v8f_mini_train.py:1173`–:1178) and AdamW's state layout are identical.
   `n_params_trainable` must equal ORC-0's 7,214,495 (MEASURED, ORC-0 `selfcheck.json`).
5. **The init** is the same file with the same sha256.

⇒ The two runs see the same batches, ages and teacher-forcing masks, from the same init with the same optimiser. **They differ only in T's ego
token.**

**What can still differ, and the control that proves or bounds each:**
* **The trainer build** (`853f2fc3` against EGOCUT's commit): **K-BUILD**, the default-off identity, and **K-STREAM** (§6).
* **The data on box 2** (cache, join, labels): **K-DATA**. EGOCUT's gate-token data-manifest sha256 must equal ORC-0's.
* **The eval tool** (ORC-0's T1 dumps came from the pinned tree `tool_81aa3e36`, digest `c8d2990e…`, whose `v8f_mini.py` is `4c5be881…`;
  EGOCUT needs the new `v8f_mini.py`): **K-TOOL**. ORC-0's four T1-kin runs and two fresh-age T0 evals are re-run on EGOCUT's tool tree and must
  reproduce the banked outputs bit for bit.
* **GPU run-to-run noise:** the same for any two box-2 runs, including a fresh EGOCUT-0. That is exactly what RF measures.

**Committed fallback, fixed now so no data chooses it:** if K-DATA fails, ORC-0 is not the control. **EGOCUT-0** (ORC-0's argv, re-trained at
EGOCUT's commit, new `--out`) is then trained at both seeds and becomes the decisive control, and ORC-0 becomes the replicate for RF in place of DAG-0.
A K-BUILD, K-STREAM or K-TOOL failure is a build defect, not a fallback: it is fixed, and nothing launches until it passes.

**One lever.** A separated effect is attributed to "T's ego token replaced by c". `--t-ego const` with `--ego-perturb-p > 0` or `--dagger-p > 0` is
refused by the gate: the levers are never combined in one arm.

## 4. The flag, the build, and its gate checks (default off; bit-identical when off)

**The flag:** `--t-ego {full, const}`, default `full`. It maps to the config field `t_ego: str = "full"` (validated in `__post_init__`, as
`o_interface` is at :204) and to `built_levers()["t_ego"]`.

**The model** (`v8f_mini.py`):
* `t_ego_c` is registered as a `None` buffer, absent from the state dict by default, so every earlier checkpoint loads and runs bit for bit. That is
  the `hacc_w` pattern (:611, :818).
* `_load_from_state_dict` creates the buffer when a state dict carries it. `set_t_ego_const(c)` checks shape [d] and finiteness.
* `run_t` reads `t_ego_c.expand(B, -1)` in the ego slot when `t_ego == "const"`. That is ONE line, the mutation anchor.
* A `const` model whose buffer is absent REFUSES at the forward.
* `built_levers()` adds `t_ego` and `t_ego_c_sha256` (the sha256 of the buffer's float32 bytes, or None).

**The trainer** (`v8f_mini_train.py`):
* `FLAG_CHECKS["t_ego"] = "lever:t_ego"`. The gate refuses a flag without an entry (:407).
* With `const`, c is read from the init FILE's `model["e_ego.bias"]` right after the init record (:1164), whether the init was applied or skipped
  for a resume. A resume therefore sets the same c, and `ckpt_last.pt` then reloads the identical bytes.
  * ⚠ It is never read from the live model: on a resume the init is not applied (:1166), so the live `e_ego.bias` is a random initialisation at
    gate time.
* **The eval tools need no edit.** Both build `V8FMiniConfig(**ck["cfg"])` and `load_state_dict(ck["model"])` strictly (`v8f_mini_eval.py:237`,
  :263; `v8f_t1_eval.py:506`, :511), and the checkpoint carries `cfg.__dict__` (`v8f_mini_train.py:1256`). So the field and the buffer reach the
  eval model.
  * A pre-change build REFUSES an EGOCUT checkpoint loudly (unknown config field).
  * A `full`-built model REFUSES its state dict (unexpected key `t_ego_c`).
  * Both are loud failures, never a silent eval with the live ego token.

**G-DVB (`bad_t_ego`):**
* the declared `t_ego` equals the built one;
* `const` requires `--init-ckpt`: without M2b's token there is nothing to freeze, and the lever is refused;
* `const`: the buffer is present, [d] and finite, and its sha256 equals the sha256 of the init checkpoint's `e_ego.bias` bytes (recorded in the
  init record);
* `full`: the buffer is absent;
* `const` with `--ego-perturb-p > 0` or `--dagger-p > 0` is refused;
* `n_params_trainable` is unchanged by the flag.

**G-LIVE:**
* `live_check`, unchanged, on the gate batch: every loss term finite, every trainable group gets a non-zero gradient, the frozen prior reads
  EXACTLY 0.0. No parameter is declared dead: `e_ego` still trains through S and O.
* **`t_ego_probe`** (new; the structure of `o_interface_probe`, :621), on a COPY, the real gate batch, its own seeds. Both branches run in every
  gate (the copy's `cfg.t_ego` is flipped), so each has its control:

| clause | const must read | full must read (the control that the probe can see) |
|---|---|---|
| (a) TOKEN: the ego slot of the `cond` that `read_t` receives | c on every row, bit for bit | `_ego(p10f[last_t])` |
| (b) T-INVARIANCE: `run_t` on a donor `p10f` (rolled by one row), with the checkpoint input held fixed | every T output (goal, lat, lon logits, cons) moves by **EXACTLY 0.0** | moves (> 0) |
| (c) T-GRADIENT: grad of T's loss terms (goal + lat + lon + cons) w.r.t. `e_ego`, checkpoint input fixed and detached | **EXACTLY 0.0** | > 0 |
| (d) STILL TRAINS: grad of the total on `read_t`, `trunk_t`, `h_lat`, `h_lon`, `h_goal`, `h_cons`, and on `e_ego` | > 0 for each | > 0 |
| (e) FROZEN: `t_ego_c` is not in `parameters()`, and its bytes after `live_check`'s 3 optimiser steps on a copy | change **EXACTLY 0.0** | buffer absent |
| (f) RESIDUAL (recorded, never gating): `forward_infer` on (i) donor `p10f` and (ii) donor `p10f` + `pose_t` + `pose_s`; max \|Δ T logits\| | reported: the size of back doors 2 and 3 | reported |
| (g) DETERMINISM: the probe run twice | difference **EXACTLY 0.0** | same |

* **Admissibility:** `_keys_read` is equal under `full` and `const`.
* **Default-off identity:** with `--t-ego` absent, a 3-step loss trace equals the pre-change trainer's bit for bit, and every other probe's record
  is unchanged.

**Deliberate-regression mutants** (build time, CPU, by mutation; no GPU arm is spent). Each must turn a named check RED:

| mutant | must go RED |
|---|---|
| M1: delete the `const` line in `run_t` (T reads the live token) | `t_ego_probe` (a), (b), (c) |
| M2: take c from the live model at gate time instead of the init file | the resume test's `t_ego_c_sha256` clause |
| M3: make `t_ego_c` a Parameter | (e), and G-DVB's `n_params_trainable` |
| M4: apply the constant in `run_s` instead of `run_t` | (a), and S's token check |
| M5: drop `t_ego` from the saved config | G-EVAL (strict load refuses the buffer key) |
| M6: add one `torch.rand` in the `const` branch | K-STREAM |
| M7: `bad_t_ego` accepts `const` without `--init-ckpt` | its unit test |

**G-EVAL:** an EGOCUT checkpoint from a 3-step synthetic run, loaded by `v8f_mini_eval.py`'s and `v8f_t1_eval.py`'s own loaders, must give:
* `built_levers()["t_ego"] == "const"` and the same buffer bytes;
* `forward_infer` bit-identical to the trainer model's on the probe batch;
* clause (b) on the loaded model.

**G-CKPT:** the round trip, plus a resume test (crash at step k, resume): the same `t_ego_c` bytes and the same loss trace as an uninterrupted run.

**G-SUITE:** a clean tree with 0 regressions, as for every launch.

## 5. Evaluation, rows, bars (ORC's and DAgger's, reused)

| where | what |
|---|---|
| **dev-box CPU (decisive)** | `v8f_t1_eval run --replan-steps 10 --limit causal --nav deadreckon --scene logged --schedule periodic --dump-ticks --dump-heads` on eval-141, **plain**, for EGOCUT × 2 seeds. Plus one `--limit logged` run each for K4 (ORC A2), and one fresh-age T0 eval each as K4's comparator (ORC A4). Plus K-TOOL's six ORC-0 re-runs, and R-FROZEN's four runs (beside) |
| box 2 | the chain's standard T0 eval (eval-141) of each EGOCUT checkpoint: R-T0, T0-CKPT, and the four families at T0 |

**Tier.** T1-kin is a **self-action OPEN loop**, never "closed loop". T0 is the guard tier.
**Estimator.** Paired episode-cluster bootstrap (`taniteval.ci`), B = 2,000, seed 0, 141 episodes (2,399 windows), per seed, EGOCUT − ORC-0, on the
same windows and matched cells.
**What each interval answers:**
* Another draw of EPISODES: the bootstrap.
* Another TRAINING RUN: the two seeds plus RF (`H-ESTIM-SEED-1`).
* Another INFERENCE RUN: does not arise if the planner is deterministic, as DAgger §5 states (`PREREG_V8F_PGT`; INHERITED, not re-measured
  here). K4 and K7 are consistent with it. They do not test a second inference run.

| row | definition | bar / direction |
|---|---|---|
| **R-TLAT / R-TLON** | T's lat / lon macro-F1 at ticks ≥ 1 (the T1-HEADS definition and reducer, via the ORC reader unchanged), EGOCUT − ORC-0 | a head **clears** at seed s iff separated-better AND point ≥ **bar(h, s) = max(2F_h, 2·RF_h,s)** |
| **RF (replicate floor)** | \|F1_h(DAG-0_s) − F1_h(ORC-0_s)\|, ticks ≥ 1, cells scored in both, point: DAgger §5's definition, reused verbatim | enters the bar |
| **R-ADE** | T1-kin plain ADE6, EGOCUT − ORC-0 | guard: not separated-worse |
| **R-T0** | T0 own-code ADE6, EGOCUT − ORC-0 | guard: point ≤ +F/2 = **0.0604 m** (F = F_M2b = 0.1207 m, `MODEL_REGISTRY.md` §17) |

* 2F = **0.0248** (lat) / **0.0146** (lon), the T0 fresh seed floors (ORC §2). Bars are inclusive (ORC A2).
* The bar can only rise above 2F, and only by the measured run-to-run gap. ⚠ ORC's own read flagged the lon 2F as likely too small (its lon row
  flipped sign across seeds, `RESULT_V8F_ORECOVER.md` §5 item 5). RF is the instrument for that; the bar is not re-chosen.
* **The control's levels**, MEASURED (`ORC_READ.json`):
  * T1-kin lat F1: ORC-0 0.5029 / 0.5172; lon: 0.2961 / 0.2618.
  * T1-kin ADE6: 2.8121 / 2.8473 m. T0 ADE6: 1.9796 / 2.0517 m.

**Beside (never decisive; each with its tier stamp, estimator and n):**
* **Tick-0 lat / lon F1:** what the cut costs on logged states. This is the arm's most likely price: T loses speed information.
* **The own-state gap**, (tick-0 F1 − ticks ≥ 1 F1), for EGOCUT against ORC-0: does the cut close T's T1 drop, or only move both ends?
* **R-FROZEN:** R-TLAT / R-TLON re-read with `--scene frozen` (`v8f_t1_eval.py:64`, :242), EGOCUT and ORC-0, both seeds. This bounds the rig
  caveat (§2): with the cameras frozen at t0, no replayed human motion reaches T after t0. The ORC reader runs with its T1-CONFIG control set to
  `scene frozen` for these runs only.
* The zero-row-offset F1 (`PROBE_LAG`'s definition) for both arms.
* Per-class TURN_L / TURN_R F1, where the loss was.
* The `R76` crawl rows M1 and L2, and the T1 off-route fraction at tick 2.
* S-POS@30 / 50 / 80 against CTRV, per run. **STRATEGIC is NOT paused for v8F** (PI 2026-10-17, R73): reported beside, deciding nothing.
* **The four families** (LONGITUDINAL incl. distance keeping, LATERAL, TACTICAL, STRATEGIC), at T0 and at T1-kin, EGOCUT − ORC-0. They use
  `RESULT_V8F_ORECOVER` §3's instruments unchanged (`v8f_paired_families.py`, `family_windows` + `paired_row`, `read_dk`).
  * A family cell that cannot be computed is reported with its reason and n, as in ORC §3.6. Example: the paired T1-kin STRATEGIC, which the paired
    instrument cannot read from tick dumps.
* EGOCUT − DAG-0 (the second control), EGOCUT − M2b at T0, and G-LIVE (f)'s residual-path sizes.

## 6. Controls (each must pass, or the read is REFUSED, except K-DATA, which routes to §3's fallback)

**Before training (the pairing family):**
* **K-BUILD:** the EGOCUT-build trainer with ORC-0's argv (flag absent) reproduces the `853f2fc3` trainer's 3-step loss trace bit for bit, on the
  synthetic world and on the real gate batch.
* **K-STREAM:** the EGOCUT-build trainer with `--t-ego full` and with `--t-ego const`, 20 steps on CPU, logs per step the sha256 of the batch's
  (ci, t, a_t, a_s) and of the two teacher-forcing masks. The two logs are identical at every step. Deliberate regression: mutant M6 must make
  them differ.
* **K-DATA:** EGOCUT's gate-token data-manifest sha256 equals ORC-0's, per seed. On failure, §3's EGOCUT-0 fallback runs.
* **K-INIT:** `--init-ckpt`'s sha256 equals ORC-0's (`601cb435…` / `6104fb15…`).
* **PAIR:** EGOCUT's argv equals ORC-0's banked argv plus `--t-ego const`, with `--out` differing; the reader checks it.
* **Launch gate:** a PASS token per run (two in all), binding the commit, that run's argv sha256 and the data manifest. As in ORC A2, the reader
  checks the binding; the HMAC key lives on the launch host.

**Reached training:**
* `summary.json` levers read `t_ego: const`, and `t_ego_c_sha256` equals the init file's `e_ego.bias` sha256.
* The final checkpoint's `t_ego_c` bytes equal c: a change of EXACTLY 0.0 after 3,000 steps.
* Its `e_ego.bias` differs from c (non-zero): S and O trained it, which proves the run trained at all.
* **K-CUT-EVAL:** clause (b) of `t_ego_probe`, run through the eval loader on the final checkpoint over 64 eval-141 windows, reads EXACTLY 0.0.
  The same probe on ORC-0's checkpoint reads > 0.

**Read:**
* The T1-HEADS reader's K1–K7 on every run, as amended by ORC A2 and A4. K4's comparator is the arm's own fresh-age T0 eval. K7 requires the
  rebuilt track to equal the dumped one.
* **K-TOOL:** ORC-0's re-runs on EGOCUT's tool tree equal the banked ticks (sha256 in `v8f-orecover-t1/runs.json`) and the banked fresh-age T0
  numbers. A failure is a default-path defect in the build: fix it, then re-run.
* **K-RF:** RF recomputed by this reader equals DAgger's read value, per head and seed.
* SAME-WINDOWS, GRID and GRID-COUNTS (2,399 / 141), as in ORC.
* A reader K1 on its own rows: (a, a) reads +0 [0, 0].

## 7. Outcomes, committed now (EGOCUT vs ORC-0)

| outcome | condition | committed next |
|---|---|---|
| **ROBUST** | R-TLAT and R-TLON both clear (§5) at both seeds; R-ADE not separated-worse at either seed; R-T0 ≤ +F/2 at both | T does not need its ego token: **adopt EGOCUT as the base.** Re-read T1 dominance over P10, and FLAT's committed repeat, on it. If the beside own-state gap still exceeds the bar on a head, next is **CHANNEL-SWAP → T-EGOCUT-2** (the NO EFFECT row's next). If R-FROZEN's points are not same-signed with R-TLAT / R-TLON at both seeds, the adoption is stated as "T1-kin with logged cameras", and its confirming read is blocked on T2 (not provisioned: a named blocker) |
| **PARTIAL** | exactly one head clears at both seeds; guards held | adopt the cut for the head that moved. One token feeds one shared trunk (`trunk_t` → `h_lat`, `h_lon`, :690–:693), so a per-head answer needs a per-head route. Next: **T-EGO-LATE**: T's trunk keeps c, and the head that did not clear reads `_ego(p10f[last_t])` through its own late linear path. M2b init, 3,000 steps, both seeds, vs ORC-0, under its own PREREG. ⚠ An architecture change of T's heads: it needs the Master Mind's / PI's go before its PREREG lands |
| **NO EFFECT** | no head clears at both seeds; guards held. A head separated-better but under its bar is NO EFFECT, reported and never called an effect | the token is not where T's own-state drop enters, or not alone. Next: **CHANNEL-SWAP** (zero training, dev-box CPU, its own PREREG): on EGOCUT and on ORC-0 at T1-kin, each of T's remaining own-state channels (the limit token, the checkpoint token with its odometry, nav progress) is swapped, one at a time, from the own value to the logged value at the matched route point r′ (DAgger §2's matcher). **The rule, fixed now:** the channel with the largest summed lat + lon F1 recovery at ticks ≥ 1 (point, mean of the two seeds) is cut next as **T-EGOCUT-2** (box 2, M2b init, 3,000 steps, both seeds, this PREREG's rows and bars, vs ORC-0) |
| **HARMS** | R-ADE separated-worse at ≥ 1 seed, or R-T0 above +F/2 at ≥ 1 seed | EGOCUT is not adopted: the token carries information the plan needs. Next: **T-EGODROP**: T's ego token replaced by c on a fraction q = 0.5 of training samples (one Bernoulli draw per sample from a private generator seeded by (seed, step), `EgoPerturb`'s rule), with the live token at inference. M2b init, 3,000 steps, both seeds, vs ORC-0, under its own PREREG |
| **REFUSED** | any control fails (K-DATA excepted: §3's fallback) | fix and re-run |

**Why these nexts, by measured effect size** (Rule Zero 5):
* **CHANNEL-SWAP before any further cut.**
  * After this arm, three lever families have been measured on T's ego channel: the input law (ORC), on-policy values (DAgger) and the token cut
    (this arm). The own-state channels still carry about 0.8–0.9 of the drop at zero row offset (ESTIMATED from T1-HEADS).
  * The largest single-channel effect on T measured so far is the speed bucket: lon F1 0.045 at tick 0 (ORC A2).
  * No measurement splits the checkpoint, odometry and nav channels. Ranking them costs about 1.5 CPU hours (ESTIMATED: 3 channels × 2 arms × 2
    seeds × ~280 s). Guessing would cost a box-2 arm. The rule picks the channel, not this expectation.
* **T-EGODROP under HARMS, ranked above a perception-side lever.**
  * HARMS says the token's information is needed. ORC, which trained on WRONG ego values, cost T0 LONGITUDINAL accuracy at both seeds:
    speed MAE +0.0590 / +0.0212 m/s, accel MAE +0.1312 / +0.0547 m/s², both separated (`RESULT_V8F_ORECOVER.md` §3.1).
  * Dropout replaces the value with c, an "unknown", never with a wrong value. So T keeps the information on logged states and also learns to
    decide without it.
  * A perception-side lever ranks below it on both counts:
    * the camera-row mismatch carries at most ~0.1 (lat) / ~0.2 (lon) of the drop (`PROBE_LAG`, ESTIMATED share, descriptive);
    * T1-kin replays the human's cameras, so a vision-side ego cue would be scored on the human's motion. T2, the tier that could score it, is not
      provisioned: a named blocker.
  * q = 0.5 is fixed now, not tuned.
* **T-EGO-LATE under PARTIAL:** see that row.

**The table is complete:**
* a failed control gives REFUSED, except K-DATA, which triggers the EGOCUT-0 fallback and re-reads;
* with the controls passing, a failed guard gives HARMS;
* with the guards held, the number of heads that clear at both seeds gives ROBUST (2), PARTIAL (1) or NO EFFECT (0).

No row ends at a refutation: each names the arm that runs next, or the named blocker.

## 8. Queue and cost

**Box 2,** only after DAgger's read and only on its NO EFFECT or HARMS:
1. Gate EGOCUT s0 and s1: two PASS tokens.
2. Train EGOCUT s0, then s1. The chain's predecessor predicate reads DAgger's read OUTCOME, not only its end marker.
3. The two standard T0 evals.

* **Cost, ESTIMATED from MEASURED run times:**
  * ORC's four fine-tunes took 2,249–2,258 s each for 3,000 steps (`summary.json` `seconds`, `v8f-orecover/M2b_s*_orc*`). The cut removes one
    `Linear(8, d)` call on T's path and adds none.
  * ⇒ training ≈ 2 × 2,255 s ≈ **1.25 h**. Gates and T0 evals: ESTIMATED ≤ 0.5 h (not measured per run).
  * **Box-2 total ≈ 1.75 h.**
* **Fallback (K-DATA fails):** + 2 EGOCUT-0 runs ≈ +1.75 h.
* **Saved by reusing ORC-0 instead of training EGOCUT-0:** ≈ 1.75 h of box 2.

**Dev-box CPU,** gated on C: ≥ 2 GB free, as each checkpoint arrives:
* EGOCUT: 2 causal + 2 logged T1-kin + 2 fresh-age T0;
* K-TOOL: the same six on ORC-0;
* R-FROZEN: four T1-kin runs;
* then the reader and the families driver.
* **Cost, ESTIMATED from MEASURED run times:**
  * ORC's T1-kin rollouts took 255–284 s each, with one at 507 s; the fresh-age T0 evals took 34–46 s (`v8f-orecover-t1/done_*.txt`, `wall_s`).
    The ORC reader took 197.6 s (`RESULT_V8F_ORECOVER` §2).
  * ⇒ 12 rollouts × ~280 s + 4 × ~40 s + reader and families ≈ **1.1 h** (≈ 1.6 h if every rollout runs at the 507 s outlier).

**Build, 0 GPU:**
* the flag, the model branch, `bad_t_ego` and `t_ego_probe`;
* the tests and mutants M1–M7, G-EVAL and G-CKPT;
* the reader, re-parametrised from `orc_read.py` (pair rule, RF bar, K-TOOL, K-RF, K-CUT-EVAL);
* the chain stages.
* ⚠ The trainer at tip `2a78f059` has no `--dagger-*` flag yet. `bad_t_ego`'s exclusivity clause needs DAgger's trainer wiring to land first, or
  to be added with it.

⚠ **Box-2 time needs the PI's go,** as O-RECOVER's and DAgger's did.

Nothing in this file may be changed after DAgger's read except by a dated amendment that says what it moves and why. The arm, the constant, the
control and its fallback, the rows, the bars, the controls and the outcome table are frozen at landing.
