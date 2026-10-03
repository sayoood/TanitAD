# RESULT — refcv7 on the NavSim suite (warmup_two_stage · navhard_two_stage · navtest v1.1)

**Stream** EvalFlyWheel NavSim · **2026-09-28** · **Model** refcv7-r101-s0 (training on Thor; launched
2026-09-28 ~00:03 Berlin; `config.json` md5 `e6512a01b9c70f0e4a4dac581621984a`) · **Pre-registered**
`SPEC.md` (sha256 `5ffb584c…`, 05:19 Berlin) + amendment A1 (sha256 `03cccc4a…`, 05:46 Berlin), both
before any refcv7 NavSim score (`raw/SPEC_PREREG_HASH.txt`) · **Code under test** the clean tree
`0c444082` (`git archive` → `C:/Users/Admin/ev7nav`).

Stamps on every refcv7 NavSim row: NavSim **open-loop benchmark** (T1-family; stage-1 loop OPEN; v2
background IDM-reactive, v1 non-reactive); ⛔ **never closed loop**; **zero-shot** (PhysicalAI-AV B1 →
nuPlan cameras, 3-camera stitch); **non-parity**; device + precision per split; model-as-trained
`"refcv7-r101-s0 as launched"` (`code/model_stamp7.py`). ⛔ **Everything below step 5,000 is
VALIDATION ONLY (SPEC §6): it proves the path and is never refcv7's NavSim performance.**

## 0. Headline (status 2026-09-28 ~07:45 Berlin — before the step-5,000 checkpoint exists)

1. ⛔ **ESCALATION — refcv7's speed-ceiling filter never reaches the plan it emits.** The E9 goal
   selection in `RefCV3Model.forward` (`refc_v3.py:2285-2298`) re-ranks the fan after the decoder
   with the reach mask ONLY, so the decoder's ceiling-filtered pick is overridden. MEASURED at step
   1,500: filter ON vs OFF = **bit-identical plans on 204/204 warmup scenes** (CPU and CUDA), the
   emitted plan over the ceiling in 2 CUDA scenes. SPEC_REFCV7 A2 is built but inert; G-DVB and
   G-LIVE cannot see it. Recorded as SPEC amendment A1 **before any refcv7 NavSim score**; priced by
   the diagnostic arm R7_CEILDECL_d. Owner: Master Mind / Arch (COMMS §1).
2. **The harness reads its known values through this package's drivers**: warmup CV / STOP / ECHO
   and navtest STOP (12,146 tokens) all **max |Δ| 0.0** vs the banked references; navhard CV
   (5,912 tokens) ⏳ running (§1).
3. **The bridge feeds refcv7 exactly as its trainer does**: KT7 — on 3 real PhysicalAI eval
   windows, `compute_losses_v3`'s own `model(...)` call and the bridge's feed carry identical
   keywords and tensors and give **bit-identical plans**; 68 tests green (incl. 6 GPU-lock protocol
   tests), 4 deliberate regressions RED (§2).
4. **The whole pipeline ran on the step-1,500 checkpoint (VALIDATION ONLY)**: warmup end to end on
   CPU (9 arms scored, every count guard PASS) and on CUDA at the bridge level, plus navtest and
   navhard input-path smokes (§3).
5. **The step-5,000 milestone waiter is ARMED** (event-driven, detached; §4). The BAR-R7-N1 reading
   will land in `raw/milestones/step5000/BARS.json`.
6. ⚠️ **Incident, mine:** a validation runner I killed by PID had just taken the GPU lock; the lock
   stayed stale 06:59–07:41 Berlin and blocked the battery and the Research Lab. Released with its
   own token; `gpu_lock.reap_own_stale()` now prevents it (COMMS §4).

## 1. The harness reads its known values through THIS package's drivers (SPEC §7 KH)

| split | control | result |
|---|---|---|
| warmup | CV_official, STOP_zero re-scored (`code/score_arm7.py`) vs E2's banked CSVs | ✅ **REPRODUCED**: 220 tokens × every column **max \|Δ\| 0.0**; S2-EPDMS-u CV 0.39713167136274696 = E2, STOP 0.5212469877807624 = E2; the three official rows \|Δ\| 0.0; count guards PASS (`raw/HARNESS_REPRO.json`) |
| warmup | ECHO_ha0_ext re-scored vs E2 | ✅ **REPRODUCED**: max \|Δ\| 0.0; S2-EPDMS-u 0.4287471809626652 = E2 (`raw/controls/KH_ECHO_warmup.json`) |
| navtest | STOP re-scored on all 12,146 tokens (`code/score_navtest7.py --official STOP`) vs W3's banked CSV | ✅ **REPRODUCED**: 12,146 tokens × 8 numeric columns **max \|Δ\| 0.0**; PDMS 61.82023 = W3; count guard PASS 12,146 / 0 / 12,146 (`raw/controls/KH_navtest_STOP.json`; 3rd attempt — the first two were aborted by E1's RAM guard at 1.8–1.9 GB free, one at 11,649/12,146, while other sessions held ~13 GB) |
| navhard | CV re-scored on all 5,912 tokens vs W7 (must give official two-stage EPDMS 0.11481608441648) | ⏳ **RUNNING** since 07:38 Berlin (≈ 3.2 h CPU; two earlier attempts RAM-guard aborted); verdict lands in `raw/floors/navhard_two_stage/FLOORS_PROVENANCE.json` `_KH_nav_check.verdict` and `raw/harness7.log` |

## 2. refcv7's inputs — explicit functions, literal / analytic tests, arms that go RED

**The model** is built ONLY by the landed loader `stack/tanitad/eval/refcv7_loader.build_model`
(blob `a8eb3e7e`): train()'s build order replayed, **STRICT load 0 missing / 0 unexpected over 1,131
keys, G-DVB 0 mismatches over 221 registry entries, `param_breakdown` equal to `config.json`**
(MEASURED on `ckpt_1500.pt`, CPU, 7.0 s). `refcv7_bridge.load_refcv7` then ASSERTS 20 NavSim-seam
premises off the BUILT object (horizons 5…60 ticks, window 8, 416×1024, 117 anchors in `alat`,
`residual_prior = ha0_ext_pose`, ceiling filter ON, 0.25 m bank present and stride-16 bank absent,
trunk AND lift masks 43 rows, no ego-state injection / LAN / nav-args / oracle agents, 2 DDIM steps).

| input | construction (`code/refcv7_bridge.py`) | control / test |
|---|---|---|
| frames, nav, v0, ego window, max speed | the refcv6 suite's functions IMPORTED from the clean tree (`boot7.py`) under refcv7 arms mapped to the R6 arm with the identical declared set / nav / max-speed source (`R6_TEMPLATE`) | literal arm table; K5/K6 declared-input seam; 2.0 m/s² and −0.2 rad/s literals, A1 dropped-frame, refusals (`tests/test_inputs7.py`) |
| **the feed** | `forward_kwargs7` = exactly `compute_losses_v3`'s keyword set; `set_ego_window(poses, 8, actions=None)` | ⭐ **KT7: on a REAL PhysicalAI eval window, the trainer's own `compute_losses_v3` `model(...)` call (captured) and the bridge's feed carry the SAME keywords and the SAME tensors and give a BIT-IDENTICAL plan and prior** (`raw/controls/KT7_trainer_feed.json`); AST pin of the trainer's keyword list |
| **the prior P** (`ha0_ext_pose`) | the model's own `prior_controls` + `prior_path` on the declared window | literals: a0 = 2.0 m/s² (const-accel), κ0 = −0.025 1/m (−0.2 rad/s at 8 m/s), floor −0.1, cap −0.3; the Euler path `[5.2, 10.9, 17.1, 23.8, 38.7, 55.6, 74.5, 95.4]` m; CV line; frame invariance; **KPR** emitted == model-free (CPU max \|Δ\| 0.0 m; CUDA 3.8e-6 m) |
| **the 0.25 m lift** | `build_lift_geometry` with every parameter read off `model._lift_bank_hires` | **KL-lift** bit-exact vs the bank's own `geometry(ep)` (a bank built exactly as train() builds it, and 3 real clips of the checkpoint's own bank); analytic: the 43 masked rows exclude road points 6.06–7.17 m ahead of a 1.85 m camera, valid beyond |
| **the ceiling** | the fed one-hot's bin limit | literals 25 mph → 13.888… m/s, 35 → 27.77…, 15 → 8.33…; unknown / withheld → +inf |

**Model-side controls** (`tests/test_model_seam7.py`, CPU fp32, step-1,500 checkpoint): **K0** same
seed ⇒ bit-identical; **KD** exact dedup == native, 4/4, max \|Δ\| 0.0 m (`raw/controls/KD_exact_dedup_cpu.json`);
**KI** frames, nav, v0 and the ego window each move ≥ 1 of 4 plans, max speed moves the v6 tactical
logits; **KPR**; **KF** (SPEC A1); **RED**: a window not ending at v0 and the refcv6 stride-16
geometry are both REFUSED by the model.

**Deliberate regressions that go RED** (`tests/test_mutation7.py`): `R7_MUTATION` reintroduces a
real defect class into the bridge and re-runs the literal suite, which must FAIL — `wrong_grid` (2 Hz
states packed as 0.1 s samples: the prior reads 10 m/s² for a 2 m/s² history), `drop_mask` (the C26
43-row mask dropped from the lift), `stride16` (refcv6's lift parameters), `kw_drop` (a forward
keyword lost): **4/4 RED, the unmutated suite GREEN**, and the runner REFUSES to score in a mutated
environment.

**Test totals (all green, 2026-09-28):** `test_inputs7.py` 43 · `test_model_seam7.py` 13 (CPU fp32,
539 s; KT7 on 3 real eval windows, KL-lift on 3 real clips) · `test_mutation7.py` 6 · `test_gpu_lock7.py`
6 = **68**. On CUDA the runner re-measures K0 / KD before every split (step 1,500: K0 PASS, KD 1.196 mm
→ the dedup is dropped on CUDA).

## 3. Pipeline validation — step 1,500 (⛔ VALIDATION ONLY, SPEC §6; never refcv7's NavSim performance)

Checkpoint `D:/refcv7_eval_kit/ckpt/ckpt_1500.pt` (the battery stream's pull of Thor's rolling
`ckpt.pt`, md5 `c35966f799b1c724515ba1281736fd48` — Thor before == after == local; step key 1,500).

**The whole path ran, end to end, through the milestone runner** (`code/run_navsim_refcv7.py
--splits warmup`, `raw/validation/step1500/`): GPU lock acquired → CUDA controls **K0 PASS, KD
FAIL (1.196 mm > 1 mm; selections 4/4 identical)** → (the runner then refused CUDA — since corrected
to drop only the dedup, COMMS D9) → CPU fp32 bridge, 7 arms + 2 derived, **KPR 0.0 m on every
row, the E9 re-derivation reproduced `sel_idx` on 204/204** → 9 official scorings, **every count
guard PASS (220 / 0 / 220)** → `summary_warmup.json`, `decomposition_warmup.json`,
`plan_deltas_warmup.json`, `BARS.json` (= `NOT_EVALUATED — VALIDATION ONLY`).
**The CUDA path** (the milestone's device) ran the same 7 arms + 2 derived on the NATIVE trunk path
(`raw/validation/step1500_cuda/bridge_warmup/`): 204 + 16 stand-ins per arm, KPR ≤ 3.8e-6 m,
replication 204/204, **0.35–0.42 s/scene**; plus bridge-only input-path smokes of **navtest** (12
tokens, the 416 unique-frame bank + log rigs + the oracle input) and **navhard** (20 stage-2 + 2
stage-1 real-frame tokens) — all rows produced, KPR ≤ 1.9e-6 m, replication 0 failures.

| arm (step 1,500, CPU fp32, 204 warmup stage-2 tokens) | S2-EPDMS-u | stop frac | vs R7_A1 (W/T/L) |
|---|---|---|---|
| R7_A1 (filter ON) | 0.4442 | 0.152 | — |
| R7_A1_s1 (inference seed 1) | 0.4577 | 0.147 | seed floor \|Δ\| 0.0136 (42/94/68) |
| R7_FILTOFF (real forward) | 0.4442 | 0.152 | **0/204/0 — bit-identical plans (SPEC A1)** |
| R7_CEILDECL_d (diagnostic) | 0.4442 | 0.152 | 0/204/0 on CPU (on CUDA the declared ceiling changed 2 picks) |
| R7_VMAXOFF | 0.4442 | 0.152 | 0/204/0 — the max-speed one-hot moves the logits (KI) but no plan at this step |
| R7_NAVOFF | 0.4622 | 0.181 | −0.0180 (30/98/76) |
| R7_BLIND | 0.3970 | 0.054 | +0.0472 (70/90/44) |
| R7_A1NT | 0.4450 | 0.152 | −0.0008 (43/105/56) |
| **PRIOR_ha0p** (model-free P) | 0.4586 | 0.201 | A1 − PRIOR **−0.0144** (52/81/71) |
| CV_official / STOP_zero / ECHO_ha0_ext | 0.3971 / 0.5212 / 0.4287 | — / 1.0 / — | A1 − STOP −0.0771, − CV +0.0471, − ECHO +0.0154 |

Plan-level reads (`plan_deltas_warmup.json`, 4 s endpoint distance to R7_A1's plan): seed replicate
median 0.451 m (p90 1.60, same selection 90.2 %); NAVOFF 0.068 m (p90 3.88); BLIND 0.273 m (p90 5.80);
A1NT 0.006 m; FILTOFF and VMAXOFF 0.000 m (bit-identical). **KP (precision floor, plans):** the same
checkpoint and seeds, CPU fp32 vs CUDA bf16-native: same selection 88.2 %, endpoint median **0.40 m**,
p90 2.88 m (`raw/validation/KP_step1500_warmup_plans.json`) — the size of the device term any
cross-checkpoint read with differing devices must be held against.
⛔ These numbers exist to size the floors and prove the path; at 1,500 of 50,400 steps they say
nothing about refcv7.

## 4. The milestone waiter (armed) and the manual command

* **Armed** 2026-09-28 07:42 Berlin: `code/milestone_waiter7.py 5000`, Windows PID **25012**
  (venv launcher 36772), state `raw/milestones/waiter_5000.json`, events `raw/milestones/waiter_5000.log`
  (earlier instances 05:52 / 06:35 / 06:46 were killed by PID while still waiting for the checkpoint,
  each to change the battery rule; their state files are kept as `*.superseded_*`).
* **Trigger (events, never a clock):** `D:/refcv7_eval_kit/ckpt/ckpt_5000.pt` + a `MD5SUMS` line naming
  it + local md5 == that line + the file's `step` key == 5000 → the battery goes first: the GPU lock
  seen held by a job naming `5000` / `5k` / `5,000` (not this suite; the Research Lab's lock is
  logged and waited out), then the lock ABSENT for 30 continuous minutes
  (fallbacks, each recorded: no battery lock within 4 h → proceed; lock never quiet within 10 h →
  proceed) → `run_navsim_refcv7.py --splits warmup,navtest,navhard --device auto`.
* **Verdict from artifacts:** `MILESTONE_SUMMARY.json` (with this md5), `summary_{warmup,navtest,
  navhard}.json`, `BARS.json` → `DONE` / `INCOMPLETE` + the missing artifacts in `waiter_5000.json`.
* **Budget (MEASURED throughput):** CUDA native 0.35–0.42 s/scene ⇒ bridges ≈ 4.4 GPU-hours (the lock
  is released between splits); CPU scorers after each split, ≥ 8 GB free per launch, ≤ 4 at once,
  RAM-guard aborts retried ×8 (navhard ≈ 3.2 h per arm, run in parallel).
* **Manual command** (same thing, by hand):
  `C:/Users/Admin/venvs/tanitad/Scripts/python.exe D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/code/run_navsim_refcv7.py --ckpt D:/refcv7_eval_kit/ckpt/ckpt_5000.pt --md5 <md5 from MD5SUMS> --splits warmup,navtest,navhard --device auto`
  (then `code/bars7.py --milestone raw/milestones/step5000`). For 15k / 30k: `milestone_waiter7.py 15000`
  etc.; the FINAL has no waiter yet (the trainer's final `ckpt.pt` needs a done-marker rule, as refcv6's
  `--fetch-final` had).

## 5. Step 5,000 — the first RESULT milestone (section written 2026-10-04 from the banked artifacts; the milestone finished 2026-09-28 20:53 Berlin)

`ckpt_5000.pt` md5 `06eb9dfde9f3783a782cebef22259ce4`; **all three splits CUDA bf16, native trunk path**
(K0 PASS, KD FAIL → exact dedup dropped; `MILESTONE_SUMMARY.json`). Stamps: NavSim open-loop benchmark
(T1-family), ⛔ never closed loop, zero-shot, non-parity; model trained from the launch tree, on which the
speed ceiling does **not** reach the emitted plan (SPEC amendment A1; SPEC_REFCV7 §26.1 — INHERITED from
the brief). Estimator for every interval: paired log-cluster bootstrap (`navsim_ci`, B 2000, seed 0) —
it answers *"another draw of LOGS?"* only; inference variance = the R7_A1_s1 floor; training variance
UNTESTED (one run). All numbers MEASURED: `raw/milestones/step5000/{BARS,summary_*}.json`.

| bar (SPEC §5) | values | margin | interval | inference-seed floor | verdict |
|---|---|---|---|---|---|
| **BAR-R7-N1** navtest PDMS ×100, 12,146 tokens | R7_A1 **65.5976** · STOP 61.8202 · CV 20.6517 · HUMAN 94.5514 | **+3.7774** | A1 − STOP **[+1.58, +5.86]**, separated (log_name AND nuplan_drive) | \|A1 − A1_s1\| 0.0849 | **PASS** |
| BAR-R7-NW1 warmup S2-EPDMS-u, 204 tokens | R7_A1 0.4487 · STOP 0.5212 · CV 0.3971 · ECHO 0.4287 | −0.0726 | none (7 logs < 8) | 0.0084 | **FAILED** |
| BAR-R7-NH1 navhard official two-stage EPDMS | R7_A1 0.1624 · STOP 0.2985 · CV 0.1148 · ECHO 0.1429 | −0.1361 | A1 − STOP [−0.1713, −0.1027] | 0.0055 | **FAILED** |

Beside the navtest bar, never instead of it: A1 − PRIOR_ha0p **+6.9364** [+5.01, +8.75] (what the learned
residual adds over its own kinematic prior); A1 − CV +44.946; **R7_CEILDECL_d − A1 +0.07** [−0.00, +0.14],
not separated (the declared ceiling, applied at the E9 argmax, is worth < 0.1 PDMS here); stop fraction
0.87 % (4 s endpoint < 1 m), median 4 s distance 20.13 m. 200-token diagnostics: VMAXOFF − A1 +0.42
[−0.01, +1.14] n.s.; FILTOFF ≡ A1 (bit-identical, amendment A1); VMAXORACLE − A1 +0.09 n.s. navhard:
A1 − PRIOR +0.0324 [+0.0019, +0.0627] (log_name interval excludes 0; the SPEC conjunction does not
separate), CEILDECL − A1 +0.0009 n.s. ⚠️ The navtest PASS is a 1-training-seed result (SPEC §4).

## 6. Step 30,000 — the runner was killed part-way; completion RUNNING (status 2026-10-04 ~01:30 Berlin)

`ckpt_30000.pt` md5 `ac4e4fab87e35b20de24d8d94910d910`; config md5 `e6512a01…` (unchanged).

**6.1 What killed it — two deaths, neither visible in its own logs (MEASURED: Windows System log, artifact mtimes).**
* **2026-10-02 12:46–12:50 Berlin — USB-storage resets** on the external drive then lettered D:
  (`UASPStor` 129 ×14, `disk` 153 ×5). The **navtest MAIN bridge died at 12:50** at R7_A1 row
  **11,074 / 12,146**, with no traceback (its log was on the resetting drive); its rate had already
  fallen from 2.58 s/scene (cumulative, row 9,400) to 3.93 (row 11,050) ≈ 11.6 s/scene over the last
  1,650 rows. R7_A1_s1 never started.
* **2026-10-02 16:55:03 Berlin — a user-initiated restart** from the Start menu (`User32` 1074; boot
  16:55:31). Every D: artifact stops at 16:55:06; the navhard PRIOR_ha0p scorer was at stage-2
  scenario 4,281 / 5,462. The drive came back as **E:** (`subst D: E:\` since, per the Master Mind).

**6.2 `NO_SEAM`, and whether it affects validity.** The navtest bridge ran on **CPU fp32** because the
GPU lock was held by `refe-final-navtest-full` for > 3 h (`GPU lock NOT acquired within 10800 s for
navtest -> CPU fp32 for the whole split`) — the runner's designed fallback (one device per split, A3),
not a defect. **`NO_SEAM` means no seam file existed**: `run_bridge7` writes an arm's seam only after its
loop finishes, and the loop was killed — so **no partial seam was ever scored, and no navtest main arm
was ever scored** (the log has no `SCORE navtest:r7s30000_R7_A1` line; the 4 main arms R7_A1,
R7_A1_s1, R7_CEILDECL_d, PRIOR_ha0p have no score directory). The banked rows are clean: **11,074
distinct tokens, all CPU fp32, KPR max 0.0 m, the E9 re-derivation reproduced `sel_idx` on 11,074 /
11,074, last line complete** (MEASURED 2026-10-04). Resuming is valid: a row is a deterministic function
of (checkpoint, arm seed, token) on one device (K0). ⚠️ **What it does change:** the 5k → 30k comparison
straddles devices on navtest and warmup (5k CUDA bf16 native; 30k CPU fp32) → read against the
precision floor KP (step 1,500: same selection 88.2 %, 4 s endpoint median 0.40 m), never as a pure
training effect; `step_compare7.py` flags it.

**6.3 Banked and final now.** warmup — all 9 arms scored PASS (CPU fp32), post-processed 2026-10-04
00:33 Berlin. navhard (CUDA bf16 native) — R7_A1, R7_A1_s1, R7_CEILDECL_d scored PASS on 2026-10-02.
navtest 200-token diagnostics (CPU) — scored PASS.

| warmup, step 30,000, CPU fp32 (S2-EPDMS-u, 204 stage-2 tokens; no interval) | value | vs R7_A1 (W/T/L) |
|---|---|---|
| **R7_A1** | **0.5224** | stop fraction 0.029; median 4 s distance 15.59 m |
| R7_A1_s1 (inference seed 1) | 0.5293 | seed floor \|Δ\| **0.0069** (48/94/62) |
| PRIOR_ha0p | 0.4586 | A1 − PRIOR **+0.0638** (99/62/43) |
| STOP / CV / ECHO | 0.5212 / 0.3971 / 0.4287 | A1 − STOP **+0.0011** (113/32/59) · − CV +0.1252 · − ECHO +0.0936 |
| R7_BLIND / R7_NAVOFF / R7_VMAXOFF / R7_A1NT | 0.2193 / 0.5190 / 0.5265 / 0.5211 | +0.3030 / +0.0034 / −0.0041 / +0.0013 |
| R7_FILTOFF / R7_CEILDECL_d | 0.5224 / 0.5224 | 0/204/0 both — FILTOFF bit-identical (A1); the declared ceiling binds in 90 scenes and changes **no** pick on CPU |

**BAR-R7-NW1 at 30k: NOT PROVEN** — margin +0.0011 over STOP, inside 2 × the seed floor (0.0139)
(read by `bars7.warmup_bar` on `summary_warmup.json`; `BARS.json` is written with the full milestone).
5k → 30k: R7_A1 0.4487 → 0.5224 (+0.0737) **across a device change** — the formal comparison lands as
`raw/milestones/step30000/compare_vs_step5000_warmup.json`.

**6.4 Completion (RUNNING, detached, its own log).** `code/complete_milestone7.py` (new): imports the
runner's own functions; never rewrites an OK seam (an `np.savez` rewrite would orphan the `seam_sha256`
of the PASS score beside it); re-bridges navtest on **CPU fp32** (the banked rows' device, read from the
rows AND the runner's own line) — R7_A1 `--derived` first (1,072 rows left, then the R7_CEILDECL_d and
PRIOR_ha0p seams), then R7_A1_s1 (12,146 rows); RAM-gated (≥ 6 GB sustained 60 s), relaunched up to 3×
if a stage exits without its seams; scores every unscored seam through the runner's ScorePool; post-
processes each split as soon as it is complete; rebuilds `MILESTONE_SUMMARY.json` from the seam
manifests (stamped `reconstructed`); runs `bars7.py` and `step_compare7.py` vs step 5,000 on all three
splits. Log `raw/milestones/step30000/complete.log` (mirror `C:/Users/Admin/qland/work/refcv7/complete_step30000.log`);
terminal marker `ZZCOMPLETE7DONEZZ`.

## 7. Step 50,400 — the FINAL checkpoint: suite launched 2026-10-04 00:32 Berlin

`ckpt_50400.pt` md5 `b418d0fc4a92a6848c246a6a7c50207b` (model-only, extracted from the final rolling
`ckpt_50400_full.pt`, md5 `d5f104ee…` = Thor's `ckpt.pt`; INHERITED from the Master Mind, md5 re-verified
by the runner at START). Arms, splits and arguments identical to steps 5,000 / 30,000 except
`--gpu-wait-s 43200` (COMMS D15). Log `raw/milestones/step50400/runner.log`.
⚠️ **The warmup split runs on CPU fp32, by my error** (COMMS escalation 6): launched from a C: working
directory, pytest printed `PASSED ::test_K0…` and the runner read K0 as failed. `pytest.ini` now pins the
rootdir (MEASURED: node ids carry `tests/` from a C: cwd), so navtest and navhard read K0 correctly.

**Status at hand-over (2026-10-04 ~01:35 Berlin; every line read from the logs named above).**
* **step 30,000** — the navtest R7_A1 resume started 01:07:37 Berlin (RAM gate 6.0 GB; CPU fp32; bridge
  pid 11988) and read 11,474 / 12,146 at ~01:32 (≈ 3.6–3.8 s/scene under contention). The navhard
  PRIOR_ha0p re-score was aborted twice by E1's RAM guard (01:09, 01:26 Berlin) while another stream's
  job held ~15 GB; the pool retries it (≤ 8 tries) once ≥ 8 GB is free. ESTIMATED: R7_A1 + derived seams
  ≈ 02:30 Berlin; R7_A1_s1 (12,146 CPU rows) ≈ 12–13 h more → ≈ 15:00 Berlin; then the 4 navtest
  scorers and `BARS.json` ≈ 16:00 Berlin at the earliest (RAM gates make these lower bounds).
* **step 50,400** — warmup seams complete 01:30:33 Berlin (9/9 OK, KPR PASS max 0.0 on every bridged
  arm; CPU fp32). The runner now waits for the GPU lock, held since 01:10:32 Berlin by the battery's
  `refcv7-g0diag-5000` (21 arms at ~305 s each → ESTIMATED release ≈ 03:00 Berlin). Then navtest on CUDA
  (≈ 3.1 h at step 5,000's measured rate), navhard on CUDA (≈ 1 h), then the scorers (warmup 9, navtest
  4 + 3 diagnostics, navhard 4; ≈ 1.5–2 h if RAM allows). ESTIMATED: `BARS.json` ≈ 09:00–10:00 Berlin.
  ⚠️ **Not yet observed live:** `CUDA controls navtest: K0=True` — the first line to check in
  `raw/milestones/step50400/runner.log` once the lock is acquired (the `pytest.ini` fix is MEASURED by
  `--collect-only` only).
