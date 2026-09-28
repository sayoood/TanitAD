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
