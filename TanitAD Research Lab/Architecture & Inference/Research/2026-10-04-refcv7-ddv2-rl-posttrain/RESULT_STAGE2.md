# WP-RL stage 2 — DiffusionDriveV2's RL stage ported to refcv7-r101-s0, controls measured, gate inputs ready

**Date:** 2026-10-04. **Evidence class:** MEASURED (ours) unless a line says otherwise.
**Tier:** every number here is T0: training-side, or a model-free known value. No number here is
PDMS, driving performance, or closed loop.

**Pre-registration.** `SPEC_RL.md`. Its sha256 log is `raw/SPEC_SHA256.txt`. v1 was hashed at
11:44:40Z, before any GPU run. A-0 was written after the first human-control failure and before
the TRAIN census was read. A-1 holds the cost.

**⛔ No RL training has run.** Stage 3 waits for the Master Mind's launch-gate PASS token
(PI rule, 2026-09-26).

---

## 1. Headline

1. **The release's RL arithmetic now drives refcv7's own sampler.** Shown on 12 real TRAIN windows
   on Thor:
   * the binding reproduces the deployed sampler **bitwise** (fan and u0, 12/12);
   * the trainer's own forward feeds it the **same** inputs (12/12);
   * the reward's per-tick states equal the decoder's own roll **bitwise** (12/12).
   No existing file was modified. 36 new tests pass, 14 of them mutation cases that go RED.
2. **Identity at step 0 holds.** An RLOFF run at lr 0 exports the cold start bitwise (1131/1131
   tensors). An RL run's export reads DIFFERENT (the must-fail arm).
3. **The registered DAC rule FAILED its own human control, and was replaced before any RL number
   existed.** On 300 TRAIN windows the human's trajectory read DAC = 1 on only **0.890 < 0.95**.
   777 of 867 failing corner-ticks are SAM3's "seen, no map class" ground (code 0). A-0's
   pre-listed rung R2 counts only edge and sidewalk as off-road. It passes at **0.983**, still sees
   a 6 m lateral shift (0.808), and is now the rule.
4. **Cost on Thor:**
   * **30.85 s per optimiser step** at the registered batch of 32 (micro 8), MEASURED over 3 steps.
     The reward is the largest term at 12.4 s/step.
   * ⇒ **K = 2,000 steps per arm** (the paper's optimiser-step count) ≈ **17.1 h of GPU per arm**,
     ≈ 69 h for all four arms. The RL arm stays ≤ 24 h.
5. **What the cold start offers RL.** T0, measured on 8 TRAIN windows (diagnose D4) and on 8–32
   windows per step (the smoke):
   * 40–50 % of anchor groups have a non-zero within-group reward spread, so GRPO has something to
     rank;
   * 1–9 % of chain samples carry a positive advantage after the ≥GT bar;
   * 46–72 % of chain samples fail a constraint (NC or DAC). They receive −1 under the release's
     branch.
   The human scores PDMS-proxy ≈ 0.91–1.0, so the ≥GT bar is demanding.

---

## 2. What was built (all NEW files; nothing existing touched)

| file | what | tests |
|---|---|---|
| `stack/tanitad/rl/ddv2_refcv7.py` | the refcv7 binding: full `_sample` capture (agents, BEV, prior); the x0 pass; deployed-ladder parity; per-tick states; DAC on the 10 cm map (R1/R2 ladder); replayed agent tracks; exact track filter; chunked window scoring; micro-batch loss; derangement | `stack/tests/test_ddv2_refcv7.py`, **36 pass** (dev box CPU and Thor) |
| `stack/scripts/ddv2_rl_refcv7.py` | the driver, subcommand by subcommand (listed below) | exercised on Thor (below) |

The driver's subcommands:

* `windows`: eligibility.
* `census`: model-free known values.
* `diagnose`: D1–D6.
* `train`: segmented. It holds the lock in-process, yields to queued jobs, checkpoints, and resumes.
* `smoke`: in-process gate smoke.
* `export`: the full refcv7 state_dict.
* `identity`.
* `heldout`: the T0 secondary read.
* `compare`: paired episode-cluster bootstrap.

The released arithmetic (`ddv2_rl.py`, pinned bitwise to the release) and the reward
(`pdm_proxy.py`) are **reused unchanged**. They are blob-identical in the launch tree `fec3a0d`
and in the tip `50efa52`.

**Runs on:** a COPY of the refcv7 launch tree on Thor (`/home/nvidia/refcv7_post/rl/tree`) plus
the three files. The model is therefore built by the code that trained it. `_roll_state`,
`_decode_ctrl` and `_sample` are byte-identical to the tip.

---

## 3. The DEVIATIONS (from SPEC_RL §3.4) and the paper-vs-code choices (§3.2), in one table

| id | paper / release | ours | forced by | expected direction |
|---|---|---|---|---|
| D-1 | 20 anchors | 117 v0-conditioned control anchors (468 chains per window) | the model | finer groups, higher cost |
| D-2 | waypoint state x/50, y/20 | residual control Δ on the `ha0_ext_pose` prior | the model | multiplicative noise scales Δ; an anchor at the prior explores only through the t = 8 noise |
| D-3 | 1-layer decoder | 4-stage F3 cascade (query detached), AdaLN, agent and BEV coupling | the model | RL reaches the last stage only; IL reaches all stages (the release averages IL over layers) |
| D-4 | x̂0 clamp ±1 and input clamp ±1 (code-only, A18) | not applied | a ±1 box on a control residual is a ±control_norm bound on residual acceleration, a different physical claim; the deployed sampler clamps nothing | MEASURED: it would bind on **20.1 % (a_lon) / 18.2 % (a_lat)** of FINAL x̂0 coordinates, which would get zero gradient (D-DDV1-CLAMP-1) |
| D-5 | no gradient clipping | max-norm 100 (spike guard) | batch 32 vs 512 | none on a stable run |
| D-6 | 4 s reward | 6 s reward (the whole plan) | plan length | more replay-agent NC/TTC zeros at long range; human control: NC 0.997, TTC 0.983 |
| D-7 | NAVSIM drivable polygons | SAM3 10 cm map, rule R2 (A-0) | no map in PhysicalAI | upper bound on compliance (unseen or unclassified ground ⇒ compliant) |
| D-8 | EP vs PDM-Closed | EP vs the human | no route or PDM-Closed | the human's EP ≡ 1; the bar is harder; EP saturates above the reference, as in NAVSIM |
| D-9 | NC/TTC lane clauses | none | no lane graph | upper bound on NC |
| D-10 | navtrain | refcv7 train split: 507,588 eligible windows / 4,256 clips | data | — |
| D-11 | 10 epochs × navtrain, batch 512 | 2,000 steps × 32 (§6) | one Thor | ~6 % of the paper's sample-visits, at its step count |
| D-12 | 16-mixed | generator fp32, frozen trunk bf16 as trained | — | — |
| PC-1 | release IL = all modes | **paper Eq. 4**: matched anchor, all steps, mean over stages | paper text; the release form collapsed our fan 93 % (H-DDV2RL-2) | the fan is preserved; `--il-form release_all_modes` stays one flag away |
| PC-2/3/4/5 | paper Eq. 7 normalisation / "collision" / no mask / λ 0.1 | release: non-zero-sample mean / NC∨DAC → −1 / ≥GT mask / λ 0.1 vs 1.0 | λ's meaning; the reward exists only in code | — |
| PC-6 | per-epoch LR stepping (warmup no-op) | warmup 10 % + cosine over steps | below one epoch | — |

---

## 4. The binding is the model — D1 / D3 (MEASURED on Thor, `raw/diagnose_train.json`, 12 TRAIN windows)

| check | result |
|---|---|
| D1 deployed-ladder parity (fan, u0) | **bitwise 12/12**, max \|Δ\| 0.0 |
| D1 `traj` ∈ fan | 12/12 |
| D1c the trainer's own `compute_losses_v3` captures the same `_sample` inputs (kv, cond, bank, agents, pad, BEV, prior) | **12/12** |
| D1b G = 4 tiling of one pass (queries independent) | max \|Δ\| 7.2e-7 (GPU kernel shapes; CPU bitwise in tests) |
| D3 per-tick states == the decoder's roll at the slot ticks | **bitwise 12/12** |
| gradient reach (one RL step) | every generator group: `traj_proj`, `time_mlp`, `layers.0–3`, `adaln.0–3`, `cascade.control_heads.0–3` |

---

## 5. The reward's known values

**Unit tests** (`test_ddv2_refcv7.py`) pin the analytic cases and their mutations. SPEC_RL §4
lists them all. Every mutation goes RED.

**On real windows, model-free** (`raw/census_train300.json`, 300 TRAIN windows, seed 0; and
`raw/census_eval40.json`):

| | R1 (registered) | **R2 (chosen, A-0)** |
|---|---|---|
| human NC = 1 | 0.997 | 0.997 |
| human **DAC = 1** | **0.890 ✗** | **0.983 ✓** |
| human TTC = 1 | 0.983 | 0.983 |
| human comfort = 1 | 0.950 | 0.950 |
| the human path shifted 6 m sideways reads DAC = 0 | 0.915 | 0.808 |
| the human path 1.5× faster: EP = 1 | 0.633 | 0.667 (the rest collide with the lead: NC 0, correct) |
| the human path 0.5× slower: mean EP | 0.450 | 0.490 |
| codes under failing human corners | 0: 777, 7: 45, 5: 26, 255: 12 | 7: 43, 5: 23, 1: 2, 0: 1 |

**On the GPU diagnose** (12 windows, R2):
* human NC 12/12, DAC 12/12, TTC 12/12, comfort 10/12;
* human-as-candidate identity 12/12.

A candidate whose front edge was placed on a moving agent reads **NC = 0 on 6 of 7** constructed
collisions. The 7th is UNVERIFIED: a 0.54 m/s approach whose first contact was probably not
frontal, which is not at-fault by the NAVSIM rule.

⚠️ EP saturation is the paper's own reward property and is kept (SPEC_RL §4). SPD is logged on
every candidate at weight 0.

---

## 6. Gate inputs from the GPU smoke (`raw/smoke/`, one process, one lock acquisition of 245 s)

**Setup.** It ran on real TRAIN windows at the paper recipe: lr 2e-4, AdamW wd 1e-4, clip 100,
matched IL, DAC rule R2, G = 4, the 10-label chain at η = 1. Integrity was judged by
`wprl_check.py` from each run's own artifacts. `--selftest` proves each check fires on its own
reintroduced defect.

| gate input | result |
|---|---|
| **Identity at step 0** (RLOFF, lr 0, 2 steps → export) | **1131/1131 tensors IDENTICAL** to the cold start. `param_delta_norm` 0.0 on both steps. |
| …its must-fail arm (RL, 4 steps → export) | **DIFFERENT**: 134 of 1131 tensors (the generator subset) |
| **Resume (G-CKPT)**: RL segmented 2 + 2 (resumed from `ckpt_latest.pt`) vs uninterrupted 4 | **same windows on every step**; reward mean and `param_delta_norm` **identical** at every step; logged loss within 5.0e-8 (float summation order of the log line) |
| **RLOFF** | policy-gradient coefficient **exactly 0.0** on 2/2 steps; parameters moved by the IL term (Δ 0.62 → 0.95); same IL value as RL at step 0 (0.523 m) |
| **RL-SHUF** | the permutations `[6,4,7,0,3,1,5,2]` and `[1,3,5,0,6,2,7,4]` are derangements. The step-0 LOSS VALUE equals RL's: it is a sum over rows and invariant to permuting whole blocks. The GRADIENT differs (norm 1.02 vs 1.18; Δ 0.6346 vs 0.6354). |
| **RL live** | `coef_rl_abs_sum > 0` and positive advantage on 4/4 steps. Gradient reaches every generator group. |
| **DAC live** | candidate DAC mean 0.34–0.51 per step (never ≡ 1); 0 windows with a missing map |
| **human controls in training** | human NC = 1 and DAC = 1 on the smoke windows (I-4 PASS) |
| **I-checks** | **PASS on all 6 runs** (identity, rl_seg, rl_full, rloff, rlshuf, timing32). Each check fired on its own mutation. |
| **cross-process resume** | the CPU dry run on Thor (`dry/tr_seg`): a new process resumed at step 1 from `ckpt_latest.pt` and exported. The lr-0 export read IDENTICAL. |
| **lock yield** (CPU, on a PRIVATE lock file, never the GPU lock; `raw/dry_yield/`) | a dummy job from another session queued on the lock. The trainer RELEASED it at the next step boundary. The dummy got the lock in the same second, held it 15 s, and the trainer RE-ACQUIRED after 13 s and finished its 3 steps: one run hash, same windows. Repeated on the final driver (md5 `b46d4014…`) with the same result. |
| **forward identity** (BASE vs the identity export, 1-in-40 eval windows, T0) | QUEUED behind the refcv8 jobs on the shared lock. Lands in `/home/nvidia/refcv7_post/rl/smoke/heldout_{base,identity}_s0_stride40.json`. The tensor-level identity above already implies it. |

**Cold-start training statistics.** These come from the RL smoke. They are T0 and describe 8–32
windows, not the corpus.
* 1–2 % of chain samples carry a positive advantage after the ≥GT bar.
* 57–72 % fail a constraint and get −1.
* Chain endpoint spread is 27–41 m.
* Grad norm is 0.6–2.5. The clip of 100 never bound.

---

## 7. Measured cost and the step budget (SPEC_RL A-1)

| measurement (Thor, MEASURED) | value |
|---|---|
| one RL step, micro 8 (`diagnose` D4) | 6.76 s / 8 windows; peak 5.53 GB (`max_memory_allocated`) |
| **timing32** (B = 32, micro 8, 3 steps, with optimiser) | **30.85 s/step**: capture 6.9, rollout 2.2, reward 12.4, grad 9.5 |
| model + 4,369-clip dataset build (CPU, once per process) | 4–9 s + 210 s |

* **K = 2,000 steps per arm** (A-1 condition 30.85 ≤ 43.2 s holds) ⇒ **≈ 17.1 h of GPU per arm**,
  ≈ 69 h for the four arms.
* Warmup is 200 steps, then cosine to 1e-6.
* A checkpoint every 25 steps is ≈ 13 min. A segment is ≤ 45 min, after which the trainer yields
  to any queued job.
* Each arm reads **64,000 TRAIN windows** (12.6 % of the 507,588 eligible; none twice). That is
  ≈ 6 % of the paper's sample-visits, at its step count.
* **Evaluation reads (stage 3):**
  * T1 battery roll, os-only, S2: ≈ 40 min per checkpoint per inference seed.
  * T0 heldout at stride 8: ≈ 15 min.
  * Five checkpoints × 2 seeds ⇒ ≈ 9 h.
* ⚠️ **The shared lock is the real schedule risk.** On 2026-10-04 afternoon four refcv8 jobs were
  queued at once, and this smoke waited **44 min** for a 4-min slot. Behind a busy refcv8 queue the
  arms take longer in wall-clock than in GPU time. They never take the GPU from a queued job for
  more than a 5-min minimum: they yield after that whenever anyone waits.
* **Cost lever:** the reward is 40 % of a step (per-window Python over `pdm_proxy`). Batching it
  across windows could save ≈ 5 h per arm. It is not done here. It would need a bitwise
  score-equality test and a re-gate.

---

### 6.1 Which code version produced which evidence (stated, not glossed)

* The GPU smoke (`smoke2`, 13:53–13:57Z) ran driver md5 `816aae32…`. The FINAL driver is
  `b46d4014…`.
* The difference is confined to the lock YIELD path:
  * `lock_holders()` reads `/proc/locks`;
  * the hand-over waits until the yielded-to job holds the lock;
  * a 120 s phantom-descriptor timeout.
* The final version was re-verified by the CPU yield test (twice) and by the 36 unit tests on
  Thor. The RL step, reward, checkpoint and resume code are byte-unchanged since the smoke.
* `ddv2_refcv7.py` (`701638df…`) and `test_ddv2_refcv7.py` (`b049a992…`) were final at smoke time.
  The test file only gained the EP/comfort controls after it.

## 8. What is NOT claimed

* No RL effect, no PDMS, no driving claim. Nothing has trained.
* The T0 cold-start statistics (§1 item 5) describe 8 windows. They are not a corpus estimate.
* The PhysicalAI ego box (Pacifica numbers) stays UNVERIFIED, as before.

## 9. Deliverable manifest

See `LANDING_READY_WPRL.txt`.
