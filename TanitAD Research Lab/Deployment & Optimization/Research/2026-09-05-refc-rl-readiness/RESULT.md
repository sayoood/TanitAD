# RESULT — REF-C RL readiness, RESUMED: the experiment is RUNNABLE-WITH-FIXES, the fixes are applied and staged, the cost is measured, nothing is launched

`TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-refc-rl-readiness/RESULT.md`
Production & Optimization (Deploy) FlyWheel · 2026-09-05 · dev-box 4060 only (STAGE 0 preflight + two probes; no arm trained) · Thor untouched except an idle check · ⛔ `tanitad-refcv3` (refcv4b live) never touched · PI, verbatim: *"can we run the rl experiments of refc to see the effect, the training fly wheel agent prepared this in the past."*
Register (same turn): `D-RL-READY-1`, `D-RL-REWARD-FLOOR-1`, `H-RL-MIN-1` (new); cites `D-RL-REFCV3`, `D-RL-PILOT-RC21`, `H-RL-THRESH-1/2`, `H-RL-REACH-1`, `D-REFC-DDAUDIT-1..6`, `H-DDA-3` (not duplicated).
Evidence classes: **MEASURED** (ours, artifact path) · **INHERITED** (another WP's measurement, not re-run) · **PUBLISHED** (banked primary) · **ESTIMATED** · **HYPOTHESIS**. Tiers: every training-side number here is **T0** or an instrument probe; the only T1 numbers are the base's registry rows, quoted as INHERITED.

**status: ANALYSIS DONE; BANKING PARTIAL (mount flapping 2026-09-05 ~04:35–05:06+ local, Drive client restarted 04:52:56 by someone else, still flapping; a bounded retry loop `raw/commit_retry_loop.sh` was left running until ~06:10).** The three register rows (`D-RL-READY-1`, `D-RL-REWARD-FLOOR-1`, `H-RL-MIN-1`) are **IN HEAD** (`VERIFY OK` on `Project Steering/GOALS_AND_CLAIMS.md`, commit B). Every other deliverable — this file, `SPEC.md`, the chain, `raw/*` (32 files), `stack/scripts/rl_refcv3_min.py`, `stack/tanitad/rl/rewards.py`, `stack/tests/test_rl_rewards.py`, `taniteval/tools/refcv3_arm.py`, the three D-SAFE-CAL-2 files — is **ON THE G: WORKTREE, md5-verified at copy time**, but **commit A has NOT landed**: `mm_commit.py` failed on `git hash-object` / `read-tree` (`could not open`, `0xC0000006 mount paged out`, `not a git repository`) across 2 + 8 attempts. **One-command re-run: `bash C:\Users\Admin\rl_readiness_wp\raw\commit_retry_A.sh`** (also in `raw/` here) — it re-copies, commits only the named paths (CAS), and prints the VERIFY table. Local authoritative copies: `C:\Users\Admin\rl_readiness_wp\` (WP), `C:\Users\Admin\refcv4b_repo\` (code), `C:\Users\Admin\rl_refcv3_min\` (run dir). ⛔ The launch chain is gated on `LAUNCH_APPROVED=1` and was not run past STAGE 0.

---

## 0. The verdict in ten lines

1. **Verdict: RUNNABLE-WITH-FIXES — and the fixes are applied, tested (RL suite 149/149) and staged.** The chain's STAGE 0 (md5 → fitlist → humanflag → surface → no-future-leak → audit → step-0 divergence → lr = 0 timing) **passes on the dev-box 4060**; STAGES 1–5 wait for approval of the cost in §5. Nothing was launched.
2. The Training FlyWheel's preparation is real: `stack/tanitad/rl/` (8 modules), 8 test files, the P-RC21 pilot line (three readouts, an anchor sweep, a reward decomposition, a threshold sweep, two D-SAFE-CAL preregs), a 78 KB research RESULT with the DDv2 equations, and the DeployFlyWheel predecessor's driver + chain + 8 fit clips + lead block (§1–§2). **None of it was in the repo's `stack/scripts/`** — the driver and chain existed only in a scratchpad and an off-Drive clone.
3. **The predecessor's script assumes surface (b)**: a surrogate Gaussian policy on the emitted offset (`offset·(1+0.1ε)`, G = 4, analytic `logp`), gradient to `core.decoder` minus the selector surfaces — **9,206,032 / 107,032,901 params (8.60 %)**, MEASURED on the real model. It does not assume a denoising density; the DD audit's finding (`D-REFC-DDAUDIT-6`) is stated as a limit, not hidden.
4. Four defects were found and fixed, each MEASURED (§3.1): a **false strict-load refusal** of the published refcv3 weights (a refcv4-b buffer), a **verdict that read a key the paired tool never writes** (structurally NULL), a **static lead** in the reward (a follower "collides" with the lead's t0 position), and a **gate that discarded its measurement** when it fired.
5. The step-0 trust-region gate fired at **9.06e-11 m²** — float-rounding scale against a frozen deepcopy (live-vs-live exactly 0), ten orders of magnitude under the pilot's real mode mismatch (2.774 m²). Rule adopted: absolute tolerance 1e-8 m², value reported (§3.2).
6. **The reward does not flag the human driver** on these clips (contact 0.000, TTC veto 0.047, n = 318 lead windows) — **but it prefers the constant-velocity floor to the human's own future on 78.3 % of them** (§3.3). Its improvement direction is toward `ha0`, which the base's own eval ranks **0.2304 m worse** than the model. This becomes the committed prediction of the pre-registered experiment: **NULL or FAIL-GUARD**.
7. Of the three RL surfaces (§4): **(b)** is the only one runnable on the frozen base today; **(a)** the anchor-confidence softmax is zero code but trains the selector (doctrine-forbidden, ceiling 0.0751 m) and would select toward the floor under this reward; **(c)** a DD-faithful sampler then true GRPO needs a **retrained base** — a training run, not a post-train.
8. Cost, MEASURED at lr = 0 on the 4060: **0.862 s/step, peak 1.78 GB → `rl` 28.7 min + `reg_echo` 28.7 min + `ctrl0` 2.9 min of training, ≈ 2.5–3 h serial with the T1 evals (§5).** Dev box only; Thor supplies 112 clips over the LAN; the training pod is never touched.
9. Escalations (§8): five integration items and three retraction drafts for the Master Mind's numbering, including the predecessor's TRAIN-C13.
10. What this does NOT establish: nothing about whether RL improves refcv3 — that is the experiment, and its two outcomes are committed in `SPEC.md` §6 before any arm.

---

## 1. What the predecessor (DeployFlyWheel, 2026-09-05 00:2x–00:35, died at the API limit) already established — MEASURED unless marked

Located at `C:\Users\Admin\rl_refcv3_min\` (data) and, for the scripts, in this session's scratchpad + `C:\Users\Admin\refcv4b_repo\stack\scripts\` (md5-identical copies `15d6c2ad…`); **not** in the G: repo (`git cat-file -e HEAD:stack/scripts/rl_refcv3_min.py` → absent).

| artifact | what it establishes |
|---|---|
| `rl_refcv3_min.py` (754 lines) | ONE driver, four modes (`preflight` / `fitlist` / `arm` / `verdict`), every analysis-time import paid at start-up (the 2026-08-11 lesson), reusing `tanitad.rl`, `refcv3_arm.py`, `paired_openloop.py`, `openloop_suite.py`; three arms `rl` / `reg_echo` / `ctrl0`; the trainable surface = `core.decoder` minus 9 selector prefixes with a 15-prefix tripwire; a no-ego-future permutation test; the regression-audit power check; a hash-asserted lr = 0 timing |
| `launch_refcv3_rl_min.sh` (137 lines) | six stages, everything past STAGE 0 gated on `LAUNCH_APPROVED=1`; every artifact asserted by bytes + a JSON parse; `--tiers os=T1,…,oracle_sel=T0`; pairing against the BANKED 40,284 dump over the shared `ha0` floor |
| `fitlist_120.txt`, `fit8.txt`, `fit8/` | 120 train-split ids (seed 0, train ∩ eval = 0), 8 clips pulled from Thor (37.8 MB / 1.2 s per clip MEASURED) |
| `fit8_lead_block.npz` + report | built by `build_lead_block_b1.py` v1 (2026-09-02) on 2026-09-04T22:32Z: 1,608 rows / 8 clips — LEAD 339 · NO_LEAD 304 · NOT_STRAIGHT 776 · NO_LABEL 189; 4/8 clips with any LEAD; 0 refusals; per-row GT distance-keeping (`gt_headway_min_m`, `gt_time_gap_min_s`, `gt_min_ttc_s`) against the MOVING lead; leads expressed in the ego **t0** frame (no ego-future encoding — checked in source, `build_lead_block_b1.py:43,60`) |
| `obs_fit8/` + `pull_manifest.json` | the 8 `obstacle.offline` parquets, sha256 each, pulled from HF in 5.5 s |
| `spec_register.json` | ⚠️ NOT RL: it is E-EGOVOC-1's register ops (the predecessor's other WP, banked separately at `…/2026-09-05-ego-input-literature/`) |
| `RESULT_D_SAFE_CAL_2.md`, `dsafe2_analyze.py`, `dsafe2_result.json` (scratchpad only) | the readout of the seed-1 D-SAFE-CAL-2 arms that ran on 2026-08-30 and were never read: **VOID (V1)** — a same-value reproduction rule across DIFFERENT seeds; the harness is fine (REG separates, D reproduces R3 to three decimals). Drafts TRAIN-C13. Stranded on one disk until this WP (banked, §9) |

What the predecessor did **not** get to: a SPEC.md (the chain cites a path that did not exist), a RESULT.md, register rows, running STAGE 0 even once (no `preflight.json` existed), and banking anything into the repo.

## 2. Everything else the Training FlyWheel prepared — found at ≥ 3 locations each

| where | what | state |
|---|---|---|
| `stack/tanitad/rl/` (G: and both clones, md5-identical) | `rewards.py` (7 bounded components, each with its degenerate policy, `FORBIDDEN_REWARD_INPUTS`, `THRESHOLD_CALIBRATION` coverage contract), `advantage.py` (intra-anchor centred "Dr. GRPO", inter-anchor truncated + veto pinned −1, REINFORCE surrogate with optional k3-KL), `anchor.py` (frozen `ReferencePolicy`, L2/L1 divergence; "kl" deliberately absent — no density), `audit.py` (selector disjointness, degenerate panel with INCONCLUSIVE state, coverage honesty), `config.py` (frozen dataclass, refuses `dpo` — no preference pairs; refuses `gt_similarity`+`w_imitation` double count), `posttrain.py` (counters asserted non-zero, done-marker, frozen trunk by default, eval-mode forward), `refcv3_adapter.py` (the surrogate sampler, TRAIN-C3-corrected score-function form) | in repo; **149 tests pass** (`stack/tests/test_rl_*.py`, 8 files) |
| `stack/scripts/` | `rl_pilot_refc21.py` (the v2.1 pilot driver), `rl_pilot_join.py` (episode ↔ obstacle.offline join), `rl_a0_coverage.py` (the reward-coverage launch gate) | in repo |
| `TanitAD Research Lab/Architecture & Inference/Research/2026-08-29-rl-posttrain-library/` | `RESULT.md` (78 KB: DDv2's objective written out — *its reward is never published*; DDv2 = DDPO_SF + GRPO's advantage − the PPO surrogate; the reward-design section; the method comparison for OUR site; the five mandatory controls), `LAUNCH_PLAN.md` (0.054 s/step, 0.56 GB peak on the 4060 for the mechanism check), `PREREG_P_RC21.md` (+2 amendments), `RESULT_P_RC21{,_SWEEP,_RERUN}.md`, `PREREG_D_SAFE_CAL{,_2}.md`, `RESULT_D_SAFE_CAL.md`, `code/` (18 scripts), `raw/` | in repo |
| `products/P4-training-pipelines/METHOD_LIBRARY.md` §2.1–2.2, §3.4 | GRPO/RLOO **dominated at the SELECTOR** (exact expectation, differentiable); the score-function estimator earns its keep only at the decoder under a non-differentiable reward | in repo (INHERITED) |
| `Project Steering/GOALS_AND_CLAIMS.md` | `D-RL-REFCV3` (IN PROGRESS), `D-RL-PILOT-RC21` (EXIT C → cause found), `H-RL-THRESH-1` (SUPPORTED), `H-RL-THRESH-2` (REFUTED worry), `H-RL-REACH-1` (partially supported) | in repo |
| `Project Steering/RETRACTION_LOG.md` | TRAIN-C3 (estimator gradient direction), C4 (audit blind at dt), C5 (readout in train mode, +175.7 %), C6/C12 (rule names the wrong object), C7 (reach on `offset`), C8, C10 (`$?` after a pipe), C11 | in repo; **TRAIN-C13 drafted by the predecessor, never logged** |
| git (G:) | 11 commits `14fb274` … `4e38ff6` (library → pilot → anchor → sweep → decomposition → d_safe) + `b5a9a2f` (9 primaries banked) | history |
| `C:\Users\Admin\tanitad-data\rl-pilot\` | the pilot arms' checkpoints + readouts (`dsafe-*`, `d2-*`, ~420 MB each) — ⚠️ single-disk | dev box only |
| absent | no `PREREG_*` for RL in `Project Steering/` (the RL preregs live in the research dir); `BACKLOG.md` 0 RL lines; the 2026-08-03 Production & Optimization charter predates the 2026-08-29 RL directive (0 RL lines) | MEASURED-ABSENCE, 2 probes each |

## 3. The assessment

### 3.0 Component by component

| component | state | evidence |
|---|---|---|
| **policy surface** | (b) surrogate Gaussian on the emitted offset — `offset_g = offset·(1+σε)`, σ 0.1 multiplicative, G 4, `logp` = Σ Gaussian log-density over (S = 8, 2), differentiable in `offset`; trainable **9,206,032 / 107,032,901 (8.60 %)** of `core.decoder`; **excluded and recorded: 4 selector tensors** (`conf_head` + grafts); tripwire on `scorer`/`phi_tac`/goal heads/`core.encoder` — preflight asserts no non-decoder and no selector tensor requires grad. ⛔ NOT DDv2's denoising density (`D-REFC-DDAUDIT-6`) — stated | `raw/preflight.json` `surface` — MEASURED |
| **checkpoint** | `ckpt_step40284_frozen.pt` md5 **`b1ed7075ff730d0993d2eaa3c86f6b56`** = registry §4.5; `model` dict bitwise-identical to `ckpt_40284_FINAL.pt` (INHERITED, registry); step 40284 asserted; local copy at `run_refcv3_viz/ckpt/`; HF `Sayood/tanitad-refc-v3` | MEASURED md5 |
| **reward** | `DEFAULT_WEIGHTS` (progress 0.30 v0-referenced · collision 1.00 · headway 0.30 graded · feasibility 0.50 · comfort 0.20); veto = contact ∨ TTC < 1.5 s outside the advantage; ctx = {v0 at t0, the lead's own track in the t0 frame, lead length}; **no ego future** — permutation test PASS; regression audit FLAGS `HACKABLE_WEIGHTS` at the scored geometry PASS; selector disjointness structural. Echo traps: SPEC §3.1. **humanflag: §3.3** | `raw/preflight.json` `no_future_leak`, `reward_audit`; `raw/humanflag_fit8.json` — MEASURED |
| **group / advantage** | intra-anchor centred (no std — Dr. GRPO), inter-anchor truncated `clamp(min 0)` with vetoed candidates pinned −1, composite `w_intra 1 + w_inter 1`; G = 4 (G < 2 refused) | `advantage.py`, tests — MEASURED (tests) |
| **KL / reference constraint** | NO KL (no density to take it against; `kl_coef` exists, off, k3 estimator); the trust region is an **L2 penalty between the live and the frozen-deepcopy mean fans on the same inputs** (`w_anchor` 1.0, full 8-slot fan); the pilot's sweep measured it monotone (drift +44.0 % at w 0 → +4.5 % at w 10, INHERITED) | `anchor.py`; `RESULT_P_RC21_SWEEP.md` — INHERITED |
| **tests** | 8 files, **149 passed** in 3.09 s (clone, `PYTHONPATH=<clone>/stack`); +1 this WP (`test_collision_time_aligned_against_moving_lead`) | MEASURED |
| **eval chain** | `openloop_suite.py` (flags verified against its argparse: `--with-oracle-sel --tiers --corpus --parity-status --train-eval-disjoint --strict --expect-step --lead-block …` all present), `paired_openloop.py` (`--a-dump/--a-arm/--b-dump/--b-arm/--floor/--n-boot/--seed/--out/--md`), `build_lead_block_b1.py` (`--clips --ego-tar --ts-tar --obs-dir --pull --chunk-map --keys --v2ep-dir --k --dt --out`), `tools/criteria_check.py` + `CRITERIA_REGISTRY.json` — all present in `refcv4b_repo` (the full off-Drive clone; `refcv4b_suite` has only `stack/`) | MEASURED |
| **corpora** | eval: 141 v2ep + manifest, labels md5 `aa12c948…`, lead block 29,556 rows / 8,341 with lead; banked base dump 4,823 win / 141 eps; fit: 120 ids, 8 local, 112 on Thor (4,714 files in its cache, 143 GB free); train labels md5 `0ff90213…` (the harness WARNS it is not the eval blob — correct: the fit clips are train-split) | MEASURED |
| **machines** | dev-box 4060 (8,188 MiB); Thor idle (only `tail -f`, GPU 0 %); pod untouched | MEASURED |
| **what is missing** | a stochastic policy with a density (surface (c)); a NAVSIM/PDM reward (rule proxies only); the STRATEGIC family (refused n = 0, as for the base); T2 (not provisioned); a paired `oracle_sel` (refused by design → fan quality from the T0 `R_ORACLE` readout); one seed | stated |

### 3.1 Four defects found in the prepared chain, each MEASURED, each fixed

| # | defect | how it showed | fix (where) | verified |
|---|---|---|---|---|
| 1 | **False strict-load refusal of the published weights.** refcv4-b registered `core.decoder.anchor_controls` as a PERSISTENT buffer (`refc.py:1168`), read only by `roll_bank` when `v0_conditioned`; refcv3 @ 40,284 predates it → `missing_keys: ['core.decoder.anchor_controls']` → `load_model` refused | STAGE 0 run 1 died at model load (`stage0.log`) | `taniteval/tools/refcv3_arm.py::load_model` tolerates exactly that key when the decoder is not v0-conditioned, records `tolerated_inert_buffers` in provenance, stays strict for everything else (`raw/patch_1_*.py` A) | STAGE 0 run 2 loads; `refcv3_arm` provenance carries the key |
| 2 | **Verdict read a phantom schema.** `mode_verdict` read `pr["families_paired_deltas"][k]["sep"/"better_is"]`; `paired_openloop.py` writes `rec["families"][FAM]["metrics"][KEY] = {delta, lo, hi, separated, verdict, …}` and never those keys → `fam = {}` → every run exits **NULL** (the false-green class, here a false null) | source read of both files | verdict rewritten on the real schema with `lower_is_better` from `TRAJ_METRICS`; VOID if the record is void or carries no family metric; fan quality from the driver's own paired `R_ORACLE` (the paired tool refuses `oracle_sel` as a model arm by design) (`raw/patch_2_driver.py` §8) | compiles; `--help`; schema cross-read |
| 3 | **Static lead.** `reward_ctx` held the lead's t0+0.2 s sample fixed over 2 s, so contact/headway/TTC compared the ego's future against a ghost: a follower at time gap < 2 s "collides" with it | unit test (a 10 m/s follower 9 m behind a 10 m/s lead: static → −1, moving → 0); on the fit8 clips it happened not to bite (§3.3); ⭐ and the preflight's OWN audit context reproduces it — its sane 10 m/s reference against a STATIC lead 15 m ahead scores **0.0** (contact), so `default_with_lead` reads FLAGGED (teleport 0.65, frozen 0.225 beat it) — recorded, not gated (`raw/preflight.json` `reward_audit`) | `rewards._collision` time-aligned branch when `lead_path` is given without `obstacles` (+ pinning test); driver `--lead-mode track` default, the lead's own 10-sample track resampled to the 0.5 s grid, `static` kept as the recorded comparison | 149/149; `humanflag` measures both |
| 4 | **A gate that raised without its measurement.** Gate 5 fired with no number attached; the cause had to be re-measured by a separate probe | STAGE 0 run 2 log | `mode_preflight` writes the partial report with `PASS: False` and the failure text on ANY raise; gate 5 records the live-vs-live noise floor and window ids (`raw/patch_4_*.py`) | `raw/preflight_FAILED_gate5_exactzero.json` carries the number |

### 3.2 Gate 5 — the step-0 trust-region divergence (MEASURED, dev-box 4060, torch per `raw/probe_step0_divergence.json`)

| measurement | value |
|---|---|
| preflight, windows [82, 285]: live vs frozen deepcopy (`anchor_traj`, max per-candidate mean squared displacement) | **9.06439e-11 m²** |
| same batch: live vs live repeat | **0** |
| probe, 2 other windows, default algorithms: live-vs-live / ref-vs-ref / live-vs-ref / logits max|Δ| / sel agreement | 0 / 0 / 0 / 0 / 1.0; no submodule in train mode; 0 state-dict keys differ |
| probe with `torch.use_deterministic_algorithms(True)` + cuDNN deterministic | identical zeros |
| the pilot's REAL mode mismatch (TRAIN-C5, INHERITED) | 2.774 m² |

Reading: 9e-11 m² is a ~1e-5 m displacement — a 1-ulp-class difference that appears only after `select_trainable` has set `requires_grad` on the live decoder and the deepcopy has re-laid-out its parameters (**HYPOTHESIS**: kernel selection differs with the tensors' allocation/alignment; not measured further because it is ten orders of magnitude below any physical effect and below the exploration noise by ~10¹⁰). **Rule adopted (patch 5): PASS iff `max_divergence_m2 ≤ 1e-8` (≈ 0.1 mm), `exact_zero` still recorded.** A genuine mode mismatch reads m²-scale and still fails.

### 3.3 `humanflag` — the reward against the demonstration, BEFORE any arm (MEASURED, 0 GPU, `raw/humanflag_fit8.json`)

Scored: the human's logged 2 s future (the same `waypoint_targets` construction the `reg_echo` arm uses), the hold-v0 straight path (the `ha0` floor) and a frozen path, under the DEFAULT reward, on every scoreable RL-fit window with a lead — **318 windows / 4 episodes** of the 8 fit clips (1,368 scoreable windows in all). NON-PARITY, T0 instrument probe.

| quantity | static lead (legacy) | track lead (default) |
|---|---|---|
| contact fires on the human | **0.000** | **0.000** |
| TTC veto fires on the human | 0.047 | 0.047 |
| `headway` < 0.5 on the human (reads as tailgating) | 0.075 | 0.075 |
| `headway` mean on the human | 0.739 | 0.729 |
| composed reward: human / hold-v0 / frozen | 0.864 / 0.889 / 0.225 | 0.861 / **0.899** / 0.225 |
| **hold-v0 ≥ human** | 70.8 % | **78.3 %** |
| frozen ≥ human | 27.7 % | 27.7 % |
| by the human's own time gap (moving lead, block): [2, 3) s n 20 · [3, 5) n 66 · [5, ∞) n 108 | TTC veto 0.15 / 0.167 / 0.0; contact 0 everywhere | same |
| time-gap quantiles 5/25/50/75/95 % | 2.93 / 3.56 / 5.80 / 8.58 / 13.26 s | |
| v0 quantiles 5/25/50/75/95 % | 0 / 0 / 2.33 / 4.28 / 6.46 m/s | |

Two readings, kept apart:
* **The H-RL-THRESH-1 class is CLEAR on this corpus** (PASS at the 0.15 ceiling): the reward does not assert that competent driving is unsafe. The static-lead defect is real in construction but does not bite here because no human window follows closer than 2.93 s and the corpus is slow (median v0 2.33 m/s; 25 %+ of lead windows at v0 = 0) — a lead 20 m ahead is not reached in 2 s. ⚠️ It must be re-measured on the 120-clip fit set (the chain does it in STAGE 0) — a faster or denser corpus would move it.
* ⭐ **The reward prefers the trivial floor to the demonstration on 78 % of lead windows.** Feasibility (0.50) and comfort (0.20) are maximal for a straight constant-speed path and progress (0.30) reads exactly 1.0 for it, so hold-v0 scores 0.899 against the human's 0.861; the human's own accelerations, curvature and jerk COST reward. This is the pilot's finding — *feasibility supplied 87 % of the reward gain by making the fan blander; the reward pays for drift* (`RESULT_P_RC21_SWEEP` §3, `RESULT_P_RC21_RERUN` §2, INHERITED) — measured at the demonstration level, before a single gradient step, on refcv3's own data. Since the base's eval ranks `ha0` **0.2304 m worse** than the model (`os` − `ha0`, separated, INHERITED registry §4.5), the reward's improvement direction points away from the model's current quality; the trust region will resist it; the predicted outcome of the honest arm is **NULL or FAIL-GUARD** (`SPEC.md` §6, committed). A reward that does not rank the constant-velocity path above the human is the precondition for expecting anything else — and that is a reward-design question, not an RL-machinery one.

## 4. The three RL surfaces, priced

| surface | what moves | cost to first number | ceiling / risk | runnable on the frozen 40,284 base? |
|---|---|---|---|---|
| **(b) Gaussian perturbation on the offset head** — the predecessor's script | `core.decoder` minus selector (9.21 M params); the fan geometry | §5 (MEASURED): STAGE 0 done; arms + eval ≈ the numbers in §5 on the 4060 | predicted NULL / FAIL-GUARD (§3.3); the method verdict transfer is the value | **YES** — this SPEC |
| **(a) policy gradient over the anchor-confidence softmax** (a categorical over N = 128 `sel_score`) | `conf_head` (385 params) + the graft gates — the SELECTOR; exact expectation Σ softmax·r, no sampling (METHOD_LIBRARY §2.1: sampling here only adds variance) | 0 new code beyond a 30-line objective; minutes on the 4060 (one fan reward per window, same as R1) | ceiling = the selection gap **0.0751 m [0.0618, 0.0884]** (`oracle_sel` − `os`, INHERITED); cannot move the fan; ⛔ doctrine: the library tripwires selector training (DDv2's named defect is selector over-reliance), and under a reward that ranks hold-v0 above the human (§3.3) it would select toward the floor. Admissible only after the reward's direction is fixed, and then as a SEPARATE pre-registration with its own echo gate (selected-path ADE vs oracle-in-fan unchanged) | yes, but not recommended |
| **(c) DD-faithful sampler (`H-DDA-3`) then true GRPO with the step density** | a new decoder path: anchored Gaussian start, DDIM 1000/scaled_linear, x0 prediction, `DDIMScheduler_with_logprob`-style step log-prob (DDv2 `diffusiondrivev2_model_rl.py`, PUBLISHED) | (i) sampler behind a flag ~1 dev-box hour (sibling's ESTIMATE, `…/refc-vs-diffusiondrive-audit/RESULT.md` §4 rank 3); (ii) v7-tiny ladder validation ~29 min/arm on the dev box × 4 arms (ValidateAIDesign §2, INHERITED); (iii) ⛔ **a RETRAINED base** — the 40,284 weights were never trained to denoise scheduled noise, so the sampler cannot be bolted onto them; the honest cost is a 40k-step training run of refcv3's class (its own run took the A40 pod from 2026-09-0x to 09-04; wall-clock per step not banked in the registry — ESTIMATED days, not hours); (iv) the GRPO stage itself ≈ (b)'s cost with a 10-step training rollout (DDv2 `step_num = 10`) ≈ 5× per step | the only surface that is DDv2's mechanism; it is also the H-DDA-3 experiment the DD audit already ranked 3rd | **NO** — needs a new base; it is where the RL question goes if (b) reads NULL |

## 5. Cost — MEASURED at lr = 0 on the dev-box RTX 4060 (`raw/preflight.json` `timing`; 20 steps, batch 2, G 4, 2 warm-up steps excluded; weights hash-asserted UNCHANGED before/after)

| item | value | class |
|---|---|---|
| step time, total (data + compute) | **0.862 s/step** (data 0.073 s + compute 0.789 s) | MEASURED |
| peak GPU (in-process `max_memory_allocated`) | **1.78 GB** of 8.19 | MEASURED |
| gradient reaches | 71 / 71 trainable tensors | MEASURED |
| `rl` 2,000 steps | **28.7 min** | derived |
| `reg_echo` 2,000 steps | **28.7 min** | derived |
| `ctrl0` 200 steps | 2.9 min | derived |
| before/after T0 readouts, 120 windows × 2 × 3 arms | ~3 min | ESTIMATED (DD audit: 201 windows × 6 configs in 67 s) |
| T1 eval: `openloop_suite.py` × 3 arms on 4,823 windows / 141 clips | **~25–40 min per arm** (~1.5 h) | ESTIMATED from the same 67 s / 201-window figure with 3 nav conditions; the suite's dev-box wall-clock is not banked — STAGE 4 measures it |
| pairing × 3 (CPU) + verdict | minutes | ESTIMATED |
| clip pull 112 × 1.2 s + fit lead block | ~4 min | MEASURED rate (predecessor) |
| **total** | **≈ 2.5–3 h of the 4060, serial**, 0 pod-hours, 0 Thor GPU | derived |

⛔ **Not launched.** `LAUNCH_APPROVED=1` is the Master Mind's / PI's approval of this figure; `launch_refcv3_rl_min.sh` exits 0 after STAGE 0 without it.

## 6. The pre-registered experiment — `SPEC.md` (ValidateAIDesign §1 schema; hypothesis `H-RL-MIN-1`)

One variable (RL stage on/off on the frozen base), four arms (base = the banked dump · `rl` · `reg_echo` the deliberate regression whose reward IS the echo · `ctrl0` lr 0), four families at T1 through the criteria-checked harness paired against the banked base over the shared `ha0` floor, the echo gate G-FAN (must fire on `reg_echo` or VOID), the ADE guard, the fan-quality guard (`R_ORACLE`), the mechanical verdict in a committed order, and the committed prediction **NULL or FAIL-GUARD** with what each outcome decides. See `SPEC.md` §4–§6.

## 7. Fix list — applied vs remaining

**Applied (clone `C:\Users\Admin\refcv4b_repo`, staged to G: per §9):** (1) harness inert-buffer tolerance; (2) time-aligned moving-lead contact + test; (3) driver `--lead-mode track`, `R_ORACLE`, `--mode humanflag`, verdict on the real schema; (4) preflight partial report on failure + noise-floor record; (5) gate 5 tolerance 1e-8 m²; (6) launch chain: `humanflag` gate in STAGE 0, `--lead-mode` on every call, work dir pinned, SPEC path real.

**Remaining before an approved launch (none blocks STAGE 0):** (r1) STAGE 1 pulls 112 clips from Thor over the LAN — `scp` in a loop; probe `ssh -n … 'echo OK'` first (the CLAUDE.md mapping trap); (r2) STAGE 2 builds `fit120_lead_block.npz` with `--pull` (needs `Keys.txt` in the clone — present); (r3) the STAGE 4 suite wall-clock is ESTIMATED — the first arm measures it, and the chain should be run with `ARMS=(ctrl0)` first to price everything on the cheapest arm; (r4) the T0 readout uses 120 EVAL windows for R_FAN/R_ORACLE — n is small for a 30 % collapse test; if G-FAN does not fire on `reg_echo`, widen `--readout-windows` before calling V2 (it is a monitor, not the T1 read); (r5) `openloop_suite --strict` refuses on any registry/criteria mismatch — run `--analyze-only` on the base dump once in the clone before STAGE 4 so a harness drift fails before the arms are paid for.

## 8. Escalations (integration requests — not written into a doc for someone to find)

1. **EvalFlyWheel / harness owner:** `refcv3_arm.load_model` now tolerates one inert buffer (fix 1). Without it **no dev-box eval of the published refcv3 weights can load** at HEAD — every `openloop_suite`, `paired_openloop`, `stratified_openloop` run on the shipped checkpoint hits it. Alternative if a harness-side allowance is unwanted: make `anchor_controls` persistent only when `v0_conditioned` in `refc.py` — but that flips the failure onto interim checkpoints saved WITH the zero buffer, so the harness-side rule is the safer one. Decide, and add a test that loads the HF refcv3 config against HEAD.
2. **TrainingFlyWheel / library owner:** `rewards._collision` gained the moving-lead branch (fix 2). It changes nothing for callers that pass `obstacles`; the pilot's `rl_pilot_refc21.py` passes `obstacles` and is unaffected.
3. **Master Mind:** the driver + chain + this WP need to land in the repo (`stack/scripts/rl_refcv3_min.py` is NEW; the predecessor's copies existed only off-Drive). Committed via `mm_commit.py` this turn if the mount holds (§9); if the length-guarded check is INCONCLUSIVE, re-verify.
4. **Master Mind:** the D-SAFE-CAL-2 readout the predecessor wrote (VOID V1) is banked into the 2026-08-29 bank with its analyzer and raw JSON; the `d2-*` and `dsafe-*` checkpoints (10 × ~420 MB) stay single-disk on the dev box — decide whether they are worth an HF private bank.
5. **PI decision (the only gate on the experiment):** approve or decline the §5 cost. Given the committed prediction (NULL/FAIL-GUARD) the honest framing is: *this run buys the transfer of the pilot's method verdict to refcv3 and closes the offset-head-RL line on the real base; it does not buy a capability gain.* If the PI wants the gain question answered, the money goes to `H-DDA-3` (surface (c)) and to a reward that does not rank hold-v0 above the human — both are design work, not launches.

### 8.1 Retraction drafts for the Master Mind's numbering (root-cause CLASS first)

* **TRAIN-C13 (predecessor's draft, unlogged):** a committed reproduction rule asked a seed-1 arm to reproduce a seed-0 number to the same value — *a rule that names the wrong object* (TRAIN-C6/C12 class, third appearance). `RESULT_D_SAFE_CAL_2.md` §3.
* **New — "an analysis reads a key its producer never writes":** the RL verdict read `families_paired_deltas`/`sep`/`better_is`; the paired tool writes `families/…/metrics/…/separated`. Every run would have exited NULL with exit 0 — the false-green class in a false-null costume; same family as the 2026-08-11 analysis-time import that destroyed a completed rollout. Rule: **a verdict reader is tested against a record the producer actually wrote** (a fixture from a real run), never against the reader's own idea of the schema.
* **New — "a persistent buffer added for one variant silently breaks strict loading of every earlier checkpoint":** `anchor_controls` (refcv4-b) vs refcv3 @ 40,284 — the harness refused the programme's own published weights on the dev box. Class: checkpoint-compatibility drift; the registry's "strict load `missing_keys: []`" for this checkpoint was true on the pod's tree and false at HEAD. Rule: a new `register_buffer(…, persistent=True)` in a shared model file needs a load test against the last published checkpoint of every variant that shares the class.
* **New — "an exact-zero gate on CUDA":** gate 5 asserted `== 0.0` and fired at 9e-11 m² on a deepcopy; the pilot passed only because its copy happened to be bit-identical. Rule: a numerical gate carries a tolerance derived from the physical effect it guards (here 2.774 m² vs 1e-8 m²), and reports the value.

## 9. Deliverable manifest — every artifact and WHERE IT LIVES

| artifact | where |
|---|---|
| this report | `TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-refc-rl-readiness/RESULT.md` (repo, committed via `mm_commit.py` — see the final report for the length-guarded verification) ; authoring copy `C:\Users\Admin\rl_readiness_wp\RESULT.md` |
| the pre-registration | `…/2026-09-05-refc-rl-readiness/SPEC.md` (repo) |
| the launch chain (patched) | `…/2026-09-05-refc-rl-readiness/launch_refcv3_rl_min.sh` (repo) = `C:\Users\Admin\rl_refcv3_min\launch_refcv3_rl_min.sh` (the run copy) |
| the driver (patched, NEW in the repo) | `stack/scripts/rl_refcv3_min.py` (repo) = `C:\Users\Admin\refcv4b_repo\stack\scripts\rl_refcv3_min.py` |
| library + harness + test edits | `stack/tanitad/rl/rewards.py`, `stack/tests/test_rl_rewards.py`, `taniteval/tools/refcv3_arm.py` (repo; clone copies identical) |
| raw: preflight (PASS) · the gate-5 failure record · humanflag · divergence probe · the five patch scripts · pre-edit md5s · register rows + insert script | `…/2026-09-05-refc-rl-readiness/raw/` (repo); run copies in `C:\Users\Admin\rl_refcv3_min\` |
| register rows `D-RL-READY-1`, `D-RL-REWARD-FLOOR-1`, `H-RL-MIN-1` | `Project Steering/GOALS_AND_CLAIMS.md` (inserted before `## D-REFCV4B-EGODROP2`, repo) |
| the predecessor's stranded D-SAFE-CAL-2 readout | `…/2026-08-29-rl-posttrain-library/RESULT_D_SAFE_CAL_2.md`, `code/dsafe2_analyze.py`, `raw/dsafe_cal_2/dsafe2_result.json` (repo) |
| the RL-fit data (8 clips 304 MB, lead block, obstacle parquets, fitlist) | `C:\Users\Admin\rl_refcv3_min\` — dev box only (regenerable: Thor cache + `build_lead_block_b1.py --pull`) |
| the base checkpoint copy | `C:\Users\Admin\rl_refcv3_min\base\ckpt_step40284_frozen.pt` (md5-verified copy of `run_refcv3_viz/ckpt/`; HF `Sayood/tanitad-refc-v3`) |
| ⛔ NOT produced | any trained arm, any checkpoint, any T1 number — the chain was not launched |

---

## 10. RE-SCOPE 2026-09-05 (the PI's correction) — STATUS HEADER, rewritten as each arm lands

**status (2026-09-05 09:12 local / 07:12Z): RE-SCOPE STARTED — nothing run yet; this header is banked FIRST (the API limit killed three agents in 36 h).** Deploy FlyWheel, third agent on this package (`a642249` = the readiness package, IN HEAD, verified by `git cat-file -e` on all five paths). ⛔ `tanitad-refcv3` (refcv4b-b1-v72-40k LIVE) never touched. GPUs at start: dev-box 4060 **100 % busy** (the withheld-bank panel, another stream) · Thor `refav1_arm.py` **running 1h46m at 97 %** (expected done ~09:00Z). Rule in force: use whichever frees FIRST, never both; poll ≤ every 5 min; launch only after a real `utilization.gpu` idle reading.

**The PI's correction, verbatim:** *"Regarding RL, I don't agree you can do only what you stated, the paper is talking about improving/eliminating trajectories leading to [collisions / infeasible outcomes]."*

**What it changes.** `H-RL-MIN-1`'s primary endpoint was the four-family distance to the human, under which the constant-velocity floor already beats the human on 78.3 % of lead windows (§3.3) — so the committed prediction was NULL. That endpoint cannot see V2's mechanism. DiffusionDriveV2's RL exists to push probability mass AWAY from collision-prone and infeasible candidates: the collision-truncated advantage (collision → −1; positive advantage only for samples above the ≥GT bar), pinned from the code in `…/Architecture & Inference/Research/2026-09-05-diffusiondrive-v2-analysis/RESULT.md` (`D-DDV2-*`). **The primary endpoint is therefore FAN SAFETY** on the EMITTED fan (top-k and the selected trajectory), before vs after RL on the frozen base: (a) time-aligned contact against the replayed `obstacle.offline` lead track; (b) TTC-below-threshold rate (2.93 s, the `H-RL-THRESH-1` class); (c) infeasible / off-reach fraction (reach band + Kamm load, `stack/tanitad/instruments/flyability.py`); (d) the confidence-head probability mass on (a)–(c). The four families stay as the SECONDARY endpoint with the original prediction retained.

**Plan (each step banked + committed as it lands):** (1) `SPEC.md` §10 re-scope, dated, original kept — with both outcomes committed in advance; (2) code: `advantage.py` gains the ≥GT mask (positive inter-anchor advantage only above the human's own reward on the same window; collision stays pinned −1), `refcv3_adapter.py` gains the two-scalar (along, lateral) noise-scale policy, the driver passes the GT bar and a `fan_safety` readout, a new analysis tool scores (a)–(d) on any `openloop_suite` dump; tests pinned; (3) STAGE 1–2 (clip pull from Thor over the LAN, fit lead block) — I/O only; (4) arms `ctrl0` → `rl` → `reg_echo` on the first free GPU, T1 evals paired against the banked refcv3-40284 dump, fan-safety scored on base and every arm; (5) verdict per the committed outcomes; register rows in the same turn.

| step | state | artifact |
|---|---|---|
| STATUS header banked | **done** (this section) | this file, commit id in the final report |
| SPEC §10 re-scope | pending | `SPEC.md` |
| code + tests | pending | `stack/tanitad/rl/{advantage,refcv3_adapter}.py`, `stack/scripts/rl_refcv3_min.py`, `taniteval/tools/fan_safety.py`, `stack/tests/` |
| STAGE 1–2 (I/O) | pending | `C:\Users\Admin\rl_refcv3_min\fit120\`, `fit120_lead_block.npz` |
| arms + T1 + fan safety | pending — GPU-gated | `raw/run/<arm>/` |
| verdict + register | pending | §11+, `GOALS_AND_CLAIMS.md` |
