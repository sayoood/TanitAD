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

---

## 11. ⭐ THE RE-SCOPED RUN — RESULT (2026-09-05, Deploy FlyWheel, third agent on this package)

**Exit: `4 FAIL-SAFETY`** — the V2-faithful RL stage makes refcv3's emitted fan **less** safe, and the
constant-reward control proves the damage comes from **the reward**, not from V2's constraint
mechanism. The committed prediction (**2 NULL-SAFETY, sub-case (i)** — "the fan is already safe,
little room") was **WRONG in the informative direction**: there was ample room, and the stage moved
into it backwards.

All four arms ran on the dev-box RTX 4060 in **28 min 29 s total** (11:45:49Z → 12:14:18Z), after
the sibling stream's withheld-bank panel released the GPU. ⛔ `tanitad-refcv3` (refcv4b-b1-v72-40k
LIVE) was never touched.

### 11.1 Validity gates first — two of the three moved, and both movements were informative

| gate | result | what it means |
|---|---|---|
| **V1 `ctrl0`** (lr 0) | ✅ **PASS** | weights hash-identical (`weights_changed: False`); **0 of 57** fan-safety metrics separated; every R\* readout delta 1e-5…1e-8, none separated |
| **V4 `ctrl_const`** (constant reward) | ⚠️ **FIRES** — re-read as an ATTRIBUTION INSTRUCTION | 14 of 57 separated. ⛔ Not a harness fault: `veto_rate_mean` **0.0897**, `final_loss` **−1.863**. The veto is applied outside the reward by design, so a constant reward gives a **veto-only** advantage, not a zero one. See RETRACTION #24 |
| **V2 G-FAN** (`reg_echo`) | ⚠️ **INCONCLUSIVE** — the arm did not produce the regression it was designed to produce | R_FAN went **+44.03 %** (4.2947 → 6.1856), not −30 %. `reg_echo` did not COLLAPSE the fan onto the logged path; it **diverged**: R3 **+16.72 m**, R_ORACLE **+15.60**, R_REACH **+20.60**, all separated |

⚠️ **On V2, stated plainly rather than argued around.** The gate exists to prove the readout can see
fan collapse. It cannot be said to have done so, because the deliberate-regression arm never
collapsed. What the run *does* establish is that the readout is not blind to a degenerate arm at all:
it flagged `reg_echo` unmistakably and in the obviously-broken direction on R3, R_ORACLE, R_REACH and
51 of 57 fan-safety metrics. ⇒ **Under SPEC §6 V2, no PASS would have been admissible from this
panel. The headline result is a FAIL, so the gate does not rescue or invalidate it** — but a PASS, had
one appeared, would have had to wait for a working collapse arm.

⚠️ **Why `reg_echo` diverged instead of collapsing, and it is a design fault in the arm:** the SPEC
gives it `w_anchor = 0.0`, i.e. **no trust region at all**, while `gt_similarity` is the only reward.
With nothing anchoring the fan, the policy ran away — the same failure the P-RC21 pilot recorded
(R3 1.97 m → 347.2 m, "every candidate left the road"). A collapse arm needs the trust region KEPT
and only the reward swapped. Work item, not a result.

### 11.2 The PRIMARY endpoint — fan safety, `rl` vs base

T0 readout, 120 fixed EVAL windows (seed 1234), 79 episodes / 30 lead episodes, paired
episode-cluster bootstrap n_boot 4,000. **Separation rule, applied identically to every arm:** CI
**strictly** excludes 0 **AND** not all-zero **AND** |delta| ≥ **1e-4** — the floor stated rather than
tuned, one conservative step above 1/(128 × 120) = 6.5e-5, the quantum of one candidate in one window.

| metric | before | after | delta | separated |
|---|---|---|---|---|
| `sel_infeasible` | 0.13333 | **0.64167** | **+0.52743** | yes |
| `sel_ttc_below` | 0.02778 | **0.13889** | **+0.13333** | yes |
| `top32_infeasible` | 0.63385 | 0.71849 | +0.08979 | yes |
| `mass_rank_ttc_below` | 0.04848 | 0.12232 | +0.08839 | yes |
| `top32_envelope` | 0.63073 | 0.71068 | +0.08459 | yes |
| `top32_kamm_over` | 0.49141 | 0.54714 | +0.06138 | yes |
| `fan_infeasible` | 0.89453 | 0.91439 | +0.02034 | yes |
| `top32_contact` | 0.03299 | 0.02691 | **−0.00729** | yes ← the one gain |
| `mass_rank_contact` | 0.00003 | 0.00028 | +0.00029 | yes ← mass ON contact went UP |
| `fan_peak_g_mean` | 4.18090 | 4.11416 | −0.07243 | yes |

**The selected trajectory went from 13.3 % infeasible to 64.2 %**, and from following closer than the
human's own 5th-percentile time gap on 2.8 % of lead windows to 13.9 %. Contact in the top-32 improved
by 0.7 points; the confidence mass placed **on** contact candidates rose.

Supporting T0 readouts, all separated: **R3 sel-ADE +0.4897 m WORSE** [+0.395, +0.590] · **R_ORACLE
(oracle-in-fan — fan QUALITY) +0.1697 WORSE** · R_REACH +0.1945 · **R1 (the composed reward itself, on
held-out EVAL windows) −0.0058** — the stage did not even improve its own objective out of sample.
R_FAN +0.0132 **not** separated, so `rl` did not collapse the fan. Selector agreement with base
**0.4083** — it changed its pick on 59 % of windows.

### 11.3 ⭐⭐ The attribution — and it is the finding

`ctrl_const` (veto only, reward identically 0.0) moved feasibility toward **BETTER**:
`fan_kamm_over` −0.0048, `top32_infeasible` −0.0143, `fan_infeasible` −0.0026, `fan_peak_g_mean`
−0.0859. The full arm moved it **WORSE**. So the two halves of the stage pull in opposite directions,
and `rl` − base attributes nothing on its own.

The contrast that isolates the reward differences the veto out. Both arms start from the same frozen
base and their BEFORE readouts are **bit-identical (max abs diff 0.000e+00 over 120 windows, asserted
in the artifact)**, so it is valid:

| `rl` − `ctrl_const` (isolates REWARD + ≥GT bar) | delta | CI | separated |
|---|---|---|---|
| `sel_infeasible` | **+0.52743** | [+0.41139, +0.63924] | yes |
| `sel_ttc_below` | **+0.13333** | [+0.03333, +0.26667] | yes |
| `top32_infeasible` | +0.10410 | [+0.07720, +0.13199] | yes |
| `top32_envelope` | +0.09738 | [+0.07015, +0.12553] | yes |
| `top32_kamm_over` | +0.07397 | [+0.05281, +0.09586] | yes |
| `fan_infeasible` | +0.02293 | [+0.01520, +0.03118] | yes |
| `fan_off_reach` | −0.00330 | [−0.00582, −0.00099] | yes |

⇒ **DiffusionDriveV2's CONSTRAINT mechanism transfers; our REWARD does not.** The veto —
collision/TTC pinned at −1, V2's actual anti-collision device — improves the fan's feasibility on its
own. The composed reward then pushes it back the other way and overwhelms the gain. Reading `rl`
against `base` alone would have credited or blamed the wrong half; this is the C6-confound family,
and the constant-only control is the only reason it is visible.

### 11.4 Run facts that condition the reading

`veto_rate_mean` 0.0975 · **`frac_above_bar_mean` 0.0456** — V2's ≥GT bar zeroed ~95 % of the positive
advantage, which is the regime V2's own released code lives in, so the update was dominated by the
veto's −1 plus a thin surviving positive mass · `use_gt_bar` True · `noise_mode` two_scalar ·
components fired: progress 2000/2000, feasibility 2000/2000, comfort 2000/2000, headway 1014,
collision 449 · 2,000 steps in **871 s**.

⚠️ **A MECHANISM, OFFERED AS A HYPOTHESIS AND NOT AS A MEASURED CLAIM:** `progress` at weight 0.30
with the reference `v0 × horizon` rewards covering ground, and the cheapest way to cover more ground
in 2 s is to accelerate harder and close on the lead — which is exactly `envelope` and `ttc_below`
going up. Testing it is a **weight ablation**, not another 2,000-step arm. It is consistent with
`D-RL-REWARD-FLOOR-2` (the reward ranks the hold-v0 floor ≥ the human on 72.5 % of lead windows) but
is **not** established by it, and it is not claimed here.

### 11.5 Cost — MEASURED, and far under the SPEC's estimate

| item | SPEC §5 estimate | MEASURED |
|---|---|---|
| `ctrl0` (200 steps) | 2.9 min | **2 min 04 s** |
| `ctrl_const` (200 steps) | — (new arm) | **2 min 07 s** |
| `rl` (2,000 steps) | 28.7 min | **15 min 09 s** |
| `reg_echo` (2,000 steps) | 28.7 min | **9 min 09 s** |
| all four arms incl. before/after readouts | ~63 min | **28 min 29 s** |

0 pod-hours, 0 Thor GPU-hours, no training run disturbed.

### 11.6 Stated limits — none silent

1. **ONE SEED.** No replication. A single-seed negative is weaker than a single-seed positive would
   have been suspicious, but it is still one seed.
2. **NON-PARITY** fit corpus (120 train-split B1 v7.2 clips), as the base itself is.
3. The primary endpoint is a **T0 readout on 120 windows** — the fan the model emits, never a driving
   claim. The T1 four-family read is §11.7.
4. `kamm_over` is a **LOWER bound**: the emitted fan is free waypoints, so the exact control-rolled
   friction instrument cannot be applied and the finite difference used instead under-reports by
   1.21–1.85× (`flyability.py`).
5. Contact is a **sampled** check on 0.5 s waypoints; only the **lead** agent is replayed, so the
   absolute contact level is a floor. Paired deltas are unaffected — the same object before and after.
6. The **echo gate is INCONCLUSIVE** (§11.1). No PASS would have been admissible from this panel.
7. `ctrl_const` is a veto-only arm by accident rather than by design; it happens to be exactly the
   arm the attribution needed, but it was not pre-registered as such.

---

## 12. ⭐⭐ THE FINISHED VERDICT (2026-09-05, Deploy FlyWheel, fourth agent on this package)

**What §12 adds to §11.** §11 banked the arms and the attribution. Three things were missing and are
supplied here: **(1)** an adjudication of the pre-registered `V4` VOID gate — §11 selected outcome
`4 FAIL-SAFETY` *past* a fired VOID gate, which SPEC §10.6's "first match in order" does not permit,
and the correction changes which comparison is quotable; **(2)** the verdict written against the
**committed** outcome text, with the all-zero contact rows QUANTIFIED rather than asserted; **(3)**
the SECONDARY endpoint — the four families at **T1** — which §11.6 forward-referenced as "§11.7" and
which did not exist. It is now measured, over the **full** corpus, at **zero GPU**.

### 12.1 `V4` — the mechanism, and exactly what it does and does not void

**MEASURED, and the mechanism is in the source, not inferred from it.** `ctrl_const` sets every
reward weight to `0.0`, and its weights nevertheless moved (`weights_changed: true`,
`veto_rate_mean` **0.0897**, `final_loss` **−1.863** — not the `0.0` a zero advantage would give).
The cause is two lines that consult the reward's **key set, never its values**:

```
stack/tanitad/rl/posttrain.py:205-209
    veto = None
    if "collision" in spec.weights:                    # KEY membership — true at weight 0.0
        veto = R.COMPONENTS["collision"](traj, ctx) < 0
    ttc = R.ttc_violation(traj, {**ctx, "ttc_min_s": cfg.ttc_min_s})   # never reads weights at all
    veto = ttc if veto is None else (veto | ttc)
```

`advantage.py:136` then centres the reward (identically `0.0` ⇒ advantage `0.0`), and `:146` pins
vetoed candidates at `veto_value = −1.0` **after** the bar, outside the centring — deliberately,
because a constraint pins where a ranking term orders. ⇒ **a constant reward does not give a zero
advantage; it gives a *veto-only* advantage at full strength.** `ctrl_const` is not a broken control;
it is an unintentionally exact **veto-only arm**. Already logged as **RETRACTION #24**, which also
tested and REFUTED the competing explanation (AdamW decoupled weight decay: predicted shrink 0.99998,
MEASURED 0.99999996, and only 47/72 tensors shrank — not the mechanism).

**Does the panel stand? It stands in part, and the split is not negotiable.** The distinction that
settles it is between an **attribution** claim and an **effect** claim:

| claim | status | why |
|---|---|---|
| *"the REWARD moved fan safety by X"* (`rl` − base) | ⛔ **VOID** — `V4`, as pre-registered | Two channels moved in **opposite** directions (§11.3): the veto improved feasibility, the reward worsened it. `rl` − base sums them and attributes nothing. `V4`'s stated consequence — *"no `rl` result above it means anything"* — is not merely triggered, it is **substantively correct**. §11.2's `rl`-vs-base table is therefore **not** quotable as a lever effect, and the exit label `4 FAIL-SAFETY` selected past it is withdrawn (see §12.7). |
| *"the reward, with the veto held fixed, moved X"* (`rl` − `ctrl_const`) | ⚠️ **ADMISSIBLE but POST-HOC** | Not pre-registered. Its validity conditions are met and re-verified by me: the four arms' BEFORE readouts are **bitwise identical** (max abs diff **0.000e+00** across all 120 windows × 57 metrics — I re-checked, I did not inherit it), same windows, same estimator. It is the correct analysis of the structure that was discovered; it is not the analysis that was promised. |
| *"this stage, as configured, produced a worse planner"* (`rl` − base at T1) | ✅ **NOT VOID** | `V4` is about which **internal channel** caused a fan-safety change. It says nothing about whether the **resulting checkpoint** drives worse. That is a before/after on a fixed pipeline, and §12.5 answers it with no attribution surgery required. |

⇒ **The panel does not need re-running for the attribution.** What it needs before any row is quoted
as a *lever* effect is a **replicate** — `H-ESTIM-SEED-1`: none of these three comparisons is
controlled for training variance, because `ctrl0` has `lr = 0` and never trains at all.

⚠️ **`V4`'s wording is the defect to fix, not the control.** *"moved ANY fan-safety metric"* is
untrue of the arm's purpose the moment a constraint channel sits outside the reward by design. The
repair for the next pre-registration is to scope the gate to what it was protecting:
**"`ctrl_const` moved any metric BY A PATH OTHER THAN THE VETO"**, with `veto_rate_mean` printed
beside it — a gate that reads its own mechanism's firing rate cannot be surprised by it.

### 12.2 The verdict, written against the committed outcome text — and it SPLITS by family

SPEC §10.6 is mechanical and first-match, so the formal exit is stated first and without softening:

> ⛔ **FORMAL EXIT: `V4` → VOID.** The constant-reward control moved 14 of 57 fan-safety metrics with
> paired separation. Every downstream fan-safety exit (1 PASS / 2 NULL / 3 TRADE-OFF / 4 FAIL) is
> reached only *after* `V4`, and `V4` fired.

`V5` (VOID-NOROOM) is then reported for completeness because it is the gate that *should* have fired
and its wording stopped it: it requires contact **`0.0000` on EVERY population**. MEASURED base:
`sel_contact` **0.0000** · `top8_contact` **0.0000** · `top32_contact` **0.0330** · `fan_contact`
**0.0974**. Two populations are zero and two are not, so `V5` did not fire **literally** — while
firing completely **in substance** on the only populations the car acts on. §12.3 quantifies that.

**What the evidence supports, per family, once the admissible comparisons are used** (post-hoc where
marked; `rl` − `ctrl_const` isolates reward + ≥GT bar; T0 readout, 120 windows / 79 episodes /
36 lead windows / 30 lead episodes; paired episode-cluster bootstrap, n_boot 4000, seed 11):

| family | outcome | evidence |
|---|---|---|
| **(a) contact** | **2 NULL-SAFETY, sub-case (i)** — *the fan was already safe*. The committed prediction was **RIGHT** | `sel_contact` and `top8_contact` are **identically 0.0000 in every arm, before and after** — 0 of 36 lead windows. `fan_contact` −0.0082 **ns**, `top32_contact` −0.0073 **ns**. §12.3 |
| **(b) TTC** | **worse, separated** | `sel_ttc_below` **+0.1333** [+0.0333, +0.2667]; `mass_rank_ttc_below` +0.0883 [+0.0009, +0.1998]. `fan_ttc_below` +0.0029 ns |
| **(c) infeasible** | **worse, separated, and large** | `sel_infeasible` **+0.5274** [+0.4114, +0.6392] — the selected path goes **13.3 % → 64.2 % infeasible**; `top32_infeasible` +0.1041; `fan_infeasible` +0.0229 |
| **(d) mass** | **worse on contact, but ONE-SIDED BY CONSTRUCTION** | `mass_rank_contact` **+2.877e-04** [+2.4e-08, +8.0e-04]. §12.4 — this metric could not have registered an improvement |

⇒ **Sub-case (i), not (ii), and the discriminator is measured rather than argued.** Outcome 2's
sub-case (ii) is *"the reward cannot see the collisions"*. It is **REFUTED**: the collision component
**fired** — `components_fired.collision` **449 / 2000** steps in `rl`, **41 / 200** in each control,
and `veto_rate_mean` 0.0897–0.0975. The reward sees collisions perfectly well. There were none to
prune on the emitted path.

### 12.3 The contact axis, QUANTIFIED — RL had nothing to grip, and this was known before the arms

**The human's own rate is the reference the outcome text demands.** MEASURED on the banked
refcv3-40284 dump over the **full** EVAL corpus — 4,823 windows / 141 episodes, **1,466 lead windows
/ 78 lead episodes** (`raw/run/base/fan_safety_dump.summary.json`, 0 GPU):

| selected-path contact | rate | vs human |
|---|---|---|
| **`g` — the HUMAN** | **0.0000** [0.0000, 0.0000] | — |
| **`os` — refcv3's deployed selection** | **0.0007** [0.0000, 0.0024] | `os − g` = **+0.0007**, **NOT separated** |
| `ha` (hold-action floor) | 0.0000 | — |
| `ha0` (hold-v0-straight floor) | 0.0055 | 8× the model |

⭐ **refcv3's selected trajectory is statistically indistinguishable from the human on collisions**,
over 1,466 lead windows and 78 episodes. The base model had already solved the problem V2's RL stage
exists to solve.

**The failure DOES exist in the fan — the ranking already excludes it.** Over the 36 lead windows of
the RL readout:

| | value |
|---|---|
| lead windows with **any** contacting candidate in the 128-fan | **10 / 36 (27.8 %)** |
| lead windows with a contacting candidate in the **top-32** | **3 / 36 (8.3 %)** |
| lead windows with a contacting candidate in the **top-8** | **0 / 36 (0.0 %)** |
| lead windows where the **selected** path contacts | **0 / 36 (0.0 %)** |
| candidates in contact per lead window (of 128) | mean **12.5**, median **0**, max **104** |
| probability mass the model puts on contact (`mass_rank_contact`) | **3.470e-05** = **0.0035 %** |

⇒ **The fan carries contact mass in its tail (9.74 % of candidates) and the model's own scorer
already assigns it 0.0035 % of the probability and never selects it.** A reward whose job is to push
mass away from collision arrived to find the mass already at 3.5e-05. There was nothing to buy.

⚠️ **This was registered BEFORE the arms ran** (`D-RL-FANSAFE-1`, verbatim: *"the re-scoped
endpoint's headroom is in (c) infeasible and (b) TTC, **not** (a) contact — stated BEFORE the arms so
it cannot be presented as a prediction afterwards"*). The honest reading is therefore not that we
discovered the null; it is that **the endpoint was chosen after its own headroom had been measured
away, and the arms were run anyway.** That is the process defect worth more than the result.

### 12.4 The two separated mass metrics — REAL, not the `H-ESTIM-SEED-1` noise floor, but one-sided

The brief asks whether `mass_rank/conf_contact` at **±0.0003** against a `1e-4` floor on 30 episodes
is an effect or the same one-seed artifact `H-ESTIM-SEED-1` documents. **Measured with `ctrl0`, and
the answer is: real, with a caveat that matters more than the significance.**

| probe | `mass_rank_contact` delta | reading |
|---|---|---|
| **`ctrl0`** (zero lever: `lr = 0`, weights hash-identical) | **−3.6e-11** [−1.1e-10, +1.7e-14] | the measurement noise floor is **~1e-10** |
| **`ctrl_const`** (trains, 200 steps, no reward information) | +1.4e-06, **ns** | |
| **`rl`** | **+2.890e-04** [+2.4e-08, +8.1e-04], separated | **~7 × 10⁶ × the noise floor** |

⇒ **Not an estimator artifact.** In relative terms the mass on contacting candidates went
**3.47e-05 → 2.76e-04, a 7.9× increase**. "A hair above the quantum" understates it: the *absolute*
number is tiny because the *base* is tiny.

⛔ **But the metric is one-sided by construction, and that is the finding.** `MIN_EFFECT = 1e-4` was
derived as a **rate** quantum — one candidate in one window, `1/(128 × 120) = 6.5e-5`. The base value
of `mass_rank_contact` is **3.470e-05**, i.e. **0.347 × the floor**. A decrease is bounded below by
zero, so the largest possible improvement is `−3.47e-05` — **2.9× smaller than the threshold that
would let it be called separated.** ⇒ **On this metric an improvement could never have been
detected and only a worsening could.** The one contact-family signal the panel was able to report was
the only direction it was able to report.

⚠️ Same family as the CLAUDE.md units trap: a floor correctly derived for one quantity, applied to
quantities it does not fit. It also mis-fits `fan_peak_g_mean` (units **g**) and
`fan_v_mean_2s_spread` (units **m/s**), where a `1e-4` rate quantum has no meaning at all — 2 of the
14 metrics that fired `V4`. ⇒ **A separation floor must be derived per-metric from that metric's own
quantum and units, and a metric whose base value sits below its own floor must be declared
UNDETECTABLE-DOWNWARD rather than reported as null.**

⚠️ **`H-ESTIM-SEED-1` still binds.** `ctrl0` bounds *measurement* noise, not *training* noise — it
never trains. No replicate arm exists. The separated rows here are **necessary, not sufficient**.

### 12.5 SECONDARY endpoint — the four families at **T1**, paired, per family, never pooled

⭐ **Recovered at ZERO GPU.** The T1 rollout **completed** (141/141 clips, `manifest.json` written)
and then died in its **analysis** step — the exact CLAUDE.md trap ("an analysis-time failure after a
completed rollout reads like a total failure while the expensive part is already paid for"). Its own
documented fix, `--analyze-only`, recovered every number without touching a GPU. ⚠️ I first counted
**127** clips from one `ls` and nearly banked "127 of 141"; two positive probes (`find`, and the
tool's own window count matching base exactly) and the completion marker `manifest.json` show the
dump is **complete**. A short count from one probe is not evidence.

`taniteval/tools/paired_openloop.py`, **T1 (self-action OPEN loop — never a driving claim)**, `rl` vs
`base` over the shared `ha0` floor. **`void: false`**, gates `G2`/`G3` pass, **4,823 shared windows /
141 episodes, 0 windows dropped on either side**, pairing controls exact (window-key GT max|A−B|
**0.000e+00** m, floor bit-comparability **0.000e+00**). Estimator: FULL-SET pooled mean, paired
episode-cluster bootstrap (cluster = clip), n_boot 2000, seed 0. Power: adequate (141 ≫ 10).

| family | metric | unit | base | **rl** | `ha0` floor | delta [95 % CI] | separated |
|---|---|---|---|---|---|---|---|
| **ADE** | `ade_m` | m | 0.4419 | **0.9433** | 0.6723 | **+0.5014** [+0.4572, +0.5464] | **yes** |
| **ADE** | `fde_m` | m | 0.9288 | **1.3477** | 1.4029 | +0.4189 [+0.3530, +0.4870] | **yes** |
| **LONGITUDINAL** | `LON_speed_mae_mps` | m/s | 0.4516 | **1.3160** | 0.4880 | **+0.8643** [+0.7975, +0.9356] | **yes** |
| **LONGITUDINAL** | `LON_along_mae_m` | m | 0.4030 | **0.8467** | 0.4705 | +0.4436 [+0.3993, +0.4906] | **yes** |
| **LONGITUDINAL** | `LON_accel_mae_mps2` | m/s² | 0.6806 | **3.2793** | 0.4786 | **+2.5987** [+2.4310, +2.7768] | **yes** |
| **LATERAL** | `LAT_cross_mae_m` | m | 0.1084 | **0.2795** | 0.3132 | +0.1711 [+0.1589, +0.1842] | **yes** |
| **LATERAL** | `LAT_heading_mae_deg` | deg | 1.3591 | **7.6017** | 2.8715 | **+5.6507** [+4.4727, +6.9142] | **yes** |
| **LATERAL** | `LAT_yaw_rate_mae_radps` | rad/s | 0.2176 | **0.5013** | 0.0476 | +0.2836 [+0.2231, +0.3516] | **yes** |
| **TACTICAL** | `TAC_traj_lat_correct` | acc ↑ | 0.9540 | **0.7730** | 0.8659 | **−0.1810** [−0.2195, −0.1459] | **yes** |
| **TACTICAL** | `TAC_traj_lon_correct` | acc ↑ | 0.7477 | **0.3838** | 0.7576 | **−0.3639** [−0.4015, −0.3278] | **yes** |
| **TACTICAL** | `TAC_declared_*` (6 rows) | acc ↑ | 0.7105 / 0.5134 … | identical | — | **0.0000 [0, 0]** | no — **STRUCTURAL** |
| **STRATEGIC** | `STR_route_correct` (3 rows) | acc ↑ | 0.7667 | identical | 0.6742 | **0.0000 [0, 0]** | no — **STRUCTURAL** |

⛔ **The committed SECONDARY prediction was NULL **or** FAIL-GUARD. The result is FAIL-GUARD, on
every trajectory-derived metric, without exception.**

⭐⭐ **And the magnitude is the headline: the RL'd planner is now WORSE THAN THE TRIVIAL
CONSTANT-VELOCITY FLOOR.** `ade_m` **0.9433 vs `ha0` 0.6723**; `LON_speed_mae` **1.3160 vs 0.4880**;
`LON_accel_mae` **3.2793 vs 0.4786** (**6.9×** the floor); `LAT_heading` **7.6017° vs 2.8715°**.
A stage that leaves the planner behind hold-v0-straight has not degraded a model, it has **destroyed**
one. ADE +113 %, speed MAE +191 %, accel MAE +382 %, heading +416 %, longitudinal manoeuvre
correctness −49 %.

⚠️ **The `0.0000 [0, 0]` rows are STRUCTURAL ZEROS and must never be read as "no harm".** The RL
trained `core.decoder` **only** (9,206,032 / 107,032,901 = 8.60 %); `core.maneuver`, `core.route`,
`tac_goal_head`, `str_goal_head` and `conf_head` are in `forbidden_prefixes` and were **frozen**.
Identical inputs through frozen heads give identical outputs — an **identity, not an estimate**
(CLAUDE.md's `H-ECHO-4` class). ⇒ the SECONDARY guard is informative on the trajectory-derived rows
and **silent** on the declared-tactical and strategic rows.

⭐ **Independent corroboration from the selector, which was never trained.** Anchor-selection profile,
base → `rl`: distinct anchors **50 → 37**, modal share **0.1482 → 0.3054**, entropy **2.8425 → 2.3241**
nats, and **agreement with the oracle-best anchor 0.5652 → 0.1702 (a 3.3× fall)**. The decoder moved
the fan so far that the *unchanged* selector now picks a near-worst candidate — which is exactly the
`sel_infeasible` 13.3 % → 64.2 % of §12.2 seen from the other side.

**Criteria checker** (`TanitAD_BenchmarkCriteria`, registry **v2.6.0**, run on the artifact):
**IN_SCOPE**, tier **T1**, **0 violations**, **3 work items** — all three the same root cause,
`strat.nav_compliance` and its two controls **REFUSED** with a reason
(`TypeError: _path_exists: path should be string`), not silently absent. It refuses identically on
both arms, so it does not bias the comparison. **Escalated in §12.7.**

### 12.6 ⭐⭐ WHAT TO DO — the deliverable, per the PI's standing correction

> *"our goals in TanitAD programme is not to refute hypotheses, it's about achieving excellent results
> and really driving autonomously … we have too many refutes"* (PI, 2026-09-05)

A null on contact plus a catastrophe on driving is not a deliverable. **The same panel contains the
positive result, and it has been sitting inside the control arm all along.**

**1. ⭐ PROMOTE THE VETO-ONLY ARM FROM CONTROL TO PRODUCT — it is the only arm that moved the fan the
right way.** `ctrl_const` — reward identically `0.0`, veto alone — improved feasibility **with no
reward at all**: `fan_peak_g_mean` **−0.0859 g**, `top32_infeasible` **−0.0143**, `top8_kamm_over`
−0.0137, `fan_kamm_over` −0.0048, `fan_infeasible` −0.0026, in **200 steps / 2 min 07 s**. That is
DiffusionDriveV2's actual anti-collision device (collision/TTC pinned at −1, outside the ranking),
and **it transfers**. The composed reward then overwhelms it and drives the planner below the
constant-velocity floor. ⇒ **Next arm: veto only, `w_anchor` kept, 2,000 steps, no composed reward.**
Same driver, one flag set, **~15 min on the dev-box 4060, 0 pod-hours.** Committed outcomes and a
**seed replicate** in the same launch, because `H-ESTIM-SEED-1` binds.

**2. Aim the reward where the headroom actually is — MEASURED, on the same corpus, vs the human:**

| selected-path metric | human `g` | refcv3 `os` | gap |
|---|---|---|---|
| `contact` | 0.0000 | 0.0007 | **ns — no headroom** |
| `envelope` (\|a\| > 4 m/s² ∨ \|κ\| > 0.2) | 0.0068 | **0.0865** | **12.7×**, separated |
| `flagged` (any) | 0.0272 | **0.1101** | **4.0×**, separated |
| `ttc_below` (2.93 s) | 0.0668 | **0.0866** | +0.0198, separated |
| `kamm_over` | 0.0000 | 0.0021 | separated |

⇒ **refcv3's deployed plan is flagged 4.0× more often than the human and leaves the comfort/dynamics
envelope 12.7× more often. That is the un-exploited gap, and it is feasibility, not collision.**

**3. ⛔ Do NOT simply add a feasibility term — this arm already had `feasibility: 0.5` and made
feasibility WORSE.** §11.4's hypothesis (`progress` at 0.30 rewards covering ground ⇒ accelerate
harder ⇒ exactly the `envelope` and `ttc_below` rises observed) is **consistent with `LON_accel_mae`
0.6806 → 3.2793 m/s², a 4.8× acceleration blow-up** — the strongest corroboration yet, and still a
hypothesis. ⭐ **The cheapest discriminating experiment needs NO training and NO GPU:** score the
banked fan's 128 candidates under `RewardSpec` and measure the **rank correlation between reward and
`envelope` violation** on the banked dump. If the reward ranks envelope-violating candidates *higher*,
the reward is disqualified at its source and no amount of RL repairs it. **Minutes, on data already
on disk.** Run this **before** any further arm.

**4. Fix the pre-registration machinery — three defects, each cheap and each caused a wrong reading
here.** (a) `V4` must be scoped to *"moved by a path other than the veto"* with `veto_rate_mean`
printed beside it. (b) `V5` must test the populations the car **acts on** (`sel`, `top8`), not "every
population" — its literal wording let a no-headroom endpoint through. (c) The separation floor must be
derived **per metric from its own quantum and units**, and any metric whose base value is below its
own floor must be stamped **UNDETECTABLE-DOWNWARD** (§12.4).

**5. Where this lever belongs in the programme: AFTER the fan work, not before it.** The endpoint has
no contact headroom, the reward is disqualified until (3) clears it, and the one component that works
(the veto) needs no reward. This is `H-DDA-3`'s territory — the vocabulary/sampler — exactly as
`D-RL-FANSAFE-1` said before the arms ran. **Recommendation to the Master Mind: run (1) and (3), both
0-pod-hour and ~15 min combined, then close the composed-reward line on refcv3 unless (3) clears it.**

### 12.7 Corrections, escalations, limits

**CORRECTION to §11 (this file), banked rather than silently edited.** §11's headline
*"Exit: `4 FAIL-SAFETY`"* selected an outcome-table exit **past a fired VOID gate**, which SPEC
§10.6's "first match in order" does not permit. **The formal exit is `V4` → VOID** (§12.1). The
*substance* of §11's finding survives on the admissible comparisons — the stage as configured is
harmful, and §12.5's T1 four-family read establishes that **without needing the attribution at all** —
but the exit **label** and the `rl`-vs-base fan-safety table of §11.2 are withdrawn as lever
evidence. Root-cause class: *an outcome selected past its own gate because the gate's failure was
diagnosed as informative* — informative it is (RETRACTION #24), and it still fired. → drafted for
`RETRACTION_LOG.md` as **#25**.

**ESCALATIONS** (raised here, not written into a doc for someone to find):
1. **`taniteval` instrument bug — `strat.nav_compliance` crashes** with
   `TypeError: _path_exists: path should be string`, taking out **3 of 3** STRATEGIC nav-compliance
   criteria rows on **every** refcv3 arm, not just this one. The STRATEGIC family is the programme's
   thesis and is currently unmeasurable through this path. **Owner: Benchmarks/Eval.**
2. **`posttrain.py:206` keys the veto on `"collision" in spec.weights`** — key membership, not weight
   value. Any future zero-weight control inherits `V4`'s failure. **Owner: Training FlyWheel.**
3. **The T1 eval chain has no `--analyze-only` retry**, so a completed 141-clip rollout was reported
   as a dead run. `run_eval.sh` should fall back to `--analyze-only` when a dump exists.

**LIMITS — none silent.** §11.6's seven stand, plus: **(8)** the T1 read is `rl` only — `reg_echo` and
`ctrl0` were never T1-evaluated, so the four families have no arm-internal control (they do have the
`ha0` floor and the frozen-head structural zeros). **(9)** `rl` − `ctrl_const` is **POST-HOC**.
**(10)** No replicate on any arm ⇒ `H-ESTIM-SEED-1`: separated is necessary, not sufficient.
**(11)** T1 is self-action **OPEN loop** — no closed-loop claim is made anywhere here.
