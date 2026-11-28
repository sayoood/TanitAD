# P4 — TRAINING PIPELINES · BACKLOG

`Owner: TanitAD_TrainingFlyWheel. Created 2026-08-23. Ranked by VALUE, not by
ease. Every item names its anchor (file:line or artifact) and its both-outcomes
where it is an experiment rather than a fix. Items marked ⛔ BLOCKED name what
unblocks them and who owns that.`

Companion documents: `SPEC.md` (the contract) · `METHOD_LIBRARY.md` (frontier
methods survey — its ranked items merge in as the **M-** series).

---

## TIER 1 — do next (gates and truth; these decide whether other work means anything)

| # | item | why it is first | anchor | est. |
|---|---|---|---|---|
| ✅ ~~**P4-1**~~ | **DONE 2026-08-30.** `o6_is_required()` reads `STAGE_GATE_SPEC` LIVE (a copied constant would keep warning after the promotion — the guard's own failure mode, reproduced inside the guard); required+unrulable ⇒ `SystemExit` before any compute, reported ⇒ banner unchanged. ⚠️ O6 is `reported` at every stage today, so the abort is dormant and armed FOR the promotion. 7 tests. ~~Make the rank-gate CAN-RULE check ABORT when O6 is a *required* probe.~~ Keep warn-only when O6 is `reported_only`. | `--spectrum-accum` defaults to **1** ⇒ 24 rows vs a 1024 ceiling ⇒ O6 **cannot rule by construction**, and INCONCLUSIVE counts as NOT-PASS, so the ladder blocks for an instrument reason. This is H-GATE-1 re-entering through a default. | `train_v6_staged.py:3190`, `:5351`, `:3501-3505`; gate semantics `:2711-2713` | S |
| **P4-2** | **Resolve the O6 floor conflict (H-RANK-16).** `O6_PARTICIPATION_FLOOR = 8.56` (n=1440 val rows) vs the E-TRUNK-3 DINOv3 reference **40.77** (5,617 frames, 130 episodes). Decide which is admissible, or make the floor instrument-scoped. | ⛔ **Every arm judged against 8.56 is being compared to a reference measured a different way.** A gate whose floor is ambiguous cannot rule honestly. | `stack/tanitad/models/v6.py` → `O6_PARTICIPATION_FLOOR` (**:1518**, was :1359); `E_TRUNK_3_LADDER.md:20` | M |
| ~~P4-3~~ | ~~champ30k DECODABILITY rung~~ | ✅ **DONE 2026-08-23.** `lead_gap_m` R² **+0.1273 [0.0866, 0.1683]**, CI wholly above zero, vs every v6F rung below zero. WORLD B does not hold for two-term+k1. Narrow + longitudinal only. | `…/raw/champ30k_decodability.json`; runner banked at `…/code/champ30k_decodability.py`; claims H-DEC-1/2/3 | — |
| **P4-3b** | ⭐ **HELD-OUT decodability re-measurement.** Rebuild the E-TRUNK-2 battery on episodes NOT in the training set (`physicalai-val-0c5f7dac3b11`), which first needs the agent/obstacle join + targets jsonl built for those episodes. | ⛔ **ALL 130 probe episodes are inside the 2,400-episode training cache — 100 % overlap** (H-DEC-3). The 5-fold holds episodes out of the RIDGE FIT, not out of TRUNK TRAINING. ⇒ every absolute number ever reported on this battery — champ30k and v6F alike — is memorisation-permissive. The cross-rung comparison survives; **the absolute number does not, and it gates v7 Scale-2.** | join needs DataFlyWheel input; battery is banked | **M** |
| **P4-3c** | **LATERAL-channel work package.** champ30k separated on `lead_gap_m` (+0.1273) and `ego_speed` (+0.1823) but sits at chance on `left_occupied` (0.5240), `vru_ahead` (0.4544), `n_agents_log`, `nearest_any_m` — and `ego_yawrate` reads **−0.4925**, actively anti-predictive. | The objective demonstrably induces LONGITUDINAL/depth perception and no lateral perception. Scaling it unchanged buys a better rangefinder, not a driver. | H-DEC-2 | **M** |
| ✅ ~~**P4-4**~~ | **DONE 2026-08-30.** `run_provenance()` in BOTH config.json and summary.json: git sha/dirty/branch, **`trainer_md5`**, torch/CUDA/cuDNN, GPU name+capability, hostname, resolved device, tz-aware start (local AND UTC — pods are UTC, the PI reads Berlin). ⭐ `trainer_md5` is the identity, not the SHA: pods have no git credentials, so HEAD is weeks stale while the tree is current. ⛔ Every probe individually guarded — provenance must never be why a run dies. 7 tests incl. all-probes-broken. ~~Add run provenance~~ — git SHA + dirty flag, **md5 of the executing trainer file** (the only meaningful identity on a file-shipped machine), torch/CUDA version, GPU model, hostname, resolved device, start time **with timezone**. | Two runs on two machines can produce byte-identical configs today. Same class as the residual-init failure, one layer out. | SPEC §2 R3–R5; inventory §C3 | M |
| ✅ ~~**P4-5**~~ | **DONE 2026-08-30 — REFUSED, not wired.** `--v2-val-cache` built a dataset nothing read; now a startup `SystemExit` naming the alternative. ⚠️ It survived because the SAME flag IS wired in `train_flagship_v4.py` / `eval_flagship_v4.py` / `probe_latent_state.py` — meaningful *somewhere*, which is why nobody noticed it was dead *here* (the `df`/`step_s` scope family). ⭐ Verified before refusing: no v6 launch line passes it (`runs.d/*.env` -> only the v5f env, which targets v4). Refusal verified by RUNNING it, not by syntax. ⭐ **AND CONFIRMED BY AN UNSCOPED REPO-WIDE SEARCH afterwards** (the scoped one had been the basis for the decision): 21 hits, **zero** a v6 trainer invocation — the JSON hits are `rollout_provenance/.../label` strings from the v4 parity checker, and the two `v7tiny_*.py` hits are PROSE. ⚠️ **And that prose is the strongest corroboration available**: both docstrings state the v7-tiny arms were trained with **NO `--v2-val-cache`**, and `v7tiny_pull_heldout.py` exists *specifically because* v6 offers no held-out path — somebody needed validation data, discovered the flag did nothing, and wrote a separate puller. That workaround IS the cost of a flag that looks wired and is not. ~~`ds_val` is dead code.~~ Built and printed at `:3795-3799`, then never referenced. `--v2-val-cache` mounts and windows a whole val corpus that **no loss, no probe and no gate reads.** Either wire it or refuse the flag. | A val cache that *looks* wired and is not is exactly the "verify by content" failure class. Anyone reading the launch line believes validation is happening. | `train_v6_staged.py:3795-3799` | S |

---

## TIER 2 — efficiency (the HAL mandate)

| # | item | both outcomes / note | anchor | est. |
|---|---|---|---|---|
| **P4-6** | **E-P4-HAL-1: measure `data_wait_s` vs `compute_s`** on Thor and dev box at champ30k settings. ⛔ MEASURE BEFORE CHANGING. ⚠️ Cannot be answered from banked logs — the trainer records `step_s`/`step_s_interval` but **no data/compute split**; this needs instrumentation around `:4178` shipping with its regression test. | *≥25 % data wait* → build the worker/prefetch path behind the HAL, re-measure paired. *<25 %* → **the zero-worker design is CORRECT**; record it so it is never re-litigated. ⭐ **INCIDENTAL EVIDENCE, 2026-08-23**: scale1 has **5.7× champ30k's encoder params** (0.97 → 5.49 M) and costs only **+14 % step time** (0.5546 vs 0.4880 s/step, both `step_s_interval` @ step 200, same machine/settings). A loop whose wall-clock barely responds to a 5.7× compute increase is **data-bound** — so the ≥25 % branch is the likely outcome. Not conclusive (encoder is only part of the FLOPs) but it raises this item's expected value. | trainer has **ZERO** DataLoader workers: `default_collate([ds_train[i] for i in idx])` (**:5563**, was :4178) | S |
| **P4-7** | Build the `Machine` HAL object (device, autocast dtype, threads, allocator, `OPS_DIR`, cache roots, LRU, sampling, grad-ckpt) — resolving AND recording every choice. | ⛔ The HAL sets knobs; it never invents per-machine defaults without recording them. That is how two "identical" arms diverge. | SPEC §4.3 | L |
| ✅ ~~**P4-8**~~ | **DONE 2026-08-30.** `resolve_amp_dtype()` probes once, announces the choice, and records it. ⛔ **THE FALLBACK IS fp32, NOT fp16 — deliberately.** fp16 autocast REQUIRES a `GradScaler`; without one gradients underflow to zero and training quietly produces a WORSE model rather than failing. An fp16 branch here would never execute on any machine we own (A40, 4060, Thor are all bf16-capable), i.e. an unexercised path in the step loop — the exact dead-flag defect P4-5 removed. fp32 is slower and CORRECT; fp16+GradScaler is its own change, with a test that exercises it. Verified on the 4060: probe → bf16. 10 tests incl. a broken-probe case and a wiring test that fails if the hardcode returns. ~~Probe is_bf16_supported~~ | `torch.bfloat16` is hardcoded with no probe and no `GradScaler`. Portability blocker for any non-Ampere+ target. ⚠️ VERIFIED BY CONTENT 2026-08-30: the training autocast is at **:5729** (`torch.autocast(dev_type, dtype=torch.bfloat16`); :945/:961 are a SEPARATE VLM path and are not this item. | `train_v6_staged.py` → `torch.autocast(... torch.bfloat16` (**:5729**, was :4278) | S |
| ✅ ~~**P4-9**~~ | **DONE 2026-08-30 — and it was NOT cosmetic.** ⭐ **MEASURED in two clean subprocesses with the variable scrubbed from the environment: module-first (fixed) → `torch.get_num_threads() == 6`; torch-first (the old order) → **16**.** A 2.67× difference, so the old placement ~7,700 lines below the import was demonstrably **INERT** — the 113-threads guard it documented did nothing. `setdefault` keeps an explicit launcher value winning, and the resolved `omp_num_threads` + `torch_num_threads` now ride in `run_provenance`. ⚠️ The test scrubs the env first: with `OMP_NUM_THREADS` already exported both orders read 6 and the test would pass while measuring nothing (TRAIN-C10's family). ~~Move OMP_NUM_THREADS~~ | ⭐ **CONFIRMED BY CONTENT 2026-08-30**: `os.environ.setdefault("OMP_NUM_THREADS", "6")` is at **:7830**, `import torch` at **:101** — the setdefault runs ~7,700 lines AFTER the import, so it cannot affect the thread pool torch already sized. The concern is real, not hypothetical. | `train_v6_staged.py` → `os.environ.setdefault("OMP_NUM_THREADS"` (**:7830**) vs `import torch` (**:101**) | S |

---

## TIER 2b — FRONTIER METHODS (the M-series; full rationale in `METHOD_LIBRARY.md` §4)

⭐ **The survey's headline changes what "adopt frontier training tech" means here:
the decision is not *which optimiser* — it is *fund which signal*.** Two structural
results (DERIVATION over code at HEAD, `METHOD_LIBRARY.md` §2) decide most rows:

1. **Our shipped `w_select` softade loss is ALREADY the exact form of the
   GRPO/RLOO estimator.** For `p=softmax(score)`, `∂/∂s_k Σᵢpᵢcᵢ = p_k(c_k − E_p[c])`
   — REINFORCE with any baseline; the group-mean/leave-one-out baseline cancels
   identically. The fan is enumerable, so we take the expectation **exactly**;
   sampling G outputs would only add variance. ⇒ **Adopting GRPO/RLOO at the
   selector would be a strictly worse re-implementation of what we ship.**
2. **Our candidate generator is differentiable to the metric cost**
   (`unicycle_rollout` in-graph; plan loss consumes `waypoints` undetached,
   `train_v6_staged.py:2422`). The score-function estimator exists because the
   environment is a black box; at the proposal site **we are the environment**.
   ⇒ RL is dominated there while the objective stays metric.

⛔ **And nothing plays the role of a preference pair.** A demonstration corpus has
one policy's output per state — **there is no negative class**. The manufacturable
negative reduces cardinal GT distance to a binary, MEASURED at **+0.0974 m (base)
/ +0.1670 m (XL), separated**. No closed loop exists. Our one on-policy experiment
was MEASURED harmful: DAgger **+0.266 m [0.008, 0.550]** closed ADE, separated,
worse than a matched-budget BC control.

| # | item | cost | why it ranks here |
|---|---|---|---|
| **M-B1** | **Name + leak-audit the return-conditioning we ALREADY ship** — declare `SPEED_BAND`, `ANCHOR_GOAL`, P2's `v_target` as RvS/DT outcome conditioning, then run the anti-echo controls **on the conditioner itself** | **S**, 0 GPU | ⭐ **best value/cost in the programme right now.** It tests a term already training, and the programme has been fooled by this exact shape **three times** (nav-echo 1.0000; T1 action echo 97.9 %→0.0 %; P1 speed echo R² 0.995→−0.72) |
| **M-B2** | **Dr. GRPO / DAPO audit of `w_select`** — pin a test that `softade` carries no std and no length normaliser; add DAPO dynamic sampling (drop degenerate fans); report `sel_norm_err_rank` + lower-tail hit rate, never ρ | **S**, 0 params | turns two frontier results into a regression test on a term we ship |
| **M-B8** | **Imagination-in-the-loop admission gate** — a pre-registered criterion the WM must pass before ANY on-policy training inside imagination | **S** | encodes the measured DAgger failure as a *rule* instead of a memory, so the next agent cannot re-run it |
| **M-B9** | **Pathwise (DRaFT-style) gradients for the DIFFERENTIABLE reward terms.** Four of the six components are differentiable through the in-graph unicycle emission, yet `tanitad.rl` routes all six through the score-function estimator — free variance reduction left on the table. | **S/M** | ⛔ **DEFERRED BY DECISION until A1 has a baseline** (Master Mind, 2026-08-29, my reasoning adopted): changing the ESTIMATOR and the REWARD in the same step destroys attributability. Found by the Phase-1 research review (E.4). |
| **M-B7** | keep the plan loss **AWR-ready** (a note + a test, so `plan_loss × exp(A/λ)` is a 3-line change the day M-B5 lands) | **XS** | cheapest option value: no module, no state-dict key |
| **M-B3** | **ChauffeurNet-style LONGITUDINAL recovery curriculum** (perturb `v0`/action history, keep the logged 6 s target) | **M** | the ONLY surveyed method attacking the ~99 %-longitudinal `cl−ol` divergence **without** a simulator. ⛔ real risk of undoing the validated speed fix (3.73→0.83 m) — the deliberate-regression arm is not optional |
| **M-B4** | ⛔ **PI DECISION** — build the preference-collection instrument (fan-comparison renderer on the existing overlay stack → pairwise ranking) | **M** | ⭐ the single item converting an entire REJECTED family into a testable one; recommend **IPO** first (metric-derived preferences are deterministic = DPO's degenerate regime) |
| **M-B5** | ⛔ **PI DECISION** (~12.4 GB + 2–3 eng-days) — ingest `obstacle.offline` so offline-RL has an `r` at all | **L** | without it CQL/IQL/AWR have no reward and their REJECTs cannot be revisited. Run the pre-registered lead-state gate FIRST — a FAIL is the cheaper outcome |
| **M-B6** | ⛔ **PI DECISION** — closed-loop environment (TanitSim) | **XL** | the only thing making the online-RL half of the mandate real. ⚠️ **Step 0 is a throughput re-measure**: "492 FPS" is a 20 k-gaussian synthetic probe; the real scene read **4.4 FPS at 1080p**. ⚠️ carry the OOD lesson — AlpaSim's one result is confounded at **3.21× reconstruction-OOD** |

⚠️ **Two corrections propagated out of the survey** (both verified against primaries):
`RESEARCH_AGENDA.md:34`'s *"gsplat 492 FPS on Thor"* must not be used to size M-B6
(its own experiment classes the real 4.4 FPS as ⛔ NOT ESTABLISHED,
`nurec-gsplat/FINDINGS.md:156`) · ⚠️ **that `corpus_overlay.py` "does not exist" — RETRACTED 2026-08-29.**
It exists at **`taniteval/taniteval/corpus_overlay.py`** (nested package path; its own
docstring declares "THE STANDARD"). Both the survey and I probed only the OUTER
`taniteval/` dir — the absence-at-one-location failure, committed inside a note about
corrections. Related: `ph0_rich_overlay.py`, `p8_bev_reel.py`, `probe_overlay.py`.

---

## TIER 3 — operational hardening

| # | item | why | anchor | est. |
|---|---|---|---|---|
| **P4-10** | **Bank a v6 run manifest** into `stack/ops/runs.d/`. Today the only manifest is for `train_flagship_v4.py`; v6 manifests are generated on the fly and never persisted. | A supervised v6 run is reproducible only if its manifest is banked with it. | `v6_chain.py:1877-1915`; `runs.d/` | S |
| **P4-11** | Make `supervise_run.sh` `OPS_DIR` machine-resolved (defaults to `/workspace`, **wrong on Thor**). | Lock/heartbeat files land in a path that does not exist on Thor. | `supervise_run.sh:32` | S |
| **P4-12** | **Ship helper with md5 gate**: one command that scp's every file a run imports and REFUSES to launch on any mismatch. | Thor has no git credentials; the measured cross-pod failure was a *partial* ship that imported fine and trained different code. | SPEC §3.2 | M |
| **P4-13** | Fix `--fps`: `getattr(a,"fps",10)` reads a flag that does not exist, hardcoding the seam dump to 10 Hz while `--dt` is live. Should be `a.dt`. | Silent wrong-frequency seam dumps. | `:4486` | S |
| **P4-14** | Delete the stale "trainer + its DataLoader workers" comment. | There are none; the comment misleads the next reader about the I/O design. | `supervise_run.sh:127-129` | XS |

---

## TIER 4 — v7 full-scale readiness (PRIORITY 3 of the mandate)

⚠️ **Gated on P4-3.** The v7 full-scale plan cannot be finalised until champ30k's
decodability is known: the recipe to scale is either "two-term+k1, scaled" (if it
acquires environment information) or "not this objective" (if WORLD B holds).

| # | item | note |
|---|---|---|
| **P4-15** | Full-scale training plan: scaled encoder under the same objective, staged gates per `/TanitAD_ValidateAIDesign`, hardware plan, checkpoint policy. | Draft after P4-3 reads out. **Both branches drafted in advance** so the readout selects rather than starts the thinking. |
| **P4-16** | **Checkpoint/snapshot policy** learned from the v6F era: fp16 snapshots every 2k, done-markers, off-box backup. | Note: fp16 snapshots are an `--init-from` artifact and are **refused as a resume point** (`:4716-4721`) — the policy must keep a full `ckpt.pt` as the resume anchor. |
| ~~P4-17~~ | ⛔ ~~BLOCKED — H-RANK-8 (`n_stack=1`) needs a cache rebuild~~ → ⭐ **MY ASSESSMENT WAS WRONG, AND IS SUPERSEDED.** Another agent is implementing it in the working tree **right now** (`stack/scripts/train_v6_staged.py`, uncommitted at 14:19) as **`--newest-frame-only`**: feed only the newest frame of the existing 3-frame stack (3 channels), so consecutive latents share NO input frames — **no cache rebuild needed**. It refuses unless `--in-channels 3`. | I had scoped this as blocked on a DataFlyWheel cache rebuild; the newest-frame route is strictly better and needs nothing from Data. ⚠️ **Thor contention** — see the handover note at the foot of this file. |
| **P4-18** | ⛔ **ESCALATED — E-ENC-3WAY gate release.** `PREREG_E_ENC_3WAY.md` §0 requires an arm to *clear* collapse first. champ30k **arrests** collapse (plateau, no fall — H-RANK-11) but does **not clear** the 8.56 floor (6.499). | Whether "collapse arrested + more training proven not to be the lever" satisfies a gate written as "collapse cleared" is a **design ruling for the Master Mind / PI**, not mine to take. Arm C (`dino-frozen`) is the cheapest and runs first if released. |

---

## TIER 5 — conservation

| # | item | why |
|---|---|---|
| **P4-19** | **Bank the E-TRUNK-2/3 decodability battery into the repo.** It currently lives ONLY in a previous session's temp scratchpad (`…\8fc25020-…\scratchpad\sp2\`): the Gram matrices, `e_trunk2_targets.jsonl` (7 MB), `e_trunk2_probe.py`, the 130-episode cache and every banked rung. | ⛔ **This is the instrument behind the programme's central T0 conclusion (WORLD B), and a temp sweep destroys it.** SPEC §10. **HIGH** severity despite Tier 5 placement — it is conservation, not progress. |
| **P4-20** | Regression test pinning `config.json` completeness (all dests + provenance block). | Stops R1 from silently narrowing later. |
| **P4-21** | Promote the validated launch procedure into `/TanitAD_TrainModel`. | Skills are the conservation mechanism — a validated procedure that stays in a transcript is lost. |

---

## DONE THIS SESSION

| item | evidence |
|---|---|
| Recovered the val-rank probe from `thor:/tmp/vp2.py` into `stack/scripts/val_rank_probe.py`, generalised, comparability contract written into the docstring, md5-verified on Thor | `c486fffc530f747334f5957bffd58c67` local == remote |
| champ30k H-RANK-10 readout + trajectory + cross-arm gate/val table | `…/2026-08-19-simwam-analysis/raw/champ30k_*.json`, `v7tiny_val_rank_5way.json` |
| Six claims resolved/added in `GOALS_AND_CLAIMS.md` (H-RANK-10..16) | same turn as the measurement, per the binding rule |

---

## ⚠️ HANDOVER — THOR CONTENTION (2026-08-23, escalate to Master Mind)

**Thor's single GPU is now occupied by `E-P4-SCALE1`** (`~/v7tiny/scale1`, launched
2026-08-23 ~14:25 Europe/Berlin, 30,000 steps, est. 4–12 h — champ30k took 4.1 h at
19.34 M; this arm is 23.87 M). Verified training by content: parity VERIFIED,
O6 gate CAN rule, X3 isolation pass, 415,002 windows.

⚠️ **Another agent is concurrently implementing H-RANK-8 (`--newest-frame-only`)**
in the working tree — `stack/scripts/train_v6_staged.py` is modified-uncommitted
(repo md5 `8299e575…`, Thor md5 `ec64c107…`). That arm will want the same GPU.

**Sequencing is a Master-Mind call.** My inputs:
- scale1 checkpoints at `--save-every` and **auto-resumes**, so killing it early is
  cheap — `resume_guard` refuses a relaunch only after the done-marker exists.
- ⛔ **Kill by explicit PID** (trainer PID `1510481` at launch); `pkill -f` self-matches.
- ⛔ **scale1 deliberately runs Thor's OWN trainer (`ec64c107…`), NOT the modified
  repo copy** — champ30k ran that binary, and a matched comparison requires it.
  Shipping the `--newest-frame-only` edit to Thor mid-run would silently change the
  arm. Ship it only when scale1 is done, or run H-RANK-8 on the dev-box RTX 4060.

---

## TIER 1b — RL POST-TRAINING, POST-EXIT-C (added 2026-08-29 after the threshold finding)

⭐ **State of the line:** P-RC21 exited at C (nothing improved across three reward
compositions and four anchor strengths) and the CAUSE is now measured — the safety
threshold flags competent human driving (H-RL-THRESH-1). Two candidate levers were
killed by their own controls in the same session: the denoise budget (extra reach
diverges) and the anchor vocabulary (clustered == random, both worse than the
existing bank). The line is **not closed**; it is waiting on a specification
decision that belongs to the PI.

| # | item | why | anchor | est. |
|---|---|---|---|---|
| **P4-10** | ⛔ **PI DECISION — run `PREREG_D_SAFE_CAL`?** Four arms, five committed outcomes, ~2 h on the 4060, Thor untouched. It is a SPECIFICATION change (what "safe" means), which is why it is not mine to launch. | Every P-RC21 null was measured against a target that flags the human on 34.9–45.1 % of windows. The nulls are uninformative about a calibrated target — ⚠️ **and uninformative is not promising**; outcome 2 closes the RL line on this base model. | `…/2026-08-29-rl-posttrain-library/PREREG_D_SAFE_CAL.md` | — |
| ✅ **P4-11** | ~~Add `R5_gt_clearance_violation_frac` + `clearance()`~~ | **DONE 2026-08-29.** The prereg's gate. R5 = the SELECTED candidate's violation rate, distinct from R2's fan-wide rate — the gap between them is where reward hacking hides, and exit 3 needs both. `clearance()` extracted as the ONE definition; absence returns **nan**, never 0/+inf, and the readout DROPS those windows so a thin join cannot report safety. 9 tests, RL suite 149 green, verified on the real model (not a mock — TRAIN-C1). | `rl/rewards.py`, `tests/test_rl_clearance.py` | — |
| **P4-12** | **`target_time_gap_s` is mis-set in the OPPOSITE direction.** T* = 2.0 s against a human median of **2.992 s** ⇒ the reward's peak asks for closer following than humans drive. | Flags 11.8 % — not urgent, and it is a graded peak rather than a barrier, so it shapes rather than vetoes. ⛔ Still a specification change: needs its own both-outcomes prereg, not a patched constant. ⚠️ Do NOT bundle it into D-SAFE-CAL — two spec changes in one experiment is the `--v2` conflation failure (ten levers on two axes, result non-attributable). | `rl/rewards.py:294`; `raw/p_rc21_rerun/threshold_sweep.json` | S |
| ✅ ~~**P4-13**~~ | **DONE 2026-08-30.** `THRESHOLD_CALIBRATION` now lives in `rewards.py` BESIDE the constants it describes (a registry in another file drifts), exported from `tanitad.rl`, with `THRESHOLD_CALIBRATION_SOURCE` naming the probe, the artifact and the NON-PARITY caveat. ⚠️ It is a COVERAGE CONTRACT, not a re-measurement — CI cannot run the corpus, so the tests enforce that *evidence exists*, not that the numbers are re-derived. ⭐ **All three guards were FALSIFIED before being trusted**: a new unregistered constant fails and the message names it; a registry value drifting from code fails; a `ctx.get` default out of sync with the registry fails. (A guard never shown to fire is the TRAIN-C8 family — presence is not validity.) 6 tests. ~~Generalise the threshold sweep into a standing instrument.~~ `code/threshold_sweep.py` is a one-off; make it a test that asserts every safety/comfort constant flags the demonstration distribution below a stated rate, and FAILS when a new constant is added without one. | ⭐ The sweep found nine sound constants and one bad one. The value is not the audit, it is that **the next constant someone adds gets the same check automatically** — otherwise this recurs and is caught by a campaign rather than by CI. | `code/threshold_sweep.py` | M |

---

## ⚠️ ANCHOR HYGIENE — every line number in this file rots (2026-08-30)

**MEASURED while executing Tier 1: ALL SIX anchors checked were stale**, some by
thousands of lines — P4-5 said `:3795`, the code was at `:4934`; P4-6 said
`:4178`, the code is at `:5563`; P4-8 said `:4278`, the code is at `:5729`; P4-9
said `:6031`, the code is at `:7830`; P4-2 said `v6.py:1359`, the code is at
`:1518`. `train_v6_staged.py` has grown ~1.4× since these were written and every
bare line reference moved with it.

⛔ **A stale anchor is the same rot class as a stale absence claim**: it looks
like evidence, it is checked by nobody, and it sends the next reader to the wrong
place — or worse, to a *plausible* wrong place. P4-8's old anchor now lands near
a **different** `torch.bfloat16` (the VLM path at `:945`/`:961`), which would have
produced a fix to the wrong call site that passed review.

⇒ **RULE FOR THIS FILE: an anchor names a SYMBOL, and carries its line only as
a convenience.** `→ `O6_PARTICIPATION_FLOOR` (:1518)` survives a refactor;
`:1359` does not. Re-verify by content before acting on any anchor here, and fix
it in the same turn you find it stale.
