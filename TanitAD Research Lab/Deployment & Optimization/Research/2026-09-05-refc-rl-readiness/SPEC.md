# SPEC — D-RL-REFCV3-MIN: the minimal pre-registered *RL stage on/off* experiment on the FROZEN refcv3 @ 40,284

`TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-refc-rl-readiness/SPEC.md`
Production & Optimization (Deploy) FlyWheel · 2026-09-05 · **PREPARED, NOT LAUNCHED** · PI, verbatim: *"can we run the rl experiments of refc to see the effect, the training fly wheel agent prepared this in the past."*
Register: hypothesis **`H-RL-MIN-1`** (inserted in `GOALS_AND_CLAIMS.md` in the same turn as this file); context rows `D-RL-REFCV3`, `D-RL-PILOT-RC21` (EXIT C), `H-RL-THRESH-1`, `H-RL-REACH-1`, `D-REFC-DDAUDIT-6`, `H-DDA-3`.
Skill: `TanitAD_ValidateAIDesign` §1 (SPEC before compute), §3 (gates in order), §4 (controls), §5 (verify by content), §6 (close the loop).

```yaml
hypothesis: H-RL-MIN-1
one_variable: RL post-training stage ON (arm `rl`) vs OFF (the banked base dump)   # nothing else differs
held_constant: [base weights (md5 b1ed7075…), decoder_steps 2, eval corpus (141 v7.2 EVAL clips, labels md5 aa12c948…),
                eval grid 2s / stride 5 / n_boot 2000 / seed 0, nav source v72, reward DEFAULT_WEIGHTS, lead model `track`,
                G 4, noise multiplicative 0.1, batch 2, lr 1e-5, steps 2000, seed 0, trainable surface (core.decoder minus 4 selector tensors)]
success:  "one LONGITUDINAL / LATERAL / TACTICAL metric improves with paired episode-cluster separation (n_boot 2000, 141 clusters)
           AND ade_m paired lo <= +0.02 m AND oracle-in-fan (R_ORACLE) not worse with separation AND R_FAN not collapsed (>= 85 % of base) — exit PASS"
failure:  "NULL (no family moves with separation) or FAIL-GUARD (ade_m lo > +0.02) — the committed PREDICTION, see §6"
controls: [ctrl0 (lr 0: weights hash-identical, readout delta must be exactly 0), reg_echo (deliberate regression: the ego GT future INSIDE the
           advantage — G-FAN MUST fire), the banked base dump (RL off), ha0 shared floor (paired_openloop --floor ha0), humanflag gate (§3.4)]
splits:   {fit: "120 train-split clips (fitlist_120.txt, seed 0, fit ∩ eval = 0 asserted twice: --mode fitlist and GATE 3 in every arm)",
           val: "the driver's T0 readout, 120 fixed EVAL windows (seed 1234) — a monitor, never a tuning set",
           test: "the 141 EVAL clips, scored once per arm by openloop_suite.py; nothing is selected on it"}
```

---

## 0. What this is, and is not

| | |
|---|---|
| **IS** | the smallest experiment that answers the PI's question on refcv3 itself: does one RL post-training stage of the decoder — DDv2-style intra-anchor GRPO + truncated inter-anchor advantage + TTC/contact veto + reference-policy trust region, `stack/tanitad/rl/` — move ANY of the four binding families on the frozen refcv3 @ 40,284, against the banked base and a shared trivial floor |
| **IS** | the transfer test of the P-RC21 METHOD verdict (`D-RL-PILOT-RC21`, EXIT C: *the anchor fixes drift; the reward cannot prune collisions; the lever is the fan*) from the v2.1 pilot to the real refcv3 base. Pilot numbers do not transfer; the method verdict is what is being tested |
| **IS NOT** | a driving claim. Training-side readouts are **T0**; the capability read is the taniteval harness, **T1 (self-action OPEN loop)**, four families, paired episode-cluster bootstrap. Nothing here is closed loop (`EVAL_DOCTRINE.md`: T2 = AlpaSim or a vehicle, not provisioned) |
| **IS NOT** | DiffusionDriveV2's GRPO. Our decoder is a deterministic anchor-refinement regressor with no sampling density (`D-REFC-DDAUDIT-1`, `-6`); the policy here is a **surrogate Gaussian on the emitted offset** (§2). Parity with DDv2's estimator is UNVERIFIED and stated |
| **IS NOT** | parity-grade. The RL-fit clips are train-split B1 v7.2 (NON-PARITY, like the base itself); only the paired margin over the shared `ha0` floor is admissible cross-arm |

## 1. Assets — every one verified BY CONTENT on 2026-09-05 (dev box)

| asset | fact | check |
|---|---|---|
| base | `ckpt_step40284_frozen.pt` 428,518,255 B, md5 **`b1ed7075ff730d0993d2eaa3c86f6b56`** = `MODEL_REGISTRY.md` §4.5 (`model` dict bitwise-identical to `ckpt_40284_FINAL.pt`); step field 40284 asserted by `--expect-step` | md5 + step, STAGE 0 |
| base config | `run_refcv3_viz/ckpt/config.json` (the trainer's own), rebuilt through `refc_v3_train.build_parser + _pin_trainer_cfg`, cross-checks pass, load strict except the ONE inert buffer (§7 fix 1) | STAGE 0 |
| base eval (RL OFF) | the banked dump `C:\Users\Admin\_wp56\dump\refcv3_40284_dump` — grid 2s, stride 5, **4,823 windows / 141 episodes**, the numbers in `MODEL_REGISTRY.md` §4.5 (`os` ADE 0.4419 [0.4098, 0.4743]; `ha0` 0.6723; `ha` 0.2996) | manifest present |
| eval corpus | `run_refcv3_ol/data/eval` 141 `.v2ep.pt` + `_v2manifest.pt`; labels `s2_labels_v7.2_eval.jsonl.gz` md5 `aa12c948f062181c3297265b51526ec5`; lead block `_lead_b1/b1_eval_lead_block.npz` (29,556 rows, 8,341 with lead, `ts_rel_s` 0.2–2.0 s) | present |
| RL-fit corpus | `fitlist_120.txt` — 120 of 4,572 train-only clip ids, seed 0, **train ∩ eval = 0** (train 4,572 / eval 147 labels); 8 already local (`fit8/`), 112 to pull from Thor `/home/nvidia/data/physicalai-b1-w120-256x640cyl` (4,714 files; MEASURED 37.8 MB in 1.2 s per clip) | STAGE 0 / STAGE 1 |
| RL-fit lead block | `fit8_lead_block.npz` (1,608 rows / 8 clips; LEAD 339); `fit120_lead_block.npz` built in STAGE 2 by `build_lead_block_b1.py --pull` (~1 min / 150 clips MEASURED by the predecessor) | STAGE 2 |
| code | `stack/tanitad/rl/` (8 modules; RL suite **149 passed**), driver `stack/scripts/rl_refcv3_min.py` (imports verified: `--help` exit 0 in 1.8 s; every analysis-time import paid at start-up), chain `launch_refcv3_rl_min.sh` (`bash -n` OK), harness `taniteval/tools/{refcv3_arm,openloop_suite,paired_openloop,build_lead_block_b1}.py`, `tools/criteria_check.py` + `products/P7-TanitEval/CRITERIA_REGISTRY.json` (asserted present at import) | tests + `--help` |
| machines | dev-box RTX 4060 (8,188 MiB) for everything; Thor `tanitad-thor-wifi` idle (only `tail -f`; supplies clips over LAN); ⛔ `tanitad-refcv3` (refcv4b LIVE) never touched | `ps`, `nvidia-smi` |

## 2. The policy surface — option (b), and why not (a) or (c) for THIS experiment

The DD audit settled that the decoder has no stochastic denoising step and therefore no log-probability (`D-REFC-DDAUDIT-1/6`). Three RL surfaces exist; this SPEC uses **(b)** because it is the only one that is both implemented and label-free today. The other two are priced in `RESULT.md` §4.

| surface | what is sampled / what gets the gradient | status |
|---|---|---|
| **(b) Gaussian perturbation on the offset head** — THIS SPEC | `offset_g = offset·(1 + 0.1·ε)`, G = 4 per anchor, `logp` analytic and differentiable in `offset` (`tanitad/rl/refcv3_adapter.py::sample_offsets`, score-function form corrected 2026-08-29, TRAIN-C3); gradient reaches **`core.decoder` minus the selector surfaces** — **9,206,032 / 107,032,901 params (8.60 %)**, 4 selector tensors excluded and recorded (`conf_head.{weight,bias}` + grafts), tripwire on `scorer`/`phi_tac`/goal heads/encoder | implemented, tested, preflight-verified on the real model |
| (a) policy gradient over the anchor-confidence softmax (a categorical over N = 128) | exact expectation Σ_i softmax(score)_i · r_i, no sampling needed; trains `conf_head` (385 params) + grafts — **the SELECTOR**, which the library's doctrine forbids training under the reward (selector over-reliance is DDv2's named defect; `select_trainable` tripwires it); cannot move the fan (oracle-in-fan unchanged by construction), so its ceiling is the selection gap **0.0751 m** (`oracle_sel` − `os`) | 0 new code; excluded here by doctrine, priced in RESULT §4 |
| (c) DD-faithful sampler (`H-DDA-3`) then true GRPO | needs a retrained base (the 40,284 weights were never trained to denoise scheduled noise) — a NEW training run, not a post-train | not runnable on the frozen base; priced in RESULT §4 |

## 3. The reward — no ego future, and the checks that prove it

**Composition** = `DEFAULT_WEIGHTS` (`rewards.py`): progress 0.30 (reference = the window's OWN v0 × horizon, floored at 5 m — never the fan, never the expert) · collision 1.00 · headway 0.30 (graded, peak at T* = 2 s) · feasibility 0.50 · comfort 0.20; `proximity` stays OUT (`THRESHOLD_CALIBRATION` marks `proximity_safe_m` MISCALIBRATED; D-SAFE-CAL VOID twice); `gt_similarity` OUT (a fan-collapse objective inside the advantage — used ONLY by the regression arm). Veto channel (outside the advantage): contact OR TTC < 1.5 s. Trust region: `w_anchor` 1.0, L2 on the FULL 8-slot mean fan against the frozen copy. Scored object: the 2 s prefix (origin + slots 0.5/1.0/1.5/2.0 s — the uniform part of refcv3's (0.5,1,1.5,2,3,4,5,6) s grid, the same index-select the `2s` eval grid uses).

**Scene context (`reward_ctx`)** = `{dt 0.5, v0 at t0, lead_path, lead_len 4.5 m}` — and nothing else. `lead_path` is the lead AGENT's own `obstacle.offline` track at t0+0.2…2.0 s expressed in the ego **t0** frame (`build_lead_block_b1.py`: yaw0/pos0 at t0 only), resampled onto the 0.5 s grid (`--lead-mode track`, default since 2026-09-05). No-lead windows carry a far sentinel held over the horizon — constant per window, so it cancels in both advantages.

### 3.1 The echo traps, named (a reward read from the GT future teaches reproduction of the GT future)

| candidate reward input | echo? | ruling |
|---|---|---|
| ego GT future (`gt_traj`, `future_poses*`, `goal_tac`) — ADE-shaped terms | ⛔ **ECHO** — inside a group-relative advantage it is a fan-collapse objective (`rewards.py` DEFAULT_WEIGHTS note); as a reward it rewards the logged path, not driving | forbidden on the honest arm (`FORBIDDEN_FUTURE_CTX`, raises); used ONLY by `reg_echo` |
| progress with an EXPERT-derived reference (e.g. "match the human's displacement") | ⛔ ECHO by the back door | refused; the reference is v0 × horizon |
| any model-produced ranking (`sel_score`, `anchor_logits`, `goal_gate`, `traj`) | ⛔ echo of the selector (the nav-echo family, 1.0000 by reproducing its own input) | `assert_selector_disjoint` raises on every reward evaluation |
| lead agent's track (other-agent future, privileged) | not an ego echo — the term's argmax is "keep 2 s behind where the lead will be", satisfied by infinitely many ego paths; unavailable at inference but admissible as a TRAINING signal (labels may use privileged; inference is vision-only) | admitted, provenance recorded in `preflight.json` |
| v0 at t0 | measured state at cycle time — BINDING-admissible | admitted |

### 3.2 The proof, run by the preflight (and its result on 2026-09-05)

1. **No-ego-future test** — every `future_*` field of a real batch is permuted and garbled (`×3.7 + 11`); `reward_ctx` must be BIT-IDENTICAL and carry no `FORBIDDEN_FUTURE_CTX` key. *(STAGE 0: PASS.)*
2. **Regression-audit power** — `HACKABLE_WEIGHTS` (progress-only) must be FLAGGED by `audit_reward` at the scored geometry (5 × 0.5 s, v0 supplied); an audit that passes it is broken (TRAIN-C4). *(STAGE 0: PASS.)*
3. **Selector disjointness** — structural, every step. *(library-enforced.)*
4. **`humanflag` — the H-RL-THRESH-1 check** (`--mode humanflag`, 0 GPU): the reward's own context scored on the HUMAN driver's logged 2 s future over every RL-fit window with a lead. PASS iff contact and the TTC veto each fire on the human ≤ 0.15. **MEASURED 2026-09-05 on fit8 (318 lead windows, 4 episodes): contact 0.000 (static AND track), TTC veto 0.047, headway < 0.5 on 7.5 % — PASS.** ⚠️ The static-lead defect (a follower reaches the lead's t0 position after one time-gap) is real in construction (`test_collision_time_aligned_against_moving_lead`) but does not bite on these clips: no human window follows closer than **2.93 s** (5th percentile; median 5.8 s) and the median v0 is **2.33 m/s**. It must be re-measured on the 120-clip fit set in STAGE 0 before any arm.
   ⭐ **And the same probe measured the reward's improvement direction:** the constant-velocity straight path (the `ha0` floor) scores **≥ the human on 78.3 %** of lead windows under `track` (70.8 % static; composed mean 0.899 vs 0.861), frozen ≥ human on 27.7 %. The reward, as composed, prefers the trivial floor to the demonstration on most windows — see §6.

## 4. The arms — ONE variable

Common to every trained arm: frozen base, `decoder_steps 2` (the deploy path), EVAL-mode forward for the policy AND the reference (`train_mode_forward False`; TRAIN-C5), `method grpo`, G 4, `normalize none` (Dr. GRPO), multiplicative noise 0.1, `ttc_min_s 1.5`, `w_imitation 0`, `kl_coef 0`, `anchor_form l2`, batch 2, seed 0, AdamW, grad-clip 1.0, RL-fit corpus = the 120-clip fit set, every knob written to the run's `config.json` by `PostTrainConfig.to_dict()`.

| arm | reward | `w_anchor` | lr | steps | role |
|---|---|---|---|---|---|
| **base** | — | — | — | 0 | RL OFF: the banked 40,284 dump (no re-run; bit-identical weights) |
| **rl** | `DEFAULT_WEIGHTS` | 1.0 | 1e-5 | 2,000 | RL ON — the one variable |
| **reg_echo** | `{gt_similarity: 1.0}` INSIDE the advantage, `w_anchor` 0 | 0.0 | 1e-5 | 2,000 | ⛔ deliberate regression: rewards the echo; **G-FAN must fire** or the panel is void |
| **ctrl0** | `DEFAULT_WEIGHTS` | 1.0 | **0.0** | 200 | reproduction control: weights hash-identical before/after, readout delta exactly 0 |

## 5. Readouts — four families, echo gate, tiers

**T0 (training-side, the driver, 120 fixed EVAL windows, seed 1234, deterministic eval mode, per-window values STORED, paired episode-cluster bootstrap 4,000 reps):** R1 composed reward of the full fan · R2 fan contact rate (lead-only) · R3 sel-ADE 2 s (selector output, READOUT ONLY) · **R_FAN** endpoint spread at 2 s (the echo gate's object) · R_REACH mean |fan − bank| · **R_ORACLE** oracle-in-fan ADE 2 s (fan quality) · `sel_idx` agreement with base.

**T1 (the binding read; `openloop_suite.py` per arm on the 141 EVAL clips, grid 2s, stride 5, n_boot 2000, seed 0, `--strict`, criteria checker 2.5.0):** ADE/FDE · **LONGITUDINAL** (speed MAE, along MAE, accel MAE, target-speed hit rates, distance-keeping headway/time-gap/TTC from the eval lead block) · **LATERAL** (heading, yaw-rate, curvature, cross-track) · **TACTICAL** (lat/lon manoeuvre correctness + declared heads vs majority-class floor) · **STRATEGIC** — ⛔ refused with reason and n = 0 exactly as the base's own eval refuses it (no `optionset`; a work item, never a pass). Arms per run: `os`, `os_navshuf`, `os_navzero` (T1), `oracle_sel` (T0), `ha`/`ha0` controls. Pairing: `paired_openloop.py --a-dump <base> --a-arm os --b-dump <arm> --b-arm os --floor ha0` → `rec["families"][FAMILY]["metrics"][KEY] = {delta, lo, hi, separated}` with direction `(arm − ha0) − (base − ha0)`; NEGATIVE favours the arm on lower-is-better metrics. ⚠️ `oracle_sel` is refused as a paired model arm by design (`ORACLE_ARMS`), so fan quality is read from the T0 `R_ORACLE` paired delta.

**Echo gate.** `reg_echo` is EXPECTED to improve R3 and R_ORACLE (it collapses the fan onto the logged path) — an ADE-only table would call it a win. The gate that catches it is **G-FAN: R_FAN falls by ≥ 30 % of base with paired separation**. If G-FAN does not fire on `reg_echo`, the readout cannot see fan collapse and **no PASS is admissible (VOID V2)**.

## 6. ⛔ Committed outcomes — selected MECHANICALLY by `rl_refcv3_min.py --mode verdict` (first match in this order)

| # | condition | exit |
|---|---|---|
| V1 | `ctrl0` absent, or its weights changed under lr 0, or any readout delta ≠ 0 | **VOID** — harness non-determinism |
| V2 | G-FAN did not fire on `reg_echo` | **VOID** — the readout is blind to fan collapse |
| V3 | the paired record for `rl` is VOID (a degenerate arm) or carries no family metric | **VOID** |
| 1 | `rl` R_FAN falls ≥ 15 % of base, separated | **FAIL-COLLAPSE** |
| 2 | `ade_m` paired lo > +0.02 m | **FAIL-GUARD** — the ADE guard |
| 3 | R_ORACLE worse, separated | **FAIL-FAN** — fan quality degraded |
| 4 | ≥ 1 LON/LAT/TAC metric improved with separation and none worsened | **PASS** — RL moved a binding family on this base |
| 5 | some improved AND some worsened | **SPLIT** — report per family, no verdict |
| 6 | `ade_m` improved with separation but R_ORACLE did not | **REJECT-SELECTOR** — the selected path moved, the fan did not (the DDv2 defect) |
| 7 | none of the above | **NULL** — the RL stage is inert on this base |

**Committed PREDICTION (written before any arm, so the outcome is read against it):** exit **7 NULL or 2 FAIL-GUARD**. Two independent measurements point the same way: (i) the v2.1 pilot's EXIT C — under a trust region the composed reward is flat and fan collision never separates at any anchor strength; (ii) §3.2's `humanflag` — the reward ranks the constant-velocity floor above the human on 78 % of lead windows, so its improvement direction from the demonstration is toward `ha0`, which the eval already ranks **0.2304 m worse** than the model (`os` − `ha0`, separated). A PASS would REFUTE the pilot's method verdict on refcv3 and would be followed by a seed-replication and a `w_anchor` ladder before any registry claim; a NULL/FAIL **CONFIRMS** it and closes the "RL on the offset head with a rule reward" line for refcv3 — the next lever is the fan (`H-DDA-3`, the vocabulary), not the scoring.

## 7. Gates IN ORDER (an arm earns the next only by clearing the previous) — and the fixes applied on 2026-09-05

STAGE 0 (always runs, trains nothing, no checkpoint): base md5 → fitlist (∩ eval = 0) → **`humanflag` PASS** → model strict-load → trainable surface (no non-decoder, no selector tensor) → no-ego-future permutation test → regression-audit power → step-0 trust-region divergence → lr = 0 timing with weights hash-asserted UNCHANGED → `preflight.json` PASS. Then `LAUNCH_APPROVED=1` gates STAGES 1–5 (clips → fit lead block → arms → T1 eval → pairing + verdict).

Fixes this resumption applied (all in the off-Drive clone `C:\Users\Admin\refcv4b_repo`, staged for the repo — see RESULT.md §7): **(1)** `refcv3_arm.load_model` tolerates the ONE inert persistent buffer `core.decoder.anchor_controls` (added by refcv4-b, read only when `v0_conditioned`) — recorded in provenance, everything else still strict; **(2)** `rewards._collision` gains a TIME-ALIGNED moving-lead branch (+ pinning test) so the lead need not be held static; **(3)** driver: `--lead-mode track`, `R_ORACLE`, `--mode humanflag`, and a verdict that reads `paired_openloop.py`'s REAL schema (the predecessor's read a key the tool never writes — structurally NULL); **(4)** the step-0 divergence gate — see RESULT.md §3 for the measured cause and the rule adopted.

## 8. Cost (MEASURED on the dev-box 4060, STAGE 0 timing at lr = 0; see RESULT.md §5 for the numbers)

Training: `steps × s/step` for `rl` (2,000) + `reg_echo` (2,000) + `ctrl0` (200) — filled from `preflight.json` `timing.est_train_min_*`. Eval: `openloop_suite.py` × 3 arms on 4,823 windows (the base's own eval took the same path on the pod; dev-box estimate in RESULT.md §5) + 3 pairings (CPU, minutes) + the clip pull (112 × ~1.2 s). ⛔ **Nothing past STAGE 0 runs without `LAUNCH_APPROVED=1` from the Master Mind / PI, which is an approval of this cost.**

## 9. Stated limits (each declared, none silent)

1. NON-PARITY fit corpus; only the paired margin over `ha0` is admissible cross-arm. 2. Surrogate Gaussian policy on the emitted offset — not a denoising density; DDv2-estimator parity UNVERIFIED (standing limit). 3. `logp` is summed over all 8 slots while the reward scores the 2 s prefix — unbiased (the extra dims are zero-mean in the score-function expectation) but higher variance. 4. Contact is a SAMPLED check on 0.5 s waypoints (r = 2 m): at 10 m/s a 5 m stride can step over a 2 m obstacle — the TTC veto is the continuous guard. 5. The lead track is privileged (other-agent future) and does not exist at inference. 6. STRATEGIC family refused (n = 0) on this eval path, as for the base. 7. Four eval waypoints ⇒ jerk from one sample; comfort is coarse. 8. One seed; a PASS needs replication before any claim.

---

## 10. ⭐ RE-SCOPE 2026-09-05 — the PI's correction. PRIMARY endpoint = FAN SAFETY

> **The PI, verbatim:** *"Regarding RL, I don't agree you can do only what you stated, the paper is
> talking about improving/eliminating trajectories leading to [collisions / infeasible outcomes]."*

⛔ **§0–§9 above are KEPT UNCHANGED as the record.** This section supersedes the **primary
endpoint**, the **arm definitions** and the **committed outcomes**; everything else in the SPEC —
assets, corpus, gates, tiers, estimator, stated limits — stands verbatim. Dated so a reader can see
which prediction was committed when.

### 10.1 What was wrong with the original endpoint

`H-RL-MIN-1`'s primary endpoint was the **four-family distance to the human**. Under the composed
reward the constant-velocity floor (`ha0`, hold-v0-straight) already scores **≥ the human on 78.3 %
of lead windows** (§3.2, MEASURED). A reward whose improvement direction points at the trivial floor
cannot move an arm *toward* the human, which is exactly why the committed prediction was **NULL**.

The PI's point is that this endpoint **cannot see V2's mechanism at all**. DiffusionDriveV2's RL
stage does not exist to reduce distance-to-human; it exists to **push probability mass away from
collision-prone and infeasible candidates**. Pinned from the released code
(`hustvl/DiffusionDriveV2@1cd12a1`, PUBLISHED-CODE, read in full by
`…/Architecture & Inference/Research/2026-09-05-diffusiondrive-v2-analysis/RESULT.md`, `D-DDV2-*`):

| ingredient | code | what it does |
|---|---|---|
| collision → **−1**, pinned | `_model_rl.py:891-902` | a colliding candidate is a CONSTRAINT violation, not a low rank — no good behaviour elsewhere buys it back |
| positive advantage **only above the ≥GT bar** | `_model_rl.py:891-893` | `reward > reward_gt − 1e-6`, then `clamp(min=0) * mask` — the GT scored under the SAME reward as a candidate |
| **two-scalar** (along, lateral) noise | `_model_rl.py:646-654` | one longitudinal + one lateral scalar per trajectory, broadcast over waypoints; the additive DDPM term is ×0 at `:640/:666` |

⇒ **Measuring an RL stage built for constraint-pruning by its distance to the human is measuring the
wrong axis.** Both ingredients above are now IMPLEMENTED and pinned (`stack/tanitad/rl/advantage.py`
`gt_bar`, `refcv3_adapter.py` `noise_mode="two_scalar"`, `stack/tests/test_rl_v2_faithful.py`,
commit `62a014f`) and are part of the `rl` arm's definition below.

### 10.2 The PRIMARY endpoint — fan safety on the EMITTED fan

Scored on the **2 s prefix** (origin + slots 0.5/1.0/1.5/2.0 s — the same index-select the `2s` eval
grid uses), on the **whole fan**, the **top-8** and **top-32** by the score the argmax actually uses,
and the **selected** trajectory. Instrument: `taniteval/tools/fan_safety.py` (commit `aabb35a`),
pinned by `stack/tests/test_fan_safety.py`.

| # | metric | definition | population |
|---|---|---|---|
| **(a)** | `contact` | time-aligned centre distance < 2 m (ego 1 m + obstacle 1 m) against the lead agent's own replayed `obstacle.offline` track — `rewards._collision` moving-lead branch | lead windows |
| **(b)** | `ttc_below` | TTC < **2.93 s** (`H-RL-THRESH-1` class — the HUMAN's own 5th-percentile time gap on the fit8 lead windows, so a candidate below it follows closer than the human ever does). `ttc_veto` at 1.5 s reported beside it | lead windows |
| **(c)** | `infeasible` | `kamm_over` (friction load > μ 0.7) ∨ `envelope` (\|a\| > 4 m/s² ∨ \|κ\| > 0.2 1/m) ∨ `off_reach` (outside the S2 reach band refcv3 does NOT apply at inference). **Each reported separately as well as OR-ed** | all windows |
| **(d)** | `mass_rank_*`, `mass_conf_*` | the probability `softmax(sel_score)` (the score the argmax uses, `reach_keep`-masked so excluded rows carry EXACTLY zero) and `softmax(anchor_logits)` (`conf_head`) put on each flag | as the flag |

**Direction:** every one of (a)–(d) is **lower-is-better**. Read as the **paired episode-cluster
bootstrap** (`taniteval/ci.py`, n_boot 2000, 141 clusters) of `arm − base` on the SAME windows, each
on its own population with `n` printed. ⛔ Never `overlapping_holdout_se`.

**The reference every "safer" claim is read against is the HUMAN's own rate** on the same windows
(`fan_safety.py --dump`, arm `g`), with the `ha`/`ha0` trivial floors beside it. A fan that is
already safer than the human has no room the RL stage could buy, and saying so requires the human's
number in the table.

### 10.3 Stated limits of the primary endpoint — declared here, not discovered later

1. ⚠️ **`kamm_over` is a LOWER bound.** The emitted fan is FREE WAYPOINTS (bank + offset), so the
   exact control-rolled `instruments/flyability.friction_load` cannot be applied and a finite
   difference over the 0.5 s prefix is used. `flyability.py`'s own docstring MEASURES that finite
   difference under-reporting by **1.21–1.85×**. Under-reporting is the dangerous direction; it is
   declared and the rate is labelled a lower bound wherever it is quoted.
2. Contact is a **SAMPLED** check on 0.5 s waypoints (§9.4 stands): at 10 m/s a 5 m stride can step
   over a 2 m obstacle. The TTC family is the continuous guard.
3. The lead track is **privileged** (another agent's future) and does not exist at inference. It is a
   TRAINING and SCORING signal only — admissible under the binding label/inference split.
4. Only the **lead** agent is replayed, not the full `obstacle.offline` scene, so (a) is a contact
   rate against the lead, not against all traffic. It is the same object before and after, so the
   paired delta is valid; the absolute level is a floor on true contact.
5. `off_reach` measures candidates outside a band refcv3 **does not apply** (`sel_reach_clamp`
   False). It is a property of the emitted fan, not a violated constraint.

### 10.4 The arms — re-scoped

| arm | reward | `use_gt_bar` | `noise_mode` | `w_anchor` | lr | steps | role |
|---|---|---|---|---|---|---|---|
| **base** | — | — | — | — | — | 0 | RL OFF: the banked 40,284 dump + its fan scored by `fan_safety` |
| **rl** | `DEFAULT_WEIGHTS` | **True** | **two_scalar** | 1.0 | 1e-5 | 2000 | RL ON, **V2-faithful** — the one variable |
| **reg_echo** | `{gt_similarity: 1.0}` INSIDE the advantage | False | two_scalar | 0.0 | 1e-5 | 2000 | ⛔ deliberate regression: the reward IS the echo. **G-FAN must fire** |
| **ctrl0** | `DEFAULT_WEIGHTS` | True | two_scalar | 1.0 | **0.0** | 200 | reproduction control: weights hash-identical, every readout delta EXACTLY 0 |
| **ctrl_const** | **all weights 0.0** | False | two_scalar | 1.0 | 1e-5 | 200 | ⭐ **constant-only control** |

⭐ **`ctrl_const` is the control the probe-panel rule requires** (CLAUDE.md 2026-08-22: four probe
failures in one afternoon, every one caught ONLY by a control that had to read a known value).
Its reward is exactly `0.0` for every candidate — MEASURED, not asserted: `RewardSpec` over these
weights returns a tensor whose unique value is `{0.0}` and whose std is `0.0`. A constant reward has
an identically-zero group-relative advantage, so the only surviving gradient is the anchor trust
region, itself zero while the live fan equals the frozen reference. ⇒ **it must read the
no-information value exactly.** It differs from `ctrl0` in what it proves: `ctrl0` disables the
OPTIMIZER (lr = 0); `ctrl_const` leaves the optimizer live and removes only the INFORMATION. **If
`ctrl_const` moves, something other than the reward is driving the update and no `rl` result above it
means anything.**

**One variable.** Against `base`, the single change is *the RL stage*. The two V2-faithful
ingredients are INSIDE the stage's definition (they are what makes it V2's stage rather than the
predecessor's), and are named here so this arm can never be read as the earlier arm with a different
result. `n` and `d` are printed for every panel row: `d` = **9,206,032 / 107,032,901 trainable params
(8.60 %)**, `n` = windows and episodes per population.

### 10.5 SECONDARY endpoint — the four families, prediction RETAINED

The four binding families (LONGITUDINAL · LATERAL · TACTICAL · STRATEGIC) stay exactly as §5
specifies, at **T1 (self-action open loop)**, paired over the shared `ha0` floor, criteria-checked.
⛔ **Their committed prediction is UNCHANGED: NULL or FAIL-GUARD**, for the reason in §6 — the
reward ranks `ha0` above the human on 78.3 % of lead windows, so its improvement direction is toward
a floor the eval already ranks 0.2304 m worse. Retaining a prediction that a re-scope could have
quietly dropped is the point: the four families are the guard that a safety gain was not bought with
driving quality.

### 10.6 ⛔ COMMITTED OUTCOMES — written before any arm ran

Selected mechanically, first match in order. VOID gates V1–V3 from §6 still apply, **plus**:

| # | condition | exit |
|---|---|---|
| **V4** | `ctrl_const` moved ANY fan-safety metric with paired separation | **VOID** — something other than the reward drives the update |
| **V5** | the base fan's contact rate is **0.0000** on every population | **VOID-NOROOM** — the endpoint has no headroom on this corpus; report the human's rate and say the experiment could not have detected an improvement |
| **1** | fan-safety mass **drops** (≥ 1 of (a)–(d) improves with paired separation, none worsens) **AND** every four-family metric stays inside its CI | **PASS — V2's mechanism TRANSFERS.** The RL stage prunes unsafe candidates on refcv3 without costing driving quality |
| **2** | no fan-safety metric moves with separation | **NULL-SAFETY.** Report WHICH of the two causes holds, with the human's own contact rate as the reference: (i) **the fan was already safe** — base rates at or below the human's, no room; or (ii) **the reward cannot see the collisions** — base rates materially above the human's while the reward's collision term never fires (`audit.report_component_coverage`), i.e. the signal exists and the reward is blind to it |
| **3** | fan safety improves with separation **AND** a four-family metric degrades beyond its CI | **TRADE-OFF.** Report both, per family, no single verdict. A safer fan bought with worse driving is a finding, not a pass |
| **4** | fan safety **worsens** with separation | **FAIL-SAFETY** — the stage makes the fan less safe; the reward's direction is wrong on this base |

**Committed PREDICTION for the PRIMARY endpoint (written before any arm):** outcome **2
NULL-SAFETY, sub-case (i)** — the base fan is already at or below the human's contact rate, so there
is little room. Basis: the `humanflag` probe MEASURED the human's own contact at **0.000** under both
lead models on 318 fit8 lead windows, and refcv3's own eval has never shown a contact-rate defect.
⚠️ This prediction is **falsifiable in a way the original was not**: if the base fan carries real
contact mass on the 141 EVAL clips, outcome 1 or 4 is reachable and the experiment is informative
either way. If V5 fires, the honest report is that the endpoint had no headroom — which is itself the
answer to whether V2's mechanism has anything to prune here.

**What each outcome decides.** PASS ⇒ the offset-head RL line stays open on refcv3 and earns a
seed-replication plus a `w_anchor` ladder before any registry claim. NULL-SAFETY(i) ⇒ the fan is not
where refcv3's problem is, and the money goes to `H-DDA-3` (the vocabulary/sampler). NULL-SAFETY(ii)
⇒ the reward is the work item, not the policy surface. TRADE-OFF ⇒ the four families are the binding
constraint and the reward needs a driving-quality term that is not `gt_similarity`. FAIL-SAFETY ⇒
the line closes on this base.

