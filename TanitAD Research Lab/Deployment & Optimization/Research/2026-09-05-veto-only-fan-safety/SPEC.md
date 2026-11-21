# SPEC — VETO-ONLY FAN SAFETY: promote refcv3's one working RL channel from accident to product

`TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-veto-only-fan-safety/SPEC.md`
Architecture & Inference FlyWheel · 2026-09-05 · dev-box RTX 4060 only · ⛔ `tanitad-refcv3` (refcv4b live) and Thor are never touched.
Register: `H-VETO-FAN-1` (new, this SPEC) · cites `D-RL-READY-1`, `D-RL-REWARD-FLOOR-1/2`, `H-RL-MIN-1`, `D-RL-FANSAFE-1`, `H-ESTIM-SEED-1`.
Evidence classes: **MEASURED** (ours + artifact path) · **INHERITED** (another WP, not re-run) · **HYPOTHESIS**. Tiers: **T0** = readout on the emitted fan (never a driving claim) · **T1** = self-action OPEN loop through the taniteval harness.

⛔ **THIS DOCUMENT IS BANKED BEFORE ANY ARM RUNS. Both outcomes are written below and neither is preferred.**

---

## 1. The PI's standing correction, and what it makes this SPEC

> *"our goals in TanitAD programme is not to refute hypotheses, its about achieving excellent results and really driving autonomously with a reference implementation. Why im saying this, we have too many refutes, and I have the feeling, we are satisfied sometimes if the hypotheses is refuted rather than solving the problem"* — PI, 2026-09-05

This is a **construction** SPEC. The deliverable is a fan that is measurably **safer** than refcv3's is today, banked with its number. A refutation is a by-product, never the product.

## 2. What we are building on (MEASURED, INHERITED from `…/2026-09-05-refc-rl-readiness/RESULT.md`)

The V2-faithful RL stage made refcv3's fan **less** safe (`sel_infeasible` 0.133 → 0.642) and its T1 planner worse than the constant-velocity floor (`ade_m` 0.4419 → 0.9433 vs `ha0` 0.6723). But the panel's *control* arm — `ctrl_const`, every reward weight `0.0` — moved feasibility the **right** way:

| `ctrl_const` − base (200 steps, 2 min 07 s) | delta |
|---|---|
| `fan_peak_g_mean` | **−0.0859 g** |
| `top32_infeasible` | **−0.0143** |
| `top8_kamm_over` | −0.0137 |
| `fan_kamm_over` | −0.0048 |
| `fan_infeasible` | −0.0026 |

⭐ **The mechanism, MEASURED in the source, not inferred:** `posttrain.py:206` keyed the veto on `"collision" in spec.weights` — KEY MEMBERSHIP, true at weight `0.0` — and `advantage.py:146` pins vetoed candidates at `−1.0` **outside** the group-relative centring. So a constant reward gives a **veto-only advantage at full strength**, not a zero one (`veto_rate_mean` **0.0897**, `final_loss` **−1.863**). `ctrl_const` was an unintentionally exact **veto-only arm**. ⇒ **DDv2's constraint channel transfers; our composed reward does not.**

⚠️ **The gap worth aiming at is NOT contact.** Selected-path contact: human `g` **0.0000**, refcv3 `os` **0.0007**, `os − g` **not separated** over 1,466 lead windows / 78 episodes — we already tie the human. The un-exploited gap is the **envelope**: `0.0068` (human) vs **`0.0865`** (refcv3) = **12.7×**, and `flagged` **4.0×**. ⛔ And do not simply add a feasibility term: the arm that made feasibility worse already carried `feasibility: 0.5`.

## 3. What is built here (the deliverables, in priority order)

| # | deliverable | GPU | why it comes first |
|---|---|---|---|
| **P1** | `stack/scripts/rl_reward_envelope_rank.py` — Spearman ρ between the composed reward and envelope/Kamm violation **over candidates within a window's own fan**, with a constant control, a random control, a self-correlation control, per-component attribution, and a **weight panel** scored on the same fan at zero extra cost. Banks the fan (`--out-npz`) so any future reward design is scorable at **0 GPU**. | one forward | ⛔ **It can disqualify the reward at source.** A group-relative advantage is a RANKING; if the reward ranks violating candidates higher, no amount of RL repairs it. |
| **P2** | The veto becomes an **explicit, recorded** configuration field: `PostTrainConfig.veto_enabled` / `veto_collision` / `veto_ttc`, composed by `posttrain.veto_mask()`, written into every run's `config.json` by `to_dict()`. Pinned by `stack/tests/test_rl_veto_explicit.py` (8 tests). New arms `veto200` / `veto2k` (veto-only **products**) and `ctrl_null` (the null `ctrl_const` was supposed to be). | 0 | A zero-weight control that is not a null is the false-green class wearing a control's clothes. It fired the pre-registered `V4` VOID gate and cost the whole `rl`-vs-base attribution. |
| **P3** | The veto-only arms as a product, **with a seed replicate**. | ~36 min | `H-ESTIM-SEED-1`: a separated CI from a one-seed arm is **necessary, not sufficient**. |
| **P4** | T1, four families, per family, never pooled, paired episode-cluster bootstrap. | ~40 min/arm | ADE alone is an INCOMPLETE result (binding). |

⭐ **Found while building P1 and fixed in the same turn (MEASURED):** `rl_refcv3_min.readout()` built its reward context with `reward_ctx()`, which is shaped for the **training** tensor `[B, N, G, S, 2]`, and handed it a `[B, N, S, 2]` **fan**. Every scene fact then broadcast against the wrong axis: `_collision`/`_headway`/`_progress` returned `[B, B, N]`, a **batch × batch outer product** in which each window's fan is scored against every window's lead. ⇒ the banked panel's `R1`/`R2` readouts (readout batch = 4) are a batch-mixed statistic. **Scoped, not blanket-voided:** training is unaffected (its tensor really does have three leading axes) and the PRIMARY fan-safety metrics are unaffected (`FS.score_paths` is called with correctly-ranked `lead5`/`v0`). Fixed by `reward_ctx(..., cand_dims=1)` plus a **positive shape assertion** in `readout`. Same family as the units trap: a correct quantity computed in the wrong SCOPE reads exactly like an answer.

## 4. The arms — committed before any of them runs

Base: `ckpt_step40284_frozen.pt`, md5 `b1ed7075ff730d0993d2eaa3c86f6b56`, step asserted 40284. Fit corpus: the 120 train-split B1 v7.2 clips (`fit120`), lead block `fit120_lead_block.npz` — **NON-PARITY, as the base itself is**. Readout: the **same fixed 120 EVAL windows** (`random.Random(1234)`), 79 episodes, 36 lead windows, so every arm is compared on identical windows.

| arm | reward weights | veto | w_anchor | lr | steps | seed | role |
|---|---|---|---|---|---|---|---|
| `ctrl_null` | all `0.0` | **OFF** | 1.0 | 1e-5 | 200 | 0 | ⛔ **the P2 proof**: `veto_rate_mean` must read **EXACTLY 0.0** and no fan-safety metric may separate |
| `veto200_s0` | all `0.0` | **ON** | 1.0 | 1e-5 | 200 | 0 | reproduces the measured `ctrl_const` gain, now by design |
| `veto200_s1` | all `0.0` | **ON** | 1.0 | 1e-5 | 200 | **1** | the **replicate**: same flags, different seed |
| `veto2k_s0` | all `0.0` | **ON** | 1.0 | 1e-5 | **2000** | 0 | the dose the brief asks for |
| `veto2k_s1` | all `0.0` | **ON** | 1.0 | 1e-5 | **2000** | **1** | the **replicate** at that dose |

Everything else is held: `group 4`, `noise 0.1`, `noise_mode two_scalar`, `batch 2`, `use_gt_bar False`, frozen trunk, `TRAINABLE_PREFIXES` = `core.decoder` minus the 9 selector prefixes, `train_mode_forward False`.

⚠️ **Two doses are run on purpose.** The measured gain exists at 200 steps. A pure constraint signal has no ranking term to trade against, so more steps could over-shoot (push every candidate away from the lead). Running both makes the dose a **measurement** rather than an assumption, and costs 4 extra minutes.

## 5. ⛔ THE NOISE FLOOR — the thing that makes a separated CI mean something

`H-ESTIM-SEED-1` (CLAUDE.md, MEASURED 2026-09-05): the episode-cluster bootstrap resamples **EPISODES with the models held fixed**, so it answers *"would another draw of episodes say this?"* and never *"would another training run say this?"*. A zero-lever replicate produced "separated" on **3 of 18** metrics — a ~17 % false-positive rate.

⇒ For every metric, the **replicate floor** is
```
floor(m) = | delta(veto_s0 - veto_s1) |            # paired, same estimator, same windows
```
and a lever effect is quotable **only** when all three hold:
1. `veto_sK − base` is separated **for both K = 0 and K = 1**, and
2. the two point estimates have the **same sign**, and
3. `min(|delta_s0|, |delta_s1|) > floor(m)`.

⚠️ A metric failing (3) is reported as **WITHIN-NOISE**, not as a null and not as a win.

## 6. ⛔ SEPARATION FLOOR — per metric, from its own quantum and units

The predecessor used a single `MIN_EFFECT = 1e-4` derived as a **rate** quantum (`1/(128 × 120)`), and it mis-fits `fan_peak_g_mean` (units **g**) and `fan_v_mean_2s_spread` (units **m/s**), where a rate quantum has no meaning. It also made `mass_rank_contact` **UNDETECTABLE-DOWNWARD**: its base value `3.470e-05` is `0.347 ×` the floor, so only a worsening could ever be reported.

⇒ Here: rate metrics keep `1e-4`; `fan_peak_g_mean` / `sel_peak_g` use **1e-3 g**; `fan_v_mean_2s_spread` uses **1e-3 m/s**. **Any metric whose base value is below its own floor is stamped `UNDETECTABLE-DOWNWARD` and is not reported as a null.**

## 7. ⭐ ACCEPTANCE — both outcomes committed in advance, neither preferred

**PRIMARY endpoint: fan feasibility**, T0 readout on the 120 fixed EVAL windows, paired episode-cluster bootstrap (`taniteval/ci.py`), n_boot 4000, seed 11. Metrics, in the order they are read:
`fan_peak_g_mean` · `top32_infeasible` · `sel_infeasible` (+ reported beside them: `fan_infeasible`, `top8_kamm_over`, `fan_kamm_over`, `top32_envelope`, `sel_ttc_below`, `mass_rank_*`).

> ### ✅ SUCCESS
> Fan feasibility **improves** — a **negative** delta on `fan_peak_g_mean` **and** on at least one of `top32_infeasible` / `sel_infeasible` — with a separated paired CI **that also clears the seed-replicate floor of §5 on that same metric**, **AND** `ade_m` at T1 does not regress by more than that same floor.
>
> ### ⛔ FAILURE
> The feasibility gain does **not** clear the replicate floor (i.e. it is within the rig's own run-to-run noise), **OR** `ade_m` regresses past that floor.

**Guards that VOID the panel before either outcome is selected, in order:**
- **G1 `ctrl_null`** — `veto_rate_mean` must be **exactly 0.0**. If it is not, P2's fix did not take and nothing below is quotable.
- **G2 pairing** — every arm's BEFORE readout must be **bitwise identical** to the base's on all 120 windows × all fan-safety metrics (max abs diff `0.000e+00`). Asserted in the artifact, not assumed.
- **G3 replicate exists** — a result quoted from a single seed is VOID by §5.

**SECONDARY endpoint: the four families at T1** (`taniteval/tools/paired_openloop.py`, self-action OPEN loop, paired episode-cluster bootstrap over the shared `ha0` floor), reported **per family, never pooled**: LONGITUDINAL (speed MAE, along MAE, accel MAE) · LATERAL (cross-track, heading, yaw-rate) · TACTICAL (traj lat/lon correctness; the `declared_*` rows are **STRUCTURAL ZEROS** — `core.maneuver`/`core.route`/the goal heads are frozen, so identical inputs through frozen heads give identical outputs; that is an identity, not an estimate) · STRATEGIC (route correctness; also structural). ADE/FDE are reported **beside** the families, never as "the result".

**Committed SECONDARY prediction (HYPOTHESIS, stated now so it cannot be claimed afterwards):** the veto is a pure constraint with no ranking term, and the trust region (`w_anchor 1.0`) is intact, so the expected T1 movement is **NULL** — no separated four-family degradation beyond the replicate floor. A separated *improvement* would be a surprise; a separated *degradation* is the FAIL-GUARD.

## 8. What this SPEC does NOT claim, whatever the result

1. Nothing about **closed loop**. T1 is self-action OPEN loop (PI ruling 2026-09-02).
2. Nothing about **contact**: refcv3's selected-path contact already ties the human and there is no headroom there (`D-RL-FANSAFE-1`, registered before the previous arms ran).
3. `kamm_over` is a **LOWER bound** — the emitted fan is free waypoints, so `flyability.friction_load` (the exact, control-rolled instrument) cannot be applied and the finite difference under-reports by 1.21–1.85×. Rank correlations in P1 are invariant to a monotone under-report; levels are not.
4. The fit corpus is **NON-PARITY** (120 train-split B1 v7.2 clips), as the base itself is.
5. Two seeds bound the rig's training noise; they do not make it small.

## 9. Cost and machine discipline

MEASURED on the same rig: 200 steps ≈ **2 min 07 s**, 2,000 steps ≈ **15 min 09 s**, T1 eval ≈ **40 min/arm**. Total ≈ 36 min of arms + T1. Dev-box RTX 4060 only, **one arm at a time** (`OMP_NUM_THREADS=6`; 7 concurrent arms once sat at 0–6 % sm for 50 min). ⛔ `tanitad-refcv3` is training refcv4b-b1-v72-40k and is never touched; Thor is not used.
