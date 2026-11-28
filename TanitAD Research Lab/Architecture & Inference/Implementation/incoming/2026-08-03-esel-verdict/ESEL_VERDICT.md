# E-SEL-0 / E-SEL-1 — THE VERDICT

**Date:** 2026-08-03 (Europe/Berlin) · **Stream:** arch-inf · **GPU cost:** ~101 s of an idle Jetson
Thor, two arms. No training launched. No training pod touched.

**Pre-registration:** `Project Steering/PREREG_D-SEL_REFC_SELECTION_SURFACE.md`
**Blob id at read time:** `52c005840ef23788aac561330fb115709fa733df` — `git ls-files -s` and
`git hash-object` **MATCH**, so §6.3's thresholds are the ones that were staged. The prereg's own
§10.1 verification passes. Every threshold below is transcribed into
`refc_sel_probe.PREREG_THRESHOLDS`; none was edited by a run.

**n = the canonical 881 windows / 40 episodes**, both arms. All four negative controls **PASS**.

---

## 0. THE VERDICT

| experiment | registered prediction | **MEASURED — `refc-base-30k`** | **MEASURED — `refc-xl-30k`** | verdict |
|---|---|---|---|---|
| **E-SEL-0** | *S1 NEEDS TRAINING* (**not separated**; C-shuffled clearly worse) | refined **WORSE** by **+0.8372 m** [+0.6915, +0.9939], **separated** | refined **WORSE** by **+0.9187 m** [+0.7778, +1.0669], **separated** | ⚠️ **OFF THE REGISTERED MAP.** No §6.3 branch covers "separated in the ADVERSE direction". **My prediction was wrong.** |
| — its sub-clause | C-shuffled clearly worse than refined (gap ≥ **0.05**) | gap **0.1103** ✔ · `rank_acc` **0.0681 = 8.7× chance** | gap **0.1120** ✔ · `rank_acc` **0.0647 = 16.6× chance** | **HELD, both arms.** **S1's premise is NOT dead** — the refined readout carries real ranking information. |
| **E-SEL-1** | *S3 LIVE at a small ρ, **0.10–0.25*** | **ρ = 0.6657** [0.6183, 0.7157]; paired vs shuffled **+0.6667** [+0.6181, +0.7187] | **ρ = 0.6212** [0.5650, 0.6791]; paired **+0.6235** [+0.5677, +0.6807] | **S3 LIVE**, both arms. Direction right, **magnitude badly understated**. ⚠️ **§4's caveat is binding.** |
| **S2** | inert on ADE; its value is compute | 73.76 % removed, oracle survives **100 %**, paired Δ **exactly 0.0** | **72.08 %** removed (reproduces the published figure exactly), paired Δ **exactly 0.0** | **Confirmed — and extended: S2 cannot rescue S1** (paired Δ exactly 0.0 for the refined ranker too). |

### The one-paragraph answer

**S1 is not a free win; it is a hole.** The confidence REF-C computes and throws away is, untrained,
a **~0.84–0.92 m WORSE** ranker than the t=0 classifier score it ships — separated, on both arms, at
canonical n. It is not noise (8.7×/16.6× chance) so the premise survives, but supervision must climb
out of a deficit rather than harvest a surplus, and **S2 fills none of it**. **S3 is the lever with
information behind it** — REF-C's world model discriminates candidates at ρ ≈ 0.62–0.67 — but that
ρ is measured against the **future frame**, which the implemented S3 never sees (§4).

> **The prereg said I would say so if I was wrong, so: I was wrong.** The registered prediction was
> *S1 NEEDS TRAINING*, whose first conjunct is *"E-SEL-0 not separated"*. E-SEL-0 **is** separated,
> in a direction the pre-registration did not enumerate. I am not reframing that into the nearest
> branch. §3.1 states exactly what it does and does not license.

---

## 1. What was run, and where

| | |
|---|---|
| arms | **`refc-base-30k`** (128 anchors) and **`refc-xl-30k`** (256), both step 29,999 |
| n | **881 windows / 40 episodes** — the canonical val protocol, both arms |
| estimator | **`paired_episode_cluster_bootstrap`** / `episode_cluster_bootstrap`, unit = **episode**, `n_boot = 2000`. ⛔ `overlapping_holdout_se` never called |
| decode | `nav_mode = follow_constant`, `steps = 2`, window 8, stride 8 — identical to the banked fans, deliberately (§6) |
| host | **Jetson AGX Thor** (`thor6`, aarch64, torch 2.13.0+cu130), GPU idle before and after |

⛔ **Both live RunPod pods are TRAINING** (`tanitad-pod5` flagship-v5f at 100 % GPU; `tanitad-pod4`
v1arch-v2bal at 36 %) and were not touched. **`tanitad-eval` is gone** — two independent probes
(`ssh tanitad-eval` and a raw `root@69.30.85.106:22073`) both return *Connection refused*, and
`FLEET_INVENTORY_2026-07-28-migration.md` records that the PI terminated it. Its `/root/models/` and
its ~27-arm per-window dump surface went with it. Thor is the box the prereg's §6.4 names as the
alternative, and it turned out to hold **everything the experiment needed**.

---

## 2. THE FOUR NEGATIVE CONTROLS (prereg §6.0) — run FIRST, all four PASS on both arms

| control | `refc-base-30k` | `refc-xl-30k` |
|---|---|---|
| **C-raster** | fed raster **(256, 256)** vs the arm's declared `grid_shape` **(8, 8) = 64 tokens**, asserted **before** any window was scored | same ✅ |
| **C-identity** | `argmax(logits) == sel_idx` on **1.0000** of windows; selected ADE **0.4728** == published **0.4728** | **0.4714** == published **0.4714** ✅ |
| **C-identity-vs-bank** | vs the eval pod's banked fan: selection agreement **1.0000**, per-window max \|fan diff\| median **~4e-04 m** | agreement **1.0000** ✅ |
| **C-oracle-floor** | **0.1914** == published **0.1914** | **0.16395** vs published **0.1640** ✅ |
| **C-shuffled** | **14.5426 m**, `frac_sel_2x_worse` **0.9798** | **13.9564 m**, **0.9894** ✅ |

⭐ **A Jetson Thor reproduces the eval pod's REF-C numbers exactly.** Cross-architecture (aarch64 vs
x86), cross-torch-build, selection agreement **1.0000** on both arms. The terminated eval pod did
not take REF-C's reproducibility with it.

### 2.1 C-shuffled is EXACT, not sampled — a property of the control, not a shortcut

Permuting a score along the candidate axis and taking `argmax` returns `π⁻¹(argmax s)`, which for a
uniform `π` is a **uniform random candidate whatever the score was**. C-shuffled is therefore
**score-independent** — the shuffled control for the refined confidence is the same distribution as
the shuffled control for the shipped logits. So it is computed as the exact per-window expectation
(the mean over candidates), removing seed noise from a control the §6.3 branches are read against.
Self-test: closed form **14.5426** vs 24 empirical draws **14.6394 ± 0.37**.

### 2.2 ⚠️ TWO CONTROLS FAILED FIRST, AND BOTH FAILURES WERE MINE

**(a) A renumbering bug that compared different clips.** One val clip read as truncated on the first
pass, and `taniteval.data.load_frames` assigns `episode_id` **by position in the list it is given** —
so handing it a list with that clip removed **renumbered every later clip**. `C-identity-vs-bank`
then reported *"agreement 0.7183, fans 22 m apart"*, which looked like a hardware difference and was
not. Caught by a **GT-based alignment assert** (the ground truth is model-independent and must be
identical for identical windows). Fixed by constructing each episode with its **original** index and
leaving the gap as a hole. ⇒ The control is now **GT-alignment-first** and refuses to report any
bank comparison until the GT matches. *(The clip then read fine on a later pass — a transient
read, not a corrupt file. Both final runs are at the full 881.)*

**(b) A `max`-based threshold that fired on arithmetic nobody selects.** 72–74 % of the fan is
outside a bounded-acceleration band, and a float difference on a candidate implying 171 km/h moves
it metres while changing nothing that is ever picked. The control now judges on **selection
agreement + the median window** and *reports* the max.

**(c) A published-value tolerance set at exactly the rounding boundary.** XL's oracle is 0.16395;
the registry stores **0.1640** at 4 dp, so a 5e-5 gate called a 5.1e-5 deviation a FAILURE. Widened
to 1.5e-4 with the reason written next to the constant — and the **exact** check is
`C-identity-vs-bank`, which reads 1.0000.

---

## 3. E-SEL-0 — the discarded refined confidence is a MUCH WORSE ranker

**881 windows / 40 episodes, paired episode-cluster bootstrap, same windows.**

| ranker | base ADE@2s | base `frac2x` | base `rank_acc` | XL ADE@2s | XL `frac2x` | XL `rank_acc` |
|---|---|---|---|---|---|---|
| **shipped** (t=0 classifier — what REF-C emits) | **0.4728** | 0.4109 | **0.3292** | **0.4714** | 0.4540 | **0.3110** |
| shipped **+ S2** band | 0.4728 | 0.4109 | 0.3292 | 0.4714 | 0.4540 | 0.3110 |
| **refined** (S1's score, **UNSUPERVISED**) | **1.3100** | 0.8695 | 0.0681 | **1.3901** | 0.8774 | 0.0647 |
| refined **+ S2** band | **1.3100** | 0.8695 | 0.0681 | **1.3901** | 0.8774 | 0.0647 |
| oracle-in-fan | 0.1914 | 0.0000 | 1.0000 | 0.1639 | 0.0000 | 1.0000 |
| shuffled (random) | 14.5426 | 0.9798 | 0.0078 | 13.9564 | 0.9894 | 0.0039 |

* **paired refined − shipped:** base **+0.8372** [+0.6915, +0.9939] · XL **+0.9187** [+0.7778, +1.0669] — **separated**, `p(Δ>0) = 1.000`, both arms.
* **paired (refined + S2) − refined: exactly +0.0000 [0.0000, 0.0000], not separated.**
* `corr(anchor, refined)` = **0.9239** (base) / **0.8998** (XL) — highly correlated with the shipped score, and decisively worse where they differ.

### 3.1 What this licenses, and what it does not

✅ **Licensed:** *there is nothing free in S1.* §6.1's question — *"is the DISCARDED refined
confidence a better ranker than the one we ship?"* — has a clean answer: **no, it is ~0.84–0.92 m
worse**, replicated on two arms.

✅ **Licensed:** *S2 does not rescue S1.* `dsel-nocons` (S1+S2+S4) inherits the **entire** deficit at
initialisation. The refined ranker is **not** failing because it prefers unflyable candidates — it
already picks inside the reachable band and picks badly there.

⛔ **NOT licensed: "S1 is dead."** The pre-registration is explicit that E-SEL-0 is a **LOWER BOUND**
on S1, not S1: these weights never trained the refined readout as a ranker. The premise sub-test
passes on both arms — `frac_sel_2x_worse` gap vs shuffled **0.1103 / 0.1120** (≥ the registered
0.05), `rank_acc` **8.7× / 16.6× chance**. The signal is there; it is pointed at a different question.

⛔ **NOT licensed: any §6.3 branch.** All four presuppose refined ≥ shipped or no separation.
**"Separated adversely" was not registered** — reported as a gap in the pre-registration rather than
forced into the nearest row.

### 3.2 The mechanism, from source — why the direction is not surprising in hindsight

`AnchoredDiffusionDecoder.forward` supervises `conf` (the t=0 pass) with the **anchor-cls CE against
the GT-nearest ORIGINAL anchor**. `refined` is the same head re-evaluated on **denoised**
trajectories with **no supervision of its own** — pre-D-SEL nothing in the loss differentiated
w.r.t. it. So the refined readout is a trained classifier **evaluated off its training
distribution**, and its degradation is a distribution-shift result. That is precisely what
`REFINED_CLS_WEIGHT = 1.0` exists to fix — and the −0.84 m is now a **measurement of how big that
shift is**, which is new information about the size of the job S1's CE has to do.

---

## 4. E-SEL-1 — S3 LIVE at ρ ≈ 0.62–0.67 — ⚠️ WITH A CAVEAT THAT MUST TRAVEL WITH IT

Per §6.2: `cons_i = law_head([pooled, fan_i])`, statistic `ρ(−‖cons_i − z_{t+5}‖², −ADE_i)` per
window, against a permuted-candidate-axis control.

| arm | Spearman ρ | paired ρ − shuffled | §6.3 trigger (separated **and** \|ρ\| ≥ 0.10) |
|---|---|---|---|
| `refc-base-30k` | **0.6657** [0.6183, 0.7157] | **+0.6667** [+0.6181, +0.7187] | ✅ **S3 LIVE** |
| `refc-xl-30k` | **0.6212** [0.5650, 0.6791] | **+0.6235** [+0.5677, +0.6807] | ✅ **S3 LIVE** |

**REF-C's world model is strongly candidate-discriminating.** That answers the question the prereg
said no amount of training can un-ask.

### 4.1 ⛔ THE CAVEAT — the registered statistic uses a signal the registered mechanism never sees

`z_{t+5}` is `encode_pooled(frame_{t+5})` — **the future frame**. The statistic asks *"does the
imagined consequence agree with what actually happened?"*

**`refc_select.consequence_scores` has no access to it.** From source
(`stack/tanitad/refs/refc_select.py:308-321`) its score is
`conf_head(layer_norm(feat_proj(law_head([ctx, fan]))))` — the decoder's own confidence head judging
the imagined latent, **with no future frame anywhere in the path**.

⇒ **ρ ≈ 0.65 is an UPPER BOUND on the information available to S3, not a prediction of S3's
delivered gain.** The registered branch ("S3 LIVE ⇒ include S3 in the retrain arm") is satisfied on
its own terms and I am not weakening it — but quoting 0.65 as *"S3's effect size"* would be the same
family as the C6 confound and the REF-A I-JEPA leak: **a measurement-time input containing something
the deployed path does not have.**

**The discriminating follow-up** (0 GPU-days, same bank, one Thor pass because `feat_proj` and
`conf_head` are checkpoint weights): re-run the correlation with the **deployable** score and report
both. That number, not 0.65, should size S3 in the retrain arm. **I did not run it.**

---

## 5. THE FOUR METRIC FAMILIES — per family, never pooled (binding)

Grid **derived**, not assumed: `wp_steps [5,10,15,20] × 0.1 s → dt = 0.5 s`. A hard-coded 0.1 s
would inflate every speed ×5 and every accel ×25 (R-2026-08-03-c).

### 5.1 ⭐ The four-family decomposition of the SELECTION gap (oracle-in-fan − shipped, paired, base)

The pre-computed **upper bound on anything a reranker of this fan can buy** — the table §7.1 would
judge the retrain against.

| family | metric | paired Δ (negative = oracle better) | separated |
|---|---|---|---|
| **LONGITUDINAL** | `speed_abs_err_mps` | **−0.2688** [−0.3436, −0.1987] | ✅ **yes** |
| | `along_abs_err_m` | **−0.2781** [−0.3515, −0.2090] | ✅ **yes** |
| | `speed_signed_err_mps` | −0.0172 [−0.1013, +0.0634] | no |
| | `along_signed_err_m` | −0.0165 [−0.0972, +0.0604] | no |
| **LATERAL** | `cross_abs_err_m` | **−0.0334** [−0.0555, −0.0151] | ✅ yes (small) |
| | `heading_abs_err_deg` | −0.1334 [−0.3126, +0.0307] | no |
| | `curvature_abs_err_1pm` | **+0.0013** [−0.0011, +0.0045] | no (**wrong sign**) |
| | `yaw_rate_abs_err_degps` | **+0.1212** [−0.1571, +0.4240] | no (**wrong sign**) |

⭐ **THE SELECTION GAP IS LONGITUDINAL: 89.28 %** (base), **87.60 %** (XL), **89.88 %**
(`refc-small-30k`). That independently replicates the programme's standing *"88.7 % of the oracle
gap is longitudinal"* — on the **selection** gap specifically, at three scales.

**What it means for the retrain.** A better ranker on REF-C's fan is a **LONGITUDINAL** ranker. This
raises the priority of prereg §9.1's VTARGET escalation and of D-TAC1's finding that the 5-way
manoeuvre softmax mixes lat+lon. And the **LATERAL guard-rail (§7.2) is tight by construction**:
even a *perfect* reranker buys only 0.033 m of cross-track and moves heading, curvature and yaw-rate
not at all — so any separated lateral regression in a retrain is a real fault, not noise.

### 5.2 Per-family status, with reason and n where not computable

| family | status | n |
|---|---|---|
| **LONGITUDINAL** — speed half | ✅ MEASURED (base, shipped): `speed_mae` 0.4460 m/s, `speed_bias` **+0.0206** m/s (signed), `along_mae` 0.4166 m, `along_bias` +0.0272 m | 881 |
| **LONGITUDINAL** — distance-keeping (headway / time-gap / TTC) | ⛔ **UNAVAILABLE**: no lead-agent track supplied. The reader exists (`2026-08-03-longitudinal-distance-keeping/build_lead_tracks.py`, over `obstacle.offline`); a fan bank carries no lead. **WORK ITEM.** | **0** |
| **LATERAL** | ✅ MEASURED: `heading_mae` 1.1460°, `curvature_mae` 0.007711 1/m, `yaw_rate_mae` 1.8506 °/s, `cross_mae` 0.1313 m | 881 |
| **TACTICAL** — goal/anchor selection | ✅ MEASURED: `rank_acc`, `sel_gap`, `frac_sel_2x_worse` — **the half D-SEL exists to move** (§3) | 881 |
| **TACTICAL** — manoeuvre decision + confusion | ⛔ **UNAVAILABLE**: a fan bank stores no decoded manoeuvre. **WORK ITEM.** | **0** |
| **STRATEGIC** | ⛔ **UNAVAILABLE**: no route/goal label, and the decode used `nav_mode='follow_constant'` so the route input was never exercised (the C6 confound). **WORK ITEM** — §7.1 needs it for the S5 arm. | **0** |

---

## 6. ⚠️ THE C6 CONFOUND IS INHERITED HERE ON PURPOSE

`nav_mode = follow_constant` means the decoder saw **one constant command for every window** — the
07-21 C6 confound, and the condition the published 0.4728 / 0.4714 were collected in. It is kept
because E-SEL-0 is a **paired** contrast against exactly those numbers, and changing the fed command
would move the baseline. It is a real limitation of *those* rows, restated rather than inherited
silently. A route-exercised re-collection is the `refc_thor_eval.py` path and is a separate question.

---

## 7. THE DEAD-PARAMETER GUARD — VERIFIED TO **FIRE**, not merely to pass

The brief required this before any S1 result is trusted. `tests/test_refc_select.py` pinned only the
happy direction; **a guard never observed to raise is indistinguishable from a guard that cannot
raise** (C13's class, and the H2 "chance comparator" post-mortem's class).

`stack/tests/test_refc_select_guard_fires.py` (**new, 3 tests, all passing**) proves it **by
contrast on one model, one batch, one set of weights**:

1. **with** the ranked-score CE → `assert_selection_params_are_alive` returns
   `{decoder.route_to_anchor.weight, decoder.cons_gate}`, both gradients > 0;
2. **with `REFINED_CLS_WEIGHT = 0`** → the same guard on the same model **RAISES**, and the message
   names both dead parameters and the mechanism (`argmax`, `loss_rcls`);
3. the zero-init weights are **bit-identical** in both cases — only the gradient separates them,
   which is exactly why inspecting the weight proves nothing.

⚠️ **Scope, so it is not over-read:** the guard scans `route_to_anchor`, `cons_gate`, `goal_gate`,
`goal_dist_gate`, `goal_head`. **S1 adds zero parameters**, so on a `--sel-refined`-only arm the
guard returns an **empty dict** — correct, and *not* evidence S1 is alive. What keeps S1 honest is
that `loss_rcls` is built whenever `sel_refined` is set. Pinned by test 3.

---

## 8. WHAT I DID NOT DO — plainly

* ⛔ **Did not measure the DEPLOYABLE consequence score** (§4.1) — the number that should actually
  size S3. Highest-value follow-up; 0 GPU-days.
* ⛔ **Did not run E-SEL-0/1 on `refc-small-30k`.** Its fan-only panel is banked (881 windows) but
  Thor holds no small checkpoint.
* ⛔ **Did not run distance-keeping, manoeuvre-confusion or the strategic family.** Reported
  UNAVAILABLE with reason and n = 0 (§5.2). Each is a work item, not a pass.
* ⛔ **Did not run any retrain arm**, launch training, touch pod4/pod5, commit, or push.
* ⚠️ **Did modify Thor's `~/TanitAD` checkout**: `refc.py`, `refc_select.py`, `refc_tactical.py`,
  `flagship_v15.py`, `refc_eval.py`, `four_families.py`, `ci.py` synced from repo HEAD, because Thor
  sat at `4954544` — which **predates D-SEL and cannot return `refined_logits` at all**. Originals
  backed up to `thor:~/_dsel_backup/`. Verified by real import + `param_breakdown` reproducing
  **104,191,577 → 104,191,962 (+385)** on aarch64.

---

## 9. 🔴 ESCALATIONS

1. **→ PI / orchestrator — §6.3's branch table needs a fifth row.** *"E-SEL-0 separated in the
   ADVERSE direction"* is not enumerated, and it is what happened on both arms. A pre-registration
   that cannot express the observed outcome is the thing to fix.
2. **→ arch-inf / eval-tools — measure the DEPLOYABLE consequence score before S3 is costed** (§4.1).
   Quoting ρ ≈ 0.65 for a path that never sees `z_{t+5}` would be a leak-shaped claim.
3. **→ ops — the 256×256 REF-C val cache now exists in ONE reachable place**:
   `thor:~/valdata/physicalai-val-0c5f7dac3b11`. The eval pod that held the other copy is
   terminated, and HF carries only the **w120 256×640 cylindrical** raster (inadmissible for REF-C —
   C-raster refuses it). ⚠️ One clip already produced a transient unreadable load. **This is a
   single-disk dependency for every REF-C number the programme publishes.**
4. **→ orchestrator — `MODEL_REGISTRY.md` §4.2 still cites `taniteval/results/refc-small-30k.json`,
   which does not exist** (prereg §9.3). The artifact is at `…/incoming/2026-07-22-refc-small-30k/`.
   `refc_sel_probe.PUBLISHED` now records the real path.

---

## 10. DELIVERABLE MANIFEST

| artifact | path | state |
|---|---|---|
| **the probe (prereg §9.2 — it did not exist)** | `stack/scripts/refc_sel_probe.py` | repo, **staged** |
| **the GPU-side augmented dump** | `stack/scripts/refc_sel_dump_refined.py` | repo, **staged** — also `thor:~/refc_sel_dump_refined.py` |
| **guard-fires tests (new, 3)** | `stack/tests/test_refc_select_guard_fires.py` | repo, **staged** |
| this verdict | `…/2026-08-03-esel-verdict/ESEL_VERDICT.md` | repo, **staged** |
| **E-SEL-0/1 results, 881 windows, 2 arms** | `…/raw/full/esel_probe_refc-{base,xl}-30k.json` | repo, **staged** |
| fan-only results, 3 arms | `…/raw/esel_probe_refc-{base,xl,small}-30k.json` | repo, **staged** |
| cross-arm summary | `…/raw/esel_cross_arm_summary.json` | repo, **staged** |
| **augmented fan banks (refined_logits + cons_score)** | `…/raw/fan_refined_refc-{base,xl}-30k.pt` | repo, **staged** — also on Thor |
| guard verification transcript | `…/raw/guard_fires_verification.txt` | repo, **staged** |
| full-suite transcript | `…/raw/pytest_full_suite.txt` | repo, **staged** |
| Thor originals (pre-sync) | `thor:~/_dsel_backup/*.bak` | ⚠️ **THOR ONLY** — restorable, not program work |

**Nothing is committed and nothing is pushed.** Nothing that took effort lives only on Thor.

---

## 11. EVIDENCE CLASS

| claim | class |
|---|---|
| E-SEL-0 paired +0.8372 (base) / +0.9187 (XL), separated; refined 1.3100 / 1.3901 vs shipped 0.4728 / 0.4714 | **MEASURED** — `raw/full/esel_probe_refc-{base,xl}-30k.json` |
| E-SEL-1 ρ 0.6657 (base) / 0.6212 (XL), paired vs shuffled separated | **MEASURED** — same files |
| S2: 73.76 % / **72.08 %** removed, oracle survives 100 %, paired Δ exactly 0.0 for BOTH rankers | **MEASURED** — same files; XL reproduces the published 0.7208 exactly |
| C-shuffled 14.5426 / 13.9564, `frac_sel_2x_worse` 0.9798 / 0.9894 | **MEASURED** — closed form + 24-seed self-test |
| longitudinal share of the selection gap 89.28 % / 87.60 % / 89.88 % | **MEASURED** — `raw/esel_cross_arm_summary.json` |
| the Thor re-decode reproduces the eval-pod banks (agreement 1.0000, both arms) | **MEASURED** — `C-identity-vs-bank` |
| the dead-parameter guard fires without the ranked-score CE | **MEASURED** — `raw/guard_fires_verification.txt` |
| `refc_select.consequence_scores` never sees `z_{t+5}` | **MEASURED (source)** — `refc_select.py:308-321` |
| *"ρ ≈ 0.65 overstates what S3 can deliver"* | **HYPOTHESIS** — the discriminating measurement is §9.2 and was **NOT run** |
| *"the oracle gap is ~92 % irreducible"* / v1.2's 8.4 % | **INHERITED** — a prose note, not a results JSON. Load-bearing for §6.3's thresholds; flagged as such by the prereg itself |

**Full suite:** `cd stack && pytest -q` → **1932 passed, 12 skipped, 2 xfailed** in 395.10 s
(`raw/pytest_full_suite.txt`). ⚠️ **The baseline is moving under this run.** The brief quoted
1900/12/2; a run mid-session read **1913**, and the final run reads **1932**. My 3 new tests account
for 3 of the +32 — the other **+29 entered the suite from other streams while this task ran**.
Reported, not reconciled: a suite count is only meaningful against a pinned commit, and this branch
is being staged into concurrently.
