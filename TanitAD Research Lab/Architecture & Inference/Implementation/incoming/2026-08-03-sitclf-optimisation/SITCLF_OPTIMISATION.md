# P8 — OPTIMISE THE SCENARIO CLASSIFIER: the deployed multimodal arm is worse than its own ego-only ablation, on all three situations

**Date** 2026-08-03 · **Substrate** dev box (RTX 4060), **0 pod GPU-h** · **Suite** `pytest -q` 1680 passed / 12 skipped / 2 xfailed

---

## 0. HEADLINE

1. ⛔ **The deployed arm is the wrong arm.** `head_img_ego` — the multimodal head the stream reports as its classifier — is **worse than `head_ego`, its own ego-only ablation, on all three situations**: adding the camera *removes* **49 % / 46 % / 21 %** of the AP. **SEPARATED** by the paired episode-cluster bootstrap on the two situations with real sample size (`lane_change` ΔAP-lift **+2.3524 [+0.8591, +4.2364]**, `intersection` **+0.9485 [+0.2237, +1.6944]**). The fix is free and needs no retrain.
2. ✅ **The mechanism is in the source, not a hypothesis.** `sc_train.py:143` fuses by `np.concatenate([img, S["E"]], 1)` — a 16-dim PCA image block normalised by its **own global mean-abs** (`:132-133`) concatenated with a 3-dim ego block scaled by a **hand-set** `EGO_SCALE = [10, 2, 0.5]` (`:38`). Two unrelated normalisations, 5.3 : 1 dimensional imbalance, one shared `nn.Linear`.
3. ⚠️ **The `WIN = 8` lever is NOT ESTABLISHED — my own point estimate did not survive its interval.** WIN 8 → 50 is **+8.1 % on `intersection`**, but ΔAP-lift **+0.2959 [−0.4489, +1.1590] — not separated** (B=2000, and B=400 agrees); on `roundabout` longer is **separated WORSE** (−1.4119 [−4.0494, −0.2234]). Reported as a hypothesis with the interval attached, **not** as a result. See §4.
4. ⛔ **Three plausible levers REFUTED by measurement**, each before any of them was written up as an improvement.
5. ✅ Two negative controls land exactly on the base rate, and the retrained head is **separated from its own permuted-feature control** (+1.9497 / +3.0409 lift), so the protocol neither manufactures nor loses signal.

---

## 1. WHAT EXISTS AND WHAT IT SCORES

**Substrate** `…/2026-07-26-situation-classifier/artifacts/heldout_frames.npz` — 308,973 held-out frames, **1,610 clip clusters**, per-frame scores for 10 arms, disjoint from the clips the heads were trained on (`train_summary.json`: 147,022 train windows). All numbers below are on the **230,083 rows** that carry a full 50-frame causal history inside their own clip, so every arm is scored on **identical rows** and the paired estimator is valid.

**MEASURED (mine, `sitclf_baseline.py` → reproduces R1's table to 5 dp):**

| arm | lane_change | roundabout | intersection |
|---|---:|---:|---:|
| `head_ego` (ego only) | **0.07707** | **0.01209** | **0.13594** |
| `head_img_ego` **← DEPLOYED** | 0.03924 | 0.00651 | 0.10692 |
| `head_img_ego_concat` | 0.03218 | 0.00470 | 0.09370 |
| `head_img` (camera only) | 0.03384 | 0.00367 | 0.08619 |
| `heur_kin` (hand heuristic) | 0.06448 | 0.00347 | 0.07124 |
| `head_img_shuf` **NULL** | 0.01710 | 0.00312 | 0.03290 |
| base rate | 0.01608 | 0.00211 | 0.03060 |
| n positives | 2,827 | 380 | 5,285 |

**`head_ego` / `head_img_ego` = 1.96× / 1.86× / 1.27×.** The camera, as fused, is strictly harmful.

---

## 2. THE BIGGEST LEVER — and the three that I refuted first

I measured four candidate levers before writing any of them up. Three died.

**⛔ REFUTED — per-clip score standardisation.** Intersection AP **0.13594 → 0.07941 (−41 %)**; lane_change 0.07707 → 0.02594. The *global* score offset carries real signal — some clips genuinely contain more intersections — and z-scoring within a clip destroys it.

**⛔ REFUTED — causal temporal filtering (EMA α∈{0.3,0.6}, trailing max k∈{5,15}).** All variants ≤ baseline (intersection best 0.13442 vs 0.13494). **Predicted in advance by a structural measurement:** the scores' lag-1 autocorrelation is **0.994–0.997** while the *label's* is **0.966** — the scores are already *smoother* than the target, so there is no high-frequency noise for a filter to remove. Mean positive run = 27–29 frames (2.7–2.9 s), consistent with the 3 s anticipation horizon.

**⛔ REFUTED — score-level late fusion as a source of camera value.** Logistic late fusion of `head_img` + `head_ego` reaches lane_change 0.08502 — but the **negative control that fuses the SHUFFLED image arm instead reaches 0.08351**. The two are indistinguishable: late fusion *recovers* the ego arm, it does not extract camera information. Consistent with the label being a pure ego function.

**✅ CONFIRMED — the fusion itself is the defect** (§1), and the causal window is a second, situation-dependent lever (§4).

---

## 3. WHAT I IMPLEMENTED

| file | what |
|---|---|
| `stack/tanitad/eval/ap_ci.py` | AP, AP-lift, and the **first admissible interval for an AP in this program** — `ap_episode_cluster_bootstrap` and its **paired** form, plus generic `stat_/paired_stat_episode_cluster_bootstrap` for any set-level statistic. |
| `stack/tanitad/eval/sitclf.py` | `clip_runs`, `causal_window`, `cluster_folds`, `CausalSitHead` (= `sc_train.SitHead` with WIN a parameter), `train_sit_head` (`run_fold`'s exact recipe), `late_fuse_scores`. |
| `stack/tests/test_ap_ci.py` (14) · `stack/tests/test_sitclf.py` (14) | see below |

**Why an AP estimator had to be written.** `taniteval.ci.episode_cluster_bootstrap` reduces *per-window* values; AP is a property of the joint ranking of the whole set and cannot be expressed that way. Every sitclf number in the hub is therefore a **bare point estimate with no interval**. The new module does **not** reimplement the resampling: `_draws` delegates to `taniteval.ci._draws`, and `test_draws_are_taniteval_draws` asserts the draws are **identical draw-for-draw**. Only the statistic changes.

**Tests that are contracts, not smoke:**
* `test_causal_window_is_causal_and_never_crosses_a_clip` — row *t* holds frames *t−win+1..t* of its **own** clip and nothing later.
* `test_head_learns_a_signal_that_needs_a_LONG_window` — a target decidable only from 12 frames back: win=16 reaches AP **1.00**, win=4 sits at **0.15** against a 0.14 base rate. If win=4 scored above chance the window plumbing would be leaking; if win=16 failed, the WIN sweep on real data would be meaningless.
* `test_cluster_bootstrap_is_wider_than_a_frame_bootstrap` — with a real per-clip random effect the cluster SE is **1.94×** the frame SE.
* `test_paired_does_not_separate_two_equivalent_arms` — the estimator's own negative control.
* `test_late_fusion_with_an_uninformative_column_does_not_collapse` — the property early concat fails.

---

## 4. MEASURED — the window sweep, architecture held FIXED

Same `CausalSitHead`, same recipe (AdamW 3e-4 / wd 0.01, masked `BCEWithLogitsLoss(pos_weight=20)`, grad-clip 1.0, 8 epochs), same rows, out-of-fit over a 2-fold split on whole **clusters**. Only WIN changes.

| arm | lane_change | roundabout | intersection |
|---|---:|---:|---:|
| ours `ego_win8` (= the pre-registered constant) | 0.06457 | 0.00604 | 0.11239 |
| ours `ego_win25` | 0.06469 | 0.00338 | 0.11620 |
| ours `ego_win50` | 0.04892 | 0.00306 | **0.12144** |
| **win50 / win8** | **−24.2 %** | −49.3 % | **+8.1 %** |
| **NEG — ego permuted across clips** | 0.01756 | 0.00207 | 0.02839 |
| base rate | 0.01608 | 0.00211 | 0.03060 |

### The intervals — and what they do to the point estimates

Paired episode-cluster bootstrap on **AP-lift**, 1,609–1,610 clusters, **B=2000** (the program default; `results_sitclf_opt.json`). An independent **B=400** replication (`results_sitclf_opt_b400.json`) returns the **same verdict on every row** — quoted in brackets where it differs — so no verdict here depends on the bootstrap size.

| paired contrast | lane_change | roundabout | intersection |
|---|---|---|---|
| **`head_ego` − `head_img_ego`** (the fix) | **+2.3524 [+0.8078, +4.3194] SEP** | +2.6493 [−0.7557, +6.9919] | **+0.9485 [+0.2054, +1.7416] SEP** |
| `ego_win50` − `ego_win8` (the window lever) | −0.9732 [−2.2976, +0.1543] | **−1.4119 [−4.0494, −0.2234] SEP-WORSE** | +0.2959 [−0.4489, +1.1590] |
| `ego_win50` − **NEG permuted** | **+1.9497 [+1.0346, +3.2683] SEP** | +0.4670 [−0.2858, +1.5414] | **+3.0409 [+2.3653, +3.9838] SEP** |
| `ego_win50` − `head_ego` | −1.7501 [−3.8574, −0.1452] | −4.2847 [−9.1325, −1.9841] | −0.4740 [−1.3744, +0.5431] |

**⚠️ SELF-CORRECTION.** My working hypothesis was that the window is a situation-dependent lever — long history helping `intersection` (a junction approach is a long deceleration profile) and hurting `lane_change` (a short, local manoeuvre). The **point estimates say exactly that**, but **the interval does not support it**: `intersection`'s +8.1 % is **+0.2959 [−0.3988, +1.1884], not separated**, and the only separated window effect anywhere is `roundabout` getting **worse**. Under the program's rule the honest statement is: **`WIN` is a live hypothesis with a directionally suggestive point estimate and no decision-grade support.** It must not be quoted as a gain and must not decide a retrain. The cheapest discriminating follow-up is a per-situation WIN sweep at the banked arms' full training-set size, where the noise floor is lower.

**✅ What the intervals DO establish:** the retrained head is separated from its own permuted-feature control on `lane_change` and `intersection` (so it genuinely learns), and the **fix in §1 is separated on both situations with real sample size**. `roundabout` separates nothing in either direction — 380 positives at a 0.00211 base rate is simply too thin, and that is reported rather than papered over.

⚠️ **My retrained arms do NOT beat the banked `head_ego`** (0.06457 vs 0.07707; 0.12144 vs 0.13594). Expected and stated: the banked heads were fitted on a **disjoint, larger** clip set (147,022 windows) with per-arm CV-selected hyperparameters, while mine use a fixed `pos_weight=20` on half of the held-out clusters. **The WIN contrast is internal to my runs and valid; the absolute level is not a claim against the banked arm.**

---

## 5. NEGATIVE CONTROLS — the metric can discriminate

| control | lane_change | roundabout | intersection | verdict |
|---|---:|---:|---:|---|
| ego features **permuted across clips**, labels untouched, full retrain | 0.01756 | 0.00207 | 0.02839 | at base rate ✅ |
| **labels permuted across clusters** | 0.01750 | — | — | at base rate ✅ |
| banked `head_img_shuf` | 0.01710 | 0.00312 | 0.03290 | at base rate ✅ |
| base rate | 0.01608 | 0.00211 | 0.03060 | — |
| late fusion of **shuffled** image + ego | 0.04058 | — | — | ≈ real-image fusion ⇒ the camera adds nothing |

The first is the strong one: it is a **full retrain** of the same architecture on ego features whose link to the label has been destroyed, and it lands on chance. The protocol cannot manufacture signal.

---

## 6. THE RECOMMENDATION

1. ⭐ **Stop deploying `head_img_ego`. Deploy `head_ego`.** Free, no retrain, **+96 % / +86 % / +27 %** AP, **separated on `lane_change` and `intersection`**. MEASURED on identical rows with the binding estimator. This is the only decision-grade recommendation here.
2. If the camera is to be kept, it must enter by **late fusion** (`late_fuse_scores`), never early concat — but §2 says expect it to add ~nothing on an ego-derived label, so the honest prior is that the camera arm needs a **label that is not a pure ego function** before it can help. That is the sibling stream's open question, and this result sharpens it.
3. ⚠️ **Do NOT retrain on the window yet.** The per-situation-WIN idea is a hypothesis whose interval includes zero (§4). Pre-register it and run the sweep at full training-set size, or leave `WIN=8`.
4. `situations.py:19-22`'s stale absence-claim should be refuted **within gen-1** (`head_img` 0.037628 vs `head_img_shuf` 0.016562 = **2.27× its own null**, from `train_summary.json`), never by the cross-generation arithmetic two documents used.

---

## 7. FOUR FAMILIES

The situation classifier is a **classification** target, so:
* **TACTICAL** — this IS the family. AP + AP-lift per situation with the paired episode-cluster bootstrap, above.
* **LONGITUDINAL / LATERAL** — **N/A for a classification target**: there is no predicted trajectory to measure speed, headway, curvature or cross-track against. Reported as N/A with the reason, per the rule, **not silently dropped**. n = 230,083 rows.
* **STRATEGIC** — **UNAVAILABLE**, n = 230,083. No route/goal label exists on PhysicalAI-AV (settled at five probes: no map, lane graph, junction annotation or route signal; egomotion is clip-local metres with no GNSS).
* ⚠️ **A sitclf → `four_families.py` adapter still does not exist** — `_decision_family` (`:166`) expects decoded decisions with `pred_key`/`gt_key` against a window dict. That is a **work item**, and `ap_ci.stat_episode_cluster_bootstrap` is the piece it was missing.

---

## 8. EVIDENCE CLASS

| claim | class |
|---|---|
| §1 baseline table, §4 sweep, §5 controls, §2 refutations | **MEASURED (mine)** — `results_sitclf_opt*.json`, `*.scores.npz` |
| `sc_train.py` line citations (`:37`, `:38`, `:132-133`, `:143`, `run_fold`) | **MEASURED (mine, read at HEAD)** |
| autocorrelation 0.994–0.997 vs 0.966, positive-run 27–29 frames | **MEASURED (mine)** |
| gen-1 `train_summary.json` values (0.037628 / 0.016562) | **MEASURED (mine, re-read)** |
| `intersection` label is an exact ego identity (400/400) | **INHERITED** (`VERIFIED_scenario_classification.md`), not re-derived here |
| no map/route in PhysicalAI-AV | **INHERITED** (CLAUDE.md five-probe settlement) |
