# P8 — the situation classifier had NO deployed path; I built one, and under the PI's vision-only ruling the camera SEPARATES

**Date** 2026-08-03 · **Substrate** dev box, **0 pod GPU-h** · **Suite** `pytest -q` **1816 passed**,
12 skipped, 2 xfailed (baseline 1722) · **Run directory**
`TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-03-sitclf-fusion-wired/`
· **Primary artifact** `results_sitclf_vision_only.json` · log `run_log_vision_only.txt`

---

## 0. HEADLINE

1. ⛔ **`late_fuse_scores` had no caller because the CLASSIFIER has no deployed path.** Three probes
   (§1): nothing under `stack/` or `taniteval/` scores a frame with a trained situation head. The
   only trainers live in hub `incoming/` folders and were never promoted. **The fix was not
   unwired — the whole component was.**
2. ✅ **I built the missing consumer**: `stack/tanitad/eval/sitclf_deploy.py` +
   `stack/scripts/fuse_situation_scores.py`, **30 tests**.
3. ⛔ **The PI re-scoped this mid-flight and it changed the answer.** *"for ground truth data of
   scenario classification you can use both ego and other label, for inference only vision."*
   Ego may DERIVE the labels; it may **not** be read at inference. That closes `head_ego`,
   `head_img_ego` **and image+ego late fusion alike** — a calibrated ego *score* at inference is
   still an ego input.
4. ⭐ **THE RESULT: vision-only is REAL and SEPARATED on all three situations.** `head_img` against
   its own permuted-feature null, paired episode-cluster bootstrap **B=2000**, 1,610 clip clusters:
   ΔAP-lift **+1.1749 [+0.7930, +1.6890]** (lane_change), **+1.4082 [+0.1369, +3.4552]**
   (roundabout), **+1.5226 [+1.1358, +1.9789]** (intersection). The camera reaches
   **2.17× / 2.58× / 2.61×** the base rate.
5. ⛔ **`late_fuse_scores` has NO established role under the ruling, and I did NOT wire it as the
   deployed path.** Fusing the two VISION arms is **not separated** on any situation
   (+0.0565 / +0.9102 / +0.1520, every CI spans zero). It is implemented, tested and available; it
   is not recommended. *A fix that exists is not a reason to ship it.*
6. 🔴 **`situations.py:19`'s "VISION ADDS NOTHING OVER EGO STATE" is RETRACTED**
   (`RETRACTION_LOG.md` **R-2026-08-03-f**) — and the correct test was in the same `.npz` all along.

---

## 1. THE CONSUMER QUESTION — answered at three probes

| # | probe | result |
|---|---|---|
| 1 | `grep -rn 'head_img_ego\|head_*.pt\|sc_meta\|scores.npz'` over `stack/` + `taniteval/` | only a docstring in `sitclf.py` |
| 2 | every situation module under `stack/` | `data/situations.py` (label DETECTORS), `scripts/emit_situation_labels.py` (label EMITTER), `eval/sitclf.py` (model classes), `eval/ap_ci.py` (estimator). **None scores a frame with a trained head.** `eval/scenarios/` is the closed-loop traffic-light/work-zone suite — a different component with a similar name |
| 3 | the trainers that produce the arms | `…/2026-07-26-situation-classifier/scripts/sc_train.py` and `…/2026-07-29-situation-classifier-v2/sc_train_v2.py` — hub `incoming/` only, never promoted |

⇒ **"The fix has no caller because the classifier has no deployed path"** is the honest finding, and
it is a bigger problem than the fusion bug. Where a fused score *would* enter, had a path existed:
`sc_train_v2.py:331` (`scores[arm] = sc[-1]`, after the per-arm loop) — **not** `:143`, which is the
feature-level concat itself.

**What I built instead of leaving it unwired** — `stack/tanitad/eval/sitclf_deploy.py`:
`ScoreBundle` / `load_score_bundle`, `fuse_modalities` (the `late_fuse_scores` call site),
`vision_only_arms` (the panel), `is_vision_only` (the ruling as a guard, not a comment),
`permute_labels_by_cluster`, `regime_strata`, `anticipation_lead_s`, `four_family_report`.

---

## 2. LABEL PROVENANCE — established from source, two probes

| probe | file:line | finding |
|---|---|---|
| 1 | `stack/scripts/emit_situation_labels.py:54-62` | reads **only** `d["poses"]`; calls `kinematics(P)`, `detect_lane_change(K)`, `detect_intersection(K)`, `anticipation_target(...)`. No frames, no objects, no map |
| 2 | `stack/tanitad/data/situations.py:97, 161, 210, 244, 284` | `kinematics(P)` takes `P [T,4] = (x,y,yaw,v)`; every detector takes only `K`. The emitter passes `cross=None`, so **even `intersection` is the turn half alone** |

⇒ **All three labels are pure deterministic functions of the ego pose track.**

⚠️ **But it is NOT a leak, and I will not claim one.** I checked the timing rather than assuming:
the head's window is **[t−0.7 s, t]** (`sc_train.py:37`, offsets −7..0) and the label's evidence
window is **[onset, onset+4 s]** with onset > t (`anticipation_target:314-316`, `LC_W_S = 4.0`) —
**disjoint; no future information reaches the head.** The accurate statement is *same-source
privileged access*: an ego head observes the label's **generating process** while the camera must
infer it from pixels. That makes the old ego-vs-vision comparison structurally unfair; it does not
make it leakage.

🔴 **One genuine boundary defect, new and previously undocumented:** `omega_pre` / `alon_pre` are
built on `np.gradient` (`situations.py:107-113`), a **centred** difference, so they read **one frame
(0.1 s) past t** — despite the source comment asserting *"STRICTLY CAUSAL"*. It bites only for onsets
at exactly t+1, but the comment overstates the guarantee and is now corrected in place.

---

## 3. THE VISION-ONLY PANEL — every arm, and why each exists

| arm | definition | why |
|---|---|---|
| `PRIMARY` | `head_img` | **the deployable arm** — camera only |
| `FUSED` | `late_fuse_scores(head_img, ridge_img)` | the fix, between VISION arms only |
| `NEG_VISION` | `head_img_shuf` | the camera's own null (features permuted ACROSS clips) — **the only admissible baseline now that ego is not a legal input** |
| `NEG_FUSED` | `late_fuse_scores(head_img_shuf, ridge_img_shuf)` | the same combiner, same parameter count, camera destroyed |
| `NEG_MACHINERY` | `late_fuse_scores(head_img)` — one column | proves the combiner's free parameters alone buy nothing |
| `NEG_LABEL` | combiner fitted on labels permuted across whole CLUSTERS | proves the protocol cannot manufacture signal |

Folds are whole clip **clusters**, so the combiner that scores a row never saw any frame of that
row's clip.

---

## 4. THE RESULT — AP and paired ΔAP-lift, B=2000, 1,610 clusters

| | lane_change | roundabout | intersection |
|---|---:|---:|---:|
| rows scored / positives | 252,826 / 4,361 | 258,540 / 721 | 249,480 / 7,620 |
| base rate | 0.01725 | 0.00279 | 0.03054 |
| **`head_img` (PRIMARY)** | **0.03741** | **0.00721** | **0.07955** |
| `ridge_img` | 0.03405 | **0.01056** | 0.07767 |
| `FUSED` | 0.03839 | 0.00975 | 0.08419 |
| `NEG_VISION` | 0.01715 | 0.00328 | 0.03304 |
| `NEG_FUSED` | 0.01884 | 0.00299 | 0.02683 |
| `NEG_MACHINERY` | 0.03678 | 0.00668 | 0.07703 |
| `NEG_LABEL` | 0.03078 | 0.00670 | 0.03374 |
| **`head_img` AP-lift** | **2.169×** | **2.585×** | **2.604×** |

**Paired episode-cluster bootstrap on AP-LIFT** (`*` = SEPARATED):

| contrast | lane_change | roundabout | intersection |
|---|---|---|---|
| **PRIMARY − NEG_VISION** *(does vision work?)* | **+1.1749 [+0.7930, +1.6890]\*** | **+1.4082 [+0.1369, +3.4552]\*** | **+1.5226 [+1.1358, +1.9789]\*** |
| FUSED − PRIMARY *(the fix)* | +0.0565 [−0.0925, +0.3072] | +0.9102 [−0.1490, +2.1881] | +0.1520 [−0.1786, +0.5513] |
| FUSED − NEG_FUSED | **+1.1336 [+0.6948, +1.7759]\*** | **+2.4227 [+0.9562, +4.8835]\*** | **+1.8778 [+1.4586, +2.4101]\*** |
| NEG_MACHINERY − PRIMARY *(machinery alone)* | −0.0365 [−0.1303, +0.0350] | −0.1901 [−0.5342, +0.0408] | −0.0825 [−0.2171, +0.0929] |
| PRIMARY − NEG_LABEL | **+0.3845 [+0.1132, +0.7198]\*** | +0.1824 [−9.1104, +3.2874] | **+1.4997 [+1.1510, +1.9377]\*** |

**The discrimination control was run FIRST and it fires on all three situations.** The machinery
control is ≤ 0 everywhere, so no separation above can be an artifact of the combiner's parameters.

⚠️ **`roundabout` stays UNPOWERED** — 721 positives at a 0.00279 base rate; its `NEG_LABEL` control
does not separate and returns the wild interval [−9.11, +3.29]. Its `PRIMARY − NEG_VISION` bound
touches +0.14. Treat every roundabout row as suggestive, not decision-grade.

⭐ **Notable:** on `roundabout` the **2,049-parameter ridge probe (0.01056) beats the 2.17 M-parameter
transformer head (0.00721)** on the *same* vision features — independent support for
`sc_train_v2.py`'s AMENDMENT A1 note that *the head has been the bottleneck, not the features*.

---

## 5. THE FOUR FAMILIES (per family, never pooled; FUSED vs PRIMARY, both vision-only)

**TACTICAL — this IS the family for a classification target.** AP-lift with its own interval:
`lane_change` PRIMARY **2.1691 [1.8118, 2.6762]**, `roundabout` **2.5846 [1.6730, 4.4652]`,
`intersection` **2.6043 [2.2527, 3.0529]**. **Anticipation lead** (fixed 5 % alarm budget, rank-based
so ties cannot inflate it): PRIMARY median **2.45 s / 2.2 s / 2.3 s**, but **117 of 171 / 17 of 26 /
179 of 269 anticipation runs draw no alarm at all** — reported rather than scored as 0 s, and it is
the honest limit of the arm.

**LONGITUDINAL & LATERAL — reported as decision quality STRATIFIED BY REGIME.** A classification
target has no predicted trajectory, so target-speed / headway / TTC and heading / curvature /
cross-track are **not computable** — stated per family with the reason rather than dropped. What *is*
answerable is whether quality holds where the ego is decelerating or turning:

| situation | family | separated strata (FUSED − PRIMARY) |
|---|---|---|
| lane_change | LONGITUDINAL | `cruise_ge8` **+0.1546 [+0.0067, +0.4943]\*** (n=151,506, 3,510 pos) |
| lane_change | LATERAL | `straight` **−0.1070 [−0.2258, −0.0422]\*** — fusion is separated-WORSE here |
| roundabout | LONGITUDINAL | `steady` +4.8295 [+0.9561, +10.1436]\* — **UNPOWERED (291 pos), not decision-grade** |
| roundabout | LATERAL | `straight` +7.1673 [+2.2056, +32.7036]\* — same caveat, interval width says it all |
| intersection | LONGITUDINAL | `accelerate` **+1.0472 [+0.1955, +2.2404]\*** (n=41,677, 1,382 pos) |
| intersection | LATERAL | none separated |

⇒ Fusion's only credible effects are **regime-local and of both signs** — it helps at cruise and on
accelerating intersection approaches, and **hurts on straight-driving lane-change anticipation**.
That is precisely the trade-off a pooled score would have hidden, and it reinforces the §0.5 verdict.

**STRATEGIC — UNAVAILABLE**, n = 252,826 / 258,540 / 249,480. No route/goal/map label exists on
PhysicalAI-AV (settled at five probes: no map, lane graph, junction annotation or route signal;
egomotion is clip-local metres with no GNSS). A **WORK ITEM**, not a pass.

---

## 6. WHAT I RECOMMEND

1. ⭐ **Deploy `head_img` as the situation classifier.** It is the only arm the ruling permits, and it
   is separated from its own null on all three situations at 2.17–2.60× base rate.
2. ⛔ **Do not ship the late-fusion arm.** Not separated on any situation, and separated-WORSE on one
   lateral stratum. `late_fuse_scores` stays implemented and tested for the day a genuinely
   independent second vision score exists; it has no role today.
3. ⚠️ **`roundabout` must not decide anything** at 721 positives — its own label control fails.
4. ⭐ **The real lever is the head, not the camera.** A 2,049-parameter ridge beats a 2.17 M-parameter
   transformer on roundabout, and 68 % of lane-change anticipation runs never alarm at a 5 % budget.
   A matched-capacity vision head (BACKLOG **B4**) is the next experiment, not more fusion.
5. ⚠️ **Correct the "STRICTLY CAUSAL" comment's guarantee** — done in `situations.py`; the
   `np.gradient` centred difference reads 0.1 s past t.

---

## 7. EVIDENCE CLASS

| claim | class |
|---|---|
| §1 three consumer probes, `sc_train_v2.py:331` entry point | **MEASURED (mine, read at HEAD)** |
| §2 label provenance, window/label disjointness, the `np.gradient` 0.1 s lookahead | **MEASURED (mine, read at HEAD)** |
| §4 every AP, AP-lift, interval | **MEASURED (mine)** — `results_sitclf_vision_only.json`, `run_log_vision_only.txt` |
| §5 four families, strata, anticipation lead | **MEASURED (mine)** — same JSON |
| baseline table reproduced before any change (head_ego 0.08699 / head_img_ego 0.04347 / …) | **MEASURED (mine)** — re-derived from `heldout_frames.npz`, not inherited |
| pre-ruling multimodal panel (`results_sitclf_fusion.json`) | **MEASURED (mine), SUPERSEDED** — kept for the record; its arms read ego at inference and are no longer deployable |
| no route/map label on PhysicalAI-AV | **INHERITED** (CLAUDE.md five-probe settlement) |
| gen-1 banked AUROC 0.703 / 0.769 and lead 1.4–2.0 s | **INHERITED**, not re-derived here |
