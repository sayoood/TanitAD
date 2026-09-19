<title>HWM capacity-matched hierarchy — row 18's external datapoint exists</title>

# Row 18 has its external datapoint: a capacity-controlled flat-vs-hierarchy table, on a FROZEN encoder

**2026-09-13 · Research Lab (LAB-RUN-012) · Architecture & Inference · serves backlog row 18 / H1b via A10-2 (pre-committed read)**
`PUBLISHED lib 2604.03208 · FULL TEXT READ (v2, 2026-06-16) · Zhang, Terver, … LeCun†, Ballas† (FAIR/NYU)`
⛔ **Tier: none of ours.** Every number is the paper's own, on manipulation and maze tasks — no driving tier, no four families. It informs a design; it decides no GPU-day on its own.

---

## 0 · Findings first

| # | finding | class |
|---|---|---|
| **F1** | ⭐⭐⭐ **A10-2's pre-committed read FIRES its first branch.** Appendix D.2 / **Table 16** is an explicit capacity control: *"we compare hierarchical models against single-level world models with matched or larger parameter counts."* **Push-T:** flat **44 M** 55 % / 17 % · flat **98 M** 35 % / 15 % · **hierarchy 94 M 78 % / 61 %** (H = 10 / 15). **Diverse Maze:** flat 54 k 85 / 63 · flat **178 k** 82 / 59 · **hierarchy 182 k 95 / 83**. ⇒ *"the datapoint row 18 needs"* — A10-1 may be re-scoped as a **replication**. | PUBLISHED lib `2604.03208` Table 16 |
| **F2** | ⭐⭐ **Scaling the FLAT model made it WORSE** (Push-T 55 → 35 % at 2.2× params). The hierarchy's edge over the size-matched flat model is **+43 pt (H = 10) / +46 pt (H = 15)** on Push-T and **+13 / +24 pt** on Maze. | PUBLISHED, same table |
| **F3** | ⭐⭐⭐ **The hierarchy works on a FROZEN encoder.** Both levels share *"a frozen DINOv2 encoder"* (Push-T) / V-JEPA 2 features (Franka); only the predictors differ (25 M low-level ViT, 75 M high-level + a macro-action encoder of dim 4). ⇒ **the first lever this month that does NOT collide with our trunk-freeze** — contrast LDAD, AITS, IQL-as-encoder-loss, Qwen-Drive Stage 1→2, OccFeat, and today's Deployment value-geometry tie. | PUBLISHED §3.2 + App. A.3 |
| **F4** | ⭐⭐ **The gain is horizon-shaped.** Push-T d = 25 / 50 / 75: flat **84 / 55 / 17 %**, hierarchy **89 / 78 / 61 %** — +5 pt where the task is short, +44 pt where it is long. **Our tactical horizon is 6 s; whether that is "d = 25" or "d = 75" in their units is the transfer question**, not a detail. | PUBLISHED Table 2 |
| **F5** | ⭐ **Compute also falls:** *"up to 3× less planning compute"* at matched success (Fig. 5). Corroborates backlog **P-7** (flat strategic rollout undeployable) and **D10-2** from an independent lab. | PUBLISHED Fig. 5 |

## 1 · ⛔ Adversarial reading — what Table 16 does NOT establish

1. **No interval, no seed count on Table 16.** Franka reports N = 5 trials per configuration with Clopper–Pearson CIs (App. Table 17); Table 16's success rates carry **neither**. ⇒ separated-by-eye, **not** separated-by-estimator. By our own rule (`H-ESTIM-SEED-1`) this is necessary-grade evidence at best.
2. **"Matched" is ±4 %, not ±1 %.** Push-T flat 98 M vs hierarchy 94 M (flat *larger*, which favours flat); Maze 178 k vs 182 k (flat 2.2 % *smaller*). A10-1's own bar is ±1 %.
3. **Capacity is matched; data and search are not.** The high-level model trains on **waypoint sequences at stride 10** and plans over **macro-actions** — more temporal context per sample and a smaller search space. The comparison isolates *parameter count*, not *hierarchy vs everything else it brings*.
4. **A flat model that degrades with 2.2× capacity invites an undertuning reading** — the paper does not say the 98 M flat model was re-tuned. The +43 pt may partly be a bad flat baseline.
5. **Goal-image MPC, not driving.** Zero-shot reaching of a goal *image*; our planner has no goal image at inference (the goal-input ruling admits a *predicted* goal point).

⇒ **Verdict: SUPPORTED, not settled.** The field now has one capacity-controlled positive result; it is exactly the claim H1b makes, on a frozen encoder, with four named confounds.

## 2 · Five-dimension analysis

| dim | |
|---|---|
| **RELEVANCE** ⭐⭐⭐ | H1b is the programme's thesis hypothesis with **no measured datapoint** (row 18); GS-5 positioning; P-7 deployment. |
| **CONSEQUENCE** | **Row 18 moves from "untested anywhere" to "supported externally, untested by us".** A10-1 is re-scoped as a replication — which lowers its information value per GPU-hour *unless* it tests what Table 16 does not (§1.3: hierarchy vs a flat model given the **same stride-10 waypoint data**). GS-5's *"unresolved on both sides"* softens to *"one controlled positive, zero driving positives"*. |
| **COMBINATION** | ⭐ With **Drive-HWM `2609.03572`** (09-10: hierarchy +0.8 PDMS, **not** params-matched, driving) this makes **two hierarchy positives on complementary axes** — one controlled but not driving, one driving but not controlled. Neither is both. **That gap is precisely TanitAD's claim to own.** With today's Deployment finding (frozen trunk ties pixels on goal-distance): HWM's subgoals are matched by **latent distance in the frozen space** — the exact quantity our trunk measured weak at. ⚠️ **That is a direct risk for transfer**, and it is measurable. |
| **CHANCES / RISKS** | **Chance:** a hierarchy lever that respects the freeze, cuts planning compute ~3×, and grows with horizon. **Risks:** (a) latent-matching subgoals on a trunk whose latent distance barely beats pixels (today's F2/F4) may fail where DINOv2 succeeded; (b) the +43 pt may be undertuned flat baselines; (c) a 6 s horizon may sit in the +5 pt regime. |
| **EXPERIMENT** | ⭐⭐ **`E-ARCH-HWM-1` — the A10-1 arm, sharpened by §1:** three arms at ±1 % params on the v7-tiny ladder — **flat**, **flat + stride-10 waypoint auxiliary data** (the missing control), **slow/fast hierarchy** — 3 seeds, four families, constant-predictor + raw-input floor. **Committed:** hierarchy − (flat + waypoints) ≥ 3 % relative ⇒ row 18 closes for us and leads the paper; < 1 % while hierarchy − flat ≥ 3 % ⇒ **the benefit is the waypoint DATA, not the hierarchy**, and the thesis is rewritten around temporal-abstraction data; < 1 % against both ⇒ A10-1's downgrade branch fires. **Pre-gate (0 GPU):** the Deployment VGEO-3 screen — if frozen-trunk latent distance cannot beat pixels on a path-length target, run the hierarchy arm with a **learned** subgoal metric, not raw latent distance. |

## 3 · What this changes (≤3)

1. ⭐⭐⭐ **Close A10-2 (✅ read) and re-specify A10-1 as `E-ARCH-HWM-1`** — add the *flat + waypoint-data* arm; without it a positive is non-attributable (the `--v2` conflation class).
2. ⭐⭐ **Update GS-5 / the paper's positioning:** *"one capacity-controlled positive outside driving (HWM), one uncontrolled positive in driving (Drive-HWM), none both — TanitAD's matched-params driving test is the missing cell."*
3. ⭐ **Record F3 for the v7f freeze owner:** hierarchy is the one lever this month that needs **no** encoder unfreeze.

## 4 · Manifest

| artifact | location |
|---|---|
| RESULT | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-13-hwm-capacity-matched-hierarchy/RESULT.md` |
| Verbatim extracts (Tables 2 & 16, frozen-encoder lines) | `repo:…/raw/hwm_extracts.md` |
| Search log | `repo:…/raw/search_log.md` |
| Primary | `repo:TanitAD Research Lab/Library/papers/2604.03208_Hierarchical-Planning-with-Latent-World-Models.pdf` (banked 2026-08-27; cited-by updated today) |
