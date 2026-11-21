# D-DATA-EFFICIENCY — the LITERATURE half

`TanitAD Research Lab · Architecture & Inference · 2026-08-30`
`PI-commissioned. The DataFlyWheel runs the EMPIRICAL half on our own corpus in parallel.`
`Work package: TanitAD Research Lab/Architecture & Inference/Research/2026-08-30-data-efficiency-thesis/`

---

## 0. THE ANSWER

**The lever is not data volume, and it is not "all of them."**

On the only single-ladder closed-loop ablation in the literature that prices architecture, recipe
and data together (**TransFuser++ Table 13**, CARLA Longest6, 3 training × 3 eval seeds), an
**ARCHITECTURE change bought +17 DS**, a **RECIPE change (augmentation) +9 DS**, and **TRIPLING
THE DATA +2 DS** at 3× the training cost. On the same page, **freezing the encoder cost −10 DS.**
Data volume was the worst lever per unit cost of everything measured there — and that ladder's
3× data was **volume without diversity** (the same routes re-run with different traffic), which
is exactly the axis the PI is asking about.

**Three things reshape that ranking, and all three are measured:**

1. ⭐ **DATA SELECTION is the largest measured data lever on driving — and the largest measured
   way to waste data.** MOSAIC (NAVSIM v2 EPDMS): a good selector reaches random's score with
   **75–85 % less data** (BRMR 0.15–0.25 ⇒ **4–6.7×**); the *uncertainty* selector needs
   **2–14.6× MORE** than random. **The spread between selectors (~97×) dwarfs the average
   benefit.** Which score you use decides which of those you get.
2. ⭐ **The architecture DOES lower the data requirement, and the best number is 5×.** PPGeo:
   self-supervised policy pretraining reaches **equal closed-loop success on CARLA with 20 % of
   the demonstrations** (73.3 ± 6.1 at 8 K vs 73.3 ± 2.3 at 40 K). ⚠️ **But every such gain has
   an expiry date in data volume** — Hansen measures the sign *flipping* by 100 demonstrations,
   and names **data augmentation from scratch** as the cheap substitute.
3. ⛔ **The counter-evidence is real: Waymo measures closed-loop failure falling as a POWER LAW
   at 447,000 hours.** And DINO-WM — the architecture closest to ours — is **catastrophically
   bad at small n (SR 0.08 at n=200)**. The published curves are steep at the bottom and flat in
   the middle, **and at 4,719 clips we are at the bottom.**

⭐ **THE LEVER I WOULD PULL FIRST IS NOT ON THE PI'S LIST.** It is a **defect in our own training
task that I measured in our source today**: our 3-frame channel-stacked encoder makes **2 of the
3 frames in the k=1 latent-prediction target already visible in the context**, so **two thirds of
that target is copy, not prediction**. DINO-WM measured the same class of shortcut collapsing
planning success **0.76 → 0.08**. It is free to test and it is **upstream of every other lever**:
if the world-model objective is 2/3 copy at k=1, no data experiment above is being scored
honestly. §6.

⭐ **THE SECOND THING I WOULD RUN IS BOTH THE PI'S QUESTION AND THE UNCLAIMED LITERATURE
EXPERIMENT — the same run.** Nobody has ablated **diversity at fixed hours on driving** (§7,
confirmed; the one near-miss is unquotable). Our own **D-SPEED-GATE** hard-excludes every clip
above **14 m/s**, and **88.7 % of our oracle gap is longitudinal**. §8.3 argues those are **the
same fact**, and §8.4 gives the 2×2 that discriminates. Cost: 3 training runs, **no new data
collection**.

---

## 1. ⭐ THE RANKED LEVER TABLE

Ranked by **measured effect per unit of cost**. ⛔ **Rows are NOT comparable in absolute value** —
each names its own benchmark and split, and **PDMS / DS / EPDMS / ADE / mAP are not
interchangeable**. Rows sharing one ablation ladder are marked **[L]** and only *those* are
rank-comparable. **TIER** is stamped per our own doctrine: **CONTROL** = closed-loop / success
rate; **PSEUDO-CL** = rule-compliance in non-reactive sim; **OPEN** = ADE/FDE/minADE;
**PERCEPTION** = mAP/mIoU; **LOSS** = cross-entropy only.

### TIER-1 — measured on a CONTROL or PSEUDO-CLOSED-LOOP metric

| # | LEVER | measured effect | metric · benchmark | n / seeds | cost | confound | source |
|---|---|---|---|---|---|---|---|
| **1** | ⭐ **SSL POLICY PRETRAINING** (architecture lowers the data requirement) | **5× fewer demos** for equal success: PPGeo @20 % (8 K) **73.3±6.1** ≡ random-init @100 % (40 K) **73.3±2.3**. And PPGeo@40 % **91.3** > ImageNet@100 % 87.3 (**2.5×**) | **CONTROL** · CARLA CoRL2017 Nav/NavDynamic, Town01→02 | 3 trials, std | one pretraining pass on **unlabelled** video | varies *downstream imitation* data under a fixed encoder — not WM-vs-imitation; sim; ±6.1 is wide | `2301.01006` |
| **2** | ⭐ **ARCHITECTURE — predictor/decoder design** | **+17 DS** (40±3 → 57±3) | **CONTROL** · CARLA Longest6 **[L]** | 3 train × 3 eval | ~0 | cumulative ladder — ordering assigns credit | `2306.07957` T13 |
| **3** | ⭐⭐ **DATA SELECTION over a pool** | **4–6.7× less data** at equal score (BRMR **0.15–0.25**); MOSAIC **84.25** vs Random **80.38** @4,000 clips | **PSEUDO-CL** · NAVSIM v2 EPDMS (OpenScene) | 3 seeds | one scoring pass over the pool | ⛔ **the same table: Uncertainty 73.46 = 6.9 pts WORSE than random, BRMR up to 14.58.** Needs a pool. ~120 h scale | `2604.08366` T1 |
| **4** | **ARCHITECTURE — TRAINED vs FROZEN encoder** | **−10 DS if frozen** | **CONTROL** · CARLA val towns | not stated | free (a flag) | a single sentence, no table | `2306.07957` §3.4 |
| **5** | **RECIPE — data augmentation** | **+9 DS** (57±3 → 66±4) | **CONTROL** · CARLA Longest6 **[L]** | 3 train × 3 eval | ~0 | cumulative ladder. ⭐ Hansen names augmentation as what *substitutes* for pretraining at scale | `2306.07957` T13 |
| **6** | ⭐ **COMPOSITION / DIVERSITY at ~fixed volume** | **RT-1: 97 % of the data, 25 % fewer TASKS → distractor generalization 83 → 42.** `2507.06219`: identical 10 % volume, **0.46 (episode) vs 0.36 (task) vs 0.33 (scenario)** | **CONTROL** · robotics manipulation | — | re-selection only | ⛔ **ROBOTICS, NOT DRIVING.** No driving equivalent exists (§7) | `2212.06817`, `2507.06219` |
| **7** | **RECIPE — auxiliary speed-prediction head** | *"substantially improve the success rate"*; reduces but does not remove the inertia failure | **CONTROL** · NoCrash, CARLA | 4 seeds | free | magnitude is a FIGURE only; seed variance up to **42 %** | `1904.08980` §5 |
| **8** | **RECIPE — two-stage training** | **+3 DS** (64±4 → 67±2) | **CONTROL** · Longest6 **[L]** | 3×3 | small | cumulative | `2306.07957` T13 |
| **9** | **INFERENCE COMPUTE — 3-seed ensemble** | **+3 DS** (70±4 → 73) | **CONTROL** · Longest6 **[L]** | 1 | **3× inference** — inadmissible on Thor | no std on the ensemble row | `2306.07957` T13 |
| **10** | ⛔ **DATA VOLUME ×3, SAME distribution** | **+2 DS** (67±2 → 69±1) | **CONTROL** · Longest6 **[L]** | 3×3 | **3× collection + 3× training** | ⭐ the extra data is the *same routes*, new traffic ⇒ this prices **volume without diversity** | `2306.07957` T13 |
| **10b** | *(same lever, easier split)* | **+6 DS** (54±1 → 60±**6**) | **CONTROL** · CARLA val towns | 3 seeds | as above | ⚠️ **the delta equals 1σ of the larger arm** | `2306.07957` T5 |
| **11** | ⛔ **DATA VOLUME from a LOW-DIVERSITY source** | **NEGATIVE** — best at **10 h of 100 h**; inertia failures rise **monotonically** with data | **CONTROL** · NoCrash, CARLA | 4 seeds (vs **42 % variance over 12**) | 10× collection wasted | the *mechanism* (Fig. 4) is far more robust than the point estimate | `1904.08980` §5 |
| **12** | **DATA VOLUME in a latent world model, low-n regime** | ⛔ **the OTHER sign: SR 0.08 → 0.72 over 25× data** (n=200 → 5,000); then 0.72 → 0.92 over the last 3.7× | **CONTROL** · PushT planning SR | no CI | linear in data | PushT ≠ driving; **but it is the architecture closest to ours** | `2411.04983` T5 |

### TIER-2 — OPEN-LOOP / PERCEPTION / LOSS. ⛔ Not comparable to Tier-1 and not a capability claim.

| # | LEVER | measured effect | metric · benchmark | confound | source |
|---|---|---|---|---|---|
| **13** | ⭐ **MODEL CAPACITY as a data substitute** | ResNet-18 → ResNet-50 (11.2 M → 24.3 M backbone) reaches the **same FDE with ~3,000 h instead of 8,192 h = 63 % less data** (lane change **78 %**) | **OPEN** · FDE, NVIDIA internal | ⛔ capacity scaling **did NOT** relieve that paper's closed-loop plateau | `2504.04338` §6.2.2 |
| **14** | **WM/forecasting PRETRAINING for label efficiency** | ViDAR **@½ labels 39.4 mAP > baseline @full 37.7** ⇒ **2×**, delta grows as labels shrink | **PERCEPTION** · nuScenes 3D det | perception, not control | `2312.17655` |
| **15** | **SSL DATA CURATION (closest analogue to WM pretraining)** | **100 M curated beats 743 M raw at equal iterations** (IN linear probe 84.7 vs 82.8; **IN-A 66.4 vs 46.9**) ⇒ **7.4×** | **PERCEPTION/repr.** · ImageNet | bootstraps from an IN-1k SSL encoder | `2405.15613` |
| **16** | **WM OBJECTIVE as CO-TRAINING** | ADE **−33.9 %** over 700 k→70 M *with* WM vs **−3.9 %** without | **OPEN** · in-house ADE, n=100 scenarios | ⛔ **+3.6 % at 70 k, −3.6 % (WORSE) at 700 k.** A LARGE-data lever | `2510.12796` T3 |
| **17** | **MODEL SIZE (compute-optimal)** | **N ∝ C^0.63** vs **D ∝ C^0.44** ⇒ params grow **1.5× as fast** as data | **LOSS + OPEN** · internal Waymo, 447 k h | ⛔ the programme's banked gloss was **backwards** (§2.4). No R² published | `2506.08228` §4.1 |
| **18** | **OBJECTIVE changes the compute allocation** | **WM: N ∝ C^0.49–0.62** · **BC: D ∝ C^0.68** ⇒ imitation is the data-hungry objective | **LOSS** (explicitly not control) · Bleeding Edge | a video game, not driving | `2411.04434` |
| **19** | **RARITY / TAIL-MINING ECONOMICS** | long-tail is **0.03 %** of driving ⇒ accumulating it by volume needs **≈3,333×** the miles (6,391,012 mi → 6,888 mi @0.1 % → ×0.30 human filter) | — (a corpus-construction ratio) | ⛔ **WOD-E2E is an EVAL benchmark.** It does NOT show curated *training* beats volume | `2510.26125` §3.3.2 |
| **20** | **TARGETED TAIL-FILLING at fixed total** | **+45.1 % samples of ONE scenario type → −9.7 % ADE on it**; +339 % → **−32.9 %** | **OPEN** · ONE-Drive internal, 2 M demos held constant | ⛔ **gains reported ONLY on the upweighted slices; the other 21 types are unreported** | `2412.02689` §V-B |
| **21** | ⛔ **DIVERSITY AT FIXED HOURS, on driving, on a driving metric** | **NO ADMISSIBLE MEASUREMENT EXISTS** | — | — | **GAP** — §7 |

### The one-sentence ranking

**Pretraining/objective (5×) ≈ selection over a pool (4–6.7×) > architecture (+17 DS) >
avoiding a frozen encoder (+10 DS) > augmentation (+9 DS) > two-stage ≈ ensemble (+3 DS) >
data volume ×3 at fixed composition (+2 DS) > data volume from a low-diversity source
(NEGATIVE) — with model capacity (63 % less data) sitting outside the comparison at
open-loop tier.**

### ⭐ THE LEVER TO PULL FIRST

**Not from this table.** Fix the **target–context frame overlap** in our own latent-prediction
objective (§6.2): it is MEASURED in our source, free, has a published analogue with a **9.5×
effect**, and is **upstream of every row above**. Then run the **fixed-hours diversity 2×2**
(§8.4), which is simultaneously the PI's question, the literature gap, and the test of our own
D-SPEED-GATE exclusion.

Of the PI's named candidates the ranked first pull is **ARCHITECTURE / OBJECTIVE**, then
**COMPOSITION (selection + diversity)** — **not volume**.

---

## 2. THE ANCHORS — RE-VERIFIED FROM THE BANKED PDFs. FIVE OF SIX NEEDED CORRECTING.

All six anchors were opened, page-counted (15/21/17/31/18/22 pages — none truncated) and
re-read. **Five of the six needed a correction or a caveat the programme was not carrying.** Only TransFuser++ survived unamended.

### ✅ 2.1 WOD-E2E (`2510.26125`) — the ratio holds, the attribution does not

**VERIFIED, §3.3.2 verbatim scope:** a case study over **6,391,012 miles** of logs yielded
**6,888 miles (0.1 %)** by automated mining; a human filtering pass at **30 % conversion**
reduced it to **0.03 %**. 4,021 segments, 20 s each, split 2,037 / 479 / 1,505.
⇒ **the ≈3,333× curation-vs-volume ratio is PRIMARY and correct** (1 / 0.0003).

⛔ **CORRECTION 1 — "Waymo, holding 447,000 h" is NOT from this paper.** WOD-E2E states
*miles* (6.39 M), never hours. **447 thousand hours is Table 1 of `2506.08228`** — a
different Waymo paper with a different corpus (5.6 M miles, 59.8 M run segments). The
programme's anchor **conflated two papers**, and the derived "447,000 h ÷ 12 h = 37,250×"
is not any paper's claim. Quote the **0.03 % → 3,333×** form; it is the defensible one.

⚠️ **CORRECTION 2 — internal inconsistency in the source.** The abstract says
"4,021 segments (approximately **12 hours**)" but §3.1 says each segment is **20 s**;
4,021 × 20 s = **22.3 h**. Even netting the test split's 12 s release it is 19.0 h.
Quote the segment count, not the hours.

⛔ **CORRECTION 3 — the scope error that matters most. WOD-E2E is an EVALUATION BENCHMARK.**
The 3,333× is the cost of *building a test set concentrated on the tail*. It is **not**
evidence that training on curated data beats training on volume. Nothing in this paper trains
a data-scaling ablation. ⇒ **The anchor may be used to argue for a curated EVAL corpus and
for tail-mining economics. It may not be used to argue our TRAINING corpus can be small.**

⭐ **TWO NEW FINDINGS THIS PAPER CARRIES AND WE HAD NOT BANKED:**

- **§4.4 Q2, verbatim conclusion: "a better ADE does not guarantee a better RFS."** Over
  **19 leaderboard submissions** they observe "only a mild positive correlation" between ADE
  and the human-preference metric. ⇒ Independent published support for our four-metric-families
  rule and for EVAL_DOCTRINE's refusal to let ADE stand as "the result".
- **§4.4 Q1 is an ARCHITECTURE × DATA interaction — and its own table does not support its
  own answer.** The paper answers "adding extra data sources helps MLLM-based models, only
  minor gains for diffusion-based". But the table reads: **HMVLM, 1 dataset → RFS 7.736;
  Poutine, 2 datasets → 7.986; AutoVLA, 3 datasets → 7.556.** ⇒ **not monotone in the number
  of data sources**, and the models also differ in backbone, CoT style and RL. ⛔ **This is
  observational across leaderboard entries, not a controlled ablation. Do not cite Q1 as
  evidence that more data sources help.** I state it because it is the only place in the
  anchor set where a driving benchmark touches the composition axis at all — and it is
  confounded past usability.

### ⚠️ 2.2 DINO-WM (`2411.04983`) — the anchor understates its own point, and the metric lesson is sharper than banked

**VERIFIED, App. A.4.1 Table 5, PushT, planning success rate (SR) — a CONTROL metric:**

| dataset size (trajectories) | SR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| n = 200 | **0.08** | 0.949 | 0.056 |
| n = 1,000 | 0.48 | 0.973 | 0.013 |
| n = 5,000 | 0.72 | 0.981 | 0.007 |
| n = 10,000 | 0.88 | 0.984 | 0.006 |
| n = 18,500 | **0.92** | 0.987 | 0.005 |

⇒ **92.5× data · SSIM +4.0 % · SR ×11.5.** The banked anchor is arithmetically correct.

⛔ **CORRECTION 4 — the anchor's LESSON is wrong as generalised.** The banked reading is
"a data lever must be judged on CONTROL, not reconstruction." But in the same table
**LPIPS improves 11.2× (0.056 → 0.005), which tracks the 11.5× control gain almost exactly.**
⇒ **The failure is SSIM specifically — it is near its ceiling and insensitive. It is not that
prediction fidelity is uninformative.** A *perceptual* fidelity metric was an excellent proxy
here. ⇒ **Restate the rule as: never use a saturating fidelity metric as the data-efficiency
score; validate any proxy against the control metric on the same sweep.**

⭐ **AND THE ANCHOR OMITS THE MOST IMPORTANT HALF FOR US: THIS IS THE STRONGEST PRO-VOLUME
EVIDENCE IN THE SET, IN THE ARCHITECTURE CLOSEST TO OURS.** A latent world model at n=200 is
not mildly worse — it is **useless (SR 0.08)**. The elbow is at n≈5,000 (SR 0.72), i.e. **most
of the gain is bought in the first 25×, and the last 3.7× buys 0.72 → 0.92.** ⇒ Under-data a
latent world model and you do not get a degraded planner, you get **no planner**. This is the
single most relevant curve in the review for §8.

⚠️ **Confounds:** PushT is a 2-D pushing task, not driving; n is *trajectories*, not hours;
single environment; no CI reported on any cell.

### ⚠️ 2.3 Codevilla ICCV 2019 (`1904.08980`) — the mechanism survives, the point estimate is fragile

**VERIFIED §5:** CARLA100, comparing **2 / 10 / 50 / 100 h** of expert demonstrations.
*"Our best results on most of the scenarios were obtained by using only 10 hours of training
data, in particular on the 'Dense Traffic' tasks and novel conditions such as New Weather and
New Town."* Fig. 4: the share of episodes failing to the **inertia problem increases with the
amount of training data**. Mechanism verified §3.2: when stopped, *"the probability it stays
static is … overwhelming in the training data,"* creating a spurious low-speed → no-acceleration
correlation. Fix: a speed-prediction branch (CILRS) — *"not a final solution."*

✅ The brief's warning is correct and I repeat it: **this is NOT an argument for high-speed data.**

⛔ **A CAVEAT THE PROGRAMME'S ANCHOR DOES NOT CARRY, and it is large. Table 3: the seed
variance of CILRS success rate on NoCrash is 23 % / 26 % / 42 % (Empty / Regular / Dense) over
12 seeds** — *"the success rate can change by up to 42 % for tasks with dynamic objects."*
The data-scaling comparison (Fig. 3) used **4 seeds**. ⇒ **"10 h beats 100 h" sits inside a
±42-point band estimated from 4 runs. Treat the POINT ESTIMATE as indicative only.** The
durable claims are (a) the *mechanism*, and (b) the *monotone* inertia trend in Fig. 4, which
is directional and far more robust than an aggregate success rate.

⚠️ **The authors name the confound themselves, and it is our axis:** the degradation is
*"the risk of overfitting to data that lacks diversity … exacerbated by the limited spatial
extent and visual variety of our environment."* ⇒ Codevilla is best read as **"volume drawn
from a low-diversity distribution can be net-negative on closed-loop generalisation"** — which
is the diversity thesis, in sim, at 100 h.

### ⛔ 2.4 Waymo scaling (`2506.08228`) — THE PROGRAMME'S GLOSS IS BACKWARDS

**VERIFIED §4.1:** for their symmetric encoder–decoder models,
**N_opt ∝ C^0.63** and **D_opt ∝ C^0.44** ⇒ *"the optimal model size should grow ∼1.5 times as
fast as the number of training examples."* At equal compute an optimal LLM is **~50× larger**
than an optimal motion-forecasting model. Corpus (Table 1): **447 thousand hours**, 5.6 M miles,
59.8 M run segments, **541 M training examples**.

⛔ **CORRECTION 5 — the banked gloss "for driving, spend on DATA not PARAMETERS" inverts the
exponents.** 0.63 > 0.44 means that as compute grows you add **parameters faster than data**.
The two defensible readings, kept apart:

1. **Within driving, as compute scales:** params grow ~1.5× as fast as data. *Pro-parameters.*
2. **Across domains at equal compute:** a driving optimum is ~50× smaller than an LLM optimum,
   i.e. driving models are relatively **more data-hungry per parameter** than LLMs. The paper's
   own hedge: *"potentially suggesting the importance of collecting more data, **or perhaps
   improving the training data sampling techniques for this domain**."*

Only reading 2 supports "spend on data", and it is a cross-domain statement, not a within-budget
allocation rule. **The registry line should be corrected.**

⚠️ **Tier:** the headline metrics are cross-entropy loss and **minADE / wADE — OPEN-LOOP**.
The paper explicitly flags that *"it is unclear if they are an unbiased estimator of closed-loop
simulation performance"*. Under EVAL_DOCTRINE most of this paper is **T0-class**. §5 covers the
closed-loop part, which is real and is the strongest counter-evidence in the review.

⚠️ The authors themselves caution the fits are under-powered: establishing a power law
*"would require a study spanning many more orders of magnitude"*, and *"a parabolic form fits
our observed data better."* By our own exponent rule (fit window + R² + n or it is inadmissible)
**no exponent from this paper may decide a GPU-day without re-derivation.**

### ✅ 2.5 TransFuser++ (`2306.07957`) — verified, and it is the review's centrepiece

**Table 5 (validation towns, LAV):** 185k → 555k frames, DS **54±1 → 60±6**. The paper's own
framing: *"a clear improvement of 6 DS via scaling … at the cost of 3× longer training."*
⚠️ **The delta equals one standard deviation of the larger arm.**

⭐ **Table 13 (CARLA Longest6, cumulative ablation, std over 3 training × 3 evaluation runs) is
the most valuable single artifact in this review, because it prices architecture, recipe and
data on ONE ladder, ONE benchmark, ONE team, with seeds:**

| ablation step | DS | Δ DS |
|---|---|---|
| TransFuser (reproduced) | 40 ± 3 | — |
| **+ transformer decoder** | 57 ± 3 | **+17** |
| **+ data augmentation** | 66 ± 4 | **+9** |
| + disentangled output | 64 ± 4 | −2 |
| + two-stage training | 67 ± 2 | +3 |
| **+ 3× data** | 69 ± 1 | **+2** |
| TF++ WP variant | 70 ± 4 | +1 |
| + 3-seed ensemble | 73 | +3 |

⚠️ **Read it with its confounds:** cumulative (each row baselines on the previous, so the
ordering partly sets the credit); CARLA sim, not real driving; and the 3× data is generated by
**re-running the same routes with different traffic** — a volume increase with **near-zero
added scene diversity**. ⭐ That last point is not a weakness for us, it is the *finding*:
**volume without diversity, priced on a closed-loop metric, is worth +2 DS.**

### ⛔ 2.6 DriveVLA-W0 (`2510.12796`) — TRUE AS STATED, AND UNFAVOURABLE TO US AS APPLIED

**VERIFIED Table 3** (in-house corpus; ADE in m and collision %, **open-loop by construction**):

| model | 70k frames | 700k frames | 70M frames |
|---|---|---|---|
| VLA (VQ) baseline | 2.8520 | 1.5424 | 1.4829 |
| **+ World Model** | 2.7482 *(+3.6 %)* | **1.5985** *(**−3.6 %, WORSE**)* | **1.0563** *(+28.8 %)* |
| VLA (ViT) baseline | 3.1524 | 1.4202 | 1.1051 |
| + World Model | 2.5268 *(+19.9 %)* | 1.3436 *(+5.4 %)* | 1.0640 *(+3.7 %)* |

The brief's statement — dense world-model supervision beats sparse action supervision and the
gap widens with scale — **is correct**. The baseline plateaus (700k → 70M is **100× data for
−3.9 % ADE**); the world-model arm does not (**−33.9 %**).

⛔ **BUT THE IMPLICATION FOR A SMALL-DATA PROGRAMME IS THE OPPOSITE OF FAVOURABLE.** The paper's
own title is *"World Models **Amplify** Data Scaling Law"* — the claim is that a world model makes
**more** data **more** valuable, not that it makes **less** data sufficient. At **70k frames**
the WM advantage is **+3.6 %**; at **700k** it is **negative** for the VQ model. **Our corpus
sits at ≈947k frames (§8), i.e. between their 700k and 70M rows and far closer to the row where
the world-model objective bought nothing.** This anchor must be carried with that sentence
attached or it will be read as supporting a thesis it does not support.

⭐ **A SECOND, MORE ACTIONABLE FINDING — Table 4: the optimal ACTION DECODER REVERSES with data
scale.** All three experts initialised from an identical pretrained VLA:

| action expert | NAVSIM (103k frames) — **PDMS** | in-house (70M frames) — **ADE (m)** |
|---|---|---|
| Query-based (continuous) | **88.4** *(best)* | 1.1248 *(worst)* |
| Flow matching | 87.2 | 1.0362 |
| Autoregressive (discrete) | 85.3 *(worst)* | **1.0069** *(best)* |

⛔ PDMS and ADE are different benchmarks and **must not be compared across the columns** — only
the **rank reversal within each column** is the finding. ⇒ **A measured ARCHITECTURE × DATA-SCALE
interaction: at small data prefer a precise continuous/query decoder; at large data prefer a
high-capacity autoregressive one.** The authors' mechanism: at small scale quantisation error
dominates; at large scale modelling capacity does. **For a small-data programme this argues
against a discrete/quantised trajectory vocabulary.**

---

---


## 3. ⭐ THE INTERACTION QUESTION — DOES THE ARCHITECTURE LOWER THE DATA REQUIREMENT?

**The PI's real question. The answer is YES, it has been measured, and the best number is 5×.
But every measurement carries an expiry date in data volume, and two of them point in
opposite directions.**

### 3.1 The one clean CLOSED-LOOP data-axis result in driving

**PPGeo (`2301.01006`)** — self-supervised *policy pretraining* from unlabelled driving video,
then imitation with a varying number of demonstrations. CARLA **CoRL2017 Nav / NavDynamic**,
Town01 → Town02 generalisation, 3 trials with std:

| arm | demos | closed-loop success |
|---|---|---|
| Random init | **100 % (40 K)** | 73.3 ± 2.3 |
| **PPGeo** | **20 % (8 K)** | **73.3 ± 6.1** |
| ImageNet init | 100 % (40 K) | 87.3 |
| **PPGeo** | **40 % (16 K)** | **91.3** |

⇒ **5× fewer demonstrations for equal closed-loop success vs random init; 2.5× vs ImageNet
init.** This is the single most direct published answer to "does the right pretraining lower
the data requirement" on a CONTROL metric in driving.

⚠️ **Scope, stated plainly:** it varies *downstream imitation* data under a *fixed* SSL encoder.
It measures "does SSL initialisation lower the demo requirement" — **not** "does a world-model
objective replace imitation". CARLA sim; ±6.1 on the matched arm is wide.

### 3.2 The objective genuinely allocates compute differently — and imitation is the data-hungry one

**Pearce et al. (`2411.04434`)**, iso-FLOP scaling of behaviour-modelling objectives on
8.6 years of *Bleeding Edge* play:

| objective | compute allocation |
|---|---|
| **World model** | **N ∝ C^0.49–0.62** (parameter-leaning) |
| **Behaviour cloning** | **D ∝ C^0.68** (data-leaning) |

⭐ **This is the cleanest measured statement that the objective changes where compute should go,
and it says a world-model objective is LESS data-hungry than imitation** — TanitAD's thesis, in
exponent form. ⛔ **But the y-axis is LOSS**, explicitly not control, and the domain is a video
game, not driving. It is suggestive, not decisive, and it may not decide a GPU-day.

### 3.3 ⛔ THE TWO RESULTS POINT IN OPPOSITE DIRECTIONS, AND THE CONTRADICTION IS THE FINDING

| paper | where a world-model / video-prediction objective helps MOST |
|---|---|
| **GR-1 / Seer** (`2312.13139`, `2412.15109`) — CALVIN ABCD→D, **CONTROL** | at **10 % data**: BC 1.04 → +video-pred 1.52 → full 2.00 = **+92 %**. At full data only **+26 %** |
| **DriveVLA-W0** (`2510.12796`) — in-house, **OPEN-LOOP ADE** | at **70 M frames**: **+28.8 %**. At 70 k: +3.6 %. At 700 k: **−3.6 % (worse)** |

⭐ **The most likely reconciliation — and it decides what WE should do:** GR-1's video prediction
is **PRETRAINING ON EXTERNAL DATA** (transfer into a small labelled set), whereas DriveVLA-W0's
world model is a **CO-TRAINING OBJECTIVE ON THE SAME DATA**. ⇒ **A world-model objective as
TRANSFER is a small-data lever; a world-model objective as EXTRA SUPERVISION on your own corpus
is a large-data lever.** We currently do the second. **This is a hypothesis about the mechanism,
not a measured fact — I could not find a paper that separates the two.** It is testable and it
is the most valuable open question this review surfaced.

### 3.4 ⚠️ EVERY DATA-EFFICIENCY GAIN FROM PRETRAINING HAS AN EXPIRY DATE — three independent measurements

| paper | tier | the decay |
|---|---|---|
| **Hansen (`2212.05749`)** — DMControl, 3 seeds | ⭐ **CONTROL** | ⛔ **THE SIGN FLIPS.** Verbatim: *"a larger number of demonstrations (100) generally favors LfS methods, whereas frozen pre-trained representations fare marginally better in the very low-data regime (10)."* Real xArm Pick: **from-scratch 55.0 vs pretrained 35.0**. Mechanism named: from-scratch **+ data augmentation** recovers most of the pretraining advantage |
| **Zoph (`2006.06882`)** — COCO detection | perception | gain **+2.6 AP @20 % → −0.8 @50 % → −1.0 @100 %**. At full COCO: random **41.1** · ImageNet 40.4 · **SimCLR 40.4**. Verbatim: *"both supervised and self-supervised pre-training methods fail to scale as the labeled dataset size grows"* |
| **ALSO (`2212.05867`)** — nuScenes / SemanticKITTI, **driving** | perception | **+4.6 mIoU @0.1 % labels → +0.6 @100 %**. And it **never matches full-label scratch with fewer labels** |
| **Hernandez (`2102.01293`)** | loss | gives the functional form: the transfer multiplier decays as **D_F^−0.82** and can go **negative** ("ossification") |

⇒ **The architecture→data-requirement substitution is real and is largest exactly where we are
(the low-data end), and it decays to zero or negative as data grows.** That is *favourable* to a
small-data programme — but it also means **the lever shrinks as we succeed**, and Hansen names
the cheap substitute: **data augmentation from scratch**.

### 3.5 Two more architecture→data substitutions, both real, both open-loop

- ⭐ **MODEL CAPACITY substitutes for data, measured on driving.** NVIDIA `2504.04338` §6.2.2:
  moving the backbone **ResNet-18 → ResNet-50** (11.2 M → 24.3 M; total 18.4 M → 31.5 M) reaches
  **the same FDE with ~3,000 h instead of 8,192 h = 63 % less data** (lane-keeping 64 %, lane
  change **78 %**, turning 68 %). ⛔ **OPEN-LOOP FDE.** And the same paper reports that capacity
  scaling **did not** relieve its closed-loop plateau — so this buys open-loop data-efficiency
  and nothing closed-loop on that task.
- **ViDAR (`2312.17655`)** — visual point-cloud forecasting pretraining, nuScenes 3D detection:
  **ViDAR at ½ labels (39.4 mAP) beats the baseline at full labels (37.7)** ⇒ **2× label
  efficiency**, with the delta *growing* as labels shrink (+4.9 / +6.5 / +6.7 / +7.3).
  ⛔ **PERCEPTION tier**, not control.

### 3.6 ⛔ THE CONFIRMED GAP AT THE HEART OF THE PI'S QUESTION

**Nobody has run: a world-model objective vs an imitation objective, on the SAME fixed dataset,
with DATA on the x-axis and a CLOSED-LOOP metric on the y-axis.** Probed four ways, all negative.

- The only paper running the actual comparison is **DITTO (`2302.03086`)** — Atari, 5 games,
  4 → 1000 expert episodes, no stated multiplier, and its world model sees N=1000 regardless of
  N_E, which confounds the axis.
- **No JEPA-family paper runs a demo-axis curve against BC.** V-JEPA 2-AC, ACT-JEPA and Drive-JEPA
  all lack one. ⚠️ **V-JEPA 2-AC's famous "62 h of unlabelled robot data" figure is NOT a
  data-efficiency claim — the paper makes none**, and its 10-trials-per-cell control results
  (Grasp-Cup 65 vs 15) have no data axis at all.
- Every MBRL sample-efficiency multiplier (DreamerV3, EfficientZero 500×) is **online RL with
  reward** against a model-free baseline the authors themselves say was not designed for
  data-efficiency. ⚠️ **DreamerV3's headline multiplier changed 130× → 10× between the arXiv v1
  and the Nature version — cite the version or do not cite it.**

⇒ **TanitAD's thesis is not refuted by the literature. It is UNMEASURED. That is the
publishable ground.**

---

## 4. ACTIVE LEARNING / CORESET / PRUNING — AND THE MODALITY TRANSFER, STATED HONESTLY

### 4.1 ⛔ THE ANCHOR EVERYONE CITES IS WEAKER THAN ITS REPUTATION

**Sorscher et al. (`2206.14486`), "Beyond neural scaling laws":** the exponential-scaling
result is **theory (a perceptron model) plus "signatures" on SVHN / CIFAR-10.** The actual
ImageNet claim is *"training on 80 % of ImageNet approximates training on 100 %"* — **20 %
pruned, not 90 %.** The authors' own caveat: *"data pruning on ImageNet may be more difficult…
because ImageNet is already carefully curated."*

⇒ ⛔ **"Beat power-law scaling by pruning" has never been demonstrated at aggressive pruning on
a large real corpus. Do not quote it as if it had.**

### 4.2 RANDOM IS A STRONG BASELINE, AND SCORE-BASED SELECTION FAILS WHERE YOU NEED IT

| finding | numbers | modality |
|---|---|---|
| **No-Free-Lunch theorems** (`2302.06960`) | random **beats** score-based methods at **≤30 % kept** | image cls |
| **DeepCore** (`2204.08499`) — 12 methods benchmarked | *"random selection is still a strong baseline"*; dominates **above 30 % kept** | image cls |
| ⛔ **Pruning eats the tail** (`2404.05579`) | Dynamic Uncertainty **DELETES ENTIRE CLASSES** at 10 % density on CIFAR-100 **while average accuracy still looks fine** | image cls |
| **Uncertainty-first loses to random on driving, TWICE** | MOSAIC EPDMS: **Uncertainty 73.46 vs Random 80.38**; nuScenes 3D det (`2205.07708`): Entropy **37.36 vs Random 42.07** mAP | driving |
| **NVIDIA production AL** (`2004.04699`, 847 k pool) | AL **69.5 vs random 69.2** = **+0.3 wMAP**. ⭐ On the authors' own **de-biased** test split, AL was *worse* than manual curation for cars (+0.28 vs +0.58) | 2D detection, real AV |
| ⭐⭐ **Uber ATG** (`2104.03956`) | *"randomly labeling regions performs better than or similar to many coarse-grained AL approaches"* — **SCENE-LEVEL selection SATURATES**; the win came from moving to **fine-grained region** selection (meanADE 2.89 → **2.29**) | LiDAR perception + prediction + planning |
| **AdaDeDup** (`2507.00049`) | nuScenes 3D det: full **0.3759** → random @70 % kept **0.3663** → AdaDeDup **0.3728**. **Random costs only 0.0096 mAP for 30 % less data** | 3D driving detection |

⛔ **The Uber ATG row is the one that binds hardest on us: TanitAD selects at EPISODE / CLIP
granularity, which is exactly the granularity measured to saturate.** If we build a curation
score, the literature says the return is at *sub-clip* granularity, not clip granularity.

### 4.3 ⭐ BUT THE POSITIVE CASE ON DRIVING IS REAL AND LARGE — MOSAIC IS THE BEST EVIDENCE IN THE REVIEW

**MOSAIC (`2604.08366`), NAVSIM v2 EPDMS** (a 9-metric rule-compliance aggregate in
non-reactive sim — pseudo-closed-loop, not open-loop ADE), 3 seeds. **BRMR = budget ratio to
match random; lower is better.**

| budget (OpenScene clips) | Random | Uncertainty | Coreset | Chameleon | **MOSAIC** |
|---|---|---|---|---|---|
| 250 | 72.84 ± 1.14 | 70.78 (**BRMR 14.58**) | 76.26 (0.20) | 72.97 (0.86) | **77.38 (0.15)** |
| 1000 | 75.84 ± 0.90 | 71.12 (BRMR 8.00) | 80.46 (0.22) | 79.08 (0.49) | **81.68 (0.18)** |
| 4000 | 80.38 ± 0.55 | **73.46 (BRMR 2.00)** | 83.63 (0.25) | 82.92 (0.39) | **84.25 (0.18)** |

⇒ **A GOOD selector reaches random's score with 75–85 % LESS DATA (BRMR 0.15–0.25 ⇒ 4–6.7×).
A BAD selector (uncertainty) needs 2–14.6× MORE than random.**

⭐⭐ **THE SHAPE OF THE ANSWER IS HERE: the spread between selectors (14.58× → 0.15×, a factor
of ~97) is far larger than the average benefit. Curation is simultaneously the biggest measured
data lever on driving AND the biggest measured way to waste data. Which score you use decides
which of those you get.**

⚠️ Scope: ~120 h nuPlan-derived pool, budgets 250–4,000 clips, **seed** SDs (not episode-cluster);
the ~1-point MOSAIC-over-Coreset margin is marginal, but the 6.9-point Uncertainty-vs-Random gap
is many SDs. `2604.08366` is a 2026 preprint.

### 4.4 THE MODALITY-TRANSFER LEDGER, since most of this literature is CIFAR/ImageNet

| line of work | demonstrated on | demonstrated on VIDEO? | demonstrated on DRIVING? |
|---|---|---|---|
| Sorscher / prototype pruning | image cls | ✗ | ✗ |
| Forgetting events (`1812.05159`) — 30 % of CIFAR-10 removable, **only 8 % of CIFAR-100** | image cls | ✗ | ✗ |
| EL2N / GraNd (`2107.07075`) — 50 % of CIFAR-10 | image cls | ✗ | ✗ |
| CRAIG / GLISTER / GradMatch | image cls (**CRAIG + GLISTER could not run on ImageNet — memory**) | ✗ | ✗ |
| Memorization (`2008.03703`) | image cls | ✗ | ✗ |
| SemDeDup / D4 / DFN / DsDm / DataComp | **CLIP + LLM, not vision-control** | ✗ | ✗ |
| Active learning | image cls | ✗ | ✅ detection (`2004.04699`, `2507.00049`), ✅ planning (`2403.02877`, `2604.08366`) |
| Coreset / submodular | image cls | ✅ video-language (`2504.14875`, keeps 5.4 %) | ✅ **`2604.08366` only** |
| **Pruning a driving VIDEO corpus, scored by a WORLD-MODEL or CLOSED-LOOP metric** | — | — | ⛔ **NOBODY** |

⚠️ **DataComp's headline (72.3 % → 79.2 % IN zero-shot from ~11 % kept) fixes the number of
samples SEEN, not the unique data** — filtered pools are *repeated*. It is not a
"less data" result in our sense.

⭐ **The closest analogue to world-model pretraining is `2405.15613`**: automatic SSL curation
where **100 M curated beats 743 M raw at equal iterations** (ImageNet linear probe 84.7 vs 82.8;
**ImageNet-A 66.4 vs 46.9**). ⇒ **7.4× less data, better result, on a self-supervised objective.**
⚠️ It bootstraps from an ImageNet-1k SSL encoder, and the metric is representation quality.

---

## 5. THE COUNTER-EVIDENCE — the strongest published case AGAINST the small-data thesis

*The PI asked for this stated plainly, not buried. Here it is.*

### 5.1 ⛔ THE SINGLE MOST DAMAGING RESULT: WAYMO MEASURES CLOSED-LOOP SCALING AT 447,000 HOURS

`2506.08228` §5.2, verified: closed-loop failures **η** (over a reactive 30 s sim, 10 Hz, 128
rollouts, first 0.1 s executed) **decrease as a power law with pretraining compute**. Verbatim:
*"closed loop performance also follows a similar scaling trend … This suggests that open loop
performance can serve as a good proxy for closed loop."* They explicitly contrast this with the
prior literature finding no open→closed transfer, and attribute the difference to their setup
being **controlled** — one architecture, one loss, only scale varying.

**This is the strongest counter-evidence in the review and it should not be argued away.**

⚠️ **Three scope facts that shrink it without dissolving it:**
1. The x-axis is **pretraining COMPUTE**, not data. Within compute-optimal scaling
   **N ∝ C^0.63 > D ∝ C^0.44** ⇒ **the faster-growing term is PARAMETERS.** The paper does not
   show that data *volume alone* drives closed-loop improvement.
2. **η is an imitation-similarity metric** — a scenario fails if the policy progresses
   significantly more or less than the human, or collides. Not a safety/comfort composite.
3. Internal Waymo eval set; externally unreproducible. The authors caution their own power-law
   fits are under-powered and that *"a parabolic form fits our observed data better."*

### 5.2 ⛔ THE ASYMMETRY IS STRUCTURAL AND IT IS THE HONEST HEADLINE

**Every pro-curation result in the literature sits on ≤32,539 clips (~120 h). Every scale result
sits on 8,192–447,000 h. No curation study has ever been run at the scale where the scaling
claims are made.** That cuts against confident claims in *both* directions, and it is the single
most important sentence in this review for calibrating how much to believe §4.3.

### 5.3 DATA VOLUME IS CATASTROPHIC TO UNDER-SUPPLY IN EXACTLY OUR ARCHITECTURE

DINO-WM (§2.2): a latent world model at **n=200 trajectories scores SR 0.08 — useless.** The
elbow is at n≈5,000. ⇒ **The failure mode of an under-fed latent world model is not a degraded
planner, it is no planner.** Anyone arguing "small data is enough" must first show we are past
the elbow.

### 5.4 CAPACITY, NOT CURATION, WAS THE CHEAP LEVER IN THE ONE PLACE BOTH WERE PRICED

NVIDIA `2504.04338`: **ResNet-18 → ResNet-50 = 63 % less data for equal FDE.** For a programme
whose framing is *sub-300 M parameters + clever curation*, the measured cheap lever in that
paper was **parameters**. ⚠️ Open-loop only, and it did not touch that paper's closed-loop plateau.

### 5.5 What the frontier's own published record does NOT contain

- **GAIA-1 (`2309.17080`, 4,700 h) and GAIA-2 (`2503.20523`, ~13,900 h) publish ZERO driving
  performance numbers** — no closed-loop, no planning metric, no collision rate; GAIA-1 reports
  no FID/FVD either. Wayve's published "scale won" case is a **generative** claim only.
- **Wayve's LINGO scaling statements are BLOG-ONLY — no primary exists.** Their peer-reviewed
  papers (`2310.01957`, `2312.14115`) contain **no data-scaling curve**.
- **Alpamayo-R1 (`2511.00088`) publishes no data-scaling curve.**
- **UniAD / VAD / SparseDrive / DiffusionDrive / Hydra-MDP contain no data-scaling ablation** —
  structurally, because they all train on fixed splits (nuScenes 700 scenes ≈ 5.5 h, NAVSIM
  navtrain 1,192 scenarios). There is nothing to scale, so the question was never asked.
- **Robotics does not supply the clean "volume wins" curve either.** RT-1's own ablation
  concludes **diversity matters more than quantity**; Open X-Embodiment is a *breadth* result,
  not a volume curve.

### 5.6 ⚠️ AND THE COUNTER-COUNTER-EVIDENCE: TWO INDEPENDENT GROUPS FIND CLOSED-LOOP DOES **NOT** SCALE

| paper | finding |
|---|---|
| **NVIDIA `2504.04338`** §6.3, verbatim | *"Despite observing consistent open-loop performance gains at larger data scales, the MDBF results improved only up to the 256-hour mark, beyond which they plateaued around 1 km."* ⇒ **32× data (256 h → 8,192 h) for ZERO closed-loop gain** — and *"neither data augmentation nor capacity scaling alleviated the plateau."* ⚠️ **Scope: two HIGHWAY scenarios with minimal traffic, primary challenge lane-keeping, in DRIVE Sim.** An easy task may simply have hit its ceiling |
| **ONE-Drive `2412.02689`** (~30,000 h, 4 M demos) | open-loop **ADE = 0.6833·X^−0.188, r = −0.963** across a 400× range — a clean power law — but verbatim *"this is not the case in closed-loop evaluation"* |

⇒ **Three groups, three corpora, and they disagree about the thing that matters.** Waymo (447 k h)
says closed-loop scales; NVIDIA (8 k h) and Li Auto (30 k h) say it plateaus.
⚠️ **Neither driving exponent is quotable bare: `−0.396`/`−0.417` are fit against HOURS with NO R²
reported; `−0.188` is fit against NUMBER OF DEMONSTRATIONS on a different normalisation (r=−0.963
⇒ R²=0.927). They may never be placed side by side.** By our own rule, neither may decide a GPU-day.

---

---


## 6. ⭐⭐ THE ARCHITECTURE SUB-AXIS: HOW SHOULD THE ENCODER HANDLE TIME?

*The coordinator's added axis. §6.1–6.5 is a defect I MEASURED in our own source today and is
new; §6.6–6.9 is the literature, and it REFUTES the hypothesis I was asked to test.*

### 6.1 What our encoder and predictor actually do — MEASURED from source, file:line

| fact | evidence |
|---|---|
| the encoder input is **3 consecutive frames CHANNEL-STACKED** (9 ch) through one patch-embed Conv2d | `stack/tanitad/data/comma2k19.py:633-640` `stack_frames()`; `stack/tanitad/data/physicalai.py:737` |
| episodes are resampled to **10 Hz**, so the stack spans **200 ms** | `stack/tanitad/data/physicalai.py:717-718` (`TARGET_HZ`) |
| `stack_frames` output index *j* holds 10 Hz frames **{j, j+1, j+2}** | `comma2k19.py:640` — `parts = [vid_u8[i : T-(n_stack-1)+i] for i in range(n_stack)]` |
| the training window is **`ep.frames[t : t+w]` — STRIDE 1**, i.e. consecutive stacked steps | `stack/tanitad/data/_contract.py:129-135` `EpisodeWindowDataset.__getitem__` |
| **`w = 6`** | `stack/tanitad/models/v6.py:4006` (`window=6`) |
| the prediction target is **`ep.frames[t+w : t+w+max_horizon]`** | `_contract.py:132-133` |
| the live trainer uses exactly this dataset | `stack/scripts/train_v6_staged.py:5090-5093` → `FlagshipWindowDataset` (`train_flagship4b.py:106`, subclass of `FailLoudWindowDataset` → `EpisodeWindowDataset`) |
| the latent target is the encoder run on the **FULL 9-channel future stack** | `train_v6_staged.py:5665-5668` — `z_flat = readout(encoder(ff.reshape(...)))`; `z_true = [z_flat[:, j] …]` |
| `z_true[0]` (**horizon k=1**) is present whenever the latent objectives are on | `train_v6_staged.py:5641` — `ff = b["future_frames"][:, :need_k]` |

### 6.2 ⛔ THE CONSEQUENCE: OUR k=1 LATENT-PREDICTION TARGET IS **2/3 ALREADY OBSERVED**

```
last observed context latent   z_{t+5}  = enc( frames { t+5, t+6, t+7 } )
first prediction target        z_{t+6}  = enc( frames { t+6, t+7, t+8 } )
                                                  ^^^^^^^^^^^^
                                    2 of 3 frames ALREADY SEEN by the context
```

| horizon k | target frames | overlap with the observed union {t … t+7} | fraction of the target that is COPY |
|---|---|---|---|
| **k = 1** | {t+6, t+7, t+8} | {t+6, t+7} | **2/3 = 67 %** |
| k = 2 | {t+7, t+8, t+9} | {t+7} | 33 % |
| k ≥ 3 | {t+8, …} | ∅ | 0 % |

⇒ **At k=1 — a term present in every configuration — two thirds of the pixel content the target
latent is computed from was already inside the context. A predictor can capture a large share of
that target's variance with a near-identity map and learn no dynamics at all.**

⚠️ **This is NOT val contamination.** It is a **shortcut in the training task**, and it is
structurally **UNMASKABLE**: the overlap lives *inside* one latent vector, so no causal attention
mask can remove it. That is what makes it different from — and worse than — the published
failure it most resembles.

### 6.3 THE PUBLISHED ANALOGUE, MEASURED AT 9.5×

DINO-WM (`2411.04983` App. A.4.2, Table 6, PushT **planning success rate** — a CONTROL metric)
ablates exactly this class of leakage:

| history length h | **without** causal mask | **with** causal mask |
|---|---|---|
| h = 1 | 0.76 | 0.76 *(equivalent by construction)* |
| h = 2 | 0.36 | 0.88 |
| h = 3 | **0.08** | **0.92** |

Verbatim: *"we see a rapid drop in the w/o mask case, since the model can cheat during training by
attending to future observations."*

⭐ **Two readings, both load-bearing:**
1. **The leak is catastrophic when present** — 0.76 → 0.08, a **9.5× collapse on a control
   metric**. Not a second-order effect.
2. ⭐ **With the leak closed, MORE temporal context AT THE PREDICTOR is a large win** —
   0.76 → 0.92 from h=1 to h=3 (**+21 % relative**), because *"longer history could better capture
   dynamics information like velocity, acceleration, and object momentum."*

### 6.4 WHAT THE COMPARABLE WORLD MODELS PUT IN THEIR ENCODERS — verified per paper

| model | frames the **ENCODER** sees | where time lives |
|---|---|---|
| **DINO-WM** `2411.04983` | ⭐ **ONE** — `z_t ~ enc(z_t \| o_t)`, DINOv2 frozen, 14×14 patches | decoder-only ViT over `z_{t−H:t−1}, a_{t−H:t−1}`; **H = 1–3** |
| **GAIA-1** `2309.17080` — ⭐ **the closest DRIVING analogue** | **ONE** — *"each image x_t … is discretized into n = 576 discrete tokens"* | **all** time in the 6.5 B autoregressive world model |
| **I-JEPA** `2301.08243` | ONE image; **no temporal component at all** | n/a |
| **DreamerV3** `2301.04104` | **ONE** — `z_t ~ q(z_t \| h_t, x_t)`; **zero occurrences of "stack"** in the paper (full-text grep) | the recurrent state `h_t` |
| **IRIS** `2209.00588` | **ONE** — K=16 tokens/image | autoregressive transformer, context **L=20** timesteps |
| **TD-MPC2** `2310.16828` | ONE obs/timestep | single-step latent MLP |
| **V-JEPA 2** `2506.09985` | **MULTI** — 3D tubelet **2×16×16**, 16 frames | frozen encoder + predictor |
| **Genie** `2402.15391` | **MULTI** — ST-ViViT, 16 frames | ST-transformer dynamics |
| **GAIA-2** `2503.20523` | MULTI — 8 frames → one temporal latent | ⚠️ **SECONDARY** (search summary, PDF not read) |
| **TanitAD (ours)** | **THREE**, channel-stacked to 9 ch through one Conv2d | predictor: 6 latents + per-step ego |

⇒ **The single-frame-encoder camp is DINO-WM, GAIA-1, DreamerV3, IRIS, TD-MPC2 — including the
only large driving world model in the list.** The multi-frame camp is the video-SSL line
(V-JEPA, VideoMAE) and Genie.

### 6.5 THE SHARP FORM: WE PAY THE OVERLAP FOR SOMETHING THE PREDICTOR ALREADY HAS

The reason to channel-stack is that a single frame carries no velocity. **But our predictor
already receives 6 latents plus per-timestep ego** — velocity is available to it regardless.
⇒ **The 3-frame stack is largely REDUNDANT with the predictor's own history, while being the
sole source of the 67 % target overlap.**

⚠️ **The scope limit, stated so the claim does not over-reach.** `_o14_pixel_target` slices
`x[:, -3:]` — the **newest frame only** (`train_v6_staged.py:1461`) — so its k=1 target (frame
t+8) *is* genuinely one step beyond the newest observed frame (t+7). ⇒ **The defect is scoped to
objectives whose target is the FULL stacked latent (`z_true_steps`, i.e. O1/O5). It does NOT
affect O14.**

⛔ **AND THE COORDINATOR'S CORRECTION IS RIGHT AND I APPLY IT.** BEVDet4D's Table 3 result lives
in a **shared ego-centric metric BEV grid**, where a cell asserts a fixed world location, so
uncompensated ego motion makes the representation self-contradictory. **Our stack is in IMAGE
SPACE, which asserts no such correspondence — inter-frame pixel displacement IS the motion
signal.** The result does not cross that line, and **nothing in §6 argues for ego-motion
compensation.** The mechanism identified here — **target–context overlap** — is entirely
different from BEVDet4D's. ⭐ §6.6 supplies the *legitimate* image-space camera-motion result
that the retracted transfer was reaching for, and it points the opposite way.


### 6.6 ⛔ THE ENTANGLEMENT HYPOTHESIS IS **NOT SUPPORTED** BY THE LITERATURE — I ASKED FOR THIS AND I REPORT IT

*The coordinator asked explicitly not to be allowed to confirm this cheaply. It did not survive.*
**The hypothesis "a 3-frame stack entangles content and dynamics at the encoder, which harms the
downstream dynamics model" is NOT SUPPORTED, and the prescription that follows from it — "drop
the stack, let the predictor do time" — is actively REFUTED as a standalone move.**

⛔ **Evidence AGAINST the hypothesis (the decisive rows):**

| paper | space | the number |
|---|---|---|
| ⭐ **Flare (`2101.01857`)** — early-vs-late temporal fusion **inside a control loop** | IMAGE | per-frame encoding + **latent concat is WORSE than frame stacking**: *"simply concatenating the latent features results in inferior performance when compared to the frame stacking heuristic."* Mechanism they propose: *"pixel-level frame stacking benefits from leveraging both the CNN and the fully connected layers … whereas latent-level stacking does not propagate temporal information back through the CNN encoder"* |
| ⭐ **Genie (`2402.15391`) Table 3** — matched params, **same dynamics + latent-action model on top**, only the encoder varies | IMAGE | spatial-only ViT (230 M) **FVD 114.5** vs **ST-ViViT (205 M) FVD 81.4 (−29 %)**, ΔtPSNR 1.39 → 1.66. **The multi-frame encoder WINS** |
| **Two-stream (`1406.2199`)** — matched data, matched arch, from scratch | IMAGE | single-frame RGB 52.3 → **11-frame RGB stack 56.4 (+4.1)**. The stack **does** add real signal |
| **ViViT (`2103.15691`)** | IMAGE | the *most* entangled model (M1, joint space-time) is the **best** on the large dataset: K400 **80.0** vs factorised 78.8 |
| **`2606.07687`** — action recovery from frozen latents, LIBERO | IMAGE | image-only per-frame latents are near-useless for inverse dynamics (Web-DINO ViT-L **−0.01** R²) vs video-pretrained (V-JEPA 2 ViT-L **0.40**, VideoMAE **0.46**) — a **0.69 R² gap at matched ~300 M**. *"inverse dynamics cannot manufacture temporal structure absent from the representation"* |

✅ **Evidence FOR (real, and it reshapes the recommendation rather than dying):**

| paper | the number |
|---|---|
| ⭐⭐ **Karpathy CVPR 2014** — Sports-1M, **the literal channel-stacking ablation** (first conv extended to 11×11×3×T, T=10; *our construction*). Train 1.1 M videos; **test 200,000 videos / 4,000,000 clips** | Video Hit@1: Single-Frame **59.3** · **Early Fusion 57.7** · Late Fusion 59.3 · Slow Fusion **60.9**. ⇒ **channel-stacking scored BELOW the single-frame baseline it extends, and was the worst of four ways to spend the same frames** |
| ⭐⭐ **Karpathy, the camera-motion finding — and it binds hardest on us** | *"motion-aware networks are more likely to underperform when there is camera motion present. We hypothesize that the CNNs struggle to learn complete invariance across all possible angles and speeds of camera translation and zoom."* The classes where fusion **loses** to single-frame are ego-motion-dominated (motor racing, rally cross; ΔAP −0.05 to −0.07). ⛔ **Every frame we train on has camera motion.** ⭐ This is the legitimate IMAGE-SPACE replacement for the retracted BEVDet4D argument — **and it says the opposite thing**: not "compensate the ego motion", but "early temporal fusion degrades where ego motion dominates" |
| ⭐ **Two-stream, the ratio** | stacking **RGB** buys **+4.1**; the identical stacking operation on an explicit **motion field** buys **+28.7**, same data, same arch. *"while multi-frame information is important, it is also important to present it to a ConvNet in an appropriate manner"* |
| **Flare's own Q1/Q3** | *"late fusion temporal information after encoding the image is preferred to early fusion"*, and **more stacked frames can degrade** (optimal was 2 on Pendulum) — *"unnecessary information that the actor and critic networks need to learn to ignore"* |
| **MCnet (`1706.08033`)** | motion encoder on **image differences** + content encoder on the last frame beats non-decomposed ConvLSTM on KTH (~20 vs ~18 PSNR at frame 20) |

### 6.7 ⭐ THE SYNTHESIS BOTH SIDES SUPPORT — AND IT CHANGES THE RECOMMENDATION

**The deciding axis is NOT "one frame or many at the encoder." It is "is the motion
representation CHEAPLY EXTRACTABLE from what the encoder emits."**

- Channel-stacking makes motion **available in principle but expensive to extract** →
  Karpathy (below single-frame), two-stream (+4.1 vs +28.7).
- Naive per-frame encoding makes it **unavailable** → Flare, the image-SSL rows of `2606.07687`.
- **Every winner adds EXPLICIT, STRUCTURED motion:** optical flow (+28.7), a thin dedicated fast
  pathway (SlowFast +3.0 over Slow-only; **+5.2 mAP on AVA where optical flow gives +1.1**),
  latent differences, tubelets *with correct init* (+0.7 — but **−5.3 with random init**),
  video-predictive pretraining (+0.69 R²).

⇒ **REVISED RECOMMENDATION. The cheapest discriminating experiment is NOT `n_stack=1`. It is
`n_stack=1` PLUS AN EXPLICIT MOTION SIGNAL** (a latent difference `z_t − z_{t−1}`, or a thin
motion path), run against the current 9-channel arm at matched data and matched params.

### 6.8 ⭐ AND THE DATA INTERACTION — THIS IS WHERE OUR SCALE MATTERS

| paper | the entanglement × data-size interaction |
|---|---|
| ⭐ **ViViT (`2103.15691`)** | K400 (~267 k videos): joint space-time **M1 80.0 > M2 78.8**. EPIC-Kitchens (**90 k clips**): **M1 43.1 < M2 43.7**. Verbatim: *"it can also overfit on smaller datasets such as Epic Kitchens, where we find our 'Factorised Encoder' to perform the best"* |
| **Genie** | full space-time C-ViViT **FVD 272.7** vs spatial-only 114.5, attributed to *"a tendency towards overfitting, necessitating strong regularization"* |
| **TimeSformer (`2102.05095`)** | leads at 25/50/75/100 % of K400, but **on SSv2 only at 75–100 %** — *"the strongest model only when using enough training videos"* |
| ⭐ **VideoMAE (`2203.12602`) — the counterweight** | a multi-frame (tubelet) encoder is **NOT intrinsically data-hungry**: *"impressive results on very small datasets (i.e., around 3k-4k videos)"* — HMDB51 **3.5 k** videos → 62.6 %. And **42 k domain-matched pretraining videos beat 240 k Kinetics (68.7 vs 68.5)** |

⭐⭐ **THE RECONCILIATION, AND IT IS THE MOST TRANSFERABLE FINDING IN §6: the data-hunger of a
temporally-entangled encoder is a property of the TRAINING OBJECTIVE, not of the tubelet.**
Supervised classification over an entangled space-time encoder overfits at 90 k clips (ViViT) and
at Genie's scale; **masked-prediction SSL over the same tubelet is data-efficient at 3.5 k videos.**
⇒ For us — at **4,719 clips** — a channel-stacked encoder is in the regime where ViViT and Genie
both report entanglement losing, **unless the objective forces the encoder to use time.**
**Which brings §6.2 back:** our objective's k=1 term does the opposite — it lets the encoder's
extra frames be *copied* rather than *used*.

### 6.9 ⛔ AND THE COMPARISON NOBODY HAS PUBLISHED

**No published work runs "single-frame encoder + strong predictor" against "multi-frame encoder +
weaker predictor" at matched data AND matched total parameters.** The three near-misses each hold
the wrong thing fixed: Genie varies the encoder with the **predictor fixed**; Flare's downstream
is a concat + MLP — **the weakest possible aggregator, and NOT the leg we occupy** (our predictor
is a 6-step sequence model with per-step ego); `2606.07687` varies the *pretraining objective*
with a fixed 2-frame probe head. ⇒ **A second unclaimed experiment, and it is the shape of the
v7-tiny ladder.**

### 6.10 REVISED E-ARCH-OVERLAP — the arms that survive the literature

| arm | change | what it isolates |
|---|---|---|
| **A0** control | as-is (`n_stack=3`, stride-1 window, horizons 1/2/4) | baseline |
| **A1** no-copy target | drop the **k=1** latent term, supervise k≥3 | ⭐ the overlap defect ALONE — untouched by any of §6.6, because no paper studies it |
| **A2** ~~single-frame~~ **single-frame + explicit motion** | `--newest-frame-only` (C=3, flag exists at `train_v6_staged.py:1450`) **+ a latent-difference channel** | the DINO-WM/GAIA-1 configuration, corrected for Flare's refutation |
| **A3** *(new, from §6.7)* | keep `n_stack=3` **and add** an explicit motion input | tests "present motion better" rather than "remove motion" |

**Controls that must read known values** (constitution §6.2): a **COPY BASELINE**
(`z_{t+k} := z_{t+5}`) — ⭐ *if A0's k=1 latent loss is not clearly better than copy, the k=1 term
is measuring copying and §6.2 is confirmed by a control*; a **time-shuffled target** arm; **n and
d printed**. Both outcomes committed in advance; a REFUTED result is logged in
`RETRACTION_LOG.md` with its class.

---

---

## 7. ⛔ THE LITERATURE GAP — CONFIRMED, WITH ONE NEAR-MISS

**The brief's claim — that no published work ablates DIVERSITY AT FIXED HOURS — is CONFIRMED for
driving and for video, and REFUTED for robotics, LLMs and image classification.**

### 7.1 The gap holds for DRIVING and for VIDEO

| domain | fixed-volume diversity ablation? |
|---|---|
| **Driving** | ⛔ **Not properly.** One near-miss (below) |
| **Video** | ⛔ **NO** — probed ~10 distinct phrasings across three agents. Nearest miss `2411.17584` does not hold totals constant; Cosmos (`2501.03575`) documents a diversity-balanced mixture (driving 11 %) but runs **no ablation** |
| **Robotics** | ✅ **YES, and it is the cleanest design in any domain** |
| **LLM** | ✅ **YES, and it is the sharpest number** |
| **Image cls** | ✅ yes, but the two studies **disagree** |

### 7.2 ⭐ THE NEAR-MISS, AND WHY IT IS A GAP MARKER RATHER THAN EVIDENCE

**`2607.04500` — "Geographic Diversity Beats Data Volume", a JEPA driving world model.**
It asks TanitAD's exact question: **matched 63 K scenarios**, 2 geographies (**0.228 ± 0.015**)
vs 1 geography (0.273 ± 0.008); and **200 K single-geo (0.264) is WORSE than 63 K diverse.**

⛔ **But it is not quotable:** the metric is the model's **own latent MSE**
(`s = ‖z_pred − z_target‖²/d`) with **no driving metric anywhere**; the 200 K arm is **n = 1 with
stated training instability**; inputs are **structured agent state, not camera**; single author,
no institution, not peer-reviewed. ⇒ **Bank it as the GAP MARKER. Its weakness is precisely the
argument for running the experiment properly.**

Two weaker driving near-misses:
- **`2405.00242`** holds **8 hours fixed** and varies CARLA weather diversity 1→2→4: success
  ~5 → ~30 → ~50. ⚠️ Figure-read, sim-only, a **weather-only** diversity axis, town UNVERIFIED.
- **ONE-Drive `2412.02689` §V-B** holds **2 M demonstrations constant** and rebalances two
  long-tail scenario types: **+45.1 % samples of one type → −9.7 % ADE on that type; +339 % →
  −32.9 %.** ⛔ **Gains are reported only on the UPWEIGHTED slices — the other 21 scenario types
  are unreported**, so the net effect on the whole distribution is unknown. This is the closest
  thing to a composition result on driving and it is half a result.

### 7.3 THE PROTOCOLS WORTH COPYING, from the domains that did run it

| domain | the design | the number |
|---|---|---|
| ⭐ **Robotics `2507.06219`** | holds trajectory count at **10 % of AgiBot World** and varies only *which* 10 % | episode-based **0.46** > task 0.36 > scenario 0.33 — **a 1.4× spread at identical volume** |
| ⭐ **RT-1 `2212.06817` §6.5** | keeps **97 % of the data**, removes **25 % of tasks** | distractor generalization **83 → 42** — *the sharpest diversity number in any literature* |
| ⭐ **LLM `2402.10891`** | fixes **10⁶ examples**, varies the number of distinct instructions | a **phase transition at ~400–500 instructions**: below it the model *never* generalizes, regardless of examples-per-instruction |
| Image `2105.05837` / `2307.12532` | fixes 100 k / 60 k images, varies diversity | **they disagree** — helps Places365 (+3.9), *hurts* iNat21 (−2.4); an exact null on iWildCam |

⭐ **The RT-1 and LLM rows are the strongest evidence for the PI's intuition anywhere in this
review — and neither is driving.** The phase-transition shape in particular is the right thing
to look for: it predicts that **below some number of distinct situations, adding hours does
nothing**, which is exactly the failure a curated small corpus is meant to avoid.

⇒ **CONCLUSION: the gap is real, it is ours to take, and the experimental protocol already
exists in three other fields. Copy `2507.06219`'s design; measure on a CONTROL metric, which is
what `2607.04500` failed to do.**

---

## 8. ⭐ WHAT THIS PREDICTS FOR US — AND THE CHEAPEST DISCRIMINATING EXPERIMENT

### 8.1 Where we actually sit, MEASURED

| our fact | value | source |
|---|---|---|
| B1 training corpus | **4,719 clips / 26.3 h** (4,713 after the val40 parity exclusion) | `GOALS_AND_CLAIMS.md` D-CORPUS-B1, manifest sha256 `5feda062a72a32ad` |
| legacy parity corpus | **2,376 episodes / 406,099 windows** ≈ 13.2 h | `MODEL_REGISTRY.md` (run logs) |
| frames at 10 Hz | **≈ 947,000** (26.3 h) | derived |
| params | **263.4 M total / 277.4 M trainable** (v1); 272.9 M / 286.3 M (v2/v3) | `MODEL_REGISTRY.md` |
| encoder | **TRAINED** (not frozen), 3-frame channel stack | source, §6.1 |
| tier | **all T0. No v7 arm has ever been evaluated at T1.** | `EVAL_DOCTRINE.md` / registry |

⚠️ **The brief's "~336 M params" does not appear in `MODEL_REGISTRY.md`** — the registry publishes
263.4 M / 277.4 M and 272.9 M / 286.3 M, under the sub-300 M constitutional constraint. **Flagged,
not adopted.** (Also: the anchor's "our 26 h" is the **B1** corpus, not the parity corpus at
13.2 h — the two must not be conflated, and D-CORPUS-B1 already forbids cross-domain comparison.)

### 8.2 What the literature predicts, placed against our scale

| reference point | its scale | where we land | the prediction |
|---|---|---|---|
| ⭐⭐ **DINO-WM elbow** (the closest architecture to ours) | SR 0.08 @ n=200 · **0.72 @ n≈5,000** · 0.92 @ 18,500 | **4,719 clips — AT the elbow, not past it** | ⛔ **We are in the steep part of the curve. Data still buys a lot here, and under-supply fails catastrophically rather than gracefully** |
| ⭐ **ViViT / Genie entanglement × data** | entangled encoder loses below ~90 k clips | **4,719 clips — far below** | our channel-stacked encoder is in the regime where entanglement is measured to hurt |
| ⭐⭐ **MOSAIC selection budgets** | pool 32,539 clips; budgets **250 / 1,000 / 4,000** | **our entire corpus = their largest budget** | ⭐ **we operate at exactly the scale where selection was measured to give 4–6.7×** |
| **NVIDIA closed-loop plateau** | MDBF plateaus at **256 h** | **26.3 h — an order of magnitude below** | ⛔ **we are NOT past the closed-loop data plateau; volume has not been ruled out** |
| **DriveVLA-W0 rows** | 70 k / 700 k / **70 M** frames | **≈947 k frames — just past their 700 k row** | ⛔ **exactly the row where the world-model objective bought nothing (VQ: −3.6 %)** |
| **Codevilla** | best at 10 h of 100 h (CARLA) | 26.3 h | past their optimum — but their corpus was sim and low-diversity |
| **PPGeo** | 5× demo reduction at 8 k–40 k demos | comparable order | SSL pretraining is a live, unexploited lever for us |

### 8.3 ⭐⭐ THE PREDICTION I WOULD STAKE THE REVIEW ON

**Our largest known defect and our largest known curation choice are the same fact.**

`GOALS_AND_CLAIMS.md` **D-SPEED-GATE** records (MEASURED, verified in source at
`stack/scripts/physicalai_r0.py:100`) that our clip selection sets `score = 0.0` unless
`2.0 ≤ mean_v ≤ 14.0` m/s — **a HARD EXCLUSION, not a down-weight — removing every motorway,
dual-carriageway and fast-arterial clip**, with the score peaking at 8.0 m/s. The programme
separately records that **88.7 % of our oracle gap is LONGITUDINAL.**

**The literature says a composition truncation produces a capability hole exactly where the
composition was truncated**, and it says so three times with numbers:
- **RT-1**: 97 % of the data, 25 % fewer *tasks* → distractor generalization **83 → 42**.
- **`2507.06219`**: identical 10 % volume, different *which* → **0.46 vs 0.33**, a 1.4× spread.
- **`2402.10891`**: at fixed 10⁶ examples, a **phase transition at ~400–500 distinct
  instructions** — below it the model *never* generalizes regardless of examples-per-instruction.

⇒ **PREDICTION (falsifiable): we are DIVERSITY-limited before we are HOURS-limited, and the
specific missing dimension is the SPEED/ROAD-CLASS band our own gate deleted.** If true, adding
hours *within* the 2–14 m/s band will move the longitudinal gap very little, while re-admitting
the excluded band at **constant total hours** will move it a lot.

⚠️ **Held honestly:** three facts argue the other way and I am not hiding them — DINO-WM says we
are at the elbow where volume matters most; NVIDIA's plateau is at 256 h and we are at 26 h; and
`D-B1-100H` shows the programme already *wants* 100 h. **It is entirely possible we are both
data- and composition-limited.** That is precisely why the experiment below is a 2×2 and not a
comparison.

### 8.4 ⭐ THE CHEAPEST EXPERIMENT THAT DISCRIMINATES — and it is also the publishable one

**E-DATA-2×2.** Hold everything but data fixed (same model, same recipe, same steps, same seed
policy). ⭐ **The discriminating experiment IS the unclaimed literature experiment of §7 — that
is not a coincidence, it is why this is the right one to run.**

|  | **NARROW** composition (current gate) | **WIDE** composition (re-admit the excluded speed/road band) |
|---|---|---|
| **13 h** | **A** — the parity-scale control | **B** |
| **26 h** | **C** | *(D — only if B or C moves)* |

- **B − A** = ⭐ **the DIVERSITY effect AT FIXED HOURS.** *This measurement does not exist in the
  published literature for driving.*
- **C − A** = the VOLUME effect at fixed composition.
- **Whichever is larger names the binding constraint.** If both are ≈0, we are **recipe-limited**
  and §6 is where the work is.

⛔ **THE CONTROL THAT MUST READ A KNOWN VALUE — non-negotiable, and it is what §4.2 exists to
force: a RANDOM-SELECTION arm at the same hours.** DeepCore, `2302.06960`, ActiveAD, NVIDIA and
MOSAIC's Uncertainty row all measure clever selectors LOSING to random. **If WIDE does not beat
RANDOM at matched hours, the finding is about our selector, not about diversity.**

**Also mandatory per the constitution and the four-families rule:**
- ⭐ **Report LONGITUDINAL metrics per family, not ADE** — this experiment's whole hypothesis is
  longitudinal, and `2510.26125` §4.4 independently measured that ADE and a preference metric
  correlate only mildly across 19 submissions.
- **Tier-stamp everything.** Every v7 arm to date is **T0**. A composition claim about *driving*
  needs **T1** (`taniteval/tools/t1_eval.py`), or it is a world-model diagnostic wearing a
  capability claim's clothes.
- **Episode-cluster bootstrap**, paired where the arms share windows.
- ⛔ **D-CORPUS-B1 forbids same-data comparison against `physicalai-train-e438721ae894`.** All
  four arms must be built inside the B1 domain.

**Cost:** 3 training runs (A, B, C) + 1 random control, on the tiny ladder. **No new data
collection** — B is a re-selection over clips PhysicalAI-AV already contains (the D-B1-100H
backlog notes ~13,300 unlabelled clips remain, and the corpus holds 306,152).

### 8.5 THE ORDER I WOULD RUN THEM

1. ⭐ **E-ARCH-OVERLAP A1 + the copy-baseline control** (§6.10). **Free, one ladder pass, and it
   gates everything else** — if the k=1 latent term is measuring copying, every data arm above
   would be scored through a partly-degenerate objective.
2. ⭐ **E-DATA-2×2 arms A and B** (the fixed-hours diversity swap) **+ the random control**. This
   is the PI's question, the literature gap, and the D-SPEED-GATE prediction, in one experiment.
3. **Arm C** (volume at fixed composition) — only needed if B − A is small.
4. **PPGeo-style SSL pretraining** on unlabelled driving video (§3.1, 5× on closed-loop CARLA) —
   the highest-value *unexploited* lever, and it is the one whose measured benefit is **largest
   exactly at our data scale** (§3.4).


---

## 9. CORRECTIONS THIS REVIEW FILES — every one is a re-read of a banked primary

⛔ **These are corrections to numbers the programme was already carrying. They must reach
`GOALS_AND_CLAIMS.md` D-DATA-EFFICIENCY and `RETRACTION_LOG.md`, or a prose fix will not reach
the next quote.**

| # | what was carried | what the primary says | class |
|---|---|---|---|
| **1** | *"Waymo, holding **447,000 h**, built a 12-hour benchmark"* — attributed to WOD-E2E | ⛔ **WRONG PAPER.** WOD-E2E states **miles** (6,391,012 in its case study), never hours. **447 thousand hours is Table 1 of `2506.08228`**, a different Waymo paper with a different corpus (5.6 M miles). The derived "447,000 ÷ 12 = 37,250×" is no paper's claim | **two primaries conflated into one derived ratio** — same family as quoting a true measurement outside its scope |
| **2** | *"a 12-hour benchmark"* | ⚠️ **the source is internally inconsistent**: abstract says ~12 h, §3.1 says 4,021 segments × **20 s** = **22.3 h**. Quote the segment count | source-internal inconsistency, propagated |
| **3** | *"curation beat volume ~3,300×"* used to argue a **small training corpus** | ✅ **the ratio is PRIMARY and correct** (0.03 % ⇒ 3,333×) but ⛔ **WOD-E2E is an EVALUATION benchmark.** It prices building a *test set* concentrated on the tail. It is **not** evidence that curated *training* data beats volume | **scope error** — a true number used for a question it does not address |
| **4** | *"a data lever must be judged on CONTROL, not reconstruction"* (from DINO-WM) | ⚠️ **over-generalised.** In the same table **LPIPS improves 11.2× (0.056→0.005), tracking the 11.5× control gain almost exactly.** The failure is **SSIM specifically** — saturated and insensitive. ⇒ restate as *"never use a SATURATING fidelity metric; validate any proxy against control on the same sweep"* | **a correct verdict generalised past its evidence** |
| **5** | *"for driving, spend on DATA not PARAMETERS"* (from Waymo) | ⛔ **INVERTED.** N_opt ∝ C^**0.63** > D_opt ∝ C^**0.44** ⇒ as compute grows you add **parameters faster than data**. Only the *cross-domain* reading ("a driving optimum is ~50× smaller than an LLM optimum") supports "spend on data", and it is not an allocation rule | **exponent read in the wrong direction** |
| **6** | *"dense world-model supervision beats sparse action supervision, gap widens with scale"* (DriveVLA-W0) — carried as support for a small-data thesis | ✅ **the statement is TRUE.** ⛔ **The implication is the opposite of favourable.** The paper's own title is *"World Models **AMPLIFY** Data Scaling Law"*. At **70 k frames** the WM advantage is **+3.6 %**; at **700 k** it is **−3.6 % (worse)**. **Our ≈947 k frames sit just past the row where it bought nothing** | **a true finding carried with its direction of implication reversed** |
| **7** | *"Sorscher: beat power-law scaling by pruning"* (widely cited, incl. in the brief) | ⛔ **NEVER DEMONSTRATED at aggressive pruning on a large corpus.** The ImageNet claim is *"training on 80 % of ImageNet approximates training on 100 %"* — **20 % pruned**. Exponential scaling is a perceptron theory plus signatures on SVHN/CIFAR-10. Authors' own caveat: *"ImageNet is already carefully curated"* | **a theory result quoted as an empirical one** |
| **8** | Codevilla *"best at 10 h of 100 h"* carried as a bare fact | ⚠️ **the point estimate is fragile**: **Table 3 reports seed variance up to 42 %** (NoCrash Dense, 12 seeds) while the scaling figure used **4 seeds**. The durable claims are the **mechanism** and the **monotone** inertia trend (Fig. 4) | **a point estimate quoted without its variance** |
| **9** | *"NO published work ablates diversity at fixed hours"* | ✅ **CONFIRMED for driving and video** (probed ~10 phrasings across three streams). ⛔ **REFUTED for robotics, LLMs and image classification**, where clean designs exist and should be copied (`2507.06219`, `2212.06817` §6.5, `2402.10891`). One driving near-miss (`2607.04500`) exists and is **not quotable** (latent-MSE only, n=1 on an arm, not peer-reviewed) | **the gap holds, with its scope narrowed and the protocol located** |
| **10** | *"our ~336 M params"* (in the commissioning brief) | ⚠️ **not in `MODEL_REGISTRY.md`**, which publishes **263.4 M total / 277.4 M trainable** (v1) and 272.9 M / 286.3 M (v2/v3), under the sub-300 M constraint. **Flagged, not adopted** | **a model fact quoted from prose rather than the registry** |
| **11** | *"our 26 h"* | ⚠️ **that is the B1 corpus (4,719 clips / 26.3 h, D-CORPUS-B1)**, not the parity corpus (**2,376 eps / 406,099 windows ≈ 13.2 h**). D-CORPUS-B1 already forbids comparing across them | **two corpora conflated** |
| **12** | *(the coordinator's own, applied)* BEVDet4D Tab. 3 transferred to an image-space frame stack | ✅ **the retraction is right and I apply it.** ⭐ **And §6.6 supplies the legitimate image-space replacement** — Karpathy's camera-motion finding — **which points the OPPOSITE way**: not *"compensate the ego motion"* but *"early temporal fusion degrades where ego motion dominates"* | **metric-space result transferred to image space (MM-C9), now superseded by an in-space result** |

### 9.1 THE HYPOTHESIS I WAS ASKED TO TEST AND COULD NOT CONFIRM

⛔ **"A 3-frame channel stack entangles content and dynamics at the encoder, harming the
downstream dynamics model" — NOT SUPPORTED.** Flare measures per-frame + latent concat as
**worse** than frame stacking inside a control loop; Genie measures a multi-frame encoder
**beating** spatial-only at matched params with the predictor held fixed (FVD 114.5 → **81.4**).
**The prescription "drop the stack" is refuted as a standalone move.** §6.6–6.7 replace it with
the claim both sides support: *9 undifferentiated channels is a measurably poor way to present
motion (Karpathy: Early Fusion 57.7 < Single-Frame 59.3; two-stream: RGB stack +4.1 vs flow
+28.7) — the fix is to make motion EXPLICIT, not to remove it.*

⭐ **The separate §6.2 finding — the 67 % target–context overlap — is UNTOUCHED by any of that
literature, because no paper studies it.** It stands as MEASURED (ours, from source) and
UNTESTED.

---

## 10. DELIVERABLE MANIFEST

| artifact | where it lives | only one place? |
|---|---|---|
| **This report** | `repo: TanitAD Research Lab/Architecture & Inference/Research/2026-08-30-data-efficiency-thesis/RESULT.md` — **staged** | no (also in the session scratchpad) |
| **33 newly banked primary PDFs** | `repo: TanitAD Research Lab/Library/papers/` + `library.json` + regenerated `LIBRARY.md` — **staged** | no (arXiv) |
| **Anchor verification quotes** | `repo: …/2026-08-30-data-efficiency-thesis/raw/anchor_verification.md` — **staged** | no |
| **Knowledge-base entries** | `repo: TanitAD Research Lab/Architecture & Inference/Research/KNOWLEDGE_BASE.md`, `…/Data Engineering/Research/KNOWLEDGE_BASE.md` — **staged** | no |
| PDF text extractions (working files) | scratchpad only — **deliberately not staged** (regenerable from the banked PDFs) | yes, and intentionally |

## 11. ⛔ ESCALATIONS — decisions and integrations that will not happen on their own

1. ⭐⭐ **E-ARCH-OVERLAP needs a SPEC.md and a slot.** The 67 % target–context overlap (§6.2) is
   MEASURED in the live trainer's own code path. **Until it is tested, every v6/v7
   latent-prediction number was scored against a possibly-degenerate objective.** It is free
   (one tiny-ladder pass, no new data) and it gates the interpretation of every data experiment.
   **This is not a "please merge" note — it is a decision the Master Mind must take or refuse.**
2. ⭐ **E-DATA-2×2 (§8.4) is the PI's question, the literature gap, and the D-SPEED-GATE test in
   one 3-arm run.** It needs the DataFlyWheel to produce the WIDE re-selection (re-admitting the
   >14 m/s band inside B1) — **that is a cross-team dependency and it is the one item here that
   cannot be done by the Research Lab alone.**
3. **Twelve corrections in §9 must land in `GOALS_AND_CLAIMS.md` D-DATA-EFFICIENCY and
   `RETRACTION_LOG.md` in the same turn as this report** — five of them are numbers the
   programme is actively quoting.
4. ⚠️ **`tools/kb_add.py` has a SILENT-FAILURE mode**: MEASURED today, it exited **0 with no
   output and banked nothing** for 2 of 5 IDs in a loop; both succeeded on a bare retry. ⛔ It
   reports success by printing, so an empty stdout is the only signal, and a `for` loop that does
   not check will silently drop papers. **`--verify` does not catch it** (a paper never banked has
   nothing to re-hash). ⇒ **Every banking loop must grep stdout for `banked` and retry**, which is
   what this pass did. Worth a guard + regression test in the tool.

## 12. EVIDENCE-CLASS LEDGER

- **MEASURED (ours, from source, file:line given):** §6.1 the entire encoder/window/target chain;
  §6.2 the overlap arithmetic; §11.4 the kb_add silent failure.
- **PUBLISHED-PRIMARY (banked PDF opened by me, page-counted, quoted):** every number in §2, and
  §3.5, §4.1, §4.3, §5.1, §5.6, §7.2's ONE-Drive row, and DINO-WM Table 6 in §6.3.
- **PUBLISHED-PRIMARY (banked; read by a parallel stream from arXiv HTML/ar5iv, NOT re-opened by
  me from the PDF):** §3.1–3.4, §4.2, §4.4, §5.5, §6.4, §6.6–6.8, §7.1, §7.3. ⚠️ **These are
  banked and re-checkable, but I did not personally re-read the PDF. Before any of them enters
  `MODEL_REGISTRY.md` or the paper, re-open the banked PDF.**
- **PUBLISHED-SECONDARY (inadmissible for the registry, flagged in place):** GAIA-2's encoder
  input shape (§6.4); Wayve LINGO scaling (blog-only — **no primary exists**, §5.5).
- **HYPOTHESIS:** §3.3's transfer-vs-co-training reconciliation; §8.3's speed-gate prediction;
  §6.2's claim that the overlap *causes* the content/prediction dissociation (the *overlap* is
  measured; its *effect* is not).
- ⛔ **No number in this report has been placed beside a number from a different benchmark.**
  Every table names its metric and split; PDMS, DS, EPDMS, ADE, FDE, mAP and RFS are kept apart.
- ⛔ **No exponent in this report may decide a GPU-day.** `−0.396`/`−0.417` (NVIDIA) are fit
  against **hours** with **no R² published**; `−0.188` (ONE-Drive) against **number of
  demonstrations** on a different normalisation (r = −0.963 ⇒ R² = 0.927); Waymo's 0.63/0.44 carry
  the authors' own warning that *"a parabolic form fits our observed data better."* **They may
  never be placed side by side.**
