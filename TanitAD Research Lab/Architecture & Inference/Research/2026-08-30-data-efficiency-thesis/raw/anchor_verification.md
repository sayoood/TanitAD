# raw/ — anchor verification quotes, read from the banked PDFs 2026-08-30

`Every quote below was extracted from the PDF in TanitAD Research Lab/Library/papers/ by`
`pypdf text extraction. Page counts confirm none is truncated (the 0-page trap).`
`This file is the quotable layer for RESULT.md — a claim not traceable here is not PRIMARY.`

| key | pages | bytes | status |
|---|---|---|---|
| `2510.26125` WOD-E2E | 15 | 8,306,889 | ✅ matches recorded bytes |
| `2411.04983` DINO-WM | 21 | 6,891,479 | ✅ |
| `1904.08980` Codevilla | 17 | 10,810,421 | ✅ |
| `2506.08228` Waymo scaling | 31 | 6,685,014 | ✅ |
| `2306.07957` TransFuser++ | 18 | 8,744,663 | ✅ |
| `2510.12796` DriveVLA-W0 | 22 | 3,330,317 | ✅ |
| `2504.04338` NVIDIA data scaling | 15 | ~1.83 MB | ✅ |
| `2604.08366` MOSAIC | 21 | ~2.24 MB | ✅ |
| `2412.02689` ONE-Drive | 8 | — | ✅ |
| `2403.02877` ActiveAD | 22 | ~1.77 MB | ✅ |

---

## `2510.26125` — WOD-E2E

**§3.3.2 Case Study (the 3,333× ratio, and the ONLY defensible form of it):**
> "we conducted a case study on a recent set of driving logs that includes a total of 6,391,012
> miles. After applying our automated mining strategy, we found that only 6,888 miles (0.1%) of
> the data fit our criteria for long-tail scenarios."
> "we perform a subsequent round of human filtering. This manual review process, which has a
> conversion rate of 30%, further refined the mined data … This final filtering step reduced the
> overall portion of long-tail scenarios to an even rarer 0.03%"

**§3.1 (the internal inconsistency):** "This dataset contains 4,021 driving segments … Each
segment is 20-second long … 2,037 segments for training, 479 segments for validation, and the
rest 1,505 segments for testing." — against the abstract's "approximately 12 hours".
4,021 × 20 s = 22.34 h.

**§4.4 Q2 (ADE does not track the preference metric):**
> "No, a better ADE does not guarantee a better RFS." — over 19 submissions, "only a mild
> positive correlation".

**§4.4 Q1 table (the architecture × data-source interaction, and why it is unusable):**
Swin-Trajectory 7.543 (1 dataset) · DiffusionLTF 7.717 (4) · UniPlan 7.779 (2) ·
Baseline/Gemini-Nano 7.528 (1) · AutoVLA 7.556 (3) · HMVLM 7.736 (1) · Poutine 7.986 (2).
⇒ **not monotone in the number of data sources**; models also differ in backbone, CoT style and
RL. Observational across leaderboard entries, not a controlled ablation.

---

## `2411.04983` — DINO-WM

**§3.1 (the encoder sees ONE frame):**
> "Observation model: z_t ~ enc_θ(z_t | o_t) — Transition model: z_{t+1} ~ p_θ(z_{t+1} |
> z_{t−H:t}, a_{t−H:t})"

**App. A.4.1 Table 5 (PushT, planning success rate):**

| n | SR | SSIM | LPIPS |
|---|---|---|---|
| 200 | 0.08 | 0.949 | 0.056 |
| 1,000 | 0.48 | 0.973 | 0.013 |
| 5,000 | 0.72 | 0.981 | 0.007 |
| 10,000 | 0.88 | 0.984 | 0.006 |
| 18,500 | 0.92 | 0.987 | 0.005 |

⇒ 92.5× data · SSIM **+4.0 %** · LPIPS **11.2× better** · SR **×11.5**.
⭐ LPIPS tracks control; SSIM is saturated. The elbow is at n≈5,000.

**App. A.4.2 Table 6 (the leakage ablation — the §6.3 analogue):**

| h | w/o mask | with mask |
|---|---|---|
| 1 | 0.76 | 0.76 |
| 2 | 0.36 | 0.88 |
| 3 | **0.08** | **0.92** |

> "we see a rapid drop in the w/o mask case, since the model can cheat during training by
> attending to future observations."
> "longer history could better capture dynamics information like velocity, acceleration, and
> object momentum."

---

## `1904.08980` — Codevilla, "Exploring the Limitations of Behavior Cloning"

**§5 Driving Dataset Biases:**
> "we compare models trained with 2, 10, 50 and 100 hours of demonstrations … Our best results on
> most of the scenarios were obtained by using only 10 hours of training data, in particular on
> the 'Dense Traffic' tasks and novel conditions such as New Weather and New Town."
> "the risk of overfitting to data that lacks diversity. This is here exacerbated by the limited
> spatial extent and visual variety of our environment"

**Fig. 4 caption:** "the percentage of episodes failed due to that inertia problem increases with
the amount of data used for training."

**§3.2 (the mechanism):**
> "When the ego vehicle is stopped (e.g., at a red traffic light), the probability it stays static
> is indeed overwhelming in the training data. This creates a spurious correlation between low
> speed and no acceleration"

⛔ **Table 3 — the caveat the programme was not carrying** (12 seeds):
CILRS Empty **23 %** · Regular **26 %** · Dense **42 %**; with ImageNet init 4 % / 12 % / 38 %.
> "the success rate can change by up to 42% for tasks with dynamic objects."
The data-scaling comparison used **4** seeds.

---

## `2506.08228` — Waymo, Scaling Laws of Motion Forecasting and Planning

**Table 1:** run segments 59.8 M · agents 373 B · **hours of driving 447 thousand** · miles
5.6 million · training examples 541 million.

**§4.1:** "N_opt ∝ C^0.63 and D_opt ∝ C^0.44. This indicates that for optimal training compute
efficiency, the optimal model size should grow ∼1.5 times as fast as the number of training
examples." ⇒ ⛔ **parameters grow FASTER than data.**

**§1:** "at the same training compute budget, an optimal LLM is ∼50 times larger than an optimal
motion forecasting model … potentially suggesting the importance of collecting more data, or
perhaps improving the training data sampling techniques for this domain."

**§5.2 (the closed-loop counter-evidence — real):**
> "closed loop performance also follows a similar scaling trend, with the number of failures η
> decreasing as a power law when scaling pretraining compute. This suggests that open loop
> performance can serve as a good proxy for closed loop."
Setup: 30 s sim at 10 Hz, mixed logged-playback + imitation-learned reactive agents, R=128
rollouts, first 0.1 s executed. η = scenarios where the policy progresses significantly more or
less than manual driving, or collides.

**Their own hedges:** "establishing a power-law relationship would require a study spanning many
more orders of magnitude of compute and more rigorous statistical methodology"; "a parabolic form
fits our observed data better"; "We hypothesize that having a simple architecture with minimal
inductive biases is an important piece to achieve this result."

---

## `2306.07957` — TransFuser++ (Hidden Biases of End-to-End Driving Models)

**§3.4 (the frozen-encoder result):**
> "We also experiment with freezing the pre-trained backbone and only training the transformer
> decoder and its heads in the second stage. This leads to a drop of 10 DS, indicating that
> end-to-end optimization is important."

**§3.4 dataset scale:** "We start with 185k training samples … and scale it up by re-running the
training routes 3 times with different traffic (555k frames) … a clear improvement of 6 DS via
scaling … this improvement comes at the cost of 3× longer training."
⭐ **the extra data is the SAME ROUTES with different traffic — volume without scene diversity.**

**Table 5** (validation towns): 185k **54 ± 1** → 555k **60 ± 6**. ⚠️ Δ = 1σ of the larger arm.

⭐ **Table 13 — Longest6 ablations, std over 3 training and 3 evaluation runs, each row baselined
on the previous:**

| step | DS | RC | Veh ↓ | Stat ↓ |
|---|---|---|---|---|
| TransFuser (reproduced) | 40 ± 3 | 82 ± 2 | 1.17 | 0.57 |
| + transformer decoder | **57 ± 3** | 90 ± 3 | 0.93 | 0.19 |
| + data augmentation | **66 ± 4** | 94 ± 2 | 0.64 | 0.07 |
| + disentangled | 64 ± 4 | 96 ± 1 | 0.88 | 0.02 |
| + two stage | 67 ± 2 | 96 ± 1 | 0.82 | 0.01 |
| **+ 3x data** | **69 ± 1** | 97 ± 1 | 0.79 | 0.00 |
| TF++ WP | 70 ± 4 | 94 ± 2 | 0.66 | 0.01 |
| + ensemble | 73 | 97 | 0.56 | 0.01 |

---

## `2510.12796` — DriveVLA-W0

**Table 3** (in-house corpus; ADE m ↓ / collision % ↓ — open-loop by construction):

| model | 70k | 700k | 70M |
|---|---|---|---|
| TransFuser-50M | 2.5893 / 0.0894 | 1.7464 / 0.0563 | 1.2627 / 0.0472 |
| TransFuser-7B | 2.5757 / 0.0839 | 2.1391 / 0.0710 | 1.2244 / 0.0539 |
| VLA (VQ) baseline | 2.8520 / 0.0982 | 1.5424 / 0.0565 | 1.4829 / 0.0488 |
| **+ World Model** | 2.7482 (+3.6 %) | **1.5985 (−3.6 %, WORSE)** | **1.0563 (+28.8 %)** |
| VLA (ViT) baseline | 3.1524 / 0.0950 | 1.4202 / 0.0462 | 1.1051 / 0.0359 |
| + World Model | 2.5268 (+19.9 %) | 1.3436 (+5.4 %) | 1.0640 (+3.7 %) |

⇒ baseline 700k→70M = **100× data for −3.9 % ADE** (a plateau); +WM = **−33.9 %** (no plateau).
⛔ **the WM advantage is +3.6 % at 70k and NEGATIVE at 700k.**

**Table 4 — the decoder reversal** (all three initialised from an identical pretrained VLA;
⛔ the two columns are DIFFERENT BENCHMARKS and may not be compared across):

| action expert | NAVSIM 103k — **PDMS ↑** | in-house 70M — **ADE ↓** |
|---|---|---|
| Query-based | **88.4** (best) | 1.1248 (worst) |
| Flow matching | 87.2 | 1.0362 |
| Autoregressive | 85.3 (worst) | **1.0069** (best) |

---

## `2504.04338` — NVIDIA, Data Scaling Laws for End-to-End Autonomous Driving

**§6.3 (the closed-loop plateau):**
> "Despite observing consistent open-loop performance gains at larger data scales, the MDBF
> results improved only up to the 256-hour mark, beyond which they plateaued around 1 km."
> "neither data augmentation nor capacity scaling alleviated the plateau in closed-loop
> performance."
⚠️ **Scope, stated by the authors:** "two highway scenarios with minimal traffic interactions,
where the primary challenge is lane-keeping", in NVIDIA DRIVE Sim.

**§6.2.1 (extrapolation):** "a 1% improvement in FDE would require approximately 4,000 hours of
additional driving data, while a 3% improvement demands around 29,000 hours. A 5% improvement,
furthermore, would necessitate an extensive 273,000 hours."
Table 3 (M2 fit on normalized metrics): FDE β 1.358, c **−0.396**, ε∞ 0.543 · ADE 1.464,
**−0.417**, 0.520 · MR2m 1.925, −0.399, 0.352. ⛔ **no R² published.**

**§6.2.2 (capacity substitutes for data):** ResNet-18 (11.2 M backbone / 18.4 M total) → ResNet-50
(24.3 M / 31.5 M): "the ResNet-50 model achieves the same FDE with only around 3000 hours of
data—a 63% reduction compared to the original 8192 hours required by ResNet-18. Similarly, lane
keeping requires 2970 hours (64% reduction), lane changing 1815 hours (78% reduction), and
turning 2597 hours (68% reduction)."

---

## `2604.08366` — MOSAIC, Scaling-Aware Data Selection

**Table 1 — validation EPDMS ↑ and BRMR ↓ (budget ratio to match random), 3 seeds:**

| budget (OpenScene / Navtrain) | method | OpenScene EPDMS | BRMR | Navtrain EPDMS | BRMR |
|---|---|---|---|---|---|
| 250 / 100 | Random | 72.84 ± 1.14 | 1.00 | 84.66 ± 0.60 | 1.00 |
| | Uncertainty | 70.78 ± 0.59 | **14.58** | 84.50 ± 0.48 | 1.47 |
| | Coreset | 76.26 ± 0.48 | 0.20 | 85.29 ± 0.47 | 0.53 |
| | MOSAIC | **77.38 ± 1.58** | **0.15** | **86.29 ± 0.43** | 0.30 |
| 1000 / 400 | Random | 75.84 ± 0.90 | 1.00 | 86.69 ± 0.20 | 1.00 |
| | Uncertainty | 71.12 ± 0.38 | 8.00 | 86.07 ± 0.75 | 2.00 |
| | Coreset | 80.46 ± 0.02 | 0.22 | 87.09 ± 0.29 | 0.79 |
| | MOSAIC | **81.68 ± 0.52** | 0.18 | **88.21 ± 0.03** | 0.38 |
| 4000 / 1600 | Random | 80.38 ± 0.55 | 1.00 | 88.62 ± 0.22 | 1.00 |
| | **Uncertainty** | **73.46 ± 0.19** | **2.00** | 87.75 ± 0.37 | 1.36 |
| | Coreset | 83.63 ± 0.36 | 0.25 | 89.30 ± 0.19 | 0.58 |
| | Chameleon | 82.92 ± 0.13 | 0.39 | 89.50 ± 0.20 | 0.62 |
| | MOSAIC | **84.25 ± 0.14** | **0.18** | **90.18 ± 0.25** | 0.37 |

⚠️ ± values are **seed** SDs, not episode-cluster. The ~1-point MOSAIC-over-Coreset margin is
marginal; the 6.9-point Uncertainty-vs-Random gap is many SDs.

---

## `2412.02689` — ONE-Drive, Data Scaling Laws for IL-based E2E AD

**Eq. 1:** `Y = 0.6833 · X^(−0.188), r = −0.963` (normalized ADE vs number of demonstrations,
10k → 4M, ~30,000 h). ⇒ R² = 0.927. ⛔ open-loop.

> "power-law … but this is not the case in closed-loop evaluation."

**§V-B (targeted tail-filling, total held at ~2M demonstrations):**

| scenario type | samples | metric (↓) |
|---|---|---|
| DOWN SUBROAD LC | 4,972 → 7,218 (+45.1 %) → 21,836 (+339.2 %) | 0.788 → 0.711 (−9.7 %) → 0.529 (−32.9 %) |
| SUBROAD → MAINROAD HIGHWAY | 643 → 1,485 (+130.9 %) → 2,569 (+299.5 %) | 0.786 → 0.653 (−16.9 %) → 0.593 (−22.8 %) |

⛔ **Gains reported ONLY on the upweighted slices; the other 21 scenario types are unreported.**

---

## `2403.02877` — ActiveAD

> "we observe that traditional Active Learning methods perform poorly, lacking any significant
> advantage over random selection."

nuScenes val, **open-loop L2 / collision %** @30 %: Random 0.78 / 0.37 · Coreset 0.73 / 0.54 ·
VAAL 0.81 / 0.35 · ActiveFT 0.78 / 0.39 · **ActiveAD 0.68 / 0.21** · VAD-Tiny @100 % 0.70 / 0.25.
⚠️ **nuScenes open-loop L2 is the metric AD-MLP (`2312.03031`) showed is dominated by ego status
— T0-class under our own doctrine.**

---

## OUR OWN SOURCE — MEASURED 2026-08-30 (the §6 chain)

| fact | file:line |
|---|---|
| `stack_frames`: output j = frames {j, j+1, j+2} | `stack/tanitad/data/comma2k19.py:640` |
| `n_stack = 3` default, 9 channels | `comma2k19.py:633`, `physicalai.py:679,737` |
| episodes resampled to `TARGET_HZ` = 10 Hz ⇒ stack spans 200 ms | `physicalai.py:717-718` |
| window slicing `ep.frames[t : t+w]`, **stride 1** | `stack/tanitad/data/_contract.py:129-135` |
| target `ep.frames[t+w : t+w+max_horizon]` | `_contract.py:132-133` |
| `window = 6` | `stack/tanitad/models/v6.py:4006` |
| live trainer builds `FlagshipWindowDataset` | `stack/scripts/train_v6_staged.py:5090-5093` |
| `FlagshipWindowDataset` ← `FailLoudWindowDataset` ← `EpisodeWindowDataset` | `stack/scripts/train_flagship4b.py:106` |
| latent target = encoder on the FULL 9-ch future stack | `train_v6_staged.py:5665-5668` |
| k=1 always included when latent objectives are on | `train_v6_staged.py:5641` |
| O14 pixel target takes the **newest frame only** (`x[:, -3:]`) ⇒ NOT affected | `train_v6_staged.py:1461` |
| `--newest-frame-only` flag exists (the A2 arm) | `train_v6_staged.py:1450` |

⇒ **z_{t+5} = enc({t+5,t+6,t+7}); z_{t+6} = enc({t+6,t+7,t+8}) ⇒ 2 of 3 target frames already
observed. Overlap: k=1 → 67 %, k=2 → 33 %, k≥3 → 0 %.**
