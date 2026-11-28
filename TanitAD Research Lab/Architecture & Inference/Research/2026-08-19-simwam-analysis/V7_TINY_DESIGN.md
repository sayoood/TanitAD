# v7-tiny — a fast validation rig that can PROVE the design works before v7 spends a GPU-week

`PROPOSAL, 2026-08-22.` PI: *"Im tired of days of training to discover bugs …
validate the design at very small scale and be very fast even if the results are
not excellent but definitely showing that we solved all our problems."*

---

## 1. ⛔ The lesson that must shape this, from today

**E-DENSE-1 was already a tiny rig, and it taught nothing** — not because it was
small, but because it was small in the **wrong dimensions**: 6.4 M parameters
against a *full-complexity* 130-clip driving corpus. Its **positive control
failed** (a student distilling DINOv3 reached 0.0991 against a 0.0909 floor), so
every other arm became uninterpretable. It cost hours and returned "the platform
cannot test this".

⇒ **UNIFORM SHRINKING PRESERVES THE FAILURE.** Halving everything keeps the same
signal-to-capacity ratio and reproduces the same null. v7-tiny has to shrink the
**task and the model together**, holding the ratios that actually decide
learnability.

## 2. ⭐ The gate — what v7-tiny must DEMONSTRATE, committed before it is built

A rig that can only return "inconclusive" is worthless here. v7-tiny passes only
if **all three** fire, and each has a measured floor already in hand:

| # | criterion | metric | pass |
|---|---|---|---|
| **G1** | the platform can learn at all | positive control (DINOv3 distillation or supervised probe) | ⭐ **must clearly beat the raw-patch floor** — E-DENSE-1 failed exactly here |
| **G2** | the predictor models dynamics | `explained_movement = 1 − ‖ẑ−z⁺‖²/‖z−z⁺‖²` (`e_rescue.py`) | **> 0**, i.e. beats HOLD. v6 measures **580× worse** |
| **G3** | the latent carries the world | E-DETECT-1 grid AP, episode-disjoint, cluster bootstrap | **above `prior` AND above `pixel`**, paired |

⚠️ **G2 is the one nobody has ever passed**, and it is the cheapest. It needs no
labels — only the model's own latents — so it should run FIRST and gate the rest.

## 3. What to shrink, and what must NOT — the ratio argument

### 3.1 ⛔ DO NOT shrink: tokens-per-vehicle

MEASURED today at 256×640: a vehicle spans **1.83 tokens at 10 m, 1.22 at 30 m,
0.61 at 60 m**. Halving resolution halves that — at 128×320 a vehicle is
**sub-token beyond ~18 m**, and no encoder can localise what does not occupy a
token. **Resolution and target range must move TOGETHER.**

⇒ If resolution halves, the BEV target must shrink from 0–60 m to **0–20 m** or
the task becomes provably impossible and G3 can never fire.

### 3.2 ⭐ DO shrink hard: scene diversity

This is the axis nobody has tuned, and the literature points straight at it.
`LeWM` (banked) reports linear-probe **r = 0.974 / 0.986 / 0.902** on Push-T and
**0.996** on Two-Room — low-diversity control environments — and its own paper
says it does WORST on OGBench-Cube *"due to the higher visual complexity"*.
Driving video is far past that. **Cutting from 130 clips to 10–20 similar clips
(one road type, one time of day) moves toward the regime where the recipe is
known to work**, which is the whole point of a validation rig.

### 3.3 DO shrink: horizon, depth, steps

| knob | v6F | v7-tiny | why |
|---|---|---|---|
| horizon | 6.0 s / 60 ticks | **1.5 s / 15 ticks** | O5 rollout cost is linear in k; 1.5 s still exposes compounding error |
| params | 336 M | **~20–25 M** | LeWM proves the recipe at **15 M** |
| steps | 30 000 | **4 000** | E-DENSE ran 6 000 in 25 min at 6.4 M |
| hierarchy | 3 levels | **operative only** | tactical/strategic are frozen in S-W anyway — they cannot be validated by a stage that does not train them |

### 3.4 ⛔ DO NOT shrink: the evaluation protocol or the guards

* **episode-disjoint folds + episode-cluster bootstrap** — otherwise the numbers
  are not comparable to anything banked, and `overlapping_holdout_se` biases the
  point estimate.
* **the floors** — `prior`, `pixel`, `oracle`. Today proved a rig without them is
  unreadable.
* **`RESIDUAL_HEAD_INIT_SCALE`** and the audit — the whole reason v7 exists.

## 4. ⭐⭐ The arm that makes the gate trustworthy: a DELIBERATE REGRESSION

v7-tiny must include an arm that **re-introduces the v6 defect on purpose**
(`residual head at default init`). If the gate does not FAIL that arm, the gate
cannot detect the bug it was built to catch, and a pass on the fixed arm means
nothing. This is the same discipline as `oracle` in E-DETECT-1 and
`test_the_guard_can_actually_fail`.

## 5. A concrete first configuration

| | value | rationale |
|---|---|---|
| frames | **192×480, patch 16 → 12×30 = 360 tokens** | keeps ≥1.3 tokens/vehicle at 20 m; 44 % of v6's token count |
| BEV target | **0–30 m, ±12 m**, 10×6 cells of 3 m | matched to the resolution above |
| corpus | **16 clips**, one road type | §3.2 |
| encoder | ViT d=256, depth 6 → ~5 M | |
| predictor | d=512, depth 6, window 6 → ~15 M | |
| horizon | 15 ticks (1.5 s) | |
| steps | 4 000 | |
| **cost** | **≈ 25–40 min/arm on the 4060** | measured: 25 min at 6.4 M / 160 tok / 6 000 steps |

⇒ **A 5-arm ablation runs in ~3 hours**, not days. That is the deliverable: an
overnight answer instead of a week.

## 6. The ablation ladder

| arm | purpose |
|---|---|
| `fixed` | v7's design: down-scaled residual init, dense target |
| ⛔ `regress-init` | **the deliberate regression** — default residual init. The gate MUST fail this |
| `pooled-target` | v6's pooled prediction target, fix retained — isolates the target from the init |
| ⭐ `distill` | positive control (G1) |
| `pixel` / `prior` | floors (G3) |

## 7. What v7-tiny CANNOT do, stated up front

* ⛔ **It is not a driving result.** T0 only. `E_DETECT_1_RESULT.md` §5.1 records
  the programme's only paired driving evidence running the *other* way.
* ⛔ **A pass does not transfer to 336 M / 2 376 episodes.** It licenses
  *proceeding* to scale, not a claim at scale.
* ⚠️ **A low-diversity corpus is a deliberate concession.** If v7-tiny passes on
  16 clips and fails on 130, that is itself the finding — and a far cheaper one
  than discovering it at 30 000 steps.

## 8. Open choices for the PI

1. **Resolution vs range** — 192×480 @ 0–30 m (above), or keep 256×640 @ 0–60 m
   and pay ~2× the compute per arm. The first is faster; the second is directly
   comparable to every banked E-DETECT-1 number.
2. **Corpus** — a curated 16-clip subset of PhysicalAI, or a synthetic/AlpaSim
   set where ground truth is exact and diversity is controllable.
3. **Whether G2 alone is enough to green-light v7.** It is the cheapest, needs no
   labels, and no model in the programme has ever passed it.
