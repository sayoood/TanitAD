# PRE-REGISTRATION — R3 design research (action-channel oscillation)

**Written:** 2026-07-27 (Europe/Berlin; the dev box clock reads 2026-07-26 23:00 UTC).
**Author:** literature-research agent, CPU/web only. ⛔ **No pod was contacted.** pod1 trains, pod2
runs an owed-controls job, pod3 builds a classifier, the eval pod runs trafficsim.
**Written BEFORE the literature synthesis.** What I had already read when writing this: our own
measured artifacts (`2026-07-26-tblind-rung1/`, `2026-07-26-tblind-ladder/`), our own training and
eval source (`stack/tanitad/train/flagship_losses.py`, `taniteval/taniteval/blindimag.py`), and
**two** web searches (CAPS, ChauffeurNet — titles/abstracts only). No conclusion below was formed
from those two.

---

## 1. The two admissible outcomes, committed in advance

I commit, before reading the literature, that **both of the following are acceptable deliverables and
I have no stake in which one the evidence produces**:

| outcome | what it means | what would produce it |
|---|---|---|
| **A — DESIGN** | a concrete objective/regulariser, precise enough to implement, with a bar it must clear and the cheap experiment that could refute it first | the literature shows a method that attacks **the pathology we actually measured** and there is measured headroom above what a free filter already reaches |
| **B — DO NOT RUN** | an explicit recommendation to cancel or defer R3, with the reason | the literature's achievable gain over a well-tuned filter is small, **or** the metric R3 is aimed at is already saturated by the free filter, **or** the method families all attack a pathology we do not have |
| **C — RE-AIM** | R3 as named is refused, but a *different* training-time objective is designed in its place | the pathology is real and trainable but "scheduled sampling on the action channel" is the wrong instrument for it |

⚠️ **Outcome C is declared in advance so that a refusal of R3-as-named cannot be dressed up as a
design, and a re-aimed design cannot be dressed up as a confirmation of the original plan.**

## 2. The bar, fixed in advance

`MEASURED` (`…/2026-07-26-tblind-rung1/artifacts/rung1_blend_curve.json`,
`rung1_interventions.json`), v1 = `flagship4b-speedjerk-30k` @ 29999, calibrated `str` readout,
599 windows / 596 episode clusters, paired episode-cluster bootstrap B=2000 seed 0:

| arm | `T_blind` | `de@2s` | `ade_0_2s` | beats CV | `T_useful@1m` |
|---|---:|---:|---:|---|---:|
| own kinematic (baseline) | 25 (2.5 s) | 1.8165 | 0.8710 | 0/185 | 1.4 s |
| **best free filter** (`blend0.75`) | **116 (11.6 s)** | 0.6842 | 0.3437 | 81/185 | 2.3 s |
| hold-last (α=1, no policy) | 115 (11.5 s) | 0.6718 | 0.3351 | 83/185 | 2.3 s |
| `gtkin` (same inverse, TRUE motion) ⚠️ privileged | 185 ⚠️ saturated | 0.4361 | 0.2552 | 179/185 | 3.0 s |

**The bar R3 must clear, stated now:** deployable `T_blind` **≥ 11.6 s at full command authority
(α = 0)** — i.e. the trained model's *own, unfiltered* action must be worth what a 75 %-damped action
is worth today. Any lower and the 59-hour run has bought nothing a one-line filter did not.

⚠️ **Second bar, declared now because it is the one that can actually be won:** if `T_blind` turns out
to be **saturated** by the filter (the α=1 endpoint is the measured ceiling, so it may be), the
admissible bar moves to the **comparator-free** statistics, where the filter family provably tops out:
`beats-CV ≤ 83/185` and `T_useful@1m ≤ 2.3 s` for **every** α, against `gtkin`'s 179/185 and 3.0 s.

## 3. What would make me recommend NOT running R3 — stated before I know

R3 is refused if **any** of these holds:

* **S1 — metric saturation.** The free filter already recovers ≥ 100 % of the metric's measured
  ceiling (`frac_of_ceiling_recovered = 1.011` is already on the record), **and** no comparator-free
  statistic with remaining headroom is attributable to the action channel.
* **S2 — wrong pathology.** The method families in the literature that R3 belongs to (scheduled
  sampling / student forcing / DAgger-line) are shown to attack **covariate shift on the state
  distribution (drift)**, while our measured failure is a **saturating oscillation in the command**,
  and no cited work demonstrates the former fixing the latter.
* **S3 — the fix is cheaper elsewhere.** A ≈ 0-marginal-cost objective, or a change to our own
  action-generation map (which is *our code*, not the model), is shown to attack the same mechanism.
* **S4 — the gradient cannot reach the pathology.** If the training-time path R3 would create is
  shown to be gradient-dead exactly where the pathology lives.
* **S5 — the mechanism it would install is a capability loss.** If the only way a student-forcing
  objective can reduce the penalty is by teaching the predictor to **ignore its action channel**,
  the metric improves and action-conditioning — the entire point of a world model for planning —
  degrades.

**Conversely, R3 is recommended** only if a cited method demonstrably (not assertedly) reduces
**command amplitude / saturation** without costing task accuracy, AND there is measured headroom on a
statistic the filter cannot reach, AND a pre-run falsifier exists that costs hours not days.

## 4. Evidence discipline, fixed in advance

* Every claim gets `PUBLISHED (cited)` / `MEASURED (ours + path)` / `INHERITED` / `ESTIMATED` /
  `HYPOTHESIS` **and** a tier `PROVISIONAL` / `CONFIRMED` / `DECISION-GRADE`.
* **A published method is not admissible as a fix for our pathology until it is checked against the
  specific pathology we measured.** I will state, for every cited method, *which* pathology it
  attacks — oscillation, drift, or both — and I will name the ones that are aimed at the wrong one
  **even when they are the famous ones**.
* Where the literature disagrees I give **both** sides. Where a paper *asserts* rather than
  *demonstrates*, I say so.
* ⚠️ **The anti-pattern I am explicitly guarding against** (this program logged a retraction for it
  today): a plausible mechanism becoming a finding. Any mechanism I propose must come with the
  counterfactual that would break it, and if I cannot state one, the mechanism is labelled
  HYPOTHESIS and does not enter the recommendation.

## 5. The falsifier standard for whatever I recommend

Whatever comes out of §1, it must ship with a **pre-run falsifier that costs hours, not the 59-hour
run** — ideally eval-only on the existing v1 checkpoint, using infrastructure that already exists
(`taniteval/taniteval/blindimag.py`'s `apply_action_filter` and the 32 MB per-window compaction that
recomputes any bar with **no GPU**). If I cannot name such an experiment, I must say so and the
recommendation drops to HYPOTHESIS.

## 6. What I am NOT going to do

* ⛔ Not touch a pod. Not run an experiment. This is reading and synthesis.
* ⛔ Not re-derive our measured numbers — they are quoted from the raw JSON with paths.
* ⛔ Not recommend an architecture change; that is not what was asked.
* ⛔ Not commit or push. `git add` only.
