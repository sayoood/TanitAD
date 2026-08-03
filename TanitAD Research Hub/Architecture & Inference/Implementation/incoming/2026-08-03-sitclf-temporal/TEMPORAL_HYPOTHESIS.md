# Is the situation classifier limited by MISSING TEMPORAL CONTENT?

**Date** 2026-08-03 · **Stream** sitclf temporal · **Substrate** dev box (RTX 4060), **0 pod GPU-h** —
no pod was touched. **Pre-registration** `./PRE_REGISTRATION.md`, written and staged **before** any
held-out number from this run was read. **Run directory** `TanitAD Research Hub/Architecture &
Inference/Implementation/incoming/2026-08-03-sitclf-temporal/`.

⛔ PI ruling 2026-08-03 honoured throughout: **labels may use ego; INFERENCE IS VISION-ONLY.** Every
deployable arm below reads the frozen v1 camera latents and nothing else. The two `CPOS_*` arms are
**privileged power controls, not deployables**, and are excluded from every ranking.

---

## 0. The headline

*(filled from `results_temporal.json` — see §4)*

---

## 1. ⭐ THE BRIEF'S PREMISE IS FALSE AS STATED — measured at two probes

The hypothesis was handed to me as: *our models keep only the LAST frame's feature map, so the
cross-attended tokens are single-instant; a single RGB frame carries no relative velocity, no closing
rate, no TTC; that would explain a capacity curve peaking at 129 parameters.*

Two independent facts, both MEASURED here, break that chain **before** any classifier is fitted.

**(a) The per-frame latent is not a single frame.** MEASURED — a real episode-cache tensor is
`frames_u8 [199, 9, 256, 256]` (probe 1: `C:/Users/Admin/tanitad-data/physicalai/_epcache/
physicalai-val-bb543bdf7836/ep_*.pt`), and `stack/tanitad/config.py:17` reads
`in_channels … 9 = camera (3-frame stack, D-015)` with `config.py:360` *"3 RGB frames at 100 ms
spacing channel-stacked"* (probe 2). **Every v1 latent already integrates 0.2 s of motion.**

**(b) The head does not read one latent.** `stack/tanitad/eval/sitclf.py:causal_window` stacks
`WIN = 8` latents at offsets −7..0. **0.7 s of latent history is already in the design matrix**, and
combined with (a) the deployed head already sees **0.9 s of motion-bearing evidence**.

⇒ The situation classifier is **not motion-blind by construction**. Whatever is true of REF-C's
single-instant cross-attention (`stack/tanitad/refs/refc.py:1112-1117`, a sibling stream's read,
INHERITED and not re-verified here) does **not** transfer to this classifier, which has a different
input path. The real questions are narrower, and they are what this study tests:

| id | mechanism | verdict |
|---|---|---|
| **H-T1** | the 0.9 s window is too short for a ~3 s manoeuvre | §4 |
| **H-T2** | rank-16 appearance PCA truncates the motion subspace away | **§2 — REFUTED, mechanistically** |
| **H-T3** | the stack spans the differences but the optimiser cannot find them | §4 + §3 |

---

## 2. ⭐ H-T2 IS REFUTED WITHOUT A SINGLE LABEL — the appearance basis keeps the motion

H-T2 is a claim about **subspaces**, so it is checkable with no classifier, no label and no AP: how
much of the frame-to-frame difference survives projection onto the appearance basis the deployed arm
actually uses? Artifact `results_subspace.json`, script `subspace_diag.py`, fold-0 fit rows
(39,793), same seeds as the main run.

| Δ lag | rank | Δ-variance kept by the **appearance** basis | by the purpose-built **motion** basis | motion advantage | mean principal cos |
|---|---:|---:|---:|---:|---:|
| 0.1 s | 16 | **0.8808** | 0.8952 | **+0.0144** | 0.9803 |
| 0.1 s | 64 | 0.9908 | 0.9931 | +0.0023 | 0.8815 |
| 0.1 s | 256 | 1.0000 | 1.0000 | +0.0000 | 0.9507 |
| 0.3 s | 16 | **0.8936** | 0.9081 | **+0.0145** | 0.9568 |
| 0.3 s | 64 | 0.9952 | 0.9960 | +0.0008 | 0.9102 |
| 0.3 s | 256 | 1.0000 | 1.0000 | +0.0000 | 0.9590 |

**At the deployed rank 16 the appearance basis already retains 88.1 % of the frame-to-frame
difference variance**, against 89.5 % for a basis fitted directly on that difference — an advantage
of **1.4 percentage points** — and the two subspaces have a mean principal cosine of **0.9803**, i.e.
they are very nearly the same subspace. By rank 64 the gap is 0.2 pp and by rank 256 it is zero.

⇒ **There is no discarded motion subspace to recover.** A motion-basis arm has essentially nothing
to find that the appearance basis has not already kept, and H-T2's mechanism does not exist in this
substrate. This is measured, label-free and independent of every modelling choice downstream.

### ⚠️ A defect in this very diagnostic, caught before publication

The first version of the calculation measured the difference's variance **about the appearance
mean** instead of about its own mean. Δ has mean ≈ 0, so subtracting a large appearance mean makes
the total dominated by ‖μ_appearance‖² — a constant offset the appearance basis reproduces almost
perfectly by construction. It reported the appearance basis holding **0.9520** of the Δ variance at
rank 16, which measured the centring and not the subspace.

⭐ **The correction moved the number AGAINST the conclusion** (0.9520 → 0.8808, making the motion
basis look relatively better) **and the conclusion survived it.** The corrected figures are the ones
above.

⛔ **`results_temporal.json` → `controls.H_T2_SUBSPACE_DIAGNOSTIC` contains the SUPERSEDED, wrongly
centred block**, because the hour-long fit had already passed that stage when the defect was found.
It is left untouched rather than edited after the fact; `results_subspace.json` carries a
`_supersedes` pointer, and **`results_subspace.json` is the quotable artifact**. `run_temporal.py`
has been fixed so a re-run is correct.

---

## 3. What the controls establish before any arm is quoted

*(filled from `results_temporal.json` — see §4)*

---

## 4. The ladder

*(filled from `results_temporal.json` via `render_tables.py` — see `tables.md`)*

---

## 5. The label pipeline's causality break

*(see §5 below, filled after the run)*

---

## 6. Manifest and status

*(filled at the end)*
