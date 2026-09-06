# PRE-REGISTRATION 3 — the MUTATION RANKING of candidate gate statistics

**status: PRE-REGISTERED. ⛔ WRITTEN AND COMMITTED BEFORE ANY CANDIDATE STATISTIC IS COMPUTED.**
**date:** 2026-09-06 · **owner:** RL gate stream (Arch + Inference FlyWheel) ·
**branch:** `agent/arch-inf-20260803` · **GPU: 0.**

⛔ **`G-REWARD` REMAINS FAILED ON ITS COMMITTED STATISTIC (THE RATE).** Nothing in this
document or its result re-scores it. This file executes clauses **3 and 4** of the Master
Mind's ruling of 2026-09-06:

> **3.** THE DISQUALIFICATION CRITERION, binding on every candidate statistic: it **must
> move** under that perturbation. **Run the mutation FIRST, on every candidate, before
> scoring any of them.** A candidate that fails it is out regardless of its score; a
> candidate that passes earns the right to be scored.
>
> **4.** The replacement is a NEW hypothesis with a NEW ID and its own bar,
> pre-registered, scored on data that did not select it.

---

## 1. Why the RATE was retired, restated so the criterion is not mistaken for a goalpost move

The rate is retired on **INSTRUMENT** grounds, not score grounds. The disqualifying
evidence is a **mutation test that is independent of the data and predates the score**:
`_headway`'s `amin` is **bit-identical** when three of five lead samples move by 5 m
(`test_rl_progress_leadcap.py::test_headway_is_blind_to_this_perturbation…`), and a
statistic that cannot see a 5 m change in the lead's position cannot measure
distance-keeping, whatever number it reports.

⭐ **The criterion generalises the test from the TERM to the STATISTIC.** The rate is a
**SIGN** statistic — `P(hold_v0 ≥ human)` — so it is blind to any change in magnitude that
does not flip a sign, which is the same *family* of defect as an order statistic that is
blind to any change that does not move the extremum. This document tests every candidate
for that blindness **before** any of them is scored.

## 2. ⛔ THE MUTATION, FIXED BEFORE ANY NUMBER IS COMPUTED

The panel's lead track is exactly `[5, 2]` — the reward prefix `GRID_S = (0.0, 0.5, 1.0,
1.5, 2.0) s` — so **"three of five lead samples" is literal here, not an analogy.**

```
PERTURB(d):  lead_track[i, 0] += d   for i in {1, 2, 3}
             lead_track[0] and lead_track[4] held EXACTLY fixed
```

* `d` in `{0.0, 5.0, 15.0}` metres. **5.0 m is the ruling's magnitude**; 0.0 is the NULL
  control; 15.0 is the scale control.
* ⭐ **The endpoint is pinned on purpose.** `progress`'s lead cap is a position query on
  `lead_path[..., -1, 0]` (`PREREG.md` §2), so pinning index 4 makes the `progress`
  channel **provably inert** under the perturbation and isolates the distance-keeping
  channel. `_headway` scores steps 1–4, so the perturbation moves **three of the four
  scored steps** and leaves the fourth — which is precisely the configuration in which
  `amin` can be bit-identical.
* ⛔ **Coordinate discipline unchanged.** The perturbation moves **POSITIONS**. No closing
  rate is introduced anywhere, and TTC remains a **VETO**, untouched.

## 3. ⛔ THE CANDIDATES, enumerated before any is computed

All are functions of the paired per-window composed gap
`g_i = composed(hold_v0)_i − composed(human)_i`, weights `{progress 0.3, collision 1.0,
headway 0.3}`, on the same rows, same ladder, same episode-cluster bootstrap.

| id | statistic | family | reading |
|---|---|---|---|
| `S1_rate` | `mean(1[g >= -1e-12])` — **`G-REWARD`'s retired statistic** | SIGN | lower is better |
| `S2_cliffs` | `mean(1[g>0]) - mean(1[g<0])` (Cliff delta) | SIGN | lower is better |
| `S3_median` | `median(g)` | ORDER | lower is better |
| `S4_q75` | `quantile(g, 0.75)` | ORDER | lower is better |
| `S5_mean` | `mean(g)` | MASS | lower is better |
| `S6_wmr` | `sum(max(g,0)) / sum(abs(g))` — **win-magnitude ratio** | MASS | lower is better |
| `S7_signedrank` | `sum_{g>0} rank(abs(g)) / sum rank(abs(g))` — Wilcoxon `r+` | RANK | lower is better |
| `S8_logmassratio` | `log(sum(max(g,0))+eps) - log(sum(max(-g,0))+eps)` | MASS | lower is better |
| `S9_skewsplit` | `median(g) - mean(g)` | MIXED | diagnostic |

⭐ `S6_wmr` and `S1_rate` share the **[0, 1] scale with 0.5 neutral**, which is what makes
a like-for-like bar possible in §6 without inventing a new number.

## 4. ⛔ THE DISQUALIFICATION CRITERION

> A candidate is **DISQUALIFIED** if `S(PERTURB(5.0)) - S(unperturbed)` is **exactly
> 0.000000** — bit-identical — on the SELECT split, with the reward's own
> `headway_reduce = "q0.25"` so that the composed reward genuinely moves.
>
> ⭐ **Ranking among the survivors is by RESPONSE-TO-NOISE RATIO**
> `RNR = abs(dS(5.0)) / SD_boot(S)`, where `SD_boot` is the candidate's own episode-cluster
> bootstrap standard deviation on the unperturbed panel. **A statistic whose response to a
> real 5 m geometry change is smaller than its own episode noise cannot decide anything**,
> so `RNR < 1` is reported as **UNDERPOWERED** even when it is not bit-identical.

⚠️ The reduction is set to `q0.25` for the ranking **on purpose and it is stated here, not
discovered later**: under `min` the *term* is blind, so a candidate could be scored blind
for the term's reason rather than its own. `min` is nonetheless run as control §5.2 so the
term's blindness is measured at panel scale rather than inherited from a unit test.

## 5. ⛔ CONTROLS THAT MUST READ KNOWN VALUES

The 2026-08-22 rule applies in full — four probe failures in one afternoon, each caught
only by a control that had to read a known value.

1. **NULL (d = 0.0).** Every candidate must read `dS` **exactly 0.000000**. A candidate
   that moves under a zero perturbation is measuring noise and the panel is void.
2. **TERM-BLINDNESS at panel scale (`headway_reduce="min"`, d = 5.0).** Report the
   fraction of lead windows whose `headway` component is **bit-identical**. ⭐ This is the
   unit test's finding re-measured on the real windows; if that fraction is ~0 the unit
   test does not generalise and the ruling's premise must be re-examined **in the report**.
3. **CAP INERTNESS.** `progress` must be bit-identical for **every** window under every `d`
   (the endpoint is pinned). A non-zero `progress` delta means the perturbation leaked into
   the ranking channel and the panel is void.
4. **MONOTONICITY.** For a genuinely responsive candidate `abs(dS(15.0)) >= abs(dS(5.0))`. A
   non-monotone response is reported as such and costs the candidate its rank.
5. **DIRECTION.** Moving the lead **away** can only relax distance-keeping, so a
   responsive candidate must move in the direction that pays the constant-velocity path
   **more**, not less. Reported per candidate; a wrong-signed response is flagged.

## 6. ⛔ THE SPLIT — "data that did not select it"

73 episodes. `SELECT` = episodes whose `sha256(clip_id)` first byte is **even**; `SCORE` =
**odd**. Fixed here, before any number, and computed from the clip id so it does not depend
on enumeration order.

* ⛔ **The mutation ranking reads the SELECT split ONLY.** The tool refuses to compute a
  candidate value on the SCORE split (`--split select`).
* ⛔ **The survivor's bar is pre-registered in a SEPARATE file
  (`PREREG_H-RL-GATE-STAT-1.md`) committed before the SCORE split is read**, and the score
  is then reported as written.
* The all-episode number is reported afterwards as **SECONDARY**, explicitly labelled as
  containing the selection data.

## 7. Estimator, tier, evidence class

* **Estimator:** episode-cluster bootstrap over episodes, `n_boot` 4,000, alpha 0.05.
  ⛔ It answers **ONE** question — *"would another draw of EPISODES say this?"* Not another
  training run (`H-ESTIM-SEED-1`), not another inference run. **No arm is trained and no
  planner samples here**; this is arithmetic over re-computed geometry on fixed rows.
* **Tier:** **T0** instrument probe, NON-PARITY RL-fit windows. ⛔ Not a driving number.
* **Evidence class:** MEASURED (ours), artifacts under this package's `raw/`.
* ⛔ **Four families:** this document scores a REWARD-GATE STATISTIC, not a policy. It is
  not an eval, produces no capability claim, and asserts no four-family table.

## 8. What is NOT changed by this document

No reward term, no weight, no arm, no guard, no banked result. The perturbation exists only
inside the ranking tool and is never written into any reward path. `G-REWARD` stays FAILED.
