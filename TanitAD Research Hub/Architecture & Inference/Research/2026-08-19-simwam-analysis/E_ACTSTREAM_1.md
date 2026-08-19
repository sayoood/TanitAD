# E-ACTSTREAM-1 — action-as-TOKEN beats action-as-CONDITIONING, but NEITHER beats persistence

`MEASURED (ours; dev-box RTX 4060)` · **T0-DIAGNOSTIC — a future-field prediction
error is a world-model fidelity number, NEVER driving performance** · latents
from `cache_s16000` (v6F-SW-30k@16000) · episode-disjoint split from
`p3_selection.json` · paired episode-cluster bootstrap, `n_boot 2000`, 70 eval
episode clusters · 3 seeds · **no load added to Thor**.

## 0. The question, and why it is the PI's

SimWAM puts action tokens in the **same self-attention stream** as the vision
tokens. Both of our arms condition instead:

* **v6** — actions enter `P_O` as a conditioning vector, never as tokens.
* **REF-A v1** — `TokenFieldPredictor` broadcasts the action over tokens and
  **concatenates-then-projects**; DINO-WM's exact scheme.

⇒ **The joint stream has never been tested in this programme.** The PI raised it;
this is the test.

⛔ **AND IT REPLACES E-SIMWAM-1, WHICH I PROPOSED WITHOUT CHECKING IT COULD RUN.**
The isolated-vs-imagination experiment is doubly blocked: (a) no v6 T1 val cache
exists locally — the 4.70 GB download is a pending PI decision — and (b) the
imagination arm cannot be launched at all, because `mpc_refine` requires
`selector="goal"` and `assert_selector_admissible` refuses every selector launch
while **SEL-1 stands REFUSED** (E-WC2, sigma/ADE 9.9915 [7.4492, 13.5119] against
the 3.0 line). ⚠️ Note what that means: with `mpc_w_consist` defaulting to 0.0 and
the selector refused, **v6 already runs SimWAM's isolated design**. That question
is settled by a refusal, not open.

## 1. Task and arms

Predict the cell field at `t+6` (stride 4 ⇒ **2.4 s**) from a 4-frame causal
window plus the ego actions applied over the horizon (accel, yaw-rate, derived
from `poses`, not from a label).

| arm | action injection |
|---|---|
| `concat` | embedding broadcast over the 16 tokens, concatenated, projected — **ours / DINO-WM** |
| `token` | embedding split into **2 tokens appended to the vision tokens** in the self-attention stream — **SimWAM** |

⚠️ **Parameter-matched by construction**, or the result is a capacity comparison
in an architecture costume: `n_act_tok=2` puts the arms within **576 params
(0.04 %)**. Same width, depth, heads, optimiser, schedule, data, windows, seeds.

## 2. ⛔ THE CONTROLS, FIRST — and they are the headline

| | MSE |
|---|---|
| **C-PERSIST** — copy the last observed frame | ⭐ **0.000005** |
| C-MEAN — train-set mean field | 0.000017 |
| **`token`** (SimWAM, 3-seed) | 0.000013 |
| **`concat`** (ours, 3-seed) | 0.000086 |
| C-ZERO | 0.000690 |

⛔ **NEITHER LEARNED ARM BEATS COPYING THE LAST FRAME** — `concat` loses to it by
**17×**, `token` by **2.6×**.

⇒ **No claim that either scheme "learns the dynamics" is admissible from this
experiment.** The first run of this experiment reported a "3.6× win" for the
token arm before these baselines were computed; that number compared **two
failures**. It is the D1 pattern exactly — a margin quoted without a trivial
control — and the control is what caught it.

⚠️ Predicting the **residual** (so persistence is the zero-prediction) was tried
and did not rescue it: `concat` 0.000086, `token` 0.000013, both still above
0.000005. The absolute-target and residual formulations give the same verdict.

## 3. ⭐ What DOES survive: the relative contrast, at every width

| width | concat | token | paired Δ (token − concat) | separated |
|---|---|---|---|---|
| d=192, L=4 (residual) | 0.000086 | **0.000013** | **−0.000073** [−0.000087, −0.000060] | ✅ |
| d=192, L=4 (absolute) | 0.000104 | **0.000029** | **−0.000074** [−0.000088, −0.000061] | ✅ |
| d=48, L=2 | 0.000443 | **0.000042** | **−0.000400** [−0.000476, −0.000328] | ✅ |
| d=32, L=2 | 0.000607 | **0.000166** | **−0.000440** [−0.000543, −0.000341] | ✅ |

**Negative = the SimWAM-style joint token stream predicts better.** The contrast
is separated in **every** configuration, at parameter parity, and is
seed-stable (e.g. d=192 token: 0.000012 / 0.000014 / 0.000011 against concat
0.000086 / 0.000078 / 0.000092 — the distributions do not overlap).

⚠️ **Shrinking the model made BOTH arms worse** (d=192 → d=32 costs concat 7× and
token 13×), so the failure to beat persistence is **not overfitting**. With 2,054
training windows the ceiling here is data, not capacity.

## 4. What this does and does not license

✅ **Licensed:** *at equal parameters and data, the joint action-token stream
extracts more from the same budget than broadcast-concat conditioning, robustly
across widths, targets and seeds.* That is a real relative architectural finding
and it supports testing the joint stream in REF-A v1's `TokenFieldPredictor`.

⛔ **NOT licensed:** any claim that either scheme models the dynamics, any
absolute number, and any transfer of the *magnitude* of the gap to a trained
arm. Both arms sit below the trivial floor.

⇒ **What would make it decisive:** the same experiment on the **full parity
corpus** (2,376 episodes) rather than a 130-clip probe cache. The instrument is
built and cheap; it is the data that is short.

## 5. Manifest

| artifact | where |
|---|---|
| `e_actstream.py` (the experiment, controls included) | `…/2026-08-19-simwam-analysis/code/` |
| `e_actstream_resid.json`, `e_act_d48.json`, `e_act_d32.json` | `…/2026-08-19-simwam-analysis/raw/` |
| this report | `…/2026-08-19-simwam-analysis/E_ACTSTREAM_1.md` |

---

# UPDATE 2026-08-19 (later) — centred, at the 6 s design horizon, with the disambiguating control

Three defects in the first run, all fixed:

1. ⛔ **The field was never centred.** MEASURED: raw `mean(Y²)` 0.000692 against a
   **centred variance of 0.000016** — **97.7 % of the magnitude is a constant
   offset**, so both arms were spending capacity reproducing a constant.
   Centring now uses the **TRAIN** mean field only, applied to both splits.
2. ⛔ **The horizon was 2.4 s, not our 6 s design point.** Re-run at `horizon=15`
   (6.0 s), which is what REF-A v1's three rates are built around.
3. ⛔ **The headline was CONFOUNDED.** `concat` differs from `token` in TWO ways
   at once — broadcast-vs-tokenised AND a learned `mix` applied to the FIELD
   tokens. A third arm now separates them.

## Results — centred, 6 s horizon, 3 seeds, parameter-matched

| arm | MSE (3-seed) | params | vs C-PERSIST |
|---|---|---|---|
| *C-PERSIST* (copy last field) | **0.000007** | — | — |
| **`token`** (SimWAM stream) | **0.000010** | 1,367,744 | 1.4× above |
| *C-MEAN* | 0.000017 | — | — |
| `concat` (broadcast + mix) | 0.000173 | 1,367,168 | 25× above |
| `add` (broadcast + add) | 0.000301 | 1,293,248 | 43× above |

paired Δ (token − concat) **−0.000163 [−0.000190, −0.000137]**, separated.

## ⭐ What the control settles

* `concat` **BEATS** `add` (0.000173 vs 0.000301) ⇒ the `mix` bottleneck **helps**;
  it is not the source of the gap.
* `token` beats **both** broadcast forms by **17–30×**.

⇒ **The advantage is the TOKENISATION, not the projection.** Putting the action
into the self-attention stream beats broadcasting it, however the broadcast is
combined. That is the SimWAM design choice isolated.

⇒ `token` now also **beats C-MEAN** (0.000010 vs 0.000017) where both broadcast
arms fail it, and sits **1.4× above C-PERSIST** — down from 2.6× before centring.

⛔ **STILL NOT DECISIVE, and the reason is unchanged:** no arm has beaten
persistence, so *"learns the dynamics"* remains unlicensed. What is now licensed
is stronger and disambiguated: **tokenised action conditioning dominates
broadcast conditioning at parameter parity, across widths, horizons, target
formulations and seeds.**

⏳ **The remaining lever is data.** A stride-1 latent dump (4× the windows,
~22.5 k frames) was launched for the final run. If `token` crosses C-PERSIST
there, the experiment becomes decisive; if it does not, the limit is the
REPRESENTATION, and §"what this measures about our readout" below is the finding.

## ⚠️ An independent finding about our readout, surfaced by the variance decomposition

| | |
|---|---|
| BETWEEN-episode variance | 0.000013 |
| WITHIN-episode variance | 0.000003 |
| **ratio** | ⛔ **4.5×** |

**The cell field is dominated by which episode you are in, not by when in the
episode you are** — a scene fingerprint more than a dynamics state. That bears
directly on the same day's ladder result (`LADDER_S16000.md`): probes were being
asked to read agent counts from a representation whose variance is mostly scene
identity, and the episode-cluster bootstrap correctly clusters exactly that
variance away. Worth its own experiment; recorded here because the decomposition
fell out of this one.

---

# FINAL — 4× data (stride-1 dump), and the verdict on both questions

The last missing lever was data. A **stride-1 latent dump** of the same 130 clips
and the same checkpoint (`v6F-SW-30k@16000`, 22,468 frames vs 5,617, 47.5 min on
the dev-box 4060) supplies **13,108 windows** against the previous 3,277.
⚠️ `STRIDE` stays 4 in the task geometry, so the TASK is unchanged — the stride-1
cache only supplies 4× more valid START positions.

## Results — 13,108 windows, centred, 6 s horizon, 3 seeds, parameter-matched

| arm | MSE (3-seed) | params | vs C-PERSIST |
|---|---|---|---|
| *C-PERSIST* | **0.00000728** | — | — |
| **`token`** (SimWAM stream) | **0.00000785** | 1,367,744 | **1.078×** |
| *C-MEAN* | 0.000017 | — | 2.3× |
| `concat` (broadcast + mix) | 0.000046 | 1,367,168 | 6.3× |
| `add` (broadcast + add) | 0.000078 | 1,293,248 | 10.7× |

paired Δ (token − concat) **−0.000038 [−0.000046, −0.000031]**, separated.

## ⛔ VERDICT 1 — no arm beats persistence, and now it is settled rather than suspected

paired Δ (**token − C-PERSIST**) = **+0.00000057 [+0.00000036, +0.00000084]**
⇒ **token LOSES to persistence, SEPARATED.**

The 4× data helped materially — token went 0.0000100 → 0.0000079 (−21 %) and the
gap to persistence narrowed from 1.4× to **1.078×** — but it did not cross.
⚠️ **NOT EXTRAPOLATED.** The full parity corpus is ~18× more episodes, far beyond
the 2× the programme's own rule permits projecting, so whether it crosses there
is an open question and not a prediction.

⇒ **"The world model learns the dynamics" remains UNLICENSED at this geometry.**

## ⭐ VERDICT 2 — the tokenisation advantage is decisive, large, and disambiguated

| contrast | ratio | reading |
|---|---|---|
| token vs concat | **5.9×** | tokenised beats broadcast+project |
| token vs add | **9.9×** | tokenised beats broadcast+add |
| concat vs add | 1.7× (concat better) | the `mix` bottleneck **helps** — it is not the cause |

Separated at every width (d=32/48/192), both target formulations (absolute and
residual), both horizons (2.4 s and 6.0 s), both data scales (3,277 and 13,108
windows), and all 3 seeds — token's three seeds at the final config are
0.0000080 / 0.0000080 / 0.0000079, a spread of 1 %.

⇒ **Putting the action into the self-attention stream dominates broadcasting it,
however the broadcast is combined.** That is SimWAM's design choice, isolated
from the projection it is bundled with, and reproduced in FEATURE space at
1.4 M parameters — three orders of magnitude below the 6 B it was published at.

## What this licenses for REF-A v1′

✅ Building and parking `refa_v1p.py` is justified: the conditioning scheme it
changes is the one the data says matters, and the effect is large enough that it
should be visible at v1's real geometry.
⛔ It does **not** license a claim that v1′ will beat v1 on driving, nor any
transfer of the *magnitude*. Both arms here sit above a trivial floor, and the
comparison ran on v6 cell fields (16 × 128), **not** on v1's DINOv3 token field
(640 × 1024). That transfer is the next experiment, and it needs DINOv3 features
for these clips.
