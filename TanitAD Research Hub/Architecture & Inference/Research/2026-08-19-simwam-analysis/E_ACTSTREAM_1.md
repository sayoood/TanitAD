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
