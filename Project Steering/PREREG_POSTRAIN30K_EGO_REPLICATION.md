# PRE-REGISTRATION — does `postrain30k`'s PREDICTED latent carry the ego's Δyaw?

**Registered** 2026-08-26 ~03:40 Europe/Berlin — **while `postrain30k` is at step
~25,000 of 30,000 and its checkpoint has not been scored.**
**Author** Master Mind · **Tier** T0-DIAGNOSTIC · **Instrument** `egostate.py`,
built and banked *before* this prediction existed.

---

## 1. What the census found, and why it is only a hypothesis

An 8-arm ego-state census (`egostate_census.json`, 20 held-out clips, 1,800 rows,
identity control **+0.9555**) asked, for each arm, whether the latent carries ego
**levels** (speed, yaw-rate) and ego **changes** (Δspeed, Δyaw over 4 ticks).

**`z_t` (encoded), t vs the time-shuffled control:**

| arm | step | speed | yaw-rate | Δspeed | Δyaw | reading |
|---|---|---|---|---|---|---|
| `rdw8p30k` | 30000 | **2.07** | **2.76** | −0.13 | 0.98 | levels only |
| `splitp30k` | 30000 | 0.65 | 1.78 | **2.05** | 1.83 | changes only |
| `ro128p30k` | 30000 | −0.49 | 1.99 | 0.12 | **2.21** | changes only |
| `champ30k` | 30000 | 0.70 | −0.74 | −0.68 | −0.80 | neither |
| `rdw8s30k` | 30000 | 0.10 | **−2.64** | −1.23 | −1.33 | neither |
| `o11p30k` | 7500 | −1.78 | 1.72 | −1.10 | 1.56 | neither |
| `postrain10k` | 10000 | **3.15** | 1.16 | 0.51 | 1.14 | levels only |
| `splitfrz10k` | 10000 | −0.47 | 1.55 | −1.94 | 1.80 | neither |

**`ẑ` (predicted) on the changes:** every arm ≤ 1.6 **except `postrain10k`, whose
Δyaw reads t 3.00** — the only arm in the census whose *prediction* carries an ego
change above threshold.

⛔⛔ **WHY THIS IS NOT YET A FINDING, AND THE CENSUS ITSELF SAYS SO.** `rdw8s30k`
reads yaw-rate **−2.64**. A latent cannot meaningfully *anti-carry* its own
yaw-rate; that cell is noise. ⇒ **This census demonstrates, on its own data, that
|t| ≈ 2.6 arises by chance here.** With 32 `z_t` cells and 16 `ẑ` cells, two or
three |t| > 2 are expected from nothing at all. **Only `postrain10k`'s 3.00 exceeds
the demonstrated noise band, and it exceeds it barely.**

⚠️ A negative control that lands at the same magnitude as your positive is the most
useful row in a table. It is why this is a pre-registration and not a claim.

---

## 2. ⭐ The prediction, committed before the data exists

`postrain30k` is **the same recipe as `postrain10k`** — Gate-A flags with
`--init-from` the DINOv3-distilled checkpoint — trained to 30,000 instead of
10,000 steps. It is running now and **has not been scored.**

> **PREDICTION: `postrain30k`'s `ẑ` will carry Δyaw at t > 2 on `egostate.py`,
> 20 held-out clips, k = 4, identity control passing.**

| outcome | criterion | conclusion |
|---|---|---|
| ⭐ **REPLICATED** | `ẑ` Δyaw **t > 2.6** (above the census's own demonstrated noise band) | The distilled-init recipe produces a prediction that carries an ego change — at two different step counts, on an instrument built beforehand. **The first positive action-relevant transition result in the programme.** |
| 🔶 **WEAK** | `2.0 < t ≤ 2.6` | Inside the noise band this census demonstrated. **Report as not separable**, do not promote. |
| ⛔ **NOT REPLICATED** | `t ≤ 2.0` | `postrain10k`'s 3.00 was one of the two-to-three false positives expected from 48 cells. **The census is entirely null and must be reported as such.** |

⚠️ **The NOT-REPLICATED branch is the likeliest one and is written first on
purpose.** Ten objective terms have failed; E-DEC-52 concluded that objective
design is exhausted on this corpus. A single t 3.00 in a 48-cell census is exactly
the shape of a result that evaporates, and pre-committing the null here is what
stops it from being narrated into a finding tomorrow.

## 3. Guards

- ⛔ **Same instrument, unmodified.** `egostate.py` is already banked and committed;
  the read uses it as-is. Any change to it before the read voids this
  pre-registration.
- ⛔ **The identity control must pass** (~1.0, it reads +0.9555). If it does not,
  **no verdict**.
- ⚠️ **Report Δspeed beside Δyaw**, and say which cleared — a prediction about one
  target must not be reported as though both did.
- ⚠️ **`postrain10k` is step 10,000 and `postrain30k` is 30,000**, so this is a
  replication *across step counts*, not a matched rerun. A negative could mean the
  effect is real at 10k and gone by 30k. Say so if it happens rather than choosing
  the flattering reading.
- ⚠️ **Every number is T0.** A replication would say the transition carries an ego
  change. It would say **nothing** about driving.

---

## ⛔ AMENDMENT, 2026-08-26 ~03:50 — BEFORE `postrain30k` WAS SCORED

**The threshold moves from t > 2.6 to t > 3.0**, and the reason is that the 2.6
came from an **anecdote** (one physically meaningless cell) while **E-DEC-54 then
MEASURED the null**: 24 draws from a latent that provably carries nothing give
\|t\| p95 **2.71** and **max 2.93** — on Δyaw, the very target of this prediction.

⇒ **`postrain10k`'s t 3.00, the observation that motivated this pre-registration,
is itself inside the null.** The census is entirely null, and the honest state of
this prediction is that **its motivating evidence has evaporated.** It is kept and
run anyway, because a pre-registration that is quietly withdrawn when its
motivation weakens teaches nothing.

| revised outcome | criterion |
|---|---|
| ⭐ REPLICATED | `ẑ` Δyaw **t > 3.0** *and* above the null max |
| 🔶 WEAK | 2.0 < t ≤ 3.0 — **inside the measured null; report as null** |
| ⛔ NOT REPLICATED | t ≤ 2.0 |

⚠️ **This amendment is legitimate only because it happened before any test data
existed** — `postrain30k` was at step ~27,600 of 30,000 and unscored. Tightening a
threshold on null data is sound; tightening it on test data is not, and the
timestamps are recorded here so the distinction is auditable.
