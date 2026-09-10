# PRE-REGISTRATION — E16 `--max-speed-input` as a single-lever arm on refcv6

**Registered 2026-09-10 · Architecture & Inference · `agent/arch-inf-20260803`**
**Status: REGISTERED, NOT RUN.** No GPU exists for this arm today. Both outcomes are
committed below **before** any training.

---

## 0 · ⛔⛔ THE DECLARATION THAT TRAVELS WITH EVERY NUMBER THIS ARM PRODUCES

> **`speed_max_input.v_max_ms` IS `g_tac.goals.SPEED_BAND.v_hi_ms`, which is the
> MAXIMUM OF THE EGO'S OWN REALISED SPEED OVER `[t0+2 s, +6 s]`.** Its provenance is
> **`ego-future`**. Quantization does not launder it: MEASURED 5-fold out-of-fold,
> clip-disjoint, n = 4,572, the raw ceiling is recoverable from the pinned 8-step bin
> plus `v0` at **R² = 0.9702** (bin alone 0.9513; `v0`-alone control 0.8789, so the
> bin adds **ΔR² = +0.0913 of the ego's own future**).

**PI ruling, verbatim, 2026-09-10:** *"regarding to max speed as input, stick to the
labels we created in the data set with the logic of minimal speed etc..."*

⭐ **This is CONSISTENT with the programme's standing position, not an exception to
it.** refcv5-v2 already feeds an oracle nav command and its own `config.json`
declares `nav_cmd_derivation: "v7.2 nav_command token (oracle, provenance
ego-future; allow_oracle_nav=True)"`. The PI's rule is that the nav command is an
**INPUT simulating the vehicle's nav system, not a training signal**. Max speed is
the same class of signal, standing in for a **speed-limit service**.

⛔ **THEREFORE THE BINDING REQUIREMENT IS DECLARATION, NOT REFUSAL — AND IT IS
MECHANICAL, NOT EDITORIAL:**

1. every emitted label block carries `oracle: true`, `provenance: "ego-future"` and a
   `derivation` string naming the source field and the window;
2. the blob is readable only through `v7_labels.oracle_max_speed`, which refuses
   without `allow_oracle_nav` on the manifest;
3. `config.json` carries **`speed_max_derivation`**, and
   `refc_v3_train._assert_speed_max_stamp` **REFUSES TO START** a run that would
   reach the config write without it — *and refuses the mirror case*, a control
   stamped as conditioned. **Proven by mutation** (§5).

⛔ **NO CAPABILITY CLAIM MAY BE CREDITED TO THIS CHANNEL WITHOUT THE STAMP QUOTED
BESIDE IT.** A result table for this arm that omits `speed_max_derivation` is
inadmissible — not "incomplete", inadmissible.

---

## 1 · THE HYPOTHESIS, AND THE ONE LEVER

**H-E16-1.** *Feeding a quantized posted-limit ceiling as an INPUT beside `v0` and
the nav token improves longitudinal behaviour relative to an otherwise identical
arm.*

**The lever is exactly one parsed-namespace key:** `max_speed_input` `False → True`.
Both arms share seed, corpus, labels, anchors, every other flag, and the same
`--eval-labels`. ⛔ A diff of the two `argparse.Namespace` objects that shows more
than that one key **invalidates the panel** and the arm must be relaunched.

⚠️ `max_speed_mode` is **not** a second lever: it defaults to `quantized` and is
inert without `--max-speed-input` (the parser refuses the mode alone). It is
recorded, not varied.

---

## 2 · THE COMMITTED CRITERIA — both outcomes, written before the run

⛔ Per the standing rule, **an eval reporting ADE alone is INCOMPLETE.** All four
families are reported, per-family, never pooled, each with the **paired
episode-cluster bootstrap** on the same windows.

| family | primary metric | bar |
|---|---|---|
| **LONGITUDINAL** ⭐ *the axis this lever targets* | target-speed accuracy **and** distance-keeping (headway / time-gap / TTC) | the pre-registered claim lives here |
| **LATERAL** | heading, **curvature**, **yaw-rate**, cross-track | must not regress |
| **TACTICAL** | manoeuvre-decision quality + tactical goal-setting | must not regress |
| **STRATEGIC** | strategic decision + goal/route setting | must not regress |
| (ADE) | horizon sweep | reported, but it is **one row of four families** |

**SUCCESS** — all three must hold:
1. the **longitudinal** family improves with a **paired** episode-cluster CI that
   excludes zero, **and**
2. the improvement exceeds the **replicate arm's** noise floor (§3), **and**
3. no other family regresses with a separated CI.

**FAILURE** — declared plainly and reported as written, then followed by the next
lever in the same run (RULE ZERO): any of (1)–(3) fails.

⛔ **The bar is not moved after seeing the data.** ⛔ No control is dropped because
it is inconvenient. ⛔ A post-hoc finding gets its own pre-registration.

---

## 3 · ⛔⛔ A SEPARATED CI IS NECESSARY AND NOT SUFFICIENT — THE REQUIRED CONTROLS

MEASURED 2026-09-05: `A0b_replicate` — same flags, **same seed**, zero levers moved,
argv-audited — produced *"separated"* differences from `A0` on **6 of 42 family
cells, a 14.3 % false-positive rate for `separated`**. The episode-cluster bootstrap
resamples **episodes with the models held fixed**; it answers *"would another draw of
EPISODES say this?"* and never *"would another TRAINING RUN say this?"*

⇒ **This arm ships four controls, and a result without them is not evidence:**

| control | what it answers | must read |
|---|---|---|
| **REPLICATE** — the ON arm's flags, run again | the rig's own run-to-run noise floor | the lever's effect must **exceed** it |
| **WITHHOLD** — `v_max_valid = 0` on every row | is the *value* doing the work, or its presence? | the no-ceiling state, exactly |
| **SHUFFLE** — ceilings permuted across clips | is it the ceiling, or any extra scalar? | no better than OFF |
| **OFF twin** | the single-lever comparison | the baseline |

⚠️ If the planner samples at inference, the replicate must also vary the **inference
seed** — three different variances ride on one interval and the report must name
which one it answered.

---

## 4 · ⛔ THE KNOWN RESIDUAL DEFECT, REGISTERED IN ADVANCE

**Snapping UP from a stopped ego reports the LOWEST limit.** MEASURED on the shipped
v8 train blob: **809 of 4,572 clips (17.7 %) land on the 20 km/h step and 932
(20.4 %) on 30 km/h** — 38.1 % of the corpus at ≤ 30 km/h, because the ego was
stopped or crawling at an intersection where a map would say 50. ⇒ **The channel can
teach "slow ego ⇒ low limit".** This is *not* discovered later and reported as a
surprise; it is the reason the SHUFFLE and WITHHOLD controls above are mandatory,
and any longitudinal win must be shown not to be this.

---

## 5 · WHAT IS ALREADY BUILT AND PROVEN (2026-09-10, before this arm runs)

| thing | evidence |
|---|---|
| v8 label release, both splits | **4,572/4,572 train · 147/147 eval**, controls 4,572 / 147 |
| ladder cross-check | **exact-rational, independently authored** from the km/h integers: 4,572/4,572 and 147/147 buckets agree, 0 mismatches |
| the rounding trap is guarded | the regression arm reproduces **2,631/4,572 (57.55 %)**, `50→70:1593 · 20→30:809 · 100→120:229` |
| units on the wire | 4,572/4,572 declare `m_s`; a stripped block is refused |
| the loader's refusal still bites | block removed from all 4,572 records → **`NOT ONE of this split's 112 windows…`**, no artifact written |
| the stamp is a precondition | stamp deleted from the config dict → **refuses to start**, no `config.json` written |
| 20-step smoke | **112/112 windows fed**, 20 distinct finite losses |

⛔ **The refusal at `refc_v3_train.py:1596` was NOT weakened.** It passes because
real data is supplied. Its teeth were re-proven, not assumed.

---

## 6 · THE ARM'S ARGV

See `TanitAD Research Lab/Architecture & Inference/Research/2026-09-10-max-speed-v8-build/RESULT.md`
§ "The arm". The ON and OFF lines are byte-identical except for the single token
`--max-speed-input`, and a namespace-diff assertion is shipped beside them.

⛔ **This arm is NOT launched.** There is no pod. It waits for a GPU and for the
PI's go.
