# RESULT — E-ARCH-SMAS-1

`2026-09-02 · Research Lab · Architecture & Inference · 0 GPU · raw/smas.json`

**Tier:** `T0-DIAGNOSTIC` — inherited from the source reads. ⛔ Not a driving
claim; not comparable to any T1 number.
**n:** 24 clips / 144 windows / 8 action variants per arm, `variant_draw = "roll
(never permutation)"`. **C0 identity control PASSES on all three arms at h1/h2/h4**
(`C0_identity_max_abs_diff = 0.0`), so all three are scoreable.

**Headline.** The MM-E19 ratio fall is **73.7 % denominator**. And the
pre-registered *"≥ 10× rise"* bar was never a 10× bar: at the observed scene
movement it silently demanded a **17.1×** rise in action response. ⛔ The verdict
on the k=60 arm **does not change** — it failed either way — but a **doubling of
action sensitivity would have read as "+17 %, flat"** under the same statistic,
so the criterion must change before L-1's curriculum arm is judged by it.

---

## The three banked arms at h1 (MEASURED, ours; `raw/smas.json` → `arms_h1`)

| arm | `o5_k` | action_spread | scene_spread | raw ratio |
|---|---|---|---|---|
| `postrain30k` (incumbent) | 8 | 0.001394 | 0.234344 | 0.005947 |
| **`k8clip05p30k`** (the one-variable reference) | **8** | 0.001319 | 0.213915 | 0.006165 |
| `k60clip05p30k` | **60** | 0.001089 | **0.366344** | 0.002972 |

⚠️ **A trap this package had to disarm to read its own inputs.** The k=8 control's
banked JSON stores its arm under the internal key **`"k60clip05p30k"`** — the
`--arm-name` default that the attribution package documents (§6a) and that only a
checkpoint md5 resolved. `code/scene_matched_criterion.py` therefore reads it by an
**explicitly named constant with the defect written beside it**, never by the name
in the file. Reading by name would have silently swapped the two arms and inverted
every conclusion below.

## F1 — ⛔ 73.7 % OF THE RATIO'S FALL IS THE DENOMINATOR

One variable (`o5_k` 8 → 60), reference `k8clip05p30k`
(`raw/smas.json` → `one_variable_decomposition_k8control_to_k60`):

```
action factor  0.8256      the action response fell   17 %
scene  factor  1.7126      the scene  response rose   71 %
ratio  factor  0.4821      reconstructed 0.4821  (err 2.17e-05)
```

Log-additive attribution — the only additive split of a quotient's movement:

| contributor | share of the ratio's change |
|---|---|
| the **action** response falling | **26.3 %** |
| ⛔ the **scene** response rising | **73.7 %** |

⇒ *"the model became action-deaf"* is not what this measures. What it measures is
**a model that got substantially better at the scene while getting modestly worse
at the action** — and a statistic that reports the first as if it were the second.

## F2 — ⛔⛔ THE PRE-REGISTERED "10×" BAR SILENTLY DEMANDED 17.1×

`PREREG_MM_E19` commits `HORIZON-WORKS` to a **≥ 10× rise in the ratio**. A ratio
bar is a bar on `action_spread / scene_spread`, so once the denominator moves by
`scene_factor` the action-side requirement moves with it:

```
effective action bar = stated ratio bar x scene_factor = 10 x 1.7126 = 17.13x
```

**MEASURED** (`raw/smas.json` → `prereg_bar`). ⇒ the arm was judged against a bar
requiring its action response to rise **17.1×**, not 10× — a 71 % harder test that
**no one set and no one recorded**. The bar was not fixed at pre-registration
time; it was fixed only once the arm's own scene spread was known, which is
exactly what pre-registration exists to prevent.

## F3 — ⭐⭐ WHAT THE RAW RATIO WOULD HAVE REPORTED FOR A *SUCCESSFUL* ARM

Holding the observed `scene_factor = 1.7126` and varying only the action response
(`raw/smas.json` → `hypothetical_reads_under_the_RAW_ratio`):

| if the action response had… | the raw ratio reads | how it would have been written up |
|---|---|---|
| halved (0.5×) | 0.292× | "collapsed" |
| **stayed exactly flat (1.0×)** | **0.584×** | ⛔ **"fell 42 % — the horizon made it worse"** |
| risen 1.5× | 0.876× | "flat / slightly down" |
| ⛔ **doubled (2.0×)** | **1.168×** | ⛔ **"+17 %, within noise — INERT"** |
| tripled (3.0×) | 1.752× | "a modest rise, far short of 10×" |

⇒ **An arm that doubled its action sensitivity would have been reported as
inert.** That is the finding, and it is about the instrument, not about any arm.
It is the same family as the `overlapping_holdout_se` correction: a statistic that
looks like an answer while its central value is set by something other than the
quantity of interest.

## F4 — ⚠️ THE FIX DOES NOT RESCUE THE k=60 ARM, AND THIS PACKAGE WILL NOT PRETEND IT DOES

Under **SMAS** — `action_spread / scene_ref`, denominator pinned to
`k8clip05p30k`'s 0.213915 (`raw/smas.json` → `smas`):

| arm | SMAS | vs the reference |
|---|---|---|
| `k8clip05p30k` | 0.006165 | — |
| `k60clip05p30k` | 0.005091 | **0.8256×** |

**H-SMAS-3 resolves to outcome B.** 0.83× is still a fall and still nowhere near
any plausible bar, so **MM-E19's verdict stands: k = 60 did not buy
action-conditioning.** What changes is:

1. the reported **magnitude** — 0.83×, not 0.48×;
2. the **attribution** — the action response fell 17 %, it did not collapse; the
   scene response rose 71 %, which is *not a defect*;
3. the **bar future arms face**, which is the reason this package exists.

⛔ **And SMAS is `action_spread` rescaled by a constant** — stated here and in the
code, not buried. The contribution is refusing to put a moving quantity in a
decision statistic's denominator, not inventing a better one.

## Controls (`raw/smas.json` → `controls`; **all pass**)

| control | required value | read |
|---|---|---|
| **no-information** — an action-blind arm (`action_spread = 0`) | exactly **0.0** under both raw ratio and SMAS | **0.0 / 0.0** ✅ |
| **known-value synthetic** — inject `action_factor 2.5`, `scene_factor 0.4` | recovered exactly | recovered to < 1e-12 ✅ |
| **empirical action-dead floor** (h2/h4, same reads) | the instrument's measured "action does nothing" | **6e-06 / 7e-06** on every arm ✅ |
| **C0 identity** (from the source reads) | PASS, all arms, all horizons | **PASS** ✅ |

⚠️ **The floor is the uncomfortable one.** At h2 and h4 the action spread is
**6–7e-06 on every arm**, against h1's ~1.3e-03 — i.e. **beyond one step the
action does essentially nothing, k = 60 included**. That is a property of *both*
arms, is unaffected by this criterion change, and remains the larger fact in the
table.

---

## ⇒ The criterion, stated so L-1's arm can be judged by it

**For any action-conditioning arm, the decision statistic is `action_spread` at a
FIXED reference denominator (SMAS), and `scene_spread` is reported alongside as a
CO-PRIMARY — never in the denominator.** Concretely:

1. **Name the reference arm** and pin `scene_ref` to it in the prereg, *before*
   the arm runs. The reference must be a genuine one-variable control.
2. **State the bar in action-side units.** *"action_spread must rise ≥ N×"* is a
   bar. *"the ratio must rise ≥ N×"* is not — F2 shows it floats.
3. **Report `scene_spread` and its factor in the same table**, with no threshold
   attached. A rise there is world-model progress and must not be able to fail an
   action arm; a *fall* there is a red flag that deserves its own investigation.
4. **Report the h≥2 floor every time.** An h1 result with a dead h2 is a
   one-step-only sensitivity and should never be written as "action-conditioning
   works".
5. **Carry the four controls above**, including the constant-only one, per
   `CLAUDE.md`'s probe-panel rule.

### What this does NOT fix — stated, per the spec's own criterion

* It does **not** make the action response larger. Both statistics agree the
  programme's action sensitivity is ~0.5 % of its scene sensitivity at h1 and
  ~0.005 % at h2.
* It does **not** address **P2(d)** (teacher-forced targets) — the leading
  candidate cause. A better criterion measures the same thing more honestly; it
  does not change the model.
* It does **not** supply an interval. These are point spreads over 144 windows
  with no episode-cluster bootstrap on the instrument, so **no CI is quoted
  rather than a wrong one**. ⚠️ That is a real gap: the 0.83× and 1.71× factors
  have no stated uncertainty, and adding a paired bootstrap to the actdiv probe
  is proposed as backlog row **L-13**.
* ⚠️ It inherits **F3 of the DataEng package** — the k=60 arm's training-window
  set is 23 % smaller and temporally truncated, so part of the 1.71× scene rise
  may be corpus, not capability. That confound sits *underneath* both statistics
  and neither removes it.
