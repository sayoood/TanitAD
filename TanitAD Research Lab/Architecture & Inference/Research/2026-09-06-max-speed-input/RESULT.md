# `--max-speed-input` — the pinned quantizer, and whether the quantization worked

**2026-09-06/07 · Architecture & Inference · `agent/arch-inf-20260803` · commit `6ae8acf`**

⛔ **TIER: NONE, AND THAT IS NOT A DODGE.** No model runs in this package. T0 is a
world-model diagnostic and T1 is self-action open loop; neither describes a
least-squares fit on two scalars or a quantizer applied to a label field. Every
number below is an **input-channel information probe** or an **artifact census**.
Stamping any of it T0/T1 would be a category error. The arm that needs a tier —
train `refcv3` with the flag ON vs OFF — is a GPU job, is **not** run here, and is
named as the integration item rather than silently substituted for.

---

## THE ONE-LINE ANSWER TO "IS THE QUANTIZATION COSMETIC?"

> **It is cosmetic as a leak fix and real as a semantics fix.**
> Quantizing to the pinned 8-step ladder removes only **24.6 %** of the residual
> future information the raw channel leaks past the ego's own speed — the bin
> plus `v0` still recovers **75.4 %** of it. But it converts a quantity that is
> **below the ego's current speed on 34.8 % of clips** (incoherent for a ceiling)
> into one that is below it on **8.4 %**, and it makes the ceiling **violable**
> (28 clips, 0.61 %) where the raw value is violated on **0 / 4,572 by
> construction**.

⇒ Wire it, in `quantized` mode, and read any ON-arm result **as if the model had
been handed a strong oracle** — because it has. The quantization is worth doing;
it is not a leak fix, and calling it one would be the wrong sentence to hand the PI.

---

## 1 · THE PINNED STEP SET, AND WHY IT IS THESE VALUES

```python
POSTED_LIMIT_STEPS_KMH = (20, 30, 50, 70, 80, 100, 120, 130)
# = (5.556, 8.333, 13.889, 19.444, 22.222, 27.778, 33.333, 36.111) m/s
```
Pinned in `stack/tanitad/refs/max_speed_input.py` with this rationale beside it.

**Two rules, and both are needed.** Values come from **road law**; membership
comes from the **corpus**. ⛔ No step is a quantile — a ladder fitted to the ego
speed histogram would re-encode exactly the information quantization is meant to
destroy. The corpus is 24 European countries + the United States (`strata.country`,
25 values, largest share 6.3 %), so the admissible values are the pan-European
posted ladder.

**Ladder selection, measured, n = 4,572 (snap UP, `q(v) = min{s ∈ S : s ≥ v}`):**

| ladder | k | H (bits) | max bin | slack p50 | slack p95 | over-ceiling | R²(v_hi ← q, v0) |
|---|---|---|---|---|---|---|---|
| `{30,50,70,100}` (PI's start) | 4 | 1.841 | 38.1 % | 10.4 km/h | 25.4 km/h | 4.90 % | 0.9293 |
| `{30,50,70,100,130}` | 5 | 1.969 | 38.1 % | 11.1 km/h | 26.3 km/h | 0.61 % | 0.9436 |
| **`{20,30,50,70,80,100,120,130}`** | **8** | **2.480** | **34.9 %** | **8.1 km/h** | **19.4 km/h** | **0.61 %** | **0.9631** |
| `… + 90, 110` | 10 | 2.559 | 34.9 % | 7.5 km/h | 19.2 km/h | 0.61 % | 0.9640 |
| every 10 km/h | 13 | 3.203 | 20.4 % | 4.9 km/h | 9.7 km/h | 0.61 % | 0.9884 |

**Does the corpus need 20/80/120? YES to all three:**
* **20 — needed.** 38.1 % of clips have `v_hi ≤ 30 km/h` (the ego is stopped or
  crawling; p1 of `v_hi` is **0.00 m/s**). Adding 20 splits that single mass into
  17.7 % / 20.4 %.
* **80 and 120 — needed.** The PI's ladder leaves 30 km/h-wide gaps above 70; they
  cut the p95 snap-up slack from **25.4 → 19.4 km/h**.
* **A step above 100 is mandatory.** 224 clips (**4.90 %**) exceed 100 km/h.

**And two measured rejections:**
* **90 and 110 excluded** — they move recoverability by **+0.0009 R²** and median
  slack by **0.6 km/h**. They buy nothing and cost resolution.
* **The every-10 ladder excluded** — at 13 steps `R²(v_hi | bin, v0) = 0.9884`,
  residual **0.80 m/s**. At that resolution the bin *is* the raw value in disguise.

⭐ **The top step is 130 km/h and it is not a clamp-to-cover.** 130 is the highest
generally-posted limit in these countries; the corpus max `v_hi` is 37.803 m/s =
**136.1 km/h**, so **28 clips (0.61 %) sit above the ceiling this channel reports**.
That is the single most important behavioural difference from the raw value: a real
limit *can* be exceeded, and `quantize_up` reports `over_ceiling=True` rather than
inventing a 140 km/h sign that exists nowhere.

---

## 2 · THE THREE MEASUREMENTS

**Evidence class for all three: MEASURED, full v8 train corpus, n = 4,572.**
The `(v0, v_hi)` join is re-derived from `egomotion_source` and checked against the
label: **4,572 / 4,572 identical, 0 mismatches** — a content assertion, not a
presence check (C140: 20.5 % of a validation was once the wrong episode).
Same-breath control: the same extractor reads **147 / 147** on the eval split.

### (ii) ⭐ THE LOAD-BEARING ONE — is the quantization cosmetic?

`v_hi_raw ← ?`, **5-fold out-of-fold**, clip-disjoint. In-sample is shown beside it
so the optimism gap is visible; the out-of-fold number is the one quoted.

| regressor | d | OOF R² | OOF RMSE | in-sample R² |
|---|---|---|---|---|
| `← v_hi` — **control that must read a known value** | 1 | **1.0000** | 0.000 m/s | 1.0000 |
| `← v0` alone — **the control the brief demands** | 1 | **0.8789** | **2.586 m/s** | 0.8789 |
| `← (shuffled bin, v0)` — **must collapse onto `v0`** | 3 | **0.8789** | 2.586 m/s | 0.8789 |
| `← bin` alone | 1 | 0.9513 | 1.640 m/s | 0.9514 |
| `← (bin, v0)` linear | 2 | 0.9631 | 1.428 m/s | 0.9631 |
| `← (bin, v0)` **per-bin** (strongest small regressor) | ≤16 | **0.9702** | **1.282 m/s** | 0.9706 |

**Reading, both directions, because only one of them is the usual one:**
* **Unrecovered:** `1 − R² = 0.0298`, residual **1.282 m/s**. The raw future value
  is *not* fully readable off the bin. **The quantization is not purely cosmetic.**
* **Residual leak:** `ΔR² = 0.9702 − 0.8789 = +0.0913`; RMSE falls 2.586 → 1.282,
  i.e. the bin removes **1.304 m/s** of the ego-only error. The ego present already
  explains 0.8789 of `v_hi`; of the **0.1211** it leaves, the bin recovers
  **0.0913 = 75.4 %**. **Quantization removes only ~1/4 of the residual leak.**
* Eval split (n = 147) agrees: `v0` alone 0.8725, `(bin, v0)` linear 0.9552.
  ⚠️ There the per-bin regressor **overfits** (OOF 0.8886 < linear 0.9552) — at
  n = 147 there is not enough data per bin. Named because an in-sample-only report
  would have quoted 0.9683 and overstated recoverability.
* **Can the model just guess the bin from its own speed?** Partly: out-of-fold
  nearest-centroid reads the bin off `v0` at **0.5840** vs a **0.3486** majority
  baseline. So 42 % of the time the bin says something `v0` does not.

### (i) Residual after quantization — the ego-coupling panel, raw vs quantized

| | corr(·, v0) | −v0 mean | −v0 median | −v0 p95 | differs > 1 m/s | **below v0** |
|---|---|---|---|---|---|---|
| `v_hi` **RAW** | 0.9375 | +1.135 | +0.401 | +6.443 | 52.7 % | **34.8 %** |
| `quantize_up(v_hi)` | 0.9167 | +3.598 | +3.160 | +9.259 | 83.7 % | **8.4 %** |

(The prior n = 1,200 panel read 0.941 / +1.08 / +0.42 / +6.12 / 51.5 %; this is the
same quantities at full n. No conflict.)

⭐ **The last column is the finding, and it was not on the brief's list.** The raw
`v_hi` is **below the ego's current speed on 34.8 % of clips** — a "maximum speed"
input that is already exceeded at t0 is not a ceiling, it is noise wearing a
ceiling's name. Snapping up cuts that to **8.4 %**. This, not the leak arithmetic,
is the strongest argument for quantizing.

### (iii) Bin occupancy — neither degenerate nor unique-per-clip

**H = 2.480 / 3.000 bits**, max bin **34.9 %** (50 km/h), **8 / 8 bins used**.

| 20 | 30 | 50 | 70 | 80 | 100 | 120 | 130 km/h |
|---|---|---|---|---|---|---|---|
| 809 (17.7 %) | 932 (20.4 %) | 1594 (34.9 %) | 619 (13.5 %) | 165 (3.6 %) | 229 (5.0 %) | 138 (3.0 %) | 86 (1.9 %) |

Neither failure mode fires: no bin holds "most of the mass" (34.9 % is the EU urban
default, which is the right answer), and bins are far from unique per clip.

---

## 3 · ⛔ THE MANDATORY EGO-ONLY ARM CONTROL — the channel is **NOT** an echo

**What this is:** linear readouts on scalars, 5-fold OOF, paired episode-cluster
bootstrap (`taniteval.ci`), **cluster = clip** (one window per clip).
⛔ `overlapping_holdout_se` is not used anywhere.

⛔⛔ **NAME THE VARIANCE QUESTION.** These intervals answer *"how much would this
difference move if a different sample of CLIPS were drawn from this corpus?"* They
say **nothing** about seed variance or split variance. A separated CI from a
one-seed arm is **necessary, not sufficient** — a pure replicate produced
"separated" differences on **6 of 42** cells (14.3 %) in this programme.

⚠️ **The longitudinal targets are contaminated by construction for the RAW arm**:
`v_hi` *is* the max of the ego's future speed over 2–6 s, so "raw beats ego" there
is a tautology. **The size of the tautology is the leak** — that is why it is
reported — but it must never be read as a capability claim.

**n = 4,572. d printed with every arm.**

### LONGITUDINAL (leads — this is a longitudinal lever) · mean |error|, lower better

| target | EGO only (d=1) | +QUANTIZED (d=3) | +RAW (d=2) | shuffled control (d=3) |
|---|---|---|---|---|
| future speed 2 s | 1.0890 [1.0575, 1.1210] | −0.2337 [−0.2595, −0.2088] **sep** | −0.4671 [−0.4954, −0.4403] **sep** | +0.0002 [−0.0006, +0.0010] not sep |
| future speed 4 s | 2.0272 [1.9718, 2.0815] | −0.6295 [−0.6802, −0.5804] **sep** | −1.2005 [−1.2504, −1.1517] **sep** | +0.0005 not sep |
| future speed 6 s | 2.6232 [2.5507, 2.6952] | −0.8335 [−0.8981, −0.7698] **sep** | −1.5868 [−1.6494, −1.5242] **sep** | +0.0002 not sep |
| along-track 2 s | 1.0857 m | −0.1576 **sep** | −0.3142 **sep** | +0.0002 not sep |
| along-track 4 s | 4.2262 m | −0.8828 **sep** | −1.6236 **sep** | +0.0007 not sep |
| along-track 6 s | 8.9494 m | −2.1132 [−2.3187, −1.9118] **sep** | −3.7023 **sep** | +0.0000 not sep |

* **Not an echo.** The quantized channel beats ego-only on every longitudinal
  target, separated, and retains **52.5 %** (speed 6 s) to **57.1 %**
  (along-track 6 s) of what the raw oracle delivers.
* ⭐ **The shuffle control collapses EXACTLY onto ego-only** on all six targets
  (not separated). The gain is clip-specific information, not extra capacity.
* Same-breath non-zero control: the constant-only arm (d=0) reads **5.87 m/s** and
  **35.6 m** — the instrument is alive.
* **Distance-keeping / TTC: NOT COMPUTABLE, n = 0.** It needs `obstacle.offline`
  agent boxes to form a gap; those are not in the v8 label record, and
  `tanitad.eval.constraints` computes the clearance ceiling from a **model**
  trajectory, which no arm here produces. Reported, not omitted.

### LATERAL · mean |error|, lower better — the channel is inert-to-harmful

| target | EGO only | +QUANTIZED | +RAW | shuffled control |
|---|---|---|---|---|
| cross-track 6 s | 6.2442 m [5.9814, 6.5193] | **+0.0081** [+0.0020, +0.0139] sep | +0.0236 sep | **+0.0507** sep |
| cross-track 4 s | 2.9124 m | +0.0023 sep | +0.0093 sep | +0.0221 sep |
| curvature 6 s | 0.0092 1/m | +0.0000 sep | +0.0000 sep | +0.0001 sep |
| yaw-rate 6 s | 0.0495 rad/s | +0.0000 not sep | −0.0000 sep | +0.0003 sep |

⚠️ **Separated but negligible, and the sign is the wrong way.** +0.0081 m on a
6.24 m baseline is **0.13 %** — the cost of one irrelevant feature's variance, not
a lateral effect. This is exactly the degenerate-separation case the estimator's own
docstring warns about: quoting "separated" here without the magnitude would mislead.

⭐ **The shuffle column is what makes this readable.** Here — unlike in the
longitudinal family — the shuffled channel does **not** collapse onto ego-only; it
costs **+0.0507 m**, **6.3× more** than the real channel's +0.0081. So the real
ceiling *does* carry a little clip-specific lateral information (enough to be much
cheaper than noise), but not enough to pay for its own variance. **A longitudinal
lever is longitudinally useful and laterally near-inert. That is the correct
outcome, and the shuffle control is the only reason we can say "near-inert" rather
than "harmful".**

### TACTICAL · accuracy, higher better

| | classes | majority | EGO (d=1) | +QUANT (d=3) | +RAW (d=2) | shuffled |
|---|---|---|---|---|---|---|
| `a_tac.lon` | 7 | 0.2719 | 0.3745 [0.3600, 0.3882] | **0.4383** (+0.0639 [+0.0477, +0.0807] **sep**) | 0.4921 (+0.1177 **sep**) | 0.3718 (−0.0026 sep) |
| `a_tac.lat` | 5 | 0.6470 | 0.6470 | 0.6470 (+0.0000 not sep) | 0.6470 (not sep) | 0.6470 |

⭐ **`a_tac.lon` is the headline non-tautological result**: the longitudinal
*manoeuvre token* — CRUISE / ACCELERATE / BRAKE_TO / ADAPT_SPEED_FOR_CURVE — is
**+6.4 accuracy points** more predictable with the quantized ceiling than with the
ego's speed alone, and the quantized channel captures **54 %** of the raw oracle's
gain. ⚠️ `a_tac.lat` is **degenerate**: every arm collapses to the majority class,
so this cell says *the readout has no lateral signal*, not *the channel is useless
laterally*. Reported with its n and its reason.

### STRATEGIC · `a_str.token`, 5 classes, majority 0.5227

| EGO | +QUANT | +RAW | shuffled |
|---|---|---|---|
| 0.4983 [0.4838, 0.5136] | 0.5101 (+0.0118 [+0.0085, +0.0151] **sep**) | 0.5324 (+0.0341 **sep**) | 0.4978 (not sep) |

⛔⛔ **READ THIS ONE WITH ITS CONTROL OR NOT AT ALL.** The ego-only and quantized
arms are **below the constant-only baseline (0.5227)**. Only `+RAW` clears it. So
"+QUANT is separated over EGO" is a comparison between two arms that are *both worse
than predicting the majority class*. The separated interval is real; the capability
claim it looks like is not. This is precisely the "true but wrong for the reader"
failure, and it is flagged rather than quoted.

---

## 4 · THE BIT-IDENTICAL OFF PATH, AND THE MUTATION THAT PROVES IT CAN FAIL

`stack/tests/test_max_speed_input.py` — **29 tests, all passing** on a fixed seed.

⭐⭐ **DEMONSTRATED RED** (`code/mutation_control_demo.py`, output banked in
`raw/mutation_control_demo.txt`). The script removes the zero-init from
`MaxSpeedConditioner`, reruns the suite, and restores the file:

```
STEP 1  BASELINE (zero-init intact)   exit=0   29 passed
STEP 2  MUTATION (zero-init removed)  exit=1   2 failed, 27 passed
STEP 3  RESTORED                      exit=0   29 passed
        bytes restored EXACTLY: True
parity tests that went RED: test_ON_at_INIT_is_also_BIT_IDENTICAL,
                            test_the_edge_is_BIT_INERT_at_init
VERDICT: the bit-identity assertions HAVE TEETH -- an equality that can fail
```

⚠️ **The precise scope, because the three states differ and an ablation must never
conflate them:**
* `test_OFF_is_BIT_IDENTICAL_on_a_fixed_seed` stays **green under the mutation** —
  correctly. With the flag OFF *nothing is fed*, so the edge cannot fire whatever
  its init. That identity is **structural**, not zero-init, and is therefore not
  mutation-sensitive. Saying it "survived the mutation control" would be a false
  strength claim.
* `test_ON_at_INIT_is_also_BIT_IDENTICAL` is the **zero-init** identity and *is*
  mutation-sensitive. It is what makes an ON-vs-OFF training comparison attributable.
* The parity predicate itself asserts it compared **≥ 3 tensors** — a predicate over
  zero tensors returns True forever.
* A **withheld** ceiling (`valid = 0`) with a **live** edge still emits the bias
  path, and that is correct — "no limit known" must stay a distinct learnable input,
  not collapse onto "the limit here is 0 m/s". What is asserted is that the withheld
  delta does **not depend on the withheld value**, with a same-breath control showing
  the edge is alive when `valid = 1`.

⛔ **What this package could NOT prove:** that someone applied the patch correctly.
`refc_v3.py` has live siblings and was **not touched**; the four-hunk diff is filed
at `WIRING_DIFF.md` with verified anchors (against `a1f5ade`) and an acceptance
checklist. That is named as an escalation, not buried.

---

## 5 · THE OUTPUT-SCORED CONSTRAINT — wired, and it fires with the flag OFF

`stack/tanitad/eval/speed_limit_scoring.py` composes the pinned quantizer into
`tanitad.eval.constraints.speed_envelope_report`. Nothing here is ever fed to a
model — *an input that does not exist cannot echo* — so it is scoreable today. The
baseline below scores the **ego's own realised path** over the 2–6 s band:
**n = 4,572 clips / 187,452 steps, manoeuvre_rate 0.3530** (the vacuity gate).

**Posted limit ALONE — the instrument check, and the whole argument in two rows:**

| ceiling | frac_over | clips with any overshoot |
|---|---|---|
| **RAW `v_hi`** | **0.000000** | **0 / 4,572** |
| **quantized** | 0.005564 | 28 |

⭐ The raw ego-derived ceiling **cannot fire** — it *is* the max of the speed being
scored. Any non-zero there would mean the join is broken. The quantized ceiling
**can** fire. That is the difference the PI is buying.

**Combined `min(posted, kinematic 0.35 g)` — the realistic envelope:**

| | frac_over | 95 % CI | frac_under_when_allowed |
|---|---|---|---|
| quantized | 0.012046 | [0.009383, 0.014644] | **0.498288** (3,758 clips / 147,738 steps) |
| raw | 0.006482 | [0.004924, 0.008173] | — |

paired (quant − raw) **+0.0056 [+0.0036, +0.0077] separated** — the quantized
ceiling is stricter overall because it adds a real 130 km/h cap the raw value never
imposes. And the human driver **under-drives on 49.8 %** of steps where the
situation allows speed: that is the honest baseline any flag-ON arm must beat.

---

## 6 · ⚠️ THE LARGEST CAVEAT, MEASURED — the low-speed distortion

Snapping **up** from a stopped or crawling ego reports the **lowest** posted limit,
which is a property of the *traffic*, not of the *road*.

| road class | n | ceiling ≤ 30 km/h | median ceiling |
|---|---|---|---|
| intersection | 861 | **75.0 %** | 30 km/h |
| urban | 2,965 | 36.3 % | 50 km/h |
| highway | 746 | 2.7 % | 100 km/h |

38.1 % of all clips receive a ceiling ≤ 30 km/h. A deployed map service would return
**50 km/h** for an urban clip whose ego never exceeded 5 km/h. **This is a
train/deploy mismatch in the opposite direction to the optimism the quantization
fixes**, it is concentrated exactly where the ego is constrained by traffic, and it
is the one thing most likely to make a flag-ON arm learn the wrong association
("slow ego ⟹ low limit" rather than "low limit ⟹ slow ego"). It is not fixable
inside this design; it would need a genuine map. **Named, not hidden.**

---

## 7 · THREE SELF-CORRECTIONS, BANKED AT THE SITE

1. **A display precision became a data precision.** The `(v0, v_hi)` join stored
   `v_hi_ms` at the programme's 4 dp convention. That rounds **down** on ~48 % of
   clips by up to **4.995e-05 m/s**, and the constraint identity "the ego cannot
   exceed its own realised max" then fired on **2,203 of 4,572** clips. I predicted
   0 and got 2,203; the diagnostic (max overshoot 4.995e-05 = exactly the 4 dp
   bound) identified my own storage, not the data. Fixed to full precision, with the
   reason commented at the site. **Round at the point of printing, never at the point
   of storing.**
2. **A control that measured something else.** The first constraint run combined the
   posted ceiling with the **kinematic** 0.35 g ceiling and read frac_over = 0.0182
   for the "vacuity control". The vacuity identity holds only for the posted ceiling
   **alone**. Both configurations are now reported separately; quoting the combined
   raw number as "the vacuity control" would have been true-but-wrong-for-the-reader.
3. **A "restore" that did not restore the bytes.** `mutation_control_demo.py`
   round-tripped the module through `read_text` / `write_text`, which strips CR on
   read and re-adds `os.linesep` on write. Its restore therefore rewrote a
   17,764-byte LF file as **18,114 bytes of CRLF** — content identical, **md5
   different** — and `pod_currency_audit.py`, which compares a box against the
   ref's *blobs*, would have called the file **POD-DIVERGED**. Caught by md5-ing
   repo against mirror at end of turn, not by the tests (all 29 stayed green
   throughout). Now `read_bytes` / `write_bytes`, with a content assertion on the
   restore itself (`bytes restored EXACTLY: True`) wired into the verdict.

---

## 8 · WHAT THE PI GETS FROM THIS INPUT THAT HE DOES NOT ALREADY HAVE

> **A ceiling the planner can be *wrong about*.** `v0` already tells the model how
> fast it is going and already predicts `v_hi` at R² 0.879; what the quantized
> posted-limit channel adds is the *road's* permission — worth **+6.4 accuracy
> points** on the longitudinal manoeuvre token over ego-only, and, uniquely, a bound
> the trajectory can actually violate (28 clips) and be scored against, which the
> raw ego-derived value can never be (0 clips, by construction).

---

## DELIVERABLE MANIFEST

| artifact | where | only one place? |
|---|---|---|
| `max_speed_input.py` — channel, pinned ladder, units refusal | `repo:stack/tanitad/refs/` | no (committed `6ae8acf`) |
| `speed_limit_scoring.py` — output-scored constraint wiring | `repo:stack/tanitad/eval/` | no |
| `test_max_speed_input.py` — 29 tests incl. OFF proof | `repo:stack/tests/` | no |
| `WIRING_DIFF.md` — **ESCALATION, patch not applied** | `repo:<pkg>/` | no |
| `extract_vhi_v0.py`, `quantization_panel.py`, `arm_control_panel.py`, `constraint_baseline.py`, `mutation_control_demo.py` | `repo:<pkg>/code/` | no |
| `vhi_v0_{train,eval}.json`, `quantization_panel_{train,eval}.json`, `arm_control_train.json`, `constraint_baseline_train.json`, `mutation_control_demo.txt` | `repo:<pkg>/raw/` | no |

`<pkg>` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-max-speed-input/`

## INTEGRATION ITEMS (escalated, not filed in a README)

1. **`WIRING_DIFF.md` needs an owner.** Four hunks in `refc_v3.py` + one loader read
   in `refc_v3_train.py`. Anchors verified at `a1f5ade`; re-derive if they moved.
2. **The real arm is unrun.** Train `refcv3` OFF vs `quantized` vs `raw`, with the
   **shuffle** and **withhold** controls beside the headline. Without them "the arm
   improved" cannot be separated from "the arm gained a parameter".
3. **Distance-keeping / TTC is uncomputed (n = 0)** — needs the `obstacle.offline`
   join, which is a different owner's artifact.
