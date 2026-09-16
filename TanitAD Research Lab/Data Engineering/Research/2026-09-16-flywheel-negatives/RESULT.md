# Caption-absence is a negative — the ruling, wired, and the cost measured

**2026-09-16 · DataFlyWheel · agent worktree `tanitad-wt-flywheel-neg`**

> ⚠️ **ESCALATION — the ruling is implemented and gated, and the biggest
> measured EXPOSURE sits on the family the widening added, not on the four the
> PI named first.** On the PI's own example, traffic lights:
> **692 / 867 = 79.8 % [77.0, 82.4]** of the clips that were asked the
> traffic-light grounding question **and showed a visible light** carry no
> traffic-light token. That is 692 clips = 15.1 % of the corpus, 2,768 cells
> that the new policy turns into negatives.
>
> ⛔ **READ THIS NUMBER CORRECTLY — it is a PRESENCE rate, not a false-negative
> rate.** The probe is a *necessary condition* ("was a light visible at all"),
> and the token is `TRAFFIC_LIGHT_REACT_*` — a **reaction**, not a detection. A
> green light governing another lane is a present light and a **correctly
> absent** reaction. So 79.8 % bounds how many of these new negatives *could*
> be wrong; **it does not establish that any of them are.** The probe cannot
> separate "no reaction happened" from "a real reaction the caption never
> narrated", and nothing on this box can. Nothing here asks to reverse the
> ruling; it asks whether the traffic-light family deserves a carve-out on the
> size of that unresolved exposure, and gives the numbers to decide with.

---

## 0. The ruling, and the precedent it overrides

**PI, 2026-09-16, verbatim:**

> *"fix the negatives with the flywheel agent for those 4 tokens, it is ok to
> intepret the absence of vlm caption as negatives. If there is no label about a
> red traffic light, that means there is no red traffic light in th
> eenvironment"*

**Widened the same day, verbatim:**

> *"the interpretation of absence as negatiove should be not only for the four
> missing"*

⇒ the policy applies to **all 15 `vlm-cot`-provenance tokens**, not only the
four that had zero negatives.

**The precedent being overridden, recorded once and not re-litigated.**
`stack/tanitad/data/v7_labels.py` adopted a provenance-derived negative policy
on MEASURED grounds: trusting a *declaration* about which tokens need perception
would have **supervised 4,534 unknowable lane-change negatives as true**
(MEASURED 2026-08-30), and 3,574 of 4,572 clips were never asked the
traffic-light grounding question at all, so their
`scene.traffic_light_visible == False` is NOT-PROBED rather than absent. The PI
has now ruled deliberately in the other direction. Both readings now sit side by
side with numbers: the old policy's reason above, the new policy's measured cost
in §4. A future reader can re-decide from the table rather than from the
argument.

**Corpus.** `s2_labels_v8.0_train.jsonl.gz`, md5 `fa89ea55dfce68403eb30300e57852ab`,
4,572 clips, schema `s2-geom-v7`, vocab `v7`. ⛔ **Not modified.** The policy is a
separate sidecar that names this md5 and is refused against any other.

---

## 1. What was built

| artifact | what it is |
|---|---|
| `cot_absence_negative_v8.0_train.json.gz` (md5 `fc2ee151620d2c7dbc2416f7f17481f1`) | the sidecar: 4,572 rows × 15 bits, keyed by `sha256(clip_id)[:12]`, plus the stamped policy record |
| `stack/scripts/build_cot_negative_sidecar.py` | its builder |
| `stack/tanitad/data/v7_labels.py` | `negatives="cot-absence-negative"`, `load_cot_negative_sidecar`, `assert_sidecar_matches_presence`, the manifest stamp |
| `stack/scripts/refc_v3_train.py` | `--cot-negative-sidecar` (the opt-in is the path; there is no boolean) |
| `stack/tests/test_cot_negative_policy.py` | 22 tests, 11 of them the mutation table |
| `stack/scripts/audit_cot_negative_guards.py` | the guard-removal audit (§6) |

**The policy is data, not a default.** The sidecar's `meta` carries the ruling
verbatim, its widening, the date, `ruled_by: PI`, the rule, the 15 token names,
the source blob's md5/records/schema/vocab, the digest algorithm *and its scope*
(the join_meta M18 rule — a recorded digest that does not say what it was taken
over is a number, not a verification), the census before and after, and the
headline measurement, so the cost travels with the permission. ⚠️ That embedded
copy states the traffic-light channel as *"a NECESSARY condition (no visible
light ⇒ no light to react to) and therefore an upper bound"* — the same framing
§4.5 spells out, so the file and this report cannot drift apart on what the
number means.

**Opt-in by name, on the `allow_oracle_nav` pattern.** There is no flag that
turns this on; a consumer must name the file. `load_cot_negative_sidecar(path,
manifest)` refuses unless `meta.source_blob_md5 == manifest.md5` (six copies of
this blob exist under three roots with differing md5s), and returns a **new
manifest carrying the stamp** — policy id, sidecar path, sidecar md5, blob md5,
ruling text, token list, before/after counts. That dict lands in `config.json`
via `LabelManifest.to_dict()` and again via `TacGoalEmitter.provenance()`, so
**a run that used this policy is identifiable from its own artifacts.** The
reverse is refused too: a sidecar passed with any other `negatives` policy
raises rather than being silently ignored — an ignored permission is a config
that lies about its own loss.

⛔ Clip ids appear only as `sha12` in every repo-bound file here. Verified by
test (`test_clip_ids_are_not_in_the_clear`) and by a re-scan of the shipped
sidecar and the banked probe features: **0 plaintext clip ids**.

---

## 2. Before → after: the full 22-token census

MEASURED via `tanitad.data.v7_labels.goal_supervision_census`, on the blob above,
at each record's own anchor. **Positives are unchanged — this policy only
decides absence.** `neg → 4,572 − pos` and `ignored → 0` for every token.

| token | provenance | pos | neg **before** | neg **after** | ign **before** | ign **after** | Δ neg |
|---|---|---:|---:|---:|---:|---:|---:|
| FOLLOW_LANE | geometry | 3,629 | 943 | 943 | 0 | 0 | — |
| TURN_L | geometry | 275 | 4,297 | 4,297 | 0 | 0 | — |
| TURN_R | geometry | 259 | 4,313 | 4,313 | 0 | 0 | — |
| YIELD_FOR_TURN_L | geometry | 21 | 4,551 | 4,551 | 0 | 0 | — |
| YIELD_FOR_TURN_R | geometry | 20 | 4,552 | 4,552 | 0 | 0 | — |
| STOP_POINT | geometry | 327 | 4,245 | 4,245 | 0 | 0 | — |
| SPEED_BAND | geometry | 4,572 | 0 | 0 | 0 | 0 | — |
| **YIELD** ★ | vlm-cot | 609 | **0** | **3,963** | 3,963 | 0 | +3,963 |
| **CORRIDOR_OFFSET** ★ | vlm-cot | 860 | **0** | **3,712** | 3,712 | 0 | +3,712 |
| **GAP_TARGET** ★ | vlm-cot | 368 | **0** | **4,204** | 4,204 | 0 | +4,204 |
| **REACT_ON_ONCOMING** ★ | vlm-cot | 333 | **0** | **4,239** | 4,239 | 0 | +4,239 |
| EVADE_IN_CORRIDOR | vlm-cot (+alpamayo-structured) | 240 | 20 | 4,332 | 4,312 | 0 | +4,312 |
| OVERTAKE_VEHICLE | vlm-cot | 20 | 3,635 | 4,552 | 917 | 0 | +917 |
| MERGE | vlm-cot | 79 | 3,629 | 4,493 | 864 | 0 | +864 |
| TAKE_EXIT_L | vlm-cot (+alpamayo-structured) | 21 | 369 | 4,551 | 4,182 | 0 | +4,182 |
| TAKE_EXIT_R | vlm-cot | 128 | 290 | 4,444 | 4,154 | 0 | +4,154 |
| TRAFFIC_LIGHT_REACT | vlm-cot | 18 | 761 | 4,554 | 3,793 | 0 | +3,793 |
| TRAFFIC_LIGHT_REACT_RED | vlm-cot | 376 | 403 | 4,196 | 3,793 | 0 | +3,793 |
| TRAFFIC_LIGHT_REACT_YELLOW | vlm-cot | 22 | 757 | 4,550 | 3,793 | 0 | +3,793 |
| TRAFFIC_LIGHT_REACT_GREEN | vlm-cot | 363 | 416 | 4,209 | 3,793 | 0 | +3,793 |
| LANE_CHANGE_L | vlm-cot | 23 | 290 | 4,549 | 4,259 | 0 | +4,259 |
| LANE_CHANGE_R | vlm-cot | 15 | 282 | 4,557 | 4,275 | 0 | +4,275 |

★ = the four the PI named first. Over the 15 CoT tokens the 68,580 (clip, token)
cells split 3,475 positive / 10,852 already-negative-by-entailment / **54,253
ignored**, and it is those **54,253 cells that become supervised negatives**.

**⇒ Every one of the 22 tokens is now fully supervised: each clip is positive or
negative for every token, and the IGNORE state disappears from the goal head.
The three-state mask the model was going to need collapses to two states.**
`IGNORE_W` survives only for the out-of-band window case, which is a different
rule (a window outside the record's tactical band describes nothing).

---

## 3. The imbalance is now the live issue

With the ignore state gone, the negative counts are no longer a policy question
but a class-balance one. Computed through the **existing** `goal_pos_weight`
path (`n_neg / n_pos`, capped at 50) rather than a new mechanism — the cap is
not cosmetic: uncapped, a single `LANE_CHANGE_R` positive would carry ~304× the
weight of a negative and dominate the batch gradient.

| token | pos | neg | positive rate | pos_weight **before** | pos_weight **after** | uncapped | on the cap? |
|---|---:|---:|---:|---:|---:|---:|:--:|
| SPEED_BAND | 4,572 | 0 | 1.0000 | 0.000 | 0.000 | — | masked |
| FOLLOW_LANE | 3,629 | 943 | 0.7937 | 0.260 | 0.260 | 0.3 | |
| CORRIDOR_OFFSET | 860 | 3,712 | 0.1881 | **0.000** | **4.316** | 4.3 | |
| YIELD | 609 | 3,963 | 0.1332 | **0.000** | **6.507** | 6.5 | |
| TRAFFIC_LIGHT_REACT_RED | 376 | 4,196 | 0.0822 | 1.072 | 11.160 | 11.2 | |
| GAP_TARGET | 368 | 4,204 | 0.0805 | **0.000** | **11.424** | 11.4 | |
| TRAFFIC_LIGHT_REACT_GREEN | 363 | 4,209 | 0.0794 | 1.146 | 11.595 | 11.6 | |
| REACT_ON_ONCOMING | 333 | 4,239 | 0.0728 | **0.000** | **12.730** | 12.7 | |
| STOP_POINT | 327 | 4,245 | 0.0715 | 12.982 | 12.982 | 13.0 | |
| TURN_L | 275 | 4,297 | 0.0601 | 15.625 | 15.625 | 15.6 | |
| TURN_R | 259 | 4,313 | 0.0566 | 16.653 | 16.653 | 16.7 | |
| EVADE_IN_CORRIDOR | 240 | 4,332 | 0.0525 | 0.083 | 18.050 | 18.1 | |
| TAKE_EXIT_R | 128 | 4,444 | 0.0280 | 2.266 | 34.719 | 34.7 | |
| MERGE | 79 | 4,493 | 0.0173 | 45.937 | 50.000 | 56.9 | ⚠️ |
| LANE_CHANGE_L | 23 | 4,549 | 0.0050 | 12.609 | 50.000 | 197.8 | ⚠️ |
| TRAFFIC_LIGHT_REACT_YELLOW | 22 | 4,550 | 0.0048 | 34.409 | 50.000 | 206.8 | ⚠️ |
| YIELD_FOR_TURN_L | 21 | 4,551 | 0.0046 | 50.000 | 50.000 | 216.7 | ⚠️ |
| TAKE_EXIT_L | 21 | 4,551 | 0.0046 | 17.571 | 50.000 | 216.7 | ⚠️ |
| OVERTAKE_VEHICLE | 20 | 4,552 | 0.0044 | 50.000 | 50.000 | 227.6 | ⚠️ |
| YIELD_FOR_TURN_R | 20 | 4,552 | 0.0044 | 50.000 | 50.000 | 227.6 | ⚠️ |
| TRAFFIC_LIGHT_REACT | 18 | 4,554 | 0.0039 | 42.278 | 50.000 | 253.0 | ⚠️ |
| LANE_CHANGE_R | 15 | 4,557 | 0.0033 | 18.800 | 50.000 | **303.8** | ⚠️ |

**⚠️ 9 of 22 tokens now sit ON the cap** (3 did before). For those nine the
**cap, not the data, sets the weight** — the objective no longer reflects the
measured imbalance, it reflects a constant. Do not raise the cap to "fix" it: at
300 one positive outweighs the batch. The honest options are a different
imbalance treatment (focal loss, negative subsampling) or accepting that these
nine are weight-saturated; both are decisions, and both are outside this task.

⛔ Note the interaction with the existing head mask. `tac_goal_head.mask_report`
switches off every class with positives but **no supervised negative** — 5 of 22
on the old policy (YIELD, CORRIDOR_OFFSET, GAP_TARGET, REACT_ON_ONCOMING and
SPEED_BAND). Under the new policy only `SPEED_BAND` still qualifies (4,572 /
4,572 positive), so the head goes **17 / 22 trainable → 21 / 22**: the **four
tokens the PI named first become trainable at all**, MEASURED. That is the
intended effect of the ruling and it is stamped in `tac_goal_stats`.

---

## 4. ⭐ How often the assumption is wrong — MEASURED

### 4.1 The instrument, and why it is shaped this way

Every probe is a **necessary condition** built from a channel the caption never
touched, not a detector:

- **ego kinematics** — `egomotion_alpamayo/<clip>.parquet` (world pose,
  velocity, quaternion at ~10 Hz), 4,572/4,572 clips available;
- **3D agent boxes** — the B1 TRAIN `obstacle.offline` → ego-frame agent join
  (`b1train_agents.jsonl.xz`, md5 `1c985e6d6ad34e605c4ebd30cb353558`, 849,263
  lines), 4,318/4,572 clips in the tactical window, 4,390 in the wide window.
  ⛔ A clip the join never covered is scored **UNKNOWN**, never "no agents" —
  folding a coverage gap in as an empty scene would manufacture exactly the kind
  of unevidenced negative this whole exercise is measuring.

Shape: `probe does not fire ⇒ the token is FALSE`. So

| quantity | meaning |
|---|---|
| **S** | fire rate on the **labelled positives** — *the instrument control* |
| **F** | fire rate on the caption-**unlabelled** clips |
| **f₀** | fire rate on a set of **certain negatives** |
| **upper bound** | `F / S` — not `F`: `S < 1` means the probe misses that share of true positives among the unlabelled too, so the raw rate under-states |
| **point estimate** | `(F − f₀) / (S − f₀)`, where f₀ exists |

⛔ **A probe with `S < 0.90` is VOID** and its bound is not quoted as a result.

**Certain-negative sets**, both from non-caption channels:
1. the frozen `TACTICAL_GOAL_EXCLUSIVE` table (a clip carrying GREEN cannot be
   RED), and
2. the **empty-scene** construction: no agent of any class with `0 < cx ≤ 80 m`,
   `|cy| ≤ 15 m` in *any* frame of the window. YIELD / GAP_TARGET /
   REACT_ON_ONCOMING / EVADE all need something to yield to, follow, meet or
   avoid.

### 4.2 The pipeline control — it passes

Three **geometry**-provenance tokens have a truth ego geometry knows
independently. Scored with the same features and the same time base:

| control token | probe | S | f₀ (certain negative) |
|---|---|---|---|
| STOP_POINT | `v_min ≤ 0.5 m/s` | **327 / 327 = 1.000** | 32/3,629 = 0.009 (FOLLOW_LANE) |
| TURN_L | `peak yaw ≥ +20°` | 257 / 275 = 0.935 | 9/259 = 0.035 (TURN_R) |
| TURN_R | `peak yaw ≤ −20°` | 249 / 259 = 0.961 | 17/275 = 0.062 (TURN_L) |

⇒ the time base (tactical band = clip seconds **[10, 14]**, derived from
`t0_s = 8.0` and `bands.tactical_s = [2, 6]`, both constant on all 4,572
records) and the feature extraction are sound. **Without this, every number
below would be noise wearing a decimal point.**

### 4.3 The window, and why two are reported

MEASURED on this blob: **492/609 YIELD, 859/860 CORRIDOR_OFFSET, 333/333
REACT_ON_ONCOMING and 288/368 GAP_TARGET carry `time_basis: "untimed"`**, and
their `t_nominal_s` is the band-midpoint *fallback*
(`t_nominal_provenance: "band-midpoint (PI 2026-08-28)"`), not an observation.
An untimed token is a claim about the **clip**, so scoring it only inside
[10, 14] asks the probe to find evidence in a window the label never promised.
Both windows are therefore reported: **tactical [10, 14] s** and **wide
[4, 18] s** (93.3 % of clips have join coverage over the wide one).

The window choice is itself measurable. The empty-scene certain-negative set
is **contaminated** in the tactical window — it contains 60/609 YIELD and
**62/333 (18.6 %) REACT_ON_ONCOMING labelled positives**, i.e. clips where the
caption asserts an oncoming reaction and the 3D-box channel sees no agent ahead
at all. In the wide window that falls to 23/609 (3.8 %) and 12/326 (3.7 %).
⇒ the tactical-window empty-scene f₀ is **VOID**; the wide-window one carries a
~4 % contamination, stated rather than hidden, and biases f₀ **down**, i.e. the
point estimates below are **conservative (too high)**.

### 4.4 Per-token results

Wide window unless stated. `n` is the denominator of `F`. CIs are Wilson 95 %.

| token | probe (necessary condition) | control **S** | **F** (unlabelled) | **f₀** | **upper bound** | point estimate | verdict |
|---|---|---|---|---|---|---|---|
| **GAP_TARGET** | a vehicle in the ±3.0 m corridor ahead for ≥5 % of the window | **0.906** (329/363) [0.872, 0.932] | **0.662** (2,665/4,027) [0.647, 0.676] | 0 † | **≤ 0.730** | — † | **PASS** |
| **REACT_ON_ONCOMING** | a vehicle with opposed heading (>135°) ahead within 50 m, \|cy\| ≤ 6 m | **0.902** (294/326) [0.865, 0.930] | **0.466** (1,893/4,064) [0.451, 0.481] | 0 † | **≤ 0.516** | — † | **PASS** |
| **CORRIDOR_OFFSET** | curvature-detrended lateral residual ≥ 0.10 m | **0.961** (826/860) [0.945, 0.972] | **0.875** (3,249/3,712) [0.864, 0.886] | none | ≤ 0.911 | — | **PASS but UNINFORMATIVE** |
| **YIELD** | speed drop ≥ 1.0 m/s in the window, or `v_min ≤ 2.0 m/s` | 0.900 (548/609) [0.873, 0.921] | **0.760** (3,010/3,963) [0.746, 0.773] | **0.636** (210/330) | ≤ 0.844 | **≈ 0.47** | **MARGINAL** (S = 0.8998) |
| **EVADE_IN_CORRIDOR** | a static vehicle or a VRU in the corridor **and** lateral residual ≥ 0.05 m | 0.806 (191/237) | 0.449 (1,863/4,153) | 0.211 (4/19) | ≤ 0.557 | ≈ 0.40 | **VOID** (S < 0.90) |
| **OVERTAKE_VEHICLE** | a moving vehicle ahead in the corridor the ego closes on / passes, and lateral residual ≥ 0.05 m | **0.263** (5/19) | 0.379 | 0.368 (n=3,477) | — | — | **VOID** — the probe cannot find the labelled overtakes |
| MERGE, TAKE_EXIT_L/R, LANE_CHANGE_L/R, TRAFFIC_LIGHT_REACT (colourless) | — | — | — | — | — | — | **UNVERIFIED** (§4.6) |
| TRAFFIC_LIGHT_REACT_RED / _YELLOW / _GREEN | grounding-box channel (§4.5) | **0.902** (175/194) | **0.798** (692/867) [0.770, 0.824] — a light-PRESENT rate, an exposure CEILING, **not** an FN rate | — | — | — | **MEASURED, semi-independent** |

† **f₀ = 0 is a tautology here, not a result.** The probe requires an agent and
the certain-negative set has none, so `(F − f₀)/(S − f₀)` collapses to the upper
bound `F/S`. The number in the "upper bound" column is the honest quantity for
these two; there is no identified point estimate.

**Reading the three that pass.**

- **GAP_TARGET ≤ 73 %.** Up to three quarters of the 4,204 clips about to be
  supervised as "no gap target" have a vehicle in the corridor ahead. The probe
  is only a *necessary* condition (having a lead vehicle is not the same as
  targeting a gap), so the true rate is lower — but nothing in the data
  identifies how much lower.
- **REACT_ON_ONCOMING ≤ 52 %.** Same shape: an oncoming vehicle within 50 m is
  necessary, not sufficient.
- **CORRIDOR_OFFSET: the bound is ~0.91 and therefore says nothing.** A
  curvature-detrended residual of 0.10 m is met by 87.5 % of unlabelled clips.
  Tightening the threshold loses the control (`S` 0.869 at 0.20 m, 0.730 at
  0.40 m) faster than it gains discrimination — the upper bound never drops
  below 0.90 anywhere in the sweep. ⇒ **the false-negative rate of
  CORRIDOR_OFFSET is not bounded by this instrument.** Reported as measured and
  uninformative rather than as a low number.
- **YIELD ≈ 47 %** is the one identified point estimate (f₀ is real, not
  tautological, because the probe is kinematic-only). Caveat, stated: the
  empty-scene population skews rural/high-speed, so its deceleration base rate
  under-states f₀ for the urban population, which pushes the estimate **up**;
  and the tactical-window control failure (§4.3) means the wide window is
  scoring some evidence outside the band the label describes.

### 4.5 Traffic lights — the PI's own example, and the largest single EXPOSURE

The only non-caption signal for the traffic-light family on this box is the
Alpamayo **grounding-box** channel (`scene.asked` / `scene.traffic_light_visible`
/ `scene.boxes_norm1000`). It is a **different question** from the
chain-of-thought narration but the **same annotator model**, so it is
**SEMI-INDEPENDENT**, and it is a necessary condition (no visible light ⇒ no
light to react to).

| quantity | value |
|---|---|
| clips in the corpus | 4,572 |
| clips **asked** the traffic-light grounding question | **998** (21.8 %) — the other 3,574 are NOT-PROBED |
| of those, with a **visible** light | 867 |
| **instrument control** — TL-token positives that were asked, and had a visible light | **175 / 194 = 0.902** ✔ |
| **clips with a visible light and NO traffic-light token** | **692 / 867 = 0.798** [0.770, 0.824] |

⇒ On the subpopulation where an independent-ish channel can see the answer,
**~80 % of visible traffic lights are not mentioned by the caption.** Under the
new policy each of those 692 clips becomes a supervised negative for **all
four** traffic-light tokens: 2,768 cells.

⛔⛔ **WHAT THIS IS, PRECISELY — AND WHAT IT IS NOT.** 79.8 % is the rate at
which a **light was PRESENT and no token was written**. The token is
`TRAFFIC_LIGHT_REACT_*`: a **reaction**, not a detection. The two come apart
routinely and legitimately —

* a green light governing **another lane or another approach** is a present
  light and a **correctly absent** reaction;
* a light 80 m away the ego never reaches inside the band, likewise;
* a light the ego simply drives under at speed with no change of behaviour.

⇒ **79.8 % is an upper bound on the exposure, not a count of wrong labels.** It
says *at most* 692 of these clips carry a reaction the caption failed to
narrate; it does **not** say that any of them do. The probe cannot separate
**"no reaction happened"** (the label is right) from **"a real reaction the
caption never narrated"** (the label is wrong) — and no channel on this box can:
separating them needs the light's **state, lane assignment and range** against
the ego's own longitudinal profile, i.e. pixels plus a map. That separation is
**UNVERIFIED and unbuildable here**, and it is the largest open question this
package leaves.

⛔ The rate may also **not** be extrapolated to the 3,574 clips never asked the
question — that would be the same "absence is evidence" move being audited here.
What is established is a bound on a defined subpopulation: **692 clips (15.1 %
of the corpus) receive traffic-light negatives whose correctness the box channel
cannot confirm.**

Two further scope limits, stated rather than absorbed: the colour is knowable
only from pixels, so the **per-colour RED / YELLOW / GREEN** rates are
**UNVERIFIED**; and the grounding-box channel is **SEMI-INDEPENDENT** — a
different question, but the same annotator model, so a systematic blind spot
could affect both.

### 4.6 What could not be probed honestly — UNVERIFIED, with the reason

| token | why no probe exists on this box |
|---|---|
| **LANE_CHANGE_L / _R** | needs a **lane reference**. `v7_labels.effective_mask` records the measured consequence (D-NUDGE-ABSORB): with no lane detector a lane change is absorbed into `NUDGE` or `LANE_KEEP` depending on yaw, so lateral offset alone cannot separate a lane change from a nudge. A lane-width proxy (signed `lat_peak_m ≥ 2.5 m`) fires on plain road curvature — it reads S = 0.435 / 0.533 on the positives against F = 0.354 / 0.380 on the unlabelled, i.e. **it barely separates them**. Reported for scale, **not** as a bound. |
| **MERGE**, **TAKE_EXIT_L / _R** | need **road topology**. A merge and an exit are defined by lane structure, and geometrically an exit is a bend. No map for these clips — `map.xodr` exists only for the NuRec scenes. |
| **TRAFFIC_LIGHT_REACT** (colourless, 18 positives) | no non-caption signal for the colourless case; the box channel cannot distinguish it from the coloured tokens. |
| per-colour RED / YELLOW / GREEN rates | colour is a pixel question; §4.5 bounds light-PRESENCE for the family, not the colour and not the reaction. |
| **the REACTION half of every TL token** | §4.5 measures whether a **light was there**. Whether the ego **reacted to it** — the thing the token actually claims — needs the light's state, lane assignment and range against the ego's longitudinal profile. Not buildable here. ⇒ the 79.8 % is a ceiling on exposure; the fraction of it that is a real mislabel is **UNVERIFIED**. |

**That is 6 of 15 tokens with no probe at all, a 7th (OVERTAKE_VEHICLE) whose
probe failed its control, and a partial for the traffic-light family — bounded
on presence, unmeasured on reaction.** For none of these is the false-negative
rate under the new policy a known quantity. Their combined positive count is
196 of 4,572 (779 including the traffic-light family).

---

## 5. What this buys and what it costs

**Buys.** The ignore state is gone and **54,253 cells become trainable**. The
head goes 17/22 → 21/22 trainable logits: the four tokens the PI named first
could only ever learn "always 1" (zero supervised negatives, `pos_weight` 0.0,
masked off by `mask_report`) and now have a real objective. This is a real and
large gain, and it is the reason the ruling was made.

**Costs, measured.** Where a probe could be built and passed its control, the
share of new negatives that *may* be wrong is **≤ 73 % (GAP_TARGET)**, **≤ 52 %
(REACT_ON_ONCOMING)**, **≈ 47 % (YIELD — the one identified point estimate)**
and **unbounded (CORRIDOR_OFFSET)**. For the traffic-light family the
comparable figure is **≤ 79.8 % exposure on the 867-clip subpopulation where it
can be checked**, and ⛔ unlike YIELD that one is a **ceiling only**: the probe
reads light-presence, the token is a reaction, and the gap between them is
unmeasured (§4.5). None of these is small; none of them is a count of wrong
labels.

**The decision this evidence supports** (the PI's, not this agent's):

1. **The traffic-light family carries the largest unresolved exposure and is
   the best candidate for a carve-out.** 692 / 867 measured, on the PI's own
   example, on a channel the annotator itself produced — and, uniquely among
   the tokens here, the gap between what the probe reads (a light was present)
   and what the label claims (a reaction occurred) is one **nothing on this box
   can close**. If any token keeps the ignore policy, it is these four; and if
   they do not, the thing worth building next is the probe that *would* close
   it — light state, lane assignment and range against the ego's longitudinal
   profile.
2. **The four originally-named tokens are the safest of the fifteen** — three of
   them at least have a measurable bound, and two of those bounds come from a
   probe that passed its control.
3. **Six tokens ship blind.** LANE_CHANGE_L/R, MERGE, TAKE_EXIT_L/R and
   colourless TRAFFIC_LIGHT_REACT have no instrument here. They are also the
   smallest classes (15–128 positives) and all sit on the `pos_weight` cap, so
   their gradient contribution is both unverified and saturated.
4. **An arm trained under this policy must be compared against one that is
   not.** The stamp makes that comparison possible; nothing in this package
   makes it automatic.

---

## 6. The guards, proven by mutation

⛔ Two independent passes, because a guard that is only *inspected* is a comment.

### 6.1 The artifact is mutated, the guard must fire

`stack/tests/test_cot_negative_policy.py` — 22 tests, all green.
`test_control_unmutated_passes` is the table's control: the clean sidecar loads
and validates, so every "raises" below fails for the right reason.

| # | mutation applied to the sidecar / call | stage | exception | must say |
|---|---|---|---|---|
| 1 | `meta.source_blob_md5` → another blob | load | `CotAbsenceNegativeRefused` | "was built over blob md5" |
| 2 | `schema` tag changed | load | `CotAbsenceNegativeRefused` | "declares schema" |
| 3 | `meta.policy` id changed | load | `CotAbsenceNegativeRefused` | "carries policy" |
| 4 | `digest_algorithm` undeclared/wrong | load | `CotAbsenceNegativeRefused` | "digest_algorithm" |
| 5 | one row not token-wide | load | `CotAbsenceNegativeRefused` | "wide" |
| 6 | `clips` emptied | load | `CotAbsenceNegativeRefused` | "empty policy" |
| 7 | a clip dropped from the sidecar | guard | `AssertionError` | "are NOT decided by the sidecar" |
| 8 | a clip present that the blob lacks | guard | `AssertionError` | "are NOT in the blob" |
| 9 | one POSITIVE bit flipped | guard | `AssertionError` | "POSITIVES disagree" |
| 10 | a token CoT-backed in the blob, absent from the sidecar | guard | `AssertionError` | "absent from the sidecar" |
| 11 | a sidecar token no longer CoT-backed in the blob | guard | `AssertionError` | "no longer vlm-cot-backed" |
| 12 | policy selected with **no** sidecar | target | `CotAbsenceNegativeRefused` | "needs the SIDECAR" |
| 13 | sidecar passed with **another** policy | target | `CotAbsenceNegativeRefused` | "would be IGNORED" |
| 14 | a clip the sidecar does not decide, at target time | target | `CotAbsenceNegativeRefused` | "does not decide this clip" |

10 and 11 are the ones that matter for the future: the blob is re-pinned
regularly, and the silent failure to guard against is a **new caption token
appearing whose absence nobody decided** — it would keep the old ignore policy
while its neighbours did not, and nothing would say so.

### 6.2 The guard is removed, its test must go red

`stack/scripts/audit_cot_negative_guards.py` deletes each check from
`v7_labels.py`, reruns only that guard's test, and requires it to **fail**; then
restores the file and re-verifies its md5. A guard whose test still passes with
the guard removed is not a guard.

**14 / 14 PROVEN load-bearing.** Control green first (the clean artifact
validates), `v7_labels.py` restored to md5 `d12469e49355ba757246825d34ca85bb`.
Full table: `raw/guard_removal_audit.json`.

### 6.3 Regression, measured against the CURRENT branch tip

⚠️ **The trainer edit was rebased.** It was cut against
`refc_v3_train.py` blob `c55137e0`; the core workstream landed **8c7d215**
(timm trunk CLI, `build_optimizer` DiffusionDrive parameter groups, `pose_hist`
plumbing) and **722539a** / **e95af00** while this was in flight, moving that
blob to `4499188e` and the branch to **e95af00**. The edit was re-applied by a
**clean 3-way merge** (`git merge-file`, base `c55137e0`, other `4499188e`,
**0 conflicts**); the diff against the tip is exactly the 5 hunks / +45 −5 lines
of this change and touches nothing the core workstream landed.
⚠️ `stack/tanitad/data/v7_labels.py` needed no move — its base blob `96e893a7`
already equals the tip's.

Counts, run on a scratch checkout of `agent/arch-inf-20260803` @ **e95af00**
(my worktree predates `tanitad.models.refcv6_diffusion`, so the trainer cannot
import there and the tip is the only honest place to score this):

| suite | pristine tip | tip + this change |
|---|---:|---:|
| the 9 v7-label / goal / trainer suites | **135 passed, 2 failed** | **135 passed, 2 failed** |
| `test_cot_negative_policy.py` | (absent) | **+22 passed** |
| **total** | **135 / 2** | **157 / 2** |

⇒ **157 − 135 = 22 = exactly this change's own tests. Zero regressions.** The
seam test that goes through the real `V3Dataset.__getitem__` tensor
(`test_trainer_dataset_reaches_the_loss_target`) **PASSED at the tip**, verified
not-skipped, as did `test_trainer_opt_in_is_by_name_and_off_by_default` and
`test_shipped_sidecar_matches_the_shipped_blob`. The default
(`negatives="measured"`) path is untouched; the OFF path carries no sidecar on
the class default, asserted by test.

⚠️ The **2 failures are pre-existing and present on the pristine tip**
(`test_v7_wiring.py::test_dry_run_exercises_the_v72_door_and_records_the_stamp`,
`::test_bptt_truncate_reaches_the_rollout_from_v6_loss_step`) —
`train_v6_staged.py` calls `stage_a_losses(..., stopgrad_factual=...)` against a
signature that does not take it. Reproduced identically on the unmodified repo
at `d221843` **and** on the unmodified tip, so they are not this change and are
not touched here.

⚠️ One read-control worth recording: scoring the tip from a `git archive stack`
extraction made `test_tactical_goal_underpowered_matches_census.py` fail 10/11
**on the pristine tip too** — it pins against
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-07-underpowered-pin/raw/vocab_fill_census.json`,
which lives outside `stack/`. With that one file restored it is 11/11 green on
both sides. A partial checkout is a missing artifact, not a regression, and the
test says so loudly rather than passing vacuously.

---

## 7. Deliverable manifest

All staged in worktree `C:/Users/Admin/tanitad-wt-flywheel-neg` (from
`agent/arch-inf-20260803` @ `d221843`). ⛔ **Not committed, not pushed, no branch
switched.**

| artifact | path | also elsewhere? |
|---|---|---|
| the sidecar | `TanitAD Research Lab/Data Engineering/Research/2026-09-16-flywheel-negatives/cot_absence_negative_v8.0_train.json.gz` | repo only |
| its builder | `stack/scripts/build_cot_negative_sidecar.py` | repo only |
| the probe (feature extractor) | `stack/scripts/probe_cot_absence_falsenegatives.py` | repo only |
| the analysis (bounds + controls) | `stack/scripts/analyze_cot_absence_falsenegatives.py` | repo only |
| guard-removal audit | `stack/scripts/audit_cot_negative_guards.py` | repo only |
| loader + permission + guards | `stack/tanitad/data/v7_labels.py` | repo only |
| trainer opt-in | `stack/scripts/refc_v3_train.py` (`--cot-negative-sidecar`) | repo only |
| tests + mutation table | `stack/tests/test_cot_negative_policy.py` | repo only |
| this report | `…/2026-09-16-flywheel-negatives/RESULT.md` | repo only |
| per-clip probe features (sha12-keyed) | `…/raw/probe_features_{tac,wide}.json.gz` | repo only |
| false-negative analysis | `…/raw/false_negative_analysis_{tac,wide}.json` | repo only |
| before-census, pos_weight table, guard audit | `…/raw/{census_before,pos_weight_before_after,guard_removal_audit}.json` | repo only |

**⚠️ In ONE place only (not in the repo, not backed up):**

| input | where | note |
|---|---|---|
| the agent join | `C:/Users/Admin/tanitad-caches/b1-train-join-20260906/b1train_agents.jsonl.xz` (317 MB, md5 `1c985e6d6ad34e605c4ebd30cb353558`) | also at `C:/Users/Admin/a40-rescue/b1_train_plus_eval_agents.jsonl.xz` (a superset, no sidecar) |
| ego motion | `C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo/*.parquet` (4,800 files) | derived from the PhysicalAI release |
| the label blob | `D:/Projects/TanitAD/TanitAD Research Lab/…/raw/s2_labels_v8.0_train.jsonl.gz` | tracked in the repo, md5 `fa89ea55dfce68403eb30300e57852ab` |

**Reproduce:**

```
python stack/scripts/probe_cot_absence_falsenegatives.py \
  --labels <blob> --ego-dir <egomotion_alpamayo> --join <b1train_agents.jsonl.xz> \
  --out feats_tac.json                       # add --win-lo 4 --win-hi 18 for wide
python stack/scripts/analyze_cot_absence_falsenegatives.py \
  --labels <blob> --feats feats_tac.json --out fn_tac.json
python stack/scripts/build_cot_negative_sidecar.py \
  --labels <blob> --fn-measurements fn_wide.json --out <sidecar>.json.gz
python -m pytest stack/tests/test_cot_negative_policy.py -q
python stack/scripts/audit_cot_negative_guards.py --out raw/guard_removal_audit.json
```

Runtime: ~50 s extraction per window, ~5 s analysis, ~5 s build, ~4 s tests,
~40 s audit. CPU only (`CUDA_VISIBLE_DEVICES=""`).

---

**Evidence classes used above:** every count and rate in §2, §3, §4 and §6 is
**MEASURED** on the named artifacts. §5's reading of those numbers is
**ANALYSIS**. No number here is ESTIMATED or HYPOTHESIS except the two labelled
"point estimate", which are model-based (`(F − f₀)/(S − f₀)`) and carry their
assumptions inline.
