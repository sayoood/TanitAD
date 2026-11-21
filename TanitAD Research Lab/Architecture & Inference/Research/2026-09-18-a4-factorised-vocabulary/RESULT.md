# A4 — the factorised path × velocity vocabulary, re-verified

**2026-09-18 · Architecture & Inference FlyWheel · 0 GPU · 0 pod touched.**
**Backlog row:** `Project Steering/BACKLOG.md:19` — *"Re-verify the factorised path × velocity
vocabulary (refuted 0-3 in the survey) … the single most valuable re-verification target."*

---

## 0. Headline — four findings, and A4 is NOT what the row says it is

1. ⭐⭐ **THE BACKLOG ROW CONFLATES TWO DIFFERENT OBJECTS, AND THE PROGRAMME HAS ALREADY SHIPPED
   ONE OF THEM.** A4's justification is *"maps 1:1 onto our LAT+LON softmax mechanism"* — the
   **manoeuvre head**. That head **was factorised, and has been unconditionally ON in every
   refcv3/v4/v5 arm since 2026-08-03** (`stack/tanitad/refs/refc_v3.py:572`, comment: *"action
   space: both arms, **never a lever**"*). The claim the survey actually refuted is about the
   **trajectory ANCHOR vocabulary** — a different tensor, a different head, and **still joint
   today** (`stack/tanitad/refs/refc.py:1480`, a `[128, 8, 2]` buffer of whole trajectories).
   ⇒ **A4 is open, but not for the reason the row gives.**

2. ⭐⭐ **"REFUTED 0-3" DOES NOT MEAN WHAT THE REGISTER HAS BEEN CARRYING.** All three verifiers
   voted `refuted: true` and **all three say the descriptive core is VERIFIED**: *"FACTUAL CORE
   VERIFIED — the claim's numbers are real"* (v0), *"FACTUAL SCAFFOLDING IS ACCURATE —
   INTERPRETIVE CORE IS REFUTED BY THE PAPER'S OWN TABLE"* (v1), *"DESCRIPTIVE HALF: VERIFIED …
   Abstract verbatim confirms the factorized vocabulary"* (v2). **What went 0-3 was the
   INFERENCE the survey attached to the paper, not the existence or the numbers of the factorised
   vocabulary.** The survey's own open question asked *"substance, or verification protocol?"* —
   the answer is **neither**.

3. ⛔ **AND THE EVIDENCE THAT WOULD SETTLE IT IS NOT IN THE REPO.** All **106 + 109**
   `resultPreview` fields across the two raw survey JSONs are **exactly 401 characters** — one
   distinct length, i.e. a uniform truncation. `result.refuted[12]` carries only
   `claim` / `vote` / `source`; there is **no `evidence` field**. The three verifiers' actual
   arguments **do not exist anywhere in the programme**, and arXiv **2603.29163 is not banked**
   (`library.json`, 0 hits for `29163`). A truncated artifact that reads like a complete one —
   the class `CLAUDE.md` already names.

4. ⭐⭐ **SO I MEASURED IT ON OUR OWN VOCABULARY INSTEAD, AND THE ANSWER IS LARGE.** Decomposing
   the shipped 128-anchor bank into (path, speed-profile) at a tolerance a **known-rank control
   recovers exactly**: the bank contains **≥ 45 distinct paths and ≥ 84 distinct speed profiles**.
   Their product is **≥ 3,780 candidates from 129 logits**, against **128 candidates from 128
   logits** today — **a ≥ 29.5× density multiplier for +1 logit**, and the product **contains
   every current anchor** (round-trip max 0.0657 m / mean 0.0043 m at a 1 m path grid), so
   oracle-in-vocabulary ADE **cannot rise**. ⛔ **That is a CEILING result, not a driving result** —
   §5 says why it must not be quoted as one, and §6 is the experiment that would earn that.

---

## 1. Where the claim lives, and what "0-3" refers to

**Evidence class: MEASURED (ours) — file, line and byte-level probes below.**

| what | where |
|---|---|
| the backlog row | `Project Steering/BACKLOG.md:19` |
| the survey's summary table | `TanitAD Research Lab/Architecture & Inference/Implementation/incoming/2026-07-29-deep-research-sota/DEEP_RESEARCH_2026-07-29.md:29` |
| the loop-state restatement | `Project Steering/LOOP_STATE.md:380` |
| the ranked-reference restatement | `Project Steering/REFERENCE_SYSTEMS_RANKED.md:78-81` |
| ⭐ **the primary record** | `…/2026-07-29-deep-research-sota/raw_report1_autonomous_driving.json`, `result.refuted[12]` + `logs[23]` + `workflowProgress[83,84,85]` |

### 1.1 The claim that was voted on — verbatim from `result.refuted[12].claim`

> *"The scaling is made affordable by FACTORISING the trajectory vocabulary into geometric paths x
> velocity profiles rather than enumerating whole trajectories: 1024 path anchors x 256 velocity
> anchors = 262,144 effective trajectories, described as 32x denser than the prior 8192 flat
> vocabulary. The ablation isolates the velocity axis independently of the path axis (512x128 ->
> 88.7, 512x256 -> 89.2, 1024x128 -> 89.5, 1024x256 -> 90.1 EPDMS) …"*

`vote: "0-3"`, `source: https://arxiv.org/abs/2603.29163` (SparseDriveV2).

⚠️ **Note the object.** It is a **TRAJECTORY ANCHOR VOCABULARY**. The sentence the backlog row
inherited — *"it prescribes exactly the decomposition our single 5-way maneuver softmax currently
collapses"* — is the survey's **own commentary**, not the paper's claim, and it is what welded A4
to the manoeuvre head.

### 1.2 What the three verifiers actually said (all that survives)

| voter | `workflowProgress` | the surviving text |
|---|---|---|
| v0 | `[83]` | *"Primary source refutes itself: **Table 6 (left)** …"* / *"FACTUAL CORE VERIFIED — the claim's numbers are real."* |
| v1 | `[84]` | *"**FACTUAL SCAFFOLDING IS ACCURATE — INTERPRETIVE CORE IS REFUTED BY THE PAPER'S OWN TABLE.** … Every descriptive number in the claim is EXACT"* |
| v2 | `[85]` | *"**DESCRIPTIVE HALF: VERIFIED.** … Abstract verbatim confirms the factorized vocabulary and '92.0 PDMS and 90.1 EPDMS on NAVSIM …'"* |

⇒ **The programme has been carrying "the factorised vocabulary is refuted" when its own primary
record says the vocabulary and its numbers were confirmed and the survey's *reading* was thrown
out.** `REFERENCE_SYSTEMS_RANKED.md:80-81` states it in the strong form (*"the factorisation went
0-3 … not admissible now"*) and that form is **not supported by the artifact it rests on**.

### 1.3 ⛔ The absence claim, with its two probes

**Claim: the full verifier evidence for the 0-3 vote does not exist in this repository.**

* **Probe A (structural, positive form).** Every `resultPreview` in **both** raw survey JSONs is
  **exactly 401 characters** — `raw_report1` **106 of 111** workflow entries, `raw_report2`
  **109 of 114**, and the set of distinct lengths is `[401]` in both.
  `result.refuted[12]` has keys `['claim', 'vote', 'source']` and **no `evidence` key**.
  Banked with both files' sha256 in `raw/survey_record_probe.json`.
* **Probe B (content, different path and name).** A repo-scoped grep for the verifiers' own
  distinctive phrase `INTERPRETIVE CORE IS REFUTED` over `TanitAD Research Lab/`,
  `Project Steering/` and `Paper/` returns **exactly one hit** — `raw_report1…json:2104`, i.e. the
  truncated preview itself. `Table 6 (left)` returns **zero** hits outside it.
* **Banking:** `TanitAD Research Lab/Library/library.json` contains **0** occurrences of `29163`;
  the primary is **not banked**, so any claim about the paper's Table 6 would be
  `PUBLISHED-SECONDARY` and inadmissible.

⇒ **This is a claim about the ARTIFACT, and it is positive in form.** It is not a claim that the
verifiers were wrong — it is that their argument is unrecoverable from what we hold.

### 1.4 The 2026-08-01 re-verification attempt — already logged as inadmissible

`Project Steering/Reports/2026-08-02-0030-return-from-limit-assessment.md:182,186-196`: the
targeted re-verification **lost 71 of 102 agents** to the weekly limit; *"1 claim refuted, 24
unverified … No claims survived."* The one claim with three valid votes was **refuted 0-2**.
⛔ Nothing from that run is quotable, and the row has been open since.

---

## 2. Is our vocabulary factorised or joint TODAY? — **both, on two different objects**

**Evidence class: MEASURED (ours), read from source at repo HEAD `d221843`.**

### 2.1 The MANOEUVRE / TACTICAL head — **FACTORISED, unconditionally, and it is not a lever**

| fact | file:line |
|---|---|
| the joint 5-way constant still exists | `stack/tanitad/refs/refc.py:188` — `N_MANEUVERS = 5` |
| the factorised widths | `refc.py:192-193` — `N_LAT_MAN = tac.N_LAT`, `N_LON_MAN = tac.N_LON` |
| …sourced from the vocabulary, = **3 × 3** | `stack/tanitad/refs/refc_tactical.py:136,138,143,145` |
| the seam | `refc.py:2859` `if cfg.factored_maneuver:` → `:2869-2870` `lat_head` / `lon_head`; **else** `:2885-2887` the single `Linear(aux_hidden, 5)` |
| ⭐ **v3/v4/v5 force it ON** | `stack/tanitad/refs/refc_v3.py:572` — `cfg.factored_maneuver = True  # action space: both arms, never a lever` (and `:750` for the smoke rung) |
| the 5-way is now a **push-forward**, not a head | `refc.py:3286` — `man_logits = tac.derive_man5_logprobs(lat_logits, lon_logits)`; the inverse at `:3303` |
| **two CE terms**, one per axis | `stack/scripts/refc_v3_train.py:2422-2423` — `loss_lat`, `loss_lon` |
| a **second** factorised surface on the v7.2 vocabulary | `refc_v3_train.py:2462-2465` — `loss_lat_tac`, `loss_lon_tac` |
| the v7 action vocabulary is itself two axes | `stack/tanitad/models/vocab_v7.py:320` `TACTICAL_LAT_ACTIONS_V7` (8) · `:328` `TACTICAL_LON_ACTIONS_V7` (8) |
| the anchor graft is split per axis | `refc.py:1558-1565` — `lat_to_anchor` + **zero-init** `lon_to_anchor` (`:1565`), applied at `:2374-2377` |

⇒ **On the manoeuvre axis, A4's premise is stale: the factorisation landed in D-TAC1 on
2026-08-03 and has shipped in every arm since.** `RefCConfig.factored_maneuver` defaults `False`
(`refc.py:701`) only so that pre-D-TAC1 checkpoints stay byte-reproducible.

⛔ **But it is wired differently in the two vocabularies, and the difference is a structural zero.**
`Project Steering/GOALS_AND_CLAIMS.md:11217` (**H19-STAMP-1**, MEASURED 2026-09-06): perturbing only
`lat_head_tac`/`lon_head_tac`, the decoder sees `maneuver_logits` **24.4632 under `kin3`** vs
**0.0 under `v7.0`**, `lat_prior` **21.8972 vs 0.0**, `anchor_logits` **18.0365 vs 0.0**. ⇒ **under
`v7.0` — which refcv4b ran — the 8 × 8 factorised action heads are trained by CE and have NO
inference-time influence on the anchor ranking at all.**

### 2.2 The TRAJECTORY ANCHOR vocabulary — **JOINT**

| fact | file:line |
|---|---|
| the bank is whole trajectories | `refc.py:1480` — `register_buffer("anchors", anchors)  # [N, S, 2]` |
| size | `refc.py:361` — `n_anchors: int = 128` (XL 256) |
| built by FPS / k-means over **whole GT windows** | `stack/scripts/build_refc_anchors.py:1-46`; the shipped 6 s bank by `…/2026-09-04-refcv4-launch/raw/scripts/build_anchors6s.py` |
| the classifier target is one argmin over the joint set | `stack/scripts/refc_v3_train.py:2383,2386` — `anchors = out["anchor_bank"]` … `a_star = dist.argmin(dim=1)` |

⇒ **There is no path axis and no velocity axis here. One index selects a whole trajectory.**
This is precisely the object SparseDriveV2 factorises, and **it is untouched.**

⚠️ **A prior PI decision sits next to this and must be named**: `stack/experiments/refb-v2/refb_v3.py:144-147` —
*"FINAL (Sayed 2026-07-18): B1 = TIME-anchored (DiffusionDrive/VADv2-faithful) over
waypoint_horizons. 'distance' (arc-length/Frenet) is kept as a switch but is NOT the chosen
direction."* A **path × velocity** vocabulary is **not** that switch — it keeps the time-anchored
output and only changes how the set is *constructed* — but §6's pre-registration says so
explicitly rather than letting a reader assume the ruling was reopened.

---

## 3. What a factorised path × velocity vocabulary changes in OUR code, and what it costs

**Evidence class: MEASURED for the counts and the arithmetic; ESTIMATED for the effort.**

| # | change | file:line | note |
|---|---|---|---|
| 1 | build **P paths × V speed profiles** instead of N trajectories | `stack/scripts/build_refc_anchors.py`, `…/build_anchors6s.py` | new builder; the existing gate scores it unchanged |
| 2 | the bank buffer becomes two tensors | `refc.py:1480` | `paths [P, G, 2]` (absolute arc-length grid) + `speeds [V, S]` |
| 3 | `anchor_logits` becomes **two** softmaxes | `refc.py` `AnchoredDiffusionDecoder` | `P + V` logits, `P·V` candidates |
| 4 | the classifier target becomes two argmins | `refc_v3_train.py:2383-2389` | `a_star` → `(p_star, v_star)`; **no label change** — the target is still the recorded future path, exactly the argument `stack/tanitad/refs/anchor_twoseg.py` already makes |
| 5 | ⭐ the **`lon_to_anchor` graft finally has a matching axis** | `refc.py:1563` | today a 3-wide longitudinal prior is grafted onto a **joint** 128-way score; with a velocity axis it grafts onto the axis it is about |
| 6 | the `v0`-conditioned bank becomes natural | `anchor_meta.py`, `--anchor-v0-cond` | the speed axis is the thing that should track `v0`; the path axis should not |

**Costs, MEASURED on the shipped bank** (`raw/factorisation_rank.json`):

| | joint (today) | factorised (this decomposition) |
|---|---|---|
| candidates | **128** | **≥ 3,780** (≥ 29.5×) |
| **classifier logits** | **128** | **129** (+1, +0.78 %) |
| bank tensor, float32 | 2,048 floats = 8 KB | 20,562 floats = **82 KB** (10.0×) |
| params added | — | the two heads replace one; ~parity, not measured here |

⚠️ **The naive "P + V stored objects vs N" parity argument is FALSE at the byte level** and is
corrected here: a path must be stored at a resolution finer than 8 slots, so the tensor grows
**10×**. It grows from 8 KB to 82 KB beside a 104 M-parameter model, i.e. the cost is real and
irrelevant. **The parity that matters is the decision surface: 129 logits vs 128.**

---

## 4. The measurement — controls first

**Instrument:** `code/factorisation_rank.py` · **output:** `raw/factorisation_rank.json`
**Substrate:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-04-refcv4-launch/raw/refc_anchors_6s_b1train_128.pt`
— **content-verified**, `sha256sum` = `68f81acf83b6b21ce533df282f78fe269e9b3b462eb82c726004821c7c1a806b`,
**identical** to the value that artifact's own `RESULT.md:34` declares. `[128, 8, 2]` float32,
slots `0.5/1/1.5/2/3/4/5/6 s`. CPU, float64, ~40 s. **No GPU, no corpus, no pod.**

### 4.1 Controls

| control | must read | read | verdict |
|---|---|---|---|
| **C2** — a bank built as a TRUE product of **8** paths × **16** speeds | (8, 16) | **(8, 16)** at `tol_path 0.5 m`, `tol_speed 1.0 m` | ✅ **PASS** |
| **C3** — 128 identical anchors | (1, 1) | **(1, 1)** at every tolerance | ✅ PASS |
| **C4** — 128 anchors distinct on both axes | (128, 128) | **(91, 23)** at C2's operating point | ⛔ **FAILS — over-merges** |
| **C2b** — within one TRUE speed group the recovered σ must have zero spread | 0 | max **0.9959 m**, mean **0.1815 m** over 448 pairs | ⚠️ **the instrument's floor** |
| **C1b** — compose each path with its OWN speed; must return the anchor | 0 | `ds` 1.0 m: max **0.0657**, mean **0.0043 m**; `ds` 0.25 m: max **0.0187**, mean **0.0011 m** | ⚠️ priced, not zero |
| **C1** — normalised-grid round trip, convergence sweep | → 0 | `G` 16 → 4096: **0.298 → 0.00092 m** max | ✅ converges ~1/G |

⭐ **C2 IS THE CONTROL THAT CHANGED THE RESULT, AND IT DID SO BY FAILING FIRST.** The first
representation normalised each path by its own length. Under it the bank read **80 distinct paths
and 128 distinct speeds**, i.e. an occupancy of 128 / 10,240 = 1.25 % and a **80× multiplier** — a
confident, publishable-looking number. **C2 read 128 paths for a bank built from 8**, which is
impossible, and exposed the mechanism: an arc of curvature κ normalised by its own length has a
shape that depends on `κ·L`, so **the path axis was silently carrying the speed axis**. The repair
is the absolute arc-length representation (`path_abs`), and only then does C2 recover (8, 16)
exactly. *(Class: the 2026-08-22 ridge-probe family — a probe that manufactures a result, caught
only by a control that must read a known value.)*

⛔ **C2b explains why the speed tolerance must be 1.0 m, and it is not a free parameter.** Arc
length recovered from **8 sparse waypoints is CHORD length**, and the chord deficit is
curvature-dependent — so two anchors that genuinely share a speed profile differ by up to **0.996 m**
in the recovered σ. ⇒ **a post-hoc decomposition of a joint bank cannot resolve the speed axis
below ~1 m at this slot density. A factorised bank must be BUILT factorised** (from controls, as
`anchor_twoseg.py` and refav1's `(a_lon, a_lat)` parameterisation already are), not derived.

⛔ **C4's failure is why every rank below is a LOWER BOUND.** No single tolerance passes both C2
and C4, because "distinct path" is not tolerance-free and C4's bank is deliberately a near-
continuum. At C2's operating point C4 loses 128 → 91. **Reported as a lower bound, in the
conservative direction.**

### 4.2 The bank

| axis | tol | distinct | incomparable pairs |
|---|---:|---:|---:|
| path (absolute arc length, common observed domain) | 0.01 m | 106 | 330 |
| | 0.10 m | 73 | 232 |
| | **0.50 m** ← C2's point | **45** | 165 |
| | 1.00 m | 34 | 141 |
| speed profile (metres) | 0.05 m | 128 | — |
| | 0.50 m | 109 | — |
| | **1.00 m** ← C2's point | **84** | — |
| | 2.00 m | 56 | — |

**Arc length spans 1.20 m to 216.95 m (mean 74.88);** final speed p5/p50/p95 = **1.66 / 10.63 /
28.95 m/s**. The bank's own sidecar records `distinct_anchors_used: 121` of 128 on the 19,602
held-out windows — **7 anchors are never the nearest to any window**.

### 4.3 The identity bound

The product set contains the diagonal `(path_i, speed_i)` for every `i`, which reproduces anchor
`i` to **C1b's error**. Therefore, on any window set,

> **oracle-in-vocabulary ADE(product) ≤ oracle-in-vocabulary ADE(joint) + ε**, with ε bounded by
> the C1b mean **0.0043 m** at a 1 m path grid (**0.0011 m** at 0.25 m).

Against the shipped gate's **0.3796 m** and the straight-line control's **0.6843 m**, ε is
**1.1 % / 0.3 %** of the quantity. ⇒ **the factorised vocabulary cannot lose on the ceiling.**

⛔⛔ **AND THAT IS EXACTLY WHY IT IS NOT A RESULT.** A vocabulary is a **ceiling**, and this
programme has already measured that a better ceiling does not deliver itself:
`GOALS_AND_CLAIMS.md` **H-RL-OBJECTIVE-1** — oracle-in-fan **0.2306 m** vs selected **0.4871 m**, a
**2.11× selection gap**, and **2.10× inside the selector's own top-32**. Multiplying the candidate
set by 29.5× makes the selection problem **harder**, not easier. And
**D-REFC-OFFSET-FEAS-1** measured that the emitted fan is already **88.8 % envelope-violating /
84.1 % over the Kamm circle**, so new candidates must be shown **feasible**, not merely near.
⇒ **A factorised vocabulary must be gated on the SELECTED trajectory and on feasibility, never on
the oracle alone.** §6 does that.

### 4.4 Reproducibility of this report itself

`code/factorisation_rank.py` run twice produced a **byte-identical** `factorisation_rank.json`
(`cmp -s`). `code/check_result.py` re-checks **every** number quoted in §1 and §4 against
`raw/factorisation_rank.json` and `raw/survey_record_probe.json` with the expectations written as
**literals** — never as expressions over the producer's own arithmetic — and carries a
**deliberate-regression arm** that lifts the C2-refuted "80 paths" reading out of its failure
narrative and asserts the guard goes **RED**. Output: `raw/check_result.txt`, **50 OK, 0 failures**.
⚠️ Determinism is not correctness; the controls in §4.1 are what make the numbers admissible.

---

## 5. Verdict on A4 as written

| A4's premise | verdict |
|---|---|
| *"maps 1:1 onto our LAT+LON softmax mechanism"* | ⛔ **STALE.** That head is factorised and has been since 2026-08-03; it is not a lever (`refc_v3.py:572`). |
| *"refuted 0-3 in the survey"* | ⛔ **MISREAD.** All three voters verified the descriptive half; the **interpretation** was refuted. The evidence for even that is a 401-char truncation and the primary is unbanked. |
| *"the single most valuable re-verification target"* | ✅ **STILL TRUE — of the TRAJECTORY VOCABULARY.** That object is joint, its ceiling is the programme's measured binding constraint, and the factorised form is a ≥ 29.5× density multiplier for +1 logit with a proven no-loss bound. |

⇒ **A4 is NOT already done, and it is NOT what the row says.** The row should be rewritten to name
the trajectory vocabulary, and the two stale restatements
(`LOOP_STATE.md:380`, `REFERENCE_SYSTEMS_RANKED.md:80-81`) corrected. **See §8 — this needs a
register edit I am not staging.**

---

## 6. PRE-REGISTRATION — `A4-PXV`: the cheapest discriminating experiment

**Written before any arm exists. Both outcomes committed. 0 GPU for stage 1.**

### 6.1 The one variable

**How the 128-slot anchor vocabulary is CONSTRUCTED: one joint k-means set of 128 whole
trajectories (incumbent) vs a product of P paths × V speed profiles.** Everything else —
corpus, split, slot grid, seed, gate, estimator — is held at the refcv4 launch's values.

⛔ **Not varied, and stated so no reader infers otherwise:** the output stays **time-anchored**
(the PI's 2026-07-18 `anchor_space` ruling is untouched); no label is regenerated (the classifier
target remains `argmin` distance to the recorded future path, `refc_v3_train.py:2383-2389`).

### 6.2 Stage 1 — the vocabulary gate (0 GPU, hours, no training)

Score the candidate vocabularies on the **existing** held-out gate:
**141 eval clips / 19,602 windows, clip-disjoint from the 4,572 train clips** (asserted on
`.v2ep.pt` filenames, intersection 0), metric **oracle-in-vocabulary ADE 0–2 s and 0–6 s**,
builder `…/2026-09-04-refcv4-launch/raw/scripts/build_anchors6s.py`.

**Arms** (all at a matched decision surface of **≈128 logits**):

| arm | construction |
|---|---|
| `J128` | **incumbent**, k-means slot-normalised, 128 joint trajectories |
| `F_64x64` | 64 paths × 64 speeds = 4,096 candidates, 128 logits |
| `F_32x96` | 32 × 96 — path-poor / speed-rich |
| `F_96x32` | 96 × 32 — path-rich / speed-poor |
| `F_45x84` | the decomposition MEASURED in §4.2, as a fixed reference point |

**Controls that must read a known value, reported before any arm:**

| control | must read |
|---|---|
| `zero path` | **14.3264 m** — reproduces the banked value or the harness is wrong |
| `straight line` (constant velocity) | **0.6843 m** (independently published 0.6780) |
| `J128` re-scored | **0.3796 m** — the incumbent must reproduce its own banked number |
| `F` diagonal-only (product restricted to the 128 diagonal pairs) | **= `J128` ± C1b's 0.0043 m** — the identity bound, asserted not assumed |
| **feasibility**, every arm | `fan_envelope` / `fan_kamm_over` via `refc_select`/`fan_safety` on the same windows; a candidate set that is more infeasible has not improved |

**Committed criteria — stage 1:**

* ✅ **PASS** — some `F` arm reaches oracle ADE 0–2 s **≤ 0.3416 m** (a **≥ 10 % relative**
  improvement on `J128`'s 0.3796) **AND** does not raise `fan_envelope` above `J128`'s value.
* ⛔ **FAIL** — **every** `F` arm is within **±3 %** of `J128` (0.3682–0.3910 m), or any arm that
  beats it does so while raising `fan_envelope`. **Then the factorisation buys ceiling but not
  feasibility, and A4 closes as REFUTED-ON-OUR-CORPUS** — which is a real answer, and the
  diagonal control proves the instrument could have shown a gain.
* ⚠️ **Between ±3 % and 10 %** is NOT a pass: it is reported as a measured effect too small to
  justify stage 2, and A4 closes as *"factorisation is real and immaterial here."*

⭐ **Why this bar and not another.** `J128` already bought **1.0882 → 0.3796 m** by changing
construction alone. A second construction change worth a retrain should be visible at the same
order; 10 % is a tenth of that and still an order above the C1b noise floor of 1.1 %.

### 6.3 Stage 2 — T1, and ONLY if stage 1 passes

**Tier: T1 primary** (action-closed loop, `taniteval/tools/t1_eval.py`). `T0` numbers may be
reported as diagnostics and **may not** carry the verdict (`EVAL_DOCTRINE.md`).

**Two arms, one flag apart:** `refcv5`-recipe with `--anchors <J128>` vs `--anchors <F_best>`,
same steps, same seed, same corpus.

**Estimator:** paired episode-cluster bootstrap (`taniteval/ci.py`), unit = episode.
⛔⛔ **A separated CI is NECESSARY AND NOT SUFFICIENT** (`H-ESTIM-SEED-1`). The panel **must**
carry a **replicate arm** — `J128` re-trained with the same flags — and the effect must exceed
that replicate's own floor. Since the decoder samples, the replicate must also vary the
**inference** seed.

**Four families, per the binding rule** — LONGITUDINAL (target-speed accuracy + distance-keeping),
LATERAL (heading, curvature, yaw-rate, cross-track), TACTICAL (per-axis decision quality and κ),
STRATEGIC (`n/a` with reason and n, as PhysicalAI carries no route signal). Per-family, never
pooled.

**Committed criteria — stage 2:**

* ✅ **PASS** — `F_best` beats `J128` on **T1 selected ADE 0–2 s**, separated **and** by more than
  the replicate floor, **and** LONGITUDINAL target-speed accuracy does not regress
  (pre-committed guard-rail, the axis carrying 88.7 % of the oracle gap), **and** `fan_envelope`
  does not rise.
* ⛔ **FAIL** — not separated, or separated within the replicate floor, or LATERAL/LONGITUDINAL
  separated-worse. **Then the ceiling gain did not survive selection**, which is the
  H-RL-OBJECTIVE-1 prediction and would make **selection, not vocabulary, the ranked next lever** —
  and that is a result, recorded as such.

### 6.4 Cost

Stage 1: **0 GPU**, ~2–4 h of CPU, needs only the B1 v7.2 `.v2ep.pt` caches already on the A40 pod.
Stage 2: **two training arms + one replicate**, i.e. **3 × the refcv5 recipe**. ⛔ Stage 2 is
**NOT authorised by this document** and must not be started without the PI.

---

## 7. Named blockers (Rule Zero §3)

| # | blocker | what would unblock it |
|---|---|---|
| 1 | **The literature half of A4 cannot be closed here.** The primary (arXiv 2603.29163) is unbanked and the verifiers' evidence is truncated to 401 chars. | `python tools/kb_add.py 2603.29163 --tag vocabulary --cited-by "<this RESULT.md>"` from a session with network, then read Table 6 directly. **Until then no claim about that paper's ablation is admissible.** |
| 2 | **Stage 1 needs the B1 v7.2 caches**, which are on the A40 pod, not the dev box. **MEASURED, two probes:** no `*.v2ep.pt` exists anywhere on the local disks (depth-3 walk of `D:`), and no `_epcache` / `val40` / `data` directory exists under the repo root. Stage 1 could otherwise have run in this turn. | a pod session; the compute is **CPU-only**, does not touch a GPU and does not disturb training |
| 3 | **Stage 2 needs 3 training arms.** | **PI authorisation** |
| 4 | **The backlog row and two restatements are wrong** and I am not staging steering edits. | §8 |

---

## 8. ⛔ ESCALATION — integration this report needs and did not do

1. **`Project Steering/BACKLOG.md:19`** — A4's justification names the manoeuvre head, which is
   settled. Rewrite to name the **trajectory anchor vocabulary**, and carry §5's verdict.
2. **`Project Steering/LOOP_STATE.md:380` and `Project Steering/REFERENCE_SYSTEMS_RANKED.md:80-81`** —
   both state the factorisation was refuted. **The primary record says the descriptive half was
   verified 3/3.** These are the sentences a future survey brief would inherit.
3. **`Project Steering/GOALS_AND_CLAIMS.md`** — this report asserts claims and the register must
   carry them **in the same turn** (`CLAUDE.md`). Proposed rows, ready to paste:
   * **`D-A4-VOCAB-JOINT`** — *the trajectory anchor vocabulary is JOINT today* — **SUPPORTED
     (MEASURED 2026-09-18)** — `refc.py:1480`, `refc_v3_train.py:2383-2386`; this RESULT.md §2.2.
   * **`D-A4-PXV-DENSITY`** — *the shipped 128-anchor bank decomposes into ≥ 45 paths × ≥ 84 speed
     profiles = ≥ 3,780 candidates at 129 logits (≥ 29.5×), containing every current anchor to
     0.0043 m mean* — **SUPPORTED (MEASURED 2026-09-18, C2 PASS, C4 gives the lower-bound
     direction)** — `raw/factorisation_rank.json`.
   * **`R-A4-REFUTATION-MISREAD`** — *"the factorised path × velocity vocabulary went 0-3" has been
     quoted as a substantive refutation; the primary record shows all three voters verified the
     descriptive half and refuted the survey's interpretation, and their argument survives only as
     a 401-char truncation* — **RETRACTION, class: a truncated artifact that reads like a complete
     one** — `raw_report1_autonomous_driving.json` `workflowProgress[83,84,85]`.
4. **`Project Steering/RETRACTION_LOG.md`** — item 3's `R-A4-REFUTATION-MISREAD` belongs there too,
   with the root-cause class.

---

## 9. Evidence classes

| claim | class |
|---|---|
| every file:line in §2, the sha256, and all of §4 | **MEASURED (ours)** — repo HEAD `d221843`, artifacts named inline |
| §4's ranks, multiplier and identity bound | **MEASURED (ours)** — `raw/factorisation_rank.json`, C2 PASS; **P and V are LOWER BOUNDS** (C4) |
| the survey's verbatim claim, votes and truncation | **MEASURED (ours)** — a property of `raw_report1_autonomous_driving.json` |
| SparseDriveV2's own numbers (Table 6, EPDMS ladder) | ⛔ **PUBLISHED-SECONDARY — INADMISSIBLE.** Primary unbanked; quoted here only as the text that was voted on |
| the gate values 0.3796 / 0.6843 / 1.0882 / 14.3264 | **MEASURED (inherited, content-verified)** — `…/2026-09-04-refcv4-launch/raw/refc_anchors_6s_b1train_128.pt.json`, whose parent tensor's sha256 I re-hashed |
| H19-STAMP-1, H-RL-OBJECTIVE-1, D-REFC-OFFSET-FEAS-1 | **INHERITED** — `GOALS_AND_CLAIMS.md`, not re-verified here |
| §3's effort and §6.4's timings | **ESTIMATED** |
| §6's expected outcomes | **HYPOTHESIS** — pre-registered, both directions committed |

**Tier:** §4 is a **vocabulary/ceiling diagnostic**. ⛔ **No T-tier applies and no driving claim is
made.** §6.3 is where a T1 claim would be earned.

---

## 10. DELIVERABLE MANIFEST

| artifact | where it lives | only one place? |
|---|---|---|
| `RESULT.md` (this file) | `repo: TanitAD Research Lab/Architecture & Inference/Research/2026-09-18-a4-factorised-vocabulary/RESULT.md` | no — staged |
| `code/factorisation_rank.py` | `repo: …/2026-09-18-a4-factorised-vocabulary/code/factorisation_rank.py` | no — staged |
| `raw/factorisation_rank.json` — every number in §4 | `repo: …/2026-09-18-a4-factorised-vocabulary/raw/factorisation_rank.json` | no — staged |
| `raw/survey_record_probe.json` — §1's verbatim claim, the three verifier previews, both source sha256s, the library-banking count | `repo: …/2026-09-18-a4-factorised-vocabulary/raw/survey_record_probe.json` | no — staged |
| `code/probe_survey.py` — the producer of the above | `repo: …/2026-09-18-a4-factorised-vocabulary/code/probe_survey.py` | no — staged |
| `code/check_result.py` + `raw/check_result.txt` — **50 assertions cross-checking every number quoted here against its artifact, expectations written as LITERALS, plus a deliberate-regression arm that must go RED** | `repo: …/2026-09-18-a4-factorised-vocabulary/` | no — staged |
| substrate `refc_anchors_6s_b1train_128.pt` | `repo: …/2026-09-04-refcv4-launch/raw/` — **pre-existing, untouched**, sha256 re-verified | no |
| the survey raw JSONs | `repo: …/2026-07-29-deep-research-sota/` — **pre-existing, untouched** | no |

⛔ **Nothing produced here lives on a pod, a worktree, or one disk.** No commit, no push, no branch
change. Staging is limited to the three paths above.

---

## ⛔ MASTER MIND REVIEW, 2026-09-18 — the conclusion STANDS, two of its quotes do NOT

Reviewed before landing, as the operating standard requires. **The finding is accepted.**
Two supporting quotations are **withdrawn**, and the reasoning is restated without them.

### ✅ Verified independently, and they carry the result

| checked | verdict |
|---|---|
| `result.refuted[12]` has keys `['claim','source','vote']` only — **no `evidence`** | ⭐ **CONFIRMED.** The verifiers' reasoning for this claim is genuinely not in the artifact. |
| The claim is **COMPOUND** — it asserts the paper factorises the vocabulary AND specific counts AND an ablation interpretation | ⭐ **CONFIRMED** by reading the `claim` string. ⇒ **a 0-3 vote cannot say WHICH conjunct failed.** This alone carries the agent's conclusion. |
| The survey ITSELF already flagged this | ⭐ **CONFIRMED, and it is the strongest evidence in the package.** The report's own open questions read: *"the refuted set contains results that map directly onto our diagnosed mechanisms — most sharply the factorised path x velocity trajectory vocabulary … Was that voted down on substance, or on verification protocol? Re-verifying arXiv 2603.29163 alone is probably worth more than the whole surviving set."* |
| `refc.py:1480` `register_buffer("anchors", …)  # [N, S, 2]` — whole trajectories | ⭐ **CONFIRMED.** The trajectory vocabulary is **JOINT**. |
| `refc_v3.py:572` `factored_maneuver = True  # never a lever`; `refc_v3_train.py:2386` `a_star = dist.argmin(dim=1)` | ⭐ **CONFIRMED.** The **manoeuvre head** is factorised; the **anchor vocabulary** is not. Two different objects. |

### ⛔ WITHDRAWN — both verifier quotations are misattributed

* *"FACTUAL SCAFFOLDING IS ACCURATE — INTERPRETIVE CORE IS REFU…"* is a **`lastToolSummary`** of a Verify agent and is **TRUNCATED mid-word**. Nothing in its context binds it to this claim.
* *"DESCRIPTIVE HALF CONFIRMED, INFERENTIAL HALF REFUTED ON FOUR COUNTS"* belongs to a **different claim entirely** — its `counterSource` is `arxiv.org/html/2506.04218` (*Pseudo-Simulation for Autonomous Driving*) plus the NAVSIM devkit metrics docs. It is about **NAVSIM v2 scoring**, not about a trajectory vocabulary.

⇒ The sentence *"all three verifiers voted refuted:true and all three say the descriptive half is verified"* is **NOT supported** and must not be quoted. A search for verifier text adjacent to any of the **18** `2603.29163` mentions found **none**.

⚠️ **ROOT-CAUSE CLASS: a quotation lifted from a large JSON by string search, attributed by proximity rather than by structure.** The file interleaves many claims' verifier traces; `find(needle)` lands wherever the words occur. ⇒ **When quoting from a multi-claim artifact, walk the STRUCTURE to the claim's own record and quote only what hangs off it** — and if that record has no `evidence` key, the honest report is *"the reasoning is absent"*, which is what the agent concluded anyway.

⭐ **Why the finding survives intact:** it never needed the quotes. The refutation's reasoning is **absent** (structurally verified), the claim is **compound** (so the vote is unattributable), and the survey **itself** asked to re-verify this exact paper above all others. That is sufficient, and it is all first-hand.

### ⚠️ Consequently NOT pasted

§8's ready-to-paste steering text is **held**, because it carries the withdrawn sentence. The
steering files still need correcting — the claim *"factorised vocabulary: refuted 0-3"* in
`BACKLOG.md`, `LOOP_STATE.md` and `REFERENCE_SYSTEMS_RANKED.md` overstates what the artifact
supports — but the correction must rest on **absence + compoundness**, never on the quotes.

<!-- A4-MASTERMIND-REVIEW-2026-09-18 -->
