# ERRATUM-1 to `PREREG_REFCV6_V2.md` — two numbers I quoted without checking them

**Date:** 2026-09-17, hours after the pre-registration landed in `cc24073` ·
**Author:** Master Mind · **Status: BINDING. It corrects the parent file; where they differ, this wins.**

⛔ **Both errors are mine and both are the same failure:** I carried numbers from a session summary
into a pre-registration instead of re-reading them from the record. A pre-registration is exactly
the document where that is least acceptable, because its numbers become the reference the arms are
judged against. Neither error changes a single criterion — the bars, the arms and the refusals all
stand — but the stated evidence behind two of them was wrong.

⭐ **What was CHECKED and is CORRECT, so the reader knows the audit was not selective:** the T-FLIP
figure. `refcv5-v2` **follows a flipped junction command on 20.5 % of windows**, and the acceptance
bar of ≥ 0.50 with true-minus-shuffled ≥ 0.38 (half the measured 0.76 ceiling) is quoted correctly.

---

## E1. §6 — "42.7 % of the TURN label's entropy is already in the nav token" is **UNSUPPORTED. WITHDRAWN.**

**There is no such measurement.** The only `42.7 %` in the steering record is an unrelated statistic:
*"256 of v1's 600 parity-val clips (42.7 %) are inside v2corpus's 9,000-clip training selection"* —
a corpus-overlap figure, not an entropy, not about nav, and not about turns. I attached a real number
to a different claim, which is worse than having no number at all: it reads as measured.

⇒ **Replace it with what WAS measured**, which is stronger than the figure it replaces:

| MEASURED | source |
|---|---|
| `refcv5-v2`'s tactical **turn recall 0.475 → 0.000 with nav removed** — the head is a **pure nav echo** | `REFCV6_CLARIFICATION.md` §3 |
| on the refav1 tactical decoder, under `nav_zero` the ranking **collapses to 0.520**, and a **NAV-ONLY predictor BEATS the model (0.684 > 0.650)** | `D-REFAV1-TAC-DECODER-PANEL`, 2026-09-03 |

⚠️ The second row is **refav1's** decoder, not refcv5's — a different model, quoted as corroboration
of the mechanism and never as a refcv5 measurement.

⭐ **The design consequence is UNCHANGED and now rests on evidence**: turn classes are excluded from
T-ZERO by design, and T-ZERO is scored on the non-turn behaviours. A head whose turn recall goes to
**exactly 0.000** without nav is not partially nav-driven — it is entirely nav-driven, which is a
better reason for the carve-out than the number I invented.

## E2. §5 — the oracle ceiling `0.4713` at 16×40 is **SUPERSEDED**. The value is **0.4762**.

`E-READOUT-CEILING-1` was **partially retracted** on 2026-09-08 (`R-2026-09-08-wpa-mirror`: the
azimuth address in `s5_indexed.py` is mirrored) and the oracle ladder was **re-read under the
corrected address**. Its absolute values are quotable again, and they are not the ones I used.

| rung | **corrected** (tie-group AP, seed 0) | what I wrote |
|---|---|---|
| 16 × 40 | **0.4762** | 0.4713 ⛔ |
| 8 × 20 | **0.3341** | 0.3341 ✅ (correct) |
| `pos_only` control | 0.0316 | — |

⚠️ **And a caveat that matters more than the third decimal:** the **replicate floor is 0.0122 AP**,
and a pure seed replicate — same flags, same address, seed only — reads **+0.0115 "separated"** at
16×40. ⇒ **differences below ~0.012 AP on this ladder are noise**, so the gap my §5 actually depends
on (0.4762 − 0.3341 = **0.1421**) survives with room to spare, and the 0.0049 I had it wrong by is
itself inside the noise floor. The conclusion — *a head on the 160 stride-32 tokens is capped below
the 0.60 bar before training starts* — is unaffected.

⛔ **`SPEC_REFCV6_V2.md` §2 and §6 carry the same stale 0.4713** and are corrected by this erratum
too; they are not separately edited, so that a reader who follows either file's citation arrives
here.

## Why this was caught

By re-reading the record to verify my own landed text, not by anything failing. ⚠️ **Three other
landings tonight quoted numbers I had re-derived from source; these two I did not.** The rule that
would have caught it is one this programme already has — *every number carries its evidence class
and its path* — and a summary is not a path.



---

## E3. §0 — "a measured **14.3 %** one-seed floor" is RIG-SPECIFIC and quotes the MILDEST number we have. **It should be 14.3 %–55.6 %, and the worst case is not a seed effect at all.**

*(Added in the same audit pass, after E1 and E2, by continuing to check rather than stopping at the
first two.)*

**What the parent file says** (§0, standing caveats): *"one-seed separation carries a measured
**14.3 %** floor"*.

**14.3 % is real** — `CLAUDE.md` records it for the **v7-tiny rig**: `separated` differences on
**6 of 42 family cells**, re-derived 2026-09-06. ⛔ **But it is the mildest figure in the record, and
quoting it alone makes the caveat about four times weaker than the evidence supports.**

⭐ **MEASURED on the WP-D rig, and it is worse in kind, not just in degree:** `D0b` — **D0's flags,
D0's SEED, ZERO levers moved** — was itself *separably worse* than `D0` on **5 of the same 9 metrics
(55.6 %)**, reproducing the headline ADE **+0.02610 at +0.02460** and **exceeding the lever on all
three longitudinal metrics**. The two checkpoints went through the identical T1 arm, 3,422 windows /
40 episodes, with `eid` / `window_start` / `gt` / `v0` element-wise identical; mean
`|pred(D0b) − pred(D0)|` = **0.1059 m**.

⇒ **The correction is not a bigger number, it is a different claim:**

| as written | as it should read |
|---|---|
| a **14.3 %** one-seed floor | a **14.3 %–55.6 %** replicate false-positive rate, **rig-dependent**, ⛔ **and refcv6's own rig is UNMEASURED** |
| attributed to **seed** variation | ⛔ the 55.6 % case moved **no seed and no lever** — it is **same-seed replicate noise**, which no seed-control arm would catch |

⭐ **This makes §0's rule STRONGER, not weaker.** "A separated interval is necessary, not sufficient"
was already the rule; what changes is that the honest way to establish a refcv6 result is a
**replicate at the same seed**, not merely a second seed — and that **the floor for these arms has
not been measured and must be**, rather than inherited from whichever rig gives the friendliest
number.

⛔ **And this is the same error as E1 and E2 in a third costume:** I quoted a figure that exists,
from a rig that is not the one under test, without the range or the provenance beside it. It is the
`true-but-wrong-for-the-reader` class — every word defensible, the conclusion the reader draws
wrong.

---

## E4. §12 — "the launch checker exits non-zero on any of these" asserts machinery that, for **4 of the 10**, DOES NOT EXIST

*(Found by continuing the audit into the parent file's claims about CODE, not only its numbers.)*

§12 lists ten refusals and introduces them as *"the launch checker exits non-zero on any of these"*.
That sentence claims a program. ⛔ **There is no single such program for the SPEC-v2 arms**, and the
ten split three ways. MEASURED 2026-09-17 by reading
`…/2026-09-10-refcv6-build/code/` (13 files) and the refcv6 test suite:

| # | refusal | status |
|---|---|---|
| 1 | arms differ in more than `one_variable`, on PARSED namespaces | ⚠️ **EXISTS, WIRED TO THE OLD ARMS.** `prelaunch_gate.check_one_variable` takes `w_agent` / `w_tac_goal` — the 2026-09-10 lever set. The v2 variable is the BACKBONE. **Needs rewiring, not writing.** |
| 2 | a missing control, or one not reading its known value EXACTLY | ⚠️ **PARTIAL.** `verdict_refcv6.py` has the stronger half — absence is `MISSING_DATA`, a verdict distinct from `FAIL`, so a family cannot be deleted to "fix" it. The *known-value* half is per-panel and not centralised. |
| 3 | an unregistered hypothesis id | ⛔ **DOES NOT EXIST.** Nothing reads `GOALS_AND_CLAIMS.md`. |
| 4 | a hyper-parameter selected on the SCORED split | ⛔ **DOES NOT EXIST.** Nothing checks it. ⚠️ **This is the most consequential of the four** — it is the one that silently manufactures a positive. |
| 5 | an estimator field naming `overlapping_holdout_se` | ✅ **EXISTS.** `verdict_refcv6.FORBIDDEN_ESTIMATORS`, plus the T1-stamp refusal. |
| 6 | a hard-coded 640 / 160 / 40 / 20 in the shape path | ✅ **EXISTS AND IS MUTATION-PROVEN.** `test_refcv6_geometry_agnostic.py` scans owned source, requires a written reason for each exemption, and has two mutations (a reintroduced literal is caught; the exemption marker is itself required). |
| 7 | `--conflict-detector on` with no live perception weight | ✅ **EXISTS** — exits 1 *before* `config.json` is written (landed `0243ce4`). |
| 8 | a trunk prefix that selects ZERO parameters | ✅ **EXISTS** (landed `0243ce4`). |
| 9 | a pooled tactical headline, or one quoting a token under n = 200 | ⚠️ **PARTIAL.** `verdict_refcv6.py` knows about pooling; the n = 200 floor is documented in `refcv6_tactical.py:52` (*10 of 22 sit under it*) but is not enforced at report time. |
| 10 | a capped `goal_pos_weight` quoted without the cap sentence | ⛔ **DOES NOT EXIST**, and ⚠️ it may not be enforceable in code at all — it is a rule about PROSE. It should either become a check on the emitted report's own fields, or be demoted from a refusal to a reporting convention. Pretending it is a refusal is what makes it never get built. |

⇒ **4 ✅ · 3 ⚠️ · 3 ⛔.**

⛔ **The correction to §12:** replace *"the launch checker exits non-zero on any of these"* with
**"these ten are the refusals the launch checker MUST implement; 4 exist today, 3 need rewiring
from the 2026-09-10 arm set, and 3 do not exist"** — and add to §11's OWED list:

> ⚠️ **OWED — the launch checker itself.** ⛔ No GPU arm may launch until refusals 3, 4 and 10 are
> built (or 10 is explicitly demoted) and 1, 2 and 9 are rewired to the SPEC-v2 arms. §8 already says
> no pod hour is spent before the D3 package and the RL lever; **this is the third item on that
> list**, and it was the one asserted as done.

⭐ **Why this matters more than the number errors above.** E1–E3 were wrong *evidence*. This is a
wrong *guarantee*: a reader of §12 would believe ten failure modes are mechanically impossible, and
would therefore not look for them. **A refusal that does not exist is worse than no refusal**,
because it displaces the attention that would have caught the thing by hand — which is the same
class as `e4af94f`, *a check that shares the defect it checks for*.

### E4 status update — 2026-09-17, `caf9ca8`

⭐ **Two of the three ⛔ rows are now ✅.** `stack/tanitad/train/prelaunch_v2.py` implements
**refusal 3** (`refuse_unregistered_hypotheses`, reading the real `GOALS_AND_CLAIMS.md`) and
**refusal 4** (`refuse_scored_split_leak` + `refuse_overlapping_splits`). 14 tests, and a
**guard-removal audit: 8 of 8 branches KILLED, 0 escaped, baseline restored green**.

⚠️ **Refusal 4 enforces what is decidable, not what is claimed.** *"Was this threshold fitted on the
scored split?"* cannot be read off a panel. What is enforced: every tuned quantity **declares** its
split (an **undeclared** one is REFUSED, not assumed innocent — silence is how a tuned threshold
passes as a constant), no declaration names the scored split, and the scored split is disjoint from
fit and val, measured on ids. ⇒ **a run can still lie; it can no longer pass by saying nothing.**

⛔ **Refusal 10 is still absent, now deliberately.** It is a rule about **prose** and cannot be
enforced against a report a human writes. It must either become a check on the panel's own fields
(a `pos_weight` entry carrying `capped: true` and a `cap_note`) or be **demoted from a refusal to a
convention** — the PI's call. Leaving it listed as a refusal with nothing behind it is the defect
this whole erratum section is about.

⇒ the §12 tally is now **6 ✅ · 3 ⚠️ (rewiring) · 1 ⛔ (decision)**. §11's OWED item stands: no GPU
arm launches until refusals 1, 2 and 9 are rewired to the SPEC-v2 arms and 10 is built or demoted.

---

## E5. §12 refusal 1 is INSUFFICIENT AS WRITTEN — a parsed-namespace diff cannot see this class of defect

*(Added 2026-09-17 after a defect was found that refusal 1, exactly as §12 specifies it, would have
passed.)*

**What §12 says:** *"arms differing in more than `one_variable` **as parsed namespaces**, not as
intent"*. The emphasis on parsed namespaces is right and was hard-won — diffing argv STRINGS misses
a flag sitting at its default, a `dest=`, a `type=` coercion.

⛔ **But it stops one layer too early.** MEASURED 2026-09-17: `_pin_trainer_cfg` rebuilds
`CNNEncoderConfig` by **hand-listing its fields**. The dataclass has **12**; the list carries **8**.
So whenever `--image-hw` is given, four fields are silently reset to their defaults:

| dropped | resets to | consequence |
|---|---|---|
| `trunk_name` | **`resnet101.a1_in1k`** | ⛔ **`--trunk-name resnet34` BUILDS RESNET101** |
| `trunk_mode` | `shared` | the inflate arm is unbuildable |
| `trunk_fuse` | `concat1x1` | ⛔ `--trunk-fuse last` — the **single-frame control** — is not the control |
| `trunk_fuse_identity` | `True` | ⛔ `--trunk-fuse-plain-init` — the **deliberate regression** — is not deliberate |

⭐ **`--image-hw 256 1024` is the PI's own standing instruction for every future training**, so
every 1024 run took this path — and `config.json` recorded **what was asked for, not what was
built**. §1's **arm B is `resnet34`**: it would have built `resnet101`, made A and B **the same
model**, and reported a clean recipe comparison between two identical arms.

⛔ **Refusal 1 as specified would have passed both.** The defect lives **between argv and the
model**: the parsed namespaces differ exactly as declared — `trunk_name` really is `resnet34` on one
side — and the **built configs do not differ at all**. A check that never looks at what was
CONSTRUCTED cannot see it.

⇒ **Refusal 1 is restated in two parts, and BOTH are required:**

1. **the parsed namespaces** differ in exactly the declared lever (as written today); **and**
2. ⭐ **the BUILT config** — the object the model is actually constructed from, after every pin and
   rebuild — differs in exactly the fields that lever is declared to move, with the values asserted
   as **literals**. A field that silently returns to its default is a VIOLATION, not a bookkeeping
   difference.

⚠️ **Part 2 is the one that bites, and it is cheap**: it is a `dataclasses.asdict` diff of
`cfg.core.encoder` (and its siblings) between the two arms, taken **after** `_pin_trainer_cfg`.

⭐ **Blast radius: ZERO banked arms** — verified: no banked `config.json` carries `trunk_name`, no
argv carries `--trunk-name`, and no refcv6 arm has ever run. ⛔ **No retraction is therefore
warranted** — nothing was claimed on an affected run. The defect was caught *before* the first arm,
which is exactly what §8's *"before any pod hour"* gate exists for, and the honest record is a
defect found and fixed rather than a result withdrawn.

⚠️ The class is one CLAUDE.md already names: *"write the expectation as a LITERAL, never as an
expression over the code under test"* — here a **hand-written field list** standing in for a
dataclass, which rots the moment the dataclass grows. The fix is `dataclasses.replace`, so the
field set cannot drift again.

---

## E6. §5's stride-16 premise — the ORACLE argument survives, the INFERENCE from it does not, and a new control becomes mandatory

*(Added 2026-09-17 from the D3 proof package, `0e947e4`.)*

**What §5 says:** *"an ORACLE on 8×20 tops out at AP 0.3341 against 0.4713 on 16×40, so a head on
the 160 stride-32 tokens is capped below our 0.60 bar before training starts."*

E2 already corrected `0.4713 → 0.4762`. D3 raises something E2 did not touch: **what that number is
evidence FOR.**

### What D3 measured, on `refcv5-v2`'s frozen trunk

| arm | AP@2m |
|---|---|
| `main_s16` seed 0 / seed 1 | 0.0361 / 0.0387 |
| `main_s32` seed 0 / seed 1 | 0.0324 / 0.0386 |
| ⛔ **`mirror_s16`** (the WRONG-address control) | **0.0388 — the highest arm in the panel** |
| `pixel` floor | 0.0299 |
| `shuf_s16` / `shuf_s32` | 0.0141 / 0.0262 |

⇒ **16×40 does not beat 8×20** (slots −0.0007 [−0.0223, +0.0277], heatmap +0.0037 [−0.0097,
+0.0119]) — **the seed spread swamps the stride difference** — and `main_s16` barely clears the
raw-pixel floor.

### ⛔⛔ The finding that outranks the stride question: the MIRROR CONTROL DID NOT LOSE

`main_s16 − mirror_s16` = **+0.0067 [−0.0048, +0.0255], not separated**, and the mirrored arm is the
**highest-scoring** one. This is the `R-2026-09-08-wpa-mirror` class. ⇒ **at this level none of these
AP numbers measures LOCALISATION at all** — a feature map read at a deliberately wrong azimuth
address scores as well as the right one, so the AP is reading something that is not "where things
are".

### ⚠️ How far this transfers, stated precisely, because it is easy to overstate

⛔ **D3 measured `refcv5-v2`'s FROZEN, from-scratch trunk. refcv6's trunk is a DIFFERENT object** —
`timm` ImageNet-initialised and, as of `dadb7e3`, **jointly trained with the map and box losses**,
which is exactly the intervention intended to make the stride-16 map carry more than it does today.
⇒ D3 **does not refute** §5 for refcv6's trunk, and anyone quoting it as though it did is repeating
the scope error this programme keeps making. It is a **strong caution about the premise's evidence**,
not a measurement of the thing §5 claims.

### What actually changes

1. ⚠️ **The oracle ladder survives as an UPPER BOUND on the ADDRESS SPACE and nothing more.** It
   prices what a *perfect* front-end could reach through each grid. It is **not** evidence that a
   real trunk's stride-16 map carries more usable signal than its stride-32 map — and on the one
   real trunk anybody has measured, it does not.
2. ⛔ **§5's inference is downgraded from MEASURED to HYPOTHESIS.** *"Perception hangs on stride-16"*
   remains the design choice, and it is defensible on address-space grounds, but the sentence *"a
   head on the 160 stride-32 tokens is capped below our 0.60 bar before training starts"* is an
   **oracle-derived expectation**, not a measured property of any trunk we have.
3. ⭐ **A NEW REFUSAL, and it is mandatory:** ⛔ **no box-head AP on these tokens may be quoted as
   localisation unless the MIRRORED-ADDRESS control is reported beside it AND is separated WORSE.**
   A panel whose mirror control ties — or wins, as it did here — is measuring something other than
   position, and its AP is inadmissible as a perception claim. This becomes **§12 refusal 11** and it
   applies to `E-REFCV6V2-PERCEP` directly.

⭐ **Why refusal 11 matters more than the stride choice.** Getting the stride wrong costs some
ceiling. Quoting an AP that the mirrored address also achieves would mean **reporting a perception
capability that does not exist** — and `E-REFCV6V2-PERCEP` is precisely a claim that the heads
"reach the trunk" and are "LEARNED". The oracle ladder cannot catch that; only the mirror can.

---

## E7. §6's tactical label figures name a policy the trainer does NOT default to

*(Added 2026-09-17 from the tactical-wiring agent, which corrected me on three counts.)*

**What §6 says:** *"Under the PI's absence-as-negative ruling, 54,253 cells convert ignore →
negative and the head mask goes **17/22 → 21/22** trainable … **9 of 22 tokens now sit ON the
`goal_pos_weight` cap of 50** (3 before)."*

That sentence is **accurate about the ruling** and **silent about which policy an arm actually
runs**. ⛔ MEASURED: `refc_v3_train.py` declares `tac_goal_negatives: str = "measured"` as its
**default**, and the absence-as-negative policy is **opt-in** behind `--cot-negative-sidecar`.

| policy | trainable | on the `goal_pos_weight` cap | under the n = 200 floor |
|---|---|---|---|
| **`measured`** — the trainer's DEFAULT | **17 / 22** | **3** | 10 |
| `cot-absence-negative` — the PI's ruling, **OPT-IN** | 21 / 22 | 9 | 10 |

⇒ **a tactical number must name its NEGATIVES POLICY, not just its split.** A reader planning an arm
from §6 as written would expect 21/22 trainable tokens and 9 capped, and would get 17/22 and 3
unless they passed the sidecar flag. ⚠️ **My brief to the implementation agent stated "21/22" as the
current state, which is worse than the prereg's silence** — it asserted the opt-in policy as the
default.

⭐ **Nothing about the criteria changes.** T-CLASS, T-FLOOR and T-ZERO are per-class and floor-aware
under either policy; what changes is that **every reported tactical figure must carry the policy
name beside it**, exactly as every interval carries its estimator.

### Two more corrections from the same source

1. ⛔ **"The max-speed builder was never run" was WRONG.** It ran on **2026-09-16** —
   `speed_max_window_v6_{train,eval}.meta.json` are banked under
   `…/2026-09-16-refcv6-tactical/`. Only the **payload** was absent from the artifacts directory I
   searched. ⚠️ This is *absence found at ONE location*, the first trap CLAUDE.md names, committed by
   me while briefing an agent about a different gap. The rebuild reproduces that meta's census **to
   the digit**, which is what makes the rebuilt payload trustworthy rather than merely present.
2. ⚠️ **The tactical decoder's parameter count depends on a flag I did not name.** **2,262,020** is
   the `d_bev=256` build. The default `d_bev=128` gives **2,229,252**, and the arm actually
   buildable today — agent-only, because the BEV half is blocked (§E8) — is **2,228,996**. Quoting
   one number without `d_bev` is the units error in a new costume.

## E8. §4's map half is STRUCTURALLY BLOCKED — the decoder learns from agents only

⛔ MEASURED: `refc_v3.RefCV3Model.forward` **never passes `bev_tokens=`** to `self.core(...)` — its
call site passes `**_core_kw`, which carries `scene_hook` alone — and the BEV encoder lives on the
**trainer's wrapper**, running **after** the core forward on `out["fmap_s16"]`. So **no BEV token
exists** at the point where the hook fires.

⇒ `--tac-decoder-d-bev > 0` **REFUSES**, and a run stamps `sources: ["agent"]` /
`bev_tokens_reach_decoder: false`.

⛔ **The PI asked for "the scene embeddings, for the agent AND THE MAP".** Half of that ask is
currently unreachable, and the refusal is the honest behaviour — a decoder silently attending to
agents only, while the record said "agent and map", is the defect this programme keeps finding.

⚠️ **Consequence for `E-REFCV6V2-TACTICAL`:** as it stands the hypothesis can only be tested in its
**agent-only** form. A behaviour that is a property of the MAP — lane keeping, corridor offset —
has no evidence to be learned from, so a per-class result on those tokens is **uninterpretable**,
not merely weak. They must be reported with that scope or excluded from the headline.

⇒ **PI DECISION** (queue item 18): unblocking requires moving the BEV encoder **into the model
forward**, and a ruling on whether the behaviour decoder may **backprop into the shared trunk**.
That second half is a design question about attribution, not a plumbing detail, and is not taken
here.
