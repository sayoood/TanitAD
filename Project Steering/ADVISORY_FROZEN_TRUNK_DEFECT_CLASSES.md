# ⛔ PROGRAMME ADVISORY — six defect classes found in REFe in one day, and why they are NOT REFe-specific

**Raised by the PI, 2026-09-20.** *"Since we found a lot of bugs and issues how the trunk is used
both as input and regarding the decoding and also 3D position encoding — check it for our models
like refcv6, v7f, refav1, even if the architectures are different."*

**Scope.** REFe is a DriveZero reproduction and shares almost no architecture with refcv6, v7f or
refav1. **That is the point.** Every class below is a property of *how a pretrained encoder is fed,
positioned, read out and guarded* — not of any particular network. Each was found by MEASUREMENT on
REFe, each had survived at least one green test suite, and each has an obvious analogue in a model
that looks nothing like REFe.

⚠️ **None of the numbers below are claims about refcv6/v7f/refav1.** They are REFe measurements,
quoted so the owning stream knows what the failure *looks like* and what size of error to expect.

---

## A · The frozen trunk's INPUT — normalisation

**MEASURED on REFe.** The DINOv3 ViT-L checkpoint's own `config.json` declares
`mean [0.485, 0.456, 0.406]`, `std [0.229, 0.224, 0.225]`. A grep over every model file returned
**zero** normalisation. Feature displacement against the correct operating point: **rel L2 0.562 /
cos 0.841** — *larger than deleting the positional encoding entirely* (0.529) and **2.9× a BGR/RGB
channel swap** (0.194).

⭐ **Why it is invisible.** Nothing errors, nothing fails to load, the loss still falls. A frozen
trunk at the wrong operating point simply returns worse features for the whole run, and any
backbone comparison measures your preprocessing instead of their encoder.

**CHECK IN YOUR MODEL:** for every pretrained encoder you load — find the checkpoint's declared
preprocessing and assert it is applied. Also check the CHANNEL ORDER (`cv2.imread` is BGR) and the
value range (`[0,1]` vs `[0,255]`).

---

## B · POSITION / GEOMETRY encoding

Three distinct failures, all in the same family: **a positional scheme that is not the one the
weights were trained with, or not the one the name claims.**

1. **A random frozen table substituted for the pretrained scheme.** DINOv3 computes rotary position
   encoding on the fly and stores **no** positional tensor. We had a `trunc_normal_` table with
   `requires_grad=False`. Arithmetic that exposed it: our backbone totalled 305,045,504 against the
   checkpoint's 303,079,424, and the difference **1,966,080 = exactly 1920 × 1024** — the table was
   the ONLY trunk tensor with no checkpoint counterpart.
2. **A "3D position embedding" that was a 2D table.** Our per-token learned table could not
   condition on camera geometry at all, while the corpus has **2 distinct intrinsics and 22–24
   distinct extrinsics** across 1,349 logs. The paper's own citation was to PETR, whose encoding is
   an MLP over *analytically computed* frustum coordinates.
3. **Correct intrinsics applied at the wrong resolution.** Native 1920-wide intrinsics on a
   960-wide input gave a **31.34° horizontal field of view instead of 62.85°**, off-centre, with x
   never crossing zero.

⚠️ **And a rotation that is self-consistent but not theirs is worse than none.** Our first rotary
implementation was internally coherent and passed all three of our assertions, while moving the
frozen features **farther** from their pretrained function (0.760) than omitting the encoding
(0.585). Only a comparison against the *released source* settled it.

**CHECK IN YOUR MODEL:** does any positional/geometric tensor exist that has **no counterpart in the
checkpoint**? Does anything named "3D" actually consume geometry (K, extrinsics), or only an index?
Are intrinsics expressed at the resolution the network actually sees?

---

## C · DECODER WIRING — the bottleneck that was not a bottleneck

**MEASURED.** Learnable registers meant to *compress* visual tokens into a small scene set were
**concatenated** instead: `cat([tokens, registers])`. Forward hooks showed every decoder layer
receiving `visual + 16` tokens with the **identical storage pointer** — **1,936 tokens instead of
16**, a 121× larger cross-attention context, and no decoder ever saw the compressed set. The
registers were 16 free bias parameters.

A second wiring defect in the same file: the scoring branch was fed the **proposal queries** rather
than the **candidate trajectory** it is supposed to score — judging intent instead of path.

**CHECK IN YOUR MODEL:** put a forward hook on every cross-attention and PRINT THE CONTEXT SHAPE.
If two modules are meant to read different tensors, assert the storage pointers differ. Do not infer
the wiring from the code; measure it.

---

## D · UNITS, RATES AND TIME BASES

Four instances in one file, each producing a plausible number:

| error | consequence |
|---|---|
| L1 distance averaged over metres **and radians** together | 1 rad priced as 1 m; a constructed case flips which proposal wins |
| 0.2 s displacement divided by the engine's 0.1 s step | every derived velocity **exactly 2× too large** |
| history assumed 10 Hz when the feature builder resamples to **0.2 s** | the seed spanned 0.4 s while dividing by 0.2 |
| a metric's window defined in frames, with the frame rate wrong | the window width depended on `--stride`, a *cost knob* |

⭐ **The only check that catches these is an IDENTITY, and it must not be built from the quantity
under test.** Our first guard compared derived speed against `displacement / (T · dt)` — but the
derived speed *is* displacement/dt, so `dt` cancels and it passed at 0.1 as readily as 0.2. The
working version compares against the **log's own recorded speed**, which knows nothing about our
cadence. ⚠️ And compare like with like: a 4 s mean against an instantaneous value differs legitimately
under acceleration, and we failed correct code that way too.

**CHECK IN YOUR MODEL:** for every derived quantity, name the rate it assumes and the object that
owns that rate. Add one identity per derived quantity, against an INDEPENDENT reference.

---

## E · READOUT AND AGGREGATION

- **Collapsing a nested result to one key.** Each calculator writes `{"reward":…, "info":…}`; the
  `reward` is weighted and saturated and flat *by design*, `info` carries the signal. Reading only
  `reward` produced *"0 of 9 signals differ"* — a statement about the reader, not the instrument.
- **Aggregating a NOMINAL category with `max`.** Codes 0/1/2/3/4 are different events, not a
  severity ladder; `max` returned 4 where the consumer tested for 1, and **100 of 1,066 rows** read
  as clean.
- **Aggregating over prefixes where the estimator is not yet valid.** The comfort estimator switches
  at 10 steps; aggregating from 3 made the whole lateral family read identical to a deliberate
  violator.
- **Stateful calculators reset every call.** Latches that accumulate over a window can never latch
  if the object is rebuilt per evaluation — two consumed components were scored as if every step
  were the first.

**CHECK IN YOUR MODEL:** enumerate what your metric code WRITES (not what you read). Ask of every
aggregation: *is this quantity ordinal?* And: *does this calculator carry state between calls?*

---

## F · GUARDS THAT CANNOT GO RED — the class that let all of the above survive

Every defect above passed a test. The tests shared these shapes:

1. **The expected value is an expression over the code under test.** `assert x == f(x)` measures
   determinism, not correctness.
2. **The probe hardcodes the value it reports** instead of calling live code — a screenshot, not an
   instrument.
3. **The test STUBS the method under test.** Our planner guard stubbed the very function that was
   broken, and the break survived three reviews.
4. **The assertion is satisfied by any member of a class.** Three rotary assertions (table gone,
   vectors moved, norm preserved) pass for *any* rotation, including a random one.
5. **An allow-list excuses an absence nobody re-reads.** One entry kept a strict loader green over
   a defect for as long as the entry existed.
6. **The regression arm restates the code it guards**, and drifts every time that code changes —
   ours drifted **three times in one day**. Fix: give the real function a flag and flip it.

⭐ **The cheapest discriminators, in order of strength:** an **analytic target** whose answer is known
by construction; an **independently authored** reference; a **mutation** that reintroduces the real
historical defect and must go RED.

---

## ⛔ TWO MORE, LANDED 2026-09-21, IN THE ADVISORY'S OWN PACKAGE

Both are class F, and both are worth quoting because they show the class does not spare the people
writing about it.

1. **`build_targets.py` crashed on its SUCCESS path.** The four-camera extension left one
   single-camera line behind -- `cam_ts.size` where `cam_ts` had become a `{channel: ndarray}`
   dict -- so the builder that feeds training raised `AttributeError` on the **first step of the
   first log**. The *refusal* path (a rig missing a camera) ran clean, which is why nothing caught
   it: the code that handled the exceptional case was exercised and the code that handled the
   normal case was not. ⚠️ The extension had been written, reviewed and documented; what was
   missing was an instrument that RAN it.
2. **The image index matched `_CAM_F0.zip` only**, so three of four camera containers were
   invisible. Worse, `FrameStore.read` returned `None` on a container `KeyError` **instead of
   falling through** to the loose-file path -- so the mere presence of a front-camera container
   made every side/rear frame unreadable *even when its loose file was sitting right there*. The
   symptom would have been a 100 % `missing_image_file` rate, i.e. a **data** problem reported
   where a **code** problem lived.

⭐ **The general instrument that closes this is now written, self-tested, and worth porting:**
`…/2026-09-20-refe-plan/refe/diag_consumer_conformance.py`.

The structural point it answers is bigger than either bug. **Every other instrument in that package
built its inputs FROM the config it was testing** -- `randn(B, cfg.n_cameras, 3, cfg.img_h,
cfg.img_w)`, `randn(B, cfg.ego_dim)`. That makes each one internally consistent and *structurally
incapable* of noticing that a CONSUMER disagrees with the config. It is the same shape as an
expected value written as an expression over the code under test: it measures determinism, not
correctness. MEASURED: an `ego_dim` change broke the trainer and the planner while **every guard
stayed green**, and the only symptom was `mat1 and mat2 shapes cannot be multiplied (2x44 and
45x256)` -- a matmul error three frames deep naming neither the bank nor the field.

The instrument's rule: **an expectation is a LITERAL, or it is read from a DIFFERENT consumer --
never from the config.** Real artifacts are read from disk; code consumers are recovered from
SOURCE by AST rather than called with config-derived inputs. It carries 8 arms and a mutation
self-test in which each arm must go red **alone**, with any legitimate collateral written out as a
literal rather than waived.

⚠️ **And the self-test caught its own author.** My first version of arm 1's mutation re-derived a
condition from the live bank (*"is the bank 8-D?"*) instead of feeding the arm a mutated input. It
read GREEN and announced itself inert -- the exact defect class the file exists for, committed
inside the file.

---

---

## ⛔ FOUR MORE, LANDED 2026-09-22 — and they add a SEVENTH class

All four came out of one day building the navtrain target path. Three are class F. The fourth is a
class this advisory did not have, and it is the most dangerous of the set because the artifact it
produces passes every check we own.

### G · A FILTER WHOSE SEMANTICS ENCODE ITS *OWN* CONSUMER'S REQUIREMENT, NOT YOURS

Two independent instances in one pipeline, each deleting a quarter to a half of the data, **in a
BIASED way, silently**. Nothing errors. The surviving bank is internally consistent. Every gate
stays green. The only symptom is that the dataset is smaller than it should be — and nobody knows
what it should be.

1. **`get_scenarios_from_db` (the devkit's own token query) discarded 25.4 % of navtrain.** It
   restricts results to `valid_scenes`: `row_num >= 3 AND row_num < cnt - 1`, i.e. the first two and
   last two SCENES of a log. I read that as cheap because a nuPlan scene is commonly ~20 frames.
   **MEASURED, a scene here is ~350 rows**, so the rule costs `4 / n_scenes` — and our DBs are short
   slices. Result: **13,565 of 18,179 frames kept, 19 logs yielding ZERO**, with the loss falling
   hardest on the SHORTEST logs. The rule is a coarse PROXY for "this frame has room for history and
   future"; our real requirement is 2.0 s back and 4.0 s forward. Replacing the proxy with the
   requirement, asserted per frame, restored **18,179 / 18,179**.
2. **A scorer "abort" discarded 44 % of navtrain frames that were MAXIMALLY usable.** The condition
   was `not (ok_curb and ndiff >= 3)` under the message *"scorer could not tell candidates apart"*.
   Splitting the counter showed **8/8 aborts were `!ok_curb` and 0/8 were `ndiff < 3`** — and those
   very frames had **19–20 signals separating**. The message was false for every one. `ok_curb`
   demanded that an analytically-aimed over-curb candidate actually read off-road: a property of
   ROAD GEOMETRY, which wide navtrain roads defeat and curated val14 scenes never did (**0 aborts
   there, ever**). ⚠️ The file's own rule, six lines above the defect, already said *"refuse a frame
   only when the scorer cannot tell the candidates apart at all"* — **the principle had been applied
   to one term and not to its neighbour.**

**CHECK IN YOUR MODEL:** for every filter you did not write — state what it was protecting ITS
author against, and whether that is your requirement. Then **count what it removes, and test whether
the removal correlates with anything** (length, geometry, city, class). A uniform 25 % loss is a
smaller problem than a biased 25 % loss, and only the correlation test distinguishes them.

### F (continued) · three more guards that could not go red

3. **A gate printed `[PASS]` on a comparison of ZERO items.** Two arms compare a bank's goals against
   a closed-loop run. navtrain has no closed-loop run, so nothing overlapped — and the expressions
   `n0 == 0 or ...` / `nlate == 0 or ...` were **vacuously true**. `G1 0/0 [PASS]`, `G2 0/0 [PASS]`.
   ⚠️ **My first fix did not work, and read green:** I gated on the reference dict being EMPTY, but
   it is loaded from a val14 run that exists on disk, so it is populated with keys that simply never
   match. **The condition is OVERLAP, not emptiness.** Testing the container instead of the
   comparison is the same mistake one level up. Arms now return `None` = N/A, are excluded from the
   verdict, and the marker itself carries the deficit (`SIGNALS_CONSISTENT_PARTIAL_2_ARM_NA`) so a
   green line cannot be quoted as a full pass.
4. **A control that could only FAIL.** An equivalence check compared `repr(EgoState)` — and
   `EgoState` defines no `__repr__`, so it compared **memory addresses**, reporting 1,852 / 1,875
   "disagreements" it would have reported however correct the code was. This is the
   `math.dist(cam_xy, cam_xy)` tautology with the sign flipped. ⭐ **A control that cannot come out
   the other way is not a control** — in either direction.
5. **A passing gate that exited non-zero.** A decorative character in a `print` raised
   `UnicodeEncodeError` on the cp1252 console **after every arm had passed**, making a green run
   indistinguishable from a failure. ASCII only in printed strings.

### And two explanations published before they were tested

Both were plausible, both were wrong, both were refuted by measurement within the hour — but both
had already been stated as the likely cause.

* *"The aborted frames are stationary ones where candidates collapse."* **REFUTED:** aborted speeds
  span **0.042–6.641 m/s** against scored **0.034–7.539 m/s**, fully overlapping, with adjacent
  pairs splitting. ⚠️ It mattered: the wrong story made the loss look benign (*boring frames*) when
  it was actually removing frames by road geometry — biased toward exactly the open, high-speed
  scenes the split is full of.
* *"A log's byte range in the tar can be bracketed by binary search."* **REFUTED:** measured at 9
  offsets, the logs are interleaved arbitrarily (5 of 9 out of order). The sortedness had been
  inferred from ONE probe that read the first 12 members — all necessarily from the first directory,
  which can say nothing about global order. *(Same shape as the R9 history-buffer false null and the
  R11 single-log control: a property seen at one operating point, generalised to the artifact.)*

⭐ **The rule both violate:** a mechanism that explains the data is not evidence for that mechanism.
State it as a hypothesis, name the measurement that would refute it, and run that measurement before
anyone plans around it.

### One more, and it is about tooling scope

6. **A tool validated on exactly one artifact, assumed for all of them.** `fetch_front_camera.py`
   range-fetches frames by reading a ZIP central directory. MEASURED from each archive's first 288
   and last 256 bytes: **only nuPlan's `mini` camera archive is a ZIP**; `test`, `val` and `train`
   are **TAR archives named `.zip`**. The tool had only ever been run on mini — and the handoff's
   `~73 GB test / ~99 GB val` figures beside it were **projected from mini, never indexed**. They
   could not have been. ⚠️ **A number that was never obtained looked exactly like a number that had
   been.** **CHECK:** for every tool, name the artifacts it has actually been RUN against, and mark
   every figure derived by projection as ESTIMATED at the point of use, not in a footnote.

---

## What each stream is asked to do

⛔ **This is not a request to re-audit everything.** It is seven specific questions, each answerable in
under an hour, for **refcv6 · v7f · refav1** and any arm with a pretrained or frozen encoder:

1. Is the checkpoint's declared **normalisation** applied? Channel order and value range too.
2. Does any **positional/geometric tensor** exist with no checkpoint counterpart? Does anything
   named "3D" consume real geometry?
3. **Hook every cross-attention** and print the context shape. Does it match the design?
4. For each derived quantity: what **rate** does it assume, and who owns that rate? One identity
   each, against an independent reference.
5. What do the metric calculators **write**, and is anything ordinal being aggregated with min/max?
   Do any of them carry state?
6. For each guard: **construct the regression it should catch** and confirm it goes RED.
7. For every FILTER you did not write (devkit, library, upstream tool): what was it protecting
   ITS author against, is that your requirement, **how many rows does it remove, and does the
   removal correlate with anything**? A biased 25 % loss and a uniform one look identical in a
   row count.

**Report findings into `GOALS_AND_CLAIMS.md` with evidence class, and log any retraction in
`RETRACTION_LOG.md` with its root-cause class — not just the correction.**

Evidence for every REFe number above:
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/` — `PAPER_CONFORMANCE_REVIEW.md`,
`FIX_VERIFICATION_REVIEW.md`, `REVIEW_3_FULL.md`, `REVIEW_4_FINAL.md`, `MODULE_SIZING_STUDY.md`,
and the `refe/diag_*.py` instruments.
