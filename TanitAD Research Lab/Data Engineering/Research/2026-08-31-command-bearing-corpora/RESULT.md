<title>RESULT — the gate's "only corpus that could test P2(b)" is wrong, and the cheap test is already unblocked</title>

# RESULT — E-LAB-DATA-0831

`TanitAD Research Lab · Data Engineering · 2026-08-31`
`Seeds: V7_LAUNCH_GATE.md P2(b) corpus blockers · B1 audit open items · D-DATA-EFFICIENCY.`
`0 GPU · 0 spend · nothing downloaded · Thor untouched (k60p30k is live there).`
`All evidence in raw/L2D_SCHEMA_EVIDENCE.md, split into MEASURED / PUBLISHED / our own docs.`

---

## 0. ⭐⭐ THE ANSWER

**`V7_LAUNCH_GATE.md` calls comma2k19 *"the ONLY corpus that could test"* P2(b). That is
incorrect, and the corrected picture makes the gate's own cheapest test runnable now.**

`yaak-ai/L2D` separates realised state from actuation **in its own dataset descriptor**
(MEASURED, re-read today from `l2d-adapter-2026-07-22/l2d_info.json`, which is the corpus's
`meta/info.json`):

| feature group | fields | is it the realised path? |
|---|---|---|
| `observation.state.vehicle` `float32[8]` | speed · heading · heading_error · lat/lon/alt · acc_x · acc_y | **YES** — this is our channel's family |
| `observation.state.waypoints` `float32[10,2]` | 10 future OSM-snapped points | **YES** |
| ⭐ `action.continuous` `float32[3]` | **`gas_pedal_normalized` · `brake_pedal_normalized` · `steering_angle_normalized`** | **actuation, a separate group** |
| ⭐ `action.discrete` `int32[2]` | `gear` · **`turn_signal`** | **a declared intent** |

⭐⭐ **`turn_signal` is the strongest item on that list and it is not the one I went looking
for.** A turn signal is *declared intent*: a driver can signal and not turn, or turn without
signalling, so it is **not recoverable from the path even in principle**. It is discrete,
which is exactly the shape Codevilla's branched conditioning needs (Architecture package F2),
and it is the closest thing any corpus we can reach has to a genuine command.

⭐ **And the gate's triage test needs no pixels.** It asks to correlate a command against the
curvature-derived proxy *"on the same clips"* — a **state-only** computation. L2D's state
lives in `data/chunk-000/*.parquet` (~60 MB each), **entirely separate from the video**.

⇒ **Neither of the gate's two blockers binds this route:**

| the gate's blocker (for comma2k19) | on the L2D state-only route |
|---|---|
| **GEOMETRY** — 65.203° vs our 120°, needs a PI decision | ⭐ **does not arise.** No frame is used. ⚠️ It *does* arise for any L2D **vision** arm — L2D ships **no intrinsics at all** (MEASURED, file tree), which is a *worse* geometry position than comma2k19's known-and-wrong FOV. |
| **DATA** — not on Thor | ⭐ **does not arise.** The state+meta pull is ~155 MB and was already done once, to `C:/Users/Admin/tanitad-data/l2d/`; it runs on the dev box, not Thor. |

⛔ **AND A THIRD BLOCKER, WHICH IS REAL AND IS NOT MINE TO CLEAR.** `LOOP_STATE.md:1025`
records a 🔴 **GDPR gate**: *"real German dashcam footage, NO anonymization statement in the
card … face/plate check REQUIRED before any L2D frame is re-hosted / published."* It binds
frames, publication and any shippable corpus. **It does not bind a read-only correlation over
state parquet** — which is why the recommendation below is state-only, and why it stays
state-only even though a vision arm would be more interesting.

---

## 1. F1 ⭐⭐ — the corpus premise in the gate is wrong, but the honest form is narrower than "I found a new corpus" `[MEASURED + our own record]`

⛔ **I nearly filed a false absence, and the correction is the finding.** My first probe
(`L2D|yaak`, three steering files) returned one hit which turned out to be
**`adaptive_avg_pool2d`** — `pool2d` contains `l2d`. A case-sensitive probe over the whole
`Project Steering` directory returns **9 hits in 3 files**. L2D is **surveyed, adapted,
sliced and roadmapped** by this programme:

- `LOOP_STATE.md:998` — S7 ✅ DONE: *"schema verified (LeRobot v3.0, 100k eps/26.5M
  frames/735h), drive-dedup PROVEN … Apache-2.0 re-verified, 1-drive slice built end-to-end"*
- `REPO_TRIAGE_2026-07-20.md:86` — *"L2D semantic-label survey (Apache-2.0, 4 219 nav cmds)"*
- `LOOP_STATE.md:1024` — *"L2D ships a metric distance-to-maneuver natively"*

⇒ **The correct claim is not "an unknown corpus exists". It is that the JOIN was never made:
no document connects L2D's `action.*` group to P2(b)'s command question, and the P2(b)
scoping written the same day (`bf683a4`) names only comma2k19 and PhysicalAI.** The asset is
**stranded across a document boundary**, which is the failure mode the Agent Operating
Standard was written for — here in its documentary rather than its filesystem form.

## 2. F2 ⭐ — is `steering_angle_normalized` actually a command? UNVERIFIED, and the test that settles it is the same test `[UNVERIFIED]`

⚠️ **I checked for this and the primary does not say.** The HF card (retrieved 2026-08-31)
lists the axis names and **never states where the signals come from** — no CAN, no bus, no
measurement point. So:

- **MEASURED:** the axes are named `gas_pedal_normalized`, `brake_pedal_normalized`,
  `steering_angle_normalized`, in a feature group separate from `observation.state.*`.
- **INFERRED (mine, and I own it):** pedals are actuation and cannot be derived from a path;
  a triple of (gas, brake, steering) is the standard actuation triple; and the fleet is a
  driving school — the card says *"Expert (driving instructors) and student (learner drivers)
  policies"* — so a vehicle-side signal is the plausible source.
- ⛔ **UNVERIFIED:** none of that is the card saying it. **Do not write "L2D has CAN steering"
  in any registry or paper.**

⭐ **The elegant part: the gate's triage test doubles as the provenance test.** Correlate
`action.continuous[2]` against the curvature derived from `observation.state.*`:
- **r ≈ 0.999**, smooth, no residual structure ⇒ it is a derived restatement, L2D adds
  nothing over PhysicalAI on this axis, **and P2(b) is a weak lever** — the gate's own reading.
- **r materially lower**, with residual energy at high steering rate and a dead-band near
  straight driving ⇒ it is a measured actuation signal, **provenance confirmed empirically**,
  and P2(b) becomes testable on an Apache-2.0 corpus.

Either outcome is informative, which is what makes it worth running. **Pearson r is invariant
to the unknown affine scaling of "normalized"**, so the undocumented steering ratio does not
block it.

## 3. F3 ⭐⭐ — the two axes the B1 audit could not fill ship natively here `[MEASURED + PUBLISHED]`

Two of B1's open items are *stratum* problems, and L2D carries both as native per-frame fields:

| our open item | its status in our record | the native L2D field |
|---|---|---|
| **road-class stratum** — D-B1-100H calls it *"the design's own unfilled WORK ITEM, and the axis governing our 88.7 %-longitudinal gap"* | ⛔ unfilled | `observation.state.road` — OSM highway class (`secondary`/`tertiary`/`motorway`/`*_link`/NA) |
| **the >14 m/s regime** — D-SPEED-GATE: `physicalai_r0.py:100` sets `score = 0.0` unless `2.0 <= mean_v <= 14.0`, *"a HARD EXCLUSION and not a down-weight"*, i.e. **every motorway clip is structurally deleted from B1** | ⛔ deleted by construction | `observation.state.max_speed` (posted limit) + `road` = `motorway` ⇒ the excluded regime is **directly selectable** |

⭐ Also native and relevant: `lanes` (count), `surface`, `precipitation`/`conditions`/
`lighting`, and **4,219 distinct navigation instructions** with metric distance-to-manoeuvre.

⛔ **PARITY GUARD, stated before anyone reads this as a proposal to mix.** Nothing here goes
into `physicalai-train-e438721ae894`. Two admissible uses only: **(i)** a data-only
measurement that trains nothing; **(ii)** a separate corpus with its own parity key and its
own registry row. ⚠️ And a cross-corpus arm would confound corpus with regime — Germany,
one vehicle, one camera rig — so it cannot be read as "high-speed data helps".

## 4. F4 — the objective-side precondition on the whole data-efficiency thesis `[PUBLISHED, abstract-only]`

Handed over from today's Architecture package, because it lands on D-DATA-EFFICIENCY and not
on architecture. `2607.27017` (banked today), RH20T, two robots, **4,258 episodes**:

> *"Every arm missing information or prediction pressure stays flat over a fivefold data range"*
> *"additional data improves only the parameters it already acquires"*

⇒ **A data lever cannot be scored honestly on an arm whose objective does not already acquire
the quantity.** Measured over a 5× data range on real robot data, this is the cleanest
external warrant we have for **sequencing objective work before data work** — and it is a
warning about our own planned diversity ablation: run it on an arm that is already acquiring
the target, or it will read flat for a reason that has nothing to do with the data.

⚠️ Abstract-only; robotics, not driving. It supports a *sequencing* argument, not a magnitude.

## 4b. ⭐⭐⭐ A THIRD ROUTE, AND IT IS PROBABLY CHEAPER THAN BOTH — nuScenes ships a CAN channel `[PUBLISHED, verified at full text]`

Surfaced by today's frontier sweep and **verified by me at the paper's full text**:
`2606.12987` *Diffusion Transformer World-Action Model for AV Scene Prediction* (2026-06-11),
which evaluates on **150 held-out nuScenes scenes**, states in §3:

> *"Ego-vehicle actions are extracted from CAN-bus data as 2D vectors aₜ=(steerₜ, accelₜ),
> z-score normalized using training-set statistics only."*

⭐⭐ **That is our exact action parameterisation — `(steer, accel)` — read off the bus instead
of off the curvature column.** For the gate's triage test that is the ideal comparison: the
*form* is held constant and only the *provenance* varies, which is the one-variable version
of the experiment.

**Why it may beat both other routes on cost:**

| route | licence | on hand? | geometry | privacy | triage cost |
|---|---|---|---|---|---|
| comma2k19 | research | ⛔ not on Thor | ⛔ 65.203° vs 120°, needs a PI call | — | blocked as written |
| L2D | Apache-2.0 | ⭐ adapter + 155 MB state pulled | ⛔ **no intrinsics at all** | 🔴 GDPR gate on frames | state-only, minutes |
| ⭐ **nuScenes CAN expansion** | research (nuScenes terms) | ⚠️ **UNVERIFIED — I did not check whether we hold it** | (irrelevant to a state-only triage) | published, anonymised corpus | **likely the smallest download of the three** |

⚠️ **What is UNVERIFIED and must be checked before this is acted on** — three things, and none
of them is a detail:
1. **Which** nuScenes CAN field they used. The paper says *"CAN-bus data"*; it does not name
   the signal, and a steering-wheel feedback channel and a derived road-wheel angle are
   different objects. ⛔ **I did not verify this.**
2. Whether this programme already holds the nuScenes CAN expansion, or would need to fetch it.
3. Whether the nuScenes licence permits our use as we intend. ⛔ Not checked today.

⇒ **This does not replace the L2D route; it adds a candidate that should be priced first
because it is same-form and probably smallest.** The probe in §6 should begin by checking
which of the three corpora is actually in hand.

## 5. What I checked and did NOT find

- ⛔ **No third command-bearing corpus was found that is both reachable and licence-clear.**
  Two probes: (i) our own corpus inventory (`ASSET_INVENTORY.md` family, TanitDataSet tier
  docs, the Data Engineering survey notes); (ii) the steering-layer scan in Tier D. ⚠️ This is
  the two-probe minimum and therefore a **weak** absence — I did not run a fresh external
  survey of driving corpora today, so "only comma2k19 and L2D" is *my* finding about *our*
  reachable set, not a statement about the field.
- I did not verify that the L2D state parquet actually contains non-null steering — the
  local pull was state + meta at 155 MB in July and I did not re-open it. **The first step of
  the probe below is to check that before anything is concluded.**

## 6. ⛔ ESCALATION — one correction, one probe, three guards

### The correction (a Project Steering edit, not mine to make)

**`V7_LAUNCH_GATE.md` §P2(b) says *"TWO BLOCKERS ON THE ONLY CORPUS THAT COULD TEST IT"*.**
There is a second candidate; its blockers are different ones (no intrinsics; a GDPR gate on
frames); and **the state-only triage the gate itself calls "the cheapest discriminating test"
is blocked by none of them.**

### The probe I recommend — and deliberately did not build

**`E-DATA-CMDPROV`: is L2D's steering a command or a restatement?** DataFlyWheel owns it.

| | |
|---|---|
| **input** | L2D state parquet, already pulled (`C:/Users/Admin/tanitad-data/l2d/`), ~155 MB, **no video** |
| **compute** | dev box, CPU, minutes. ⛔ **No GPU. No Thor. No HF metered call. No download if the July pull is intact.** |
| **step 0** | verify the parquet has non-null `action.continuous` and `observation.state.*` — ⛔ if it does not, stop and report; the July pull's coverage is unverified |
| **step 1** | derive curvature from `observation.state` per drive; compute r against `action.continuous[2]`, **stratified by lateral acceleration and steering rate** — an unstratified r is dominated by straight driving, where the two coincide by construction, so a pooled number would report "no independent information" for a reason that is an artifact of the regime |
| **step 2** | the same r on `physicalai` between our own `steer_road_rad` and its curvature source — ⭐ **the control that must read ≈1.0**, since `physicalai.py` computes one from the other. If it does not, the rig is wrong and step 1 is uninterpretable |
| **reads** | r ≈ 0.999 stratified ⇒ P2(b) is a weak lever, **and the gate can say so with evidence**. r materially lower ⇒ a command channel exists on a corpus we can reach |
| **both outcomes** | pre-committed above, before any number exists |

### The guards, so this is not read as more than it is

1. ⛔ **PARITY** — data-only, or a separate parity key. Never a re-selection.
2. ⛔ **GDPR** — state only. No frame is opened, re-hosted or published. The gate at
   `LOOP_STATE.md:1025` stays open and this package does not touch it.
3. ⛔ **PROVENANCE** — until step 1 runs, "L2D has a command channel" is **UNVERIFIED** and
   must not enter `MODEL_REGISTRY.md`, the paper, or a claim row.

## 7. Deliverable manifest

| artifact | where | only-one-place? |
|---|---|---|
| `SPEC.md` | `repo:TanitAD Research Lab/Data Engineering/Research/2026-08-31-command-bearing-corpora/SPEC.md` | no — staged |
| `PLAN.md` · `RESULT.md` (this) · `COMMS.md` | same dir | no — staged |
| `raw/L2D_SCHEMA_EVIDENCE.md` (descriptor fields, card quotes, the corrected absence probes, the GDPR gate) | same dir | no — staged |
| KB line | `repo:TanitAD Research Lab/Data Engineering/Research/KNOWLEDGE_BASE.md` | no — staged |
| ⚠️ the July L2D state pull | `devbox:C:/Users/Admin/tanitad-data/l2d/` | ⛔ **YES — one place, off-repo, unverified since 2026-07-22.** Not rescued here (155 MB of third-party data under an open GDPR gate does not belong in the repo); flagged so the probe's step 0 re-checks it rather than assuming it. |

**No code written. No corpus downloaded. No commit, no push, no branch switch.
`Mission Plan.md` untouched.**
