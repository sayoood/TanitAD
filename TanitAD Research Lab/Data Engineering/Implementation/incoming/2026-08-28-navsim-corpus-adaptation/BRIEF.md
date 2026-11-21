# COMMISSION — NavSim → TanitAD corpus adaptation (DataFlyWheel)

**Commissioned by** TanitAD_EvalFlyWheel on **PI direction, 2026-08-28**:
*"We need to adapt the data to what we need, ask the data fly wheel agent to extend
the data pipeline for it and generate a eval data set in production grade."*

⛔ **Carries the `Project Steering/AGENT_OPERATING_STANDARD.md` preamble by
reference**: (1) STAGE, NEVER PUSH; (2) end with a DELIVERABLE MANIFEST naming where
every artifact lives; (3) ESCALATE integration, never bury a "please merge" in a doc.

---

## 1. Why this exists — the readiness gap, measured

We can run NavSim's harness but we have **no model that can consume NavSim's input**.
NavSim hands an agent 8 cameras at 1920×1080 **pinhole**; our arms are trained on a
`CanonicalFrame`. Pointing a PhysicalAI-trained checkpoint at NavSim zero-shot would
produce a number that measures the **domain gap, not our capability** — and publishing
that would be exactly the non-comparable figure our `navsim.modality_label` gate exists
to prevent.

The PI's ruling is therefore to **adapt the data**, not the model. That is your product
(P2 data pipelines), and this brief states the eval-side contract it must satisfy.

**Already done, so you do not redo it:**
- devkit cloned and pinned: `autonomousvision/navsim@0a380a9` (2025-10-27), local at
  `C:\Users\Admin\navsim\devkit`.
- `warmup_two_stage` downloaded and **content-verified**: 1.16 GB, 4454 entries, at
  `C:\Users\Admin\navsim\data\navsim_v2.2_warmup_two_stage.tar.gz`.
- Protocol fully specified: `products/P7-TanitEval/benchmarks/NAVSIM_PROTOCOL.md`
  (900 lines, primaries banked).
- An eval-side adapter skeleton exists: `taniteval/adapters/navsim.py` (53 tests, but
  **synthetic fixtures only — it has never seen real data**). It is standalone numpy and
  does NOT import the devkit.

⚠️ **MEASURED sizes** (HTTP content-length, not the docs table, which is wrong in both
directions): `warmup_two_stage` **1.16 GB** · nuPlan maps **0.97 GB** · OpenScene test
metadata **0.48 GB** · camera shards ×32 **~106 GB** (5 sampled, 2.33–4.46 GB) · LiDAR
shards ×32 **~84 GB** · `navhard_two_stage` **38.5 GB** (docs say 31 — they are LOW).
⭐ **LiDAR is a wholly separate download and the navsim-v2 archives contain NO LiDAR at
all** (verified: the warmup archive holds 8 `CAM_*` directories and nothing else). So
camera-only is structurally supported and saves ~84 GB — it is not a compromise.

---

## 2. ⭐ THE HIGHEST-VALUE PART: the build can close two blocking gates

This is not an image-conversion job. Two of the three NavSim gates in
`products/P7-TanitEval/CRITERIA_REGISTRY.json` v2.3.0 currently **block any NavSim
number from being reported**, and both are closable *in the data build* — which is far
cleaner than closing them downstream.

### 2.1 `navsim.estimator_unit` — the overlapping-scene problem

⛔ Our decision-grade interval is the **episode-cluster bootstrap**. It does not
transfer: NavSim's unit is a **scene token**, and **NavSim scenes are explicitly allowed
to OVERLAP** (PUB-DEVKIT `docs/splits.md`: *"NavSim splits contain overlapping
scenes"*). Resampling overlapping units as if independent understates variance — the
same family as `overlapping_holdout_se`, which this programme already retracted for
biasing the **point estimate** bidirectionally, up to a sign flip.

⇒ **Deliverable:** a **non-overlapping grouping key** per sample — the NavSim analogue
of our episode id — derived from the source **log / vehicle / time-contiguity**, such
that two samples sharing a key are dependent and two with different keys are not.
Publish the key, how it was derived, and the resulting group-count and group-size
distribution. If a clean grouping is impossible for some subset, say which and why —
an honest `UNAVAILABLE` beats a grouping that quietly overlaps.

### 2.2 `navsim.ego_enforcement` — make vision-only structurally true

⛔ MEASURED from the devkit: **there is NO switch that removes ego status.** `EgoStatus`
(`ego_velocity`, `ego_acceleration`, `ego_pose`) is always populated on `AgentInput` and
nothing in the framework verifies an agent declined to read it. Our vision-only rule (PI,
2026-08-03, binding) is therefore ours alone to enforce.

⇒ **Deliverable:** build the eval set so ego is **physically absent from the inference
stream** — ego/privileged fields in a **separate sidecar** the model's input path cannot
reach, not merely a flag. Labels MAY use ego (that is allowed and expected); *inference*
may not. Done this way, "we did not read ego" stops being an assertion and becomes a
property of the artifact — which is exactly what the gate demands.

---

## 3. The target geometry — a DECISION, not a constant I can hand you

⚠️ **Read this before assuming.** I checked the source rather than trusting prose, and
the programme's own frames disagree in a way that matters:

- `stack/tanitad/data/calib.py` → `CANONICAL_256 = CanonicalFrame(height=256, width=256,
  f_ref=266.0, projection="pinhole")`, and its docstring says **"Every number this
  program has published is this."**
- `stack/tanitad/data/physicalai.py` supports `PROJECTION_MODE_CYLINDRICAL` and the
  corpus's native wide frame; the FOV geometry there is **cylindrical** (column linear in
  azimuth), where `HFOV = 2*(W/2)/f_ref`.

⛔ **The pinhole FOV formula is WRONG on a cylindrical projection** — it reads 92.6° where
the true field is 120°, and the corpus rig is literally named `camera_front_wide_120fov`.
State the projection before using any camera formula.

⇒ **You must state which `CanonicalFrame` the NavSim corpus targets, and why**, then
build for that frame explicitly. `calib.py` already makes the frame a first-class object
whose tag travels config → cache build → trainer → encoder, so a cylindrical build can
never collide with a crop build of the same size. **Use that mechanism; do not
re-declare a geometry as a literal** — the docstring records that threading geometry
through default arguments is what produced this programme's unreproducible v4 numbers.

**The conversion itself:** NavSim `cam_f0` is pinhole with published intrinsics/extrinsics
per `Camera`. Reproject into the chosen frame — an explicit reprojection, ⛔ never a
resize. Verify with the harness's own frame checker (`lateral.assert_axis_convention`,
`taniteval/taniteval/lateral.py:170`).

---

## 4. What "production grade" means here (your charter's own criteria)

1. **Provenance manifest** — every sample traceable to its NavSim scene token, source log,
   split, and devkit SHA.
2. **Licence class** stated once. NavSim/OpenScene/nuPlan are research-use; the programme
   has research character (PI, standing) — ⛔ do not hedge, do not refuse.
3. **Skip-hash / corpus identity** so the set is quotable and cross-arm comparable.
4. ⛔ **PARITY: this is a NEW corpus and MUST be flagged NON-PARITY explicitly.** It is not
   `physicalai-train-e438721ae894` / skip-hash `f09e44db`. Anything that lets a NavSim set
   be mistaken for the parity corpus must be refused.
5. **Verify by CONTENT** — never by file count, name, or exit code. Assert on the bytes:
   sample rows, require non-zero, print the mean. ⚠️ MEASURED precedent: a decode that
   raised into a pre-allocated memmap left a full-size file of **all zeros** while the job
   exited 0 — and because it was the *floor* arm, every trunk would have appeared to beat
   it. A silent false-positive generator, not a missing number.
6. **Deterministic and reproducible** from a recorded recipe.

---

## 5. Pre-registered outcomes (§3 work-package schema — commit BOTH before building)

| outcome | what it means | what we do |
|---|---|---|
| **A — clean grouping exists** | a non-overlapping key covers ≥95 % of samples | build it; NavSim CIs become admissible and `navsim.estimator_unit` closes |
| **B — grouping is partial** | a clean key covers only part of the split | build it, report coverage per subset, and the gate stays open for the remainder with its reason and n |
| **C — no valid grouping** | overlap is irreducible | ⛔ **we report NavSim point estimates with an explicit `UNAVAILABLE` interval, and say so publicly.** This is an acceptable outcome — it is not a failure, and it must not be papered over with an invented CI |

⚠️ Outcome C is genuinely acceptable. Do not bend the data to manufacture A.

---

## 6. Scope — and what is explicitly NOT yours

**In scope:** the pipeline extension + a production-grade **EVAL** dataset from
`warmup_two_stage` first (1.16 GB, already on disk — prove the whole path on it), then
`navhard_two_stage` (38.5 GB) and/or `navtest` (~106 GB camera-only) on the PI's go.

**NOT in scope, do not start:**
- ⛔ Training data / `navtrain` (445 GB) — a separate decision.
- ⛔ Downloading `navtest` or `navhard` — the PI has approved provisioning, but start with
  the warmup split that is already local; the large pulls are mine to run and are
  sequenced behind a working end-to-end path.
- ⛔ Any change to `taniteval/adapters/navsim.py` or the criteria registry — those are
  EvalFlyWheel-owned. Tell me what the adapter must change and I will change it.
- ⛔ No metered compute, no HF GPU. Local only.

---

## 7. Deliverable

A work package under
`TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-08-28-navsim-corpus-adaptation/`
with `SPEC.md` (both outcomes), `code/`, `tests/`, `raw/`, `RESULT.md`, `COMMS.md`; the
pipeline extension staged in `stack/tanitad/data/`; and the built eval set with its
manifest. Every claim carries its evidence class. **Stage, never push.**

⭐ **Report back to the EvalFlyWheel with:** the grouping key and its coverage, the frame
you targeted and why, the ego-separation mechanism, and what the adapter must change.
Those four things are what unblock a real NavSim number.
