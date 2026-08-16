# STALE BLOCKER SWEEP — a root-cause-class elimination, plus the fourth rot of one prose number

- **Date:** 2026-08-16 · **Discipline:** Architecture & Inference · **Status:** PENDING orchestrator triage
- **Evidence class:** MEASURED (ours, from source + raw artifacts in this repo) unless a row says otherwise.
- **Compute:** CPU-only, dev box, $0. No GPU touched (Thor is training a 336M model on the only GPU).

---

## 0. The class being eliminated

> **"A document that states a blocker is never revisited when the blocker clears."**

This is not a hypothetical. It has now cost the programme three times:

| # | incident | cost |
|---|---|---|
| 1 | An orthogonality instrument whose merge request lived in a README nobody re-read | **10 days** unmerged |
| 2 | `…/incoming/2026-08-03-longitudinal-distance-keeping/INTAKE.md:73-75` — *"Until that lands, arm evals will still report the family UNAVAILABLE"* | **a whole agent commissioned 2026-08-16 to rebuild an instrument that already existed** |
| 3 | `CLAUDE.md`'s *"our ingest reads N of 36 features"* | **rotted 4×**, propagating a wrong number into **17 documents + 1 code docstring** |

⭐ **The finding that ties the sweep together: incidents 2 and 3 have the SAME TRIGGER.**
`obstacle.offline` became a real read on **2026-08-03**. That single landing invalidated *both* the
INTAKE's blocker line *and* the prose feature count. **Neither was revisited**, because nothing in the
programme points *forward* from a blocker to the thing that clears it.

⚠️ **The asymmetry is the mechanism, and it is worth stating precisely.** The closing work *did* cite
the blocker: `taniteval/taniteval/lead_source.py`'s own docstring names this INTAKE's open work item
as its reason for existing. **The reference is one-directional.** The successor knows about the
predecessor; the predecessor never learns it has been superseded. Every instance in the table above
has this shape. ⇒ *A "blocked on X" line is a claim with an expiry date and no alarm attached.*

---

## 1. VERDICT TABLE — the flagship item (lead agent's own probes)

| doc:line | claim | VERDICT | clearing evidence |
|---|---|---|---|
| `…/incoming/2026-08-03-longitudinal-distance-keeping/INTAKE.md:73-76` | *"Wiring it into the eval path (val40 windows → `win["lead"]`) needs the `obstacle.offline` chunks for the 40 val episodes… Until that lands, arm evals will still report the family UNAVAILABLE"* | ⏹ **CLEARED 2026-08-03**, measured through 2026-08-14 | **3 independent probes**, below |
| same file `:3` | *"**Status:** PENDING orchestrator triage"* | ⏹ **CLEARED — landed & in production use**; the ORCHESTRATOR VERDICT block was simply never filled in | an empty verdict block is **not** evidence of rejection |

**Probes for the flagship row (all MEASURED):**

1. **The wiring exists.** `taniteval/taniteval/lead_source.py` (2026-08-03, 21 KB) — *"Turn
   `obstacle.offline` into the `win["lead"]` block `four_families` consumes"*; plus
   `taniteval/tools/build_lead_block.py` for the corpus build.
2. **It was fed, on exactly the 40 val episodes the INTAKE named.**
   `…/Data Engineering/…/incoming/2026-08-04-instrument-durability/raw/val40_lead_report.json`:
   **`n_episodes: 40`, `canonical_881: true`, registration `n_ok: 40 / n_failed: 0`, counts
   `LEAD 270 / NO_LEAD 551 / NO_LABEL 60`**, integrity `poses_sha256_match_all: true`.
3. **The registry closed it.** `Project Steering/MODEL_REGISTRY.md:1187-1190` — ~~`distance_keeping`
   UNAVAILABLE~~ struck, raw `pod4:/workspace/evalout/v1arch_oodval_q90_4fam_LEAD.json`,
   `families_unavailable=[]`.

⚠️ **This blocker generated TWO stale statements, not one.** The registry itself records that the
**2026-08-06 claim "lead block not on pod4" was ITSELF a stale absence-claim** — the block already
existed on pod4. So the same underlying fact was got wrong in both directions within three days.
*(A stale "it does not exist" and a stale "it is blocked" are the same defect.)*

**Closed in place** (annotated, history preserved — nothing deleted).

---

## 2. THE FOURTH ROT — `"our ingest reads N of 36 features"`

### 2.1 The finding, stated loudly as instructed

⛔ **The number was wrong again. `CLAUDE.md` said 5; the correct program-wide count is 6.**

Sequence: **`"2 of 36"` → 4 (2026-07-26) → 5 (2026-08-16) → 6 (2026-08-16, this sweep).**
Four rots, all inside *the very rule that warns about stale absence-claims*.

### 2.2 The real root cause is not carelessness — the SUBJECT was never defined

There is no single right answer, which is exactly why a single number kept rotting. **"Our ingest"
names three different things:**

| layer | count | features | source |
|---|---|---|---|
| `physicalai_r0.py` — r0 clip selection | **2** | `egomotion`, `camera_front_wide_120fov` | `stack/scripts/physicalai_r0.py:36-38` |
| `physicalai.py` — the **episode build** | **5** | + `camera_intrinsics`, `sensor_extrinsics`, `vehicle_dimensions` | `stack/tanitad/data/physicalai.py:232-235`, consumed at `:354` / `:407` / `:456` |
| **program-wide** (incl. the pod-side join) | **6** | + `obstacle.offline` | `stack/scripts/build_obstacle_join.py:148`, `stack/scripts/lead_state_gate.py:308-338`, `stack/tanitad/data/bev_raster.py` |

⚠️ **`obstacle.offline` is deliberately OUTSIDE the episode build** — the join is a pod-side step, which
is why `grep obstacle stack/tanitad/data/physicalai.py` still returns **zero matches** (verified).
So *"the episode build reads 5"* is **true**; *"our ingest reads 5"* is **false as of 2026-08-03**.

⇒ **The durable fix is not a better number, it is naming the layer.** Every future statement must read
*"the episode build reads 5 of 36"*, never the bare *"our ingest"*.

### 2.3 Verification that each of the 5 is a REAL read

A defined-but-unused constant would inflate the count, so this was checked rather than assumed:
all three calibration constants are passed to `_calib_chunk_path(...)` — `_CALIB_INTR` at `:354`,
`_CALIB_VEH` at `:407`, `_CALIB_EXTR` at `:456` — and `labels/egomotion` is read at `:471-472`.
The test below asserts this, so a constant going unused breaks the build.

### 2.4 The pin

**`stack/tests/test_physicalai_feature_readset.py` — 8 tests, green.** It asserts:

1. the three counts **and the exact feature names**, read from live source;
2. that each calibration constant is genuinely *consumed*, not merely declared;
3. that `obstacle.offline` is read at **≥2 of 3** known sites (absence at one location is not absence);
4. that `obstacle.offline` stays **out** of the episode build — the 5-vs-6 split is load-bearing;
5. a **drift detector** that fails when a *new* HF feature path appears in `physicalai.py`. This is the
   case all four rots actually were.

⭐ **The guard is proved able to fail.** `test_drift_detector_actually_fires` runs the detector against
a synthetic source containing `labels/lidar_front/…` and asserts it trips — because this programme has
repeatedly been burned by guards structurally unable to report the answer they are cited for (the C13
class). It also pins the two false-positive classes actually hit while writing it (trailing sentence
punctuation, and the local `calibration/physicalai_*.csv` sidecars, which are not dataset features).

**Every failure message names the documents to update**, so the next change is mechanical rather than a
re-derivation.

### 2.5 Blast radius of the stale "4"

**17 documents + 1 code docstring.** Highest-value, in order:

| site | why it matters |
|---|---|
| `CLAUDE.md:266` | the canonical statement — ✅ **FIXED by this sweep** |
| `Project Steering/EVAL_PROTOCOL_OODVAL_2026-08-05.md:143` | ⛔ a **PROTOCOL** — steers every future eval, and its *"UNAVAILABLE… a WORK ITEM, not a pass"* is **doubly stale** (count wrong AND blocker cleared). Handed to the Project Steering stream with evidence. |
| `Project Steering/V6F_PLANNER_DESIGN.md:536` | live design doc |
| `Project Steering/Gates/flagship-v5-retrain.PREP.md:58` | carries the complement, *"32-of-36"* → should be **30-of-36** |
| ⚠️ `stack/tanitad/data/bev_raster.py:12` | **the count rotted INTO CODE.** See §4 — escalated, not edited (another agent owns this file). |
| ~11 dated write-ups under `**/incoming/**` | history, low blast radius |

---

## 3. VERDICT TABLE — the wider sweep

*Three parallel streams swept the corpus (48 `INTAKE.md`, 121 `Project Steering/*.md`, ~507
non-INTAKE `incoming/**` docs). Their verdict tables are consolidated below.*

<!-- STREAM TABLES: filled in as the streams report -->

---

## 4. ⚠️ ESCALATION — things this sweep could NOT fix itself

1. ⛔ **`stack/tanitad/data/bev_raster.py:12` carries the stale "4 of 36" inside a module docstring.**
   That file is owned by a live agent this session, so it was **not edited**. It needs the one-word
   fix (`4` → the layer-qualified count) by whoever owns it next. **A stale count in code outlives a
   stale count in prose, because nobody greps docstrings.**
2. **The one-directional-reference problem is structural and unsolved.** Annotating today's stale lines
   fixes the instances, not the class. The cheap durable mechanism would be: *a doc that states a
   blocker names the artifact whose existence clears it*, so the check is a `test -f`, not a re-reading.
   That is a real work item, not a rhetorical flourish — it is the same fix shape as §2.4.
3. **Empty ORCHESTRATOR VERDICT blocks read as "rejected" but mean "nobody filled it in."** At least
   one landed-and-in-production package (`2026-08-03-longitudinal-distance-keeping`) has an empty
   verdict block. Worth a sweep of its own.

---

## 5. Deliverables

| artifact | repo path |
|---|---|
| This writeup | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-16-stale-blocker-sweep/STALE_BLOCKER_SWEEP.md` |
| The pin (8 tests, green) | `stack/tests/test_physicalai_feature_readset.py` |
| Corrected canonical count + layer table | `CLAUDE.md` (§"Absence found at ONE location is not absence") |
| Flagship blocker closed in place | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-03-longitudinal-distance-keeping/INTAKE.md` |

**Suite:** `PYTHONUTF8=1 python -m pytest -q -p no:cacheprovider` from `stack/`.
Baseline at HEAD `8e215b3`: 2919 passed / 0 failed / 17 skipped / 2 xfailed.
<!-- SUITE RESULT: filled in below -->

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** integrate / integrate-with-changes / defer / reject
- **Reason:**
- **Landed at:**
