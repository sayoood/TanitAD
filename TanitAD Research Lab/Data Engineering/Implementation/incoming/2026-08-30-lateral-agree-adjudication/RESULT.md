# D-LAT-AGREE — the lateral "41.7 % disagreement" adjudicated

**Date** 2026-08-30 · **Owner** DataFlyWheel · **Trigger** PI: *"regarding the
lateral contradiction, give it to the data flywheel to check it and fix it
BEFORE WE MOVE"* — blocker on v7f / refc_v3 / refa_v1 / refd.
**Blob:** `s2_labels_v7.jsonl.gz` md5 **`ee44875916ae7c0ac002c6716b9658ea`** —
THE RELEASE (six copies exist across three roots; this is the pinned one).

## Verdict

**The 41.7 % is not a disagreement rate. Two thirds of it is an instrument
defect plus a sensitivity difference. The genuine contradiction rate is 4.89 %.**

| class | n | % | what it is |
|---|---|---|---|
| **BOTH_AGREE** | 2,827 | 64.02 % | same side, or both straight |
| **ONE_SIDED_ALPAMAYO** | 868 | 19.66 % | Alpamayo names a side, geometry says straight |
| **ONE_SIDED_GEOMETRY** | 505 | 11.44 % | geometry names a side, Alpamayo says straight |
| ⛔ **HARD_CONTRADICTION** | **216** | **4.89 %** | **both name a side and the sides are OPPOSITE** |

⭐ **CONTROL (must read a known value):** `side=straight` + `LANE_KEEP` is
unambiguously agreement. All **1,994** land in `BOTH_AGREE` with shipped
`agree=True`, 100 %, no other class. **The decomposition reads the known cell
correctly** — without this it would be a table that cannot be wrong.

## 1. The flag itself was broken — MEASURED, localised, fixed

`alpamayo.lateral.agree` did not mean what its name says. It was **wrong on 992
of 4,416 records (22.5 %), in BOTH directions**:

| | shipped True | shipped False | should be | wrong |
|---|---|---|---|---|
| BOTH_AGREE | 2,205 | 622 | all True | **622 real agreements discarded as noise** |
| ONE_SIDED_GEOMETRY | 370 | 135 | all False | **370 real contradictions hidden** |
| HARD_CONTRADICTION | 0 | 216 | all False | 0 |
| ONE_SIDED_ALPAMAYO | 0 | 868 | all False | 0 |

**ROOT CAUSE (verified in source, `s2_geom_emit_v7.py:786-789`):** the geometry
side was derived from a scan of the **GOAL** tokens for a `TURN_` /
`YIELD_FOR_TURN_` prefix, defaulting to `"straight"`. **NUDGE_L/NUDGE_R are
ACTIONS, never turn goals**, so every non-turn clip was compared as "straight".

⚠️ **The EvalFlyWheel's hypothesis — "the NUDGE suffix is dropped" — pointed at
`side_of`, and fixing `side_of` would have been a NO-OP.** `side_of` handles the
NUDGE suffix correctly (`NUDGE_L` → left) and was simply **never called with a
NUDGE class**. The defect was the CALLER passing the wrong quantity, and
`fuse_lateral`'s own docstring already says to pass `side_of(<class>)`. Their
read from the output was right about the symptom and wrong about the site —
which is why they asked for source verification before any change. Good call.

**FIX:** the call site now passes `AF.side_of(goals["_lat"])` — the geometry's
actual lateral class, which was already threaded in one line above for the
structured layer — and **raises** rather than defaulting to `"straight"` if it
is absent, so the silent fallback cannot be re-created.

**REGRESSION TEST:** `stack/tests/test_alpamayo_lateral_agree.py`, with the four
inverted cells as fixtures (right+NUDGE_R must be True; straight+NUDGE_L must be
False) ⭐ **plus a test that reconstructs the OLD goal-scan logic and asserts it
still reproduces the inversion** — so the guard is shown to distinguish the
broken implementation from the fixed one. A guard never seen to fail proves
nothing, and this flag was wrong corpus-wide without anyone noticing.

## 2. Adjudication: which source wins — it is already settled, and already applied

**GEOMETRY WINS. It always did, and no label needs changing.**

* `fuse_lateral`'s contract: *"Geometry decides; this corroborates … a second
  opinion, never an override."* Alpamayo's lateral axis is **69.9 % vs 41.6 %
  chance** (lift ×1.68, C142) — informative, not authoritative.
* `a_tac.lat` is produced by the geometric extractor. **`alpamayo_agree` is
  written into the record and read by NOTHING** — verified across `stack/` and
  `taniteval/`: no trainer, model, head or eval consumes it. It is a pure
  diagnostic.

⇒ **The 216 hard contradictions change no label and block no training.** They
are a *quality signal about Alpamayo*, not a defect in our supervision.

**Is 216 consistent with Alpamayo simply being wrong sometimes?** Among the
1,049 records where **both** sources name a side, they agree on **833 = 79.4 %**.
That is *above* Alpamayo's own 69.9 % lateral accuracy, i.e. exactly what a
69.9 %-accurate second opinion produces against an authoritative first one.
**No excess contradiction to explain.**

⭐ **AND THE CONVENTION IS NOT FLIPPED** — the check that would have made this
serious: `right→NUDGE_R` **353** vs `right→NUDGE_L` **68**, and `left→NUDGE_L`
**269** vs `left→NUDGE_R` **24**. Both rows are strongly correct-signed. A
flipped convention would invert those ratios.

## 3. The 1,373 one-sided records are a THRESHOLD difference, not a contradiction

`NUDGE` triggers at **`abs(lat) ≥ 1.0 m` AND `abs(peak) ≥ 5°`**. A CoT narrating
"driving straight" while the ego drifts 1.2 m is **two instruments with
different sensitivity**, not two sources contradicting. This is the largest
block (31.09 %) and it is benign **by measurement**: it is precisely the cells
where one side says `straight`.

## 4. Auditability gap — CLOSED

`ego_manoeuvre.py` decided NUDGE on `abs(lat[j]) >= NUDGE_LAT_M` and **discarded
`lat`**, so the deciding evidence for every NUDGE label was unrecoverable.

* ⚠️ **Correction to the brief: `peak` was NOT missing** — `peak_yaw_deg` is
  already a `Manoeuvre` field. Only the lateral offset was lost. (It does not
  reach the record either; `manoeuvre_sequence` carries per-segment `dyaw_deg`
  and `radius_m`, which is not the same quantity.)
* **`lat_peak_m` is now a first-class `Manoeuvre` field** — signed peak lateral
  offset in metres, `+` = left — and the lateral track is computed
  **unconditionally**, so turns carry it too rather than only the branch where it
  happens to decide the class.
* **Verified against constructed inputs with a control**: a 2.0 m nudge reads
  **1.998 m**; a dead-straight run reads **0.0 m**.
* ⭐ This is also the only field that can distinguish a 1 m wobble from an
  absorbed lane change, since NUDGE has **no upper bound** (D-NUDGE-ABSORB).

## 5. Two things I did NOT do, deliberately

* **No re-emission of the released labels.** The fix changes a diagnostic with
  zero consumers; re-emitting would invalidate the validated corpus
  (`ee448759…`) that Thor is building from, for no training benefit. ⇒ the
  corrected `agree` and `lat_peak_m` land in the **next** extraction, and the
  release keeps its provenance. **This is a recommendation, not a decision —
  the PI may prefer a v7.1 metadata re-emit.**
* **No "fix" to lane-change absorption.** 39/39 LANE_CHANGE goals labelled
  `NUDGE_*` is the PI's 2026-08-16 ruling applied consistently (D-NUDGE-ABSORB),
  and the 19/39 opposite-direction reading is **indistinguishable from the base
  rate** (binomial 19/39 vs p₀=0.417 → p=0.42; n=39 cannot separate 48.7 % from
  41.7 %). Not a separate finding.

## Does the lane-detector package close this?

**No — and it is worth being precise about what it would close.** The extractor
names "the lane-detector reference" at `s2_geom_emit_v7.py:261` and `:863` as the
instrument for lane-relative questions. A lane detector would settle **the
one-sided classes** (is a 1.2 m drift a lane change or a wobble? — a
lane-relative offset answers it, an ego-relative one cannot) and would let
`NUDGE` be split from `LANE_CHANGE` on evidence. It would **not** settle the 216
hard contradictions, which are Alpamayo text errors against a geometry we
already trust. ⇒ the detector remains the right instrument for the **31 %**, and
is unnecessary for the **4.89 %**.

## Deliverable manifest

| artifact | where |
|---|---|
| `RESULT.md` (this) | repo, this package |
| `raw/lat_decomposition.json` | repo — counts, cells, control verdict |
| `code/lat_decompose.py` | repo — the decomposition, re-runnable against any blob (asserts the pinned md5) |
| the `agree` fix | repo `stack/scripts/s2_geom_emit_v7.py:786+` |
| the regression test | repo `stack/tests/test_alpamayo_lateral_agree.py` |
| `lat_peak_m` persistence | repo `stack/tanitad/data/ego_manoeuvre.py` |
| suite | `242 passed, 1 xfailed` on `-k "manoeuvre or alpamayo or lateral or ego"` |
