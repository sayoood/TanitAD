<title>SPEC — is comma2k19 really the only corpus that can test P2(b)? (2026-08-31)</title>

# SPEC — E-LAB-DATA-0831

`TanitAD Research Lab · Data Engineering · 2026-08-31`
`Seeds: V7_LAUNCH_GATE.md P2(b) corpus blockers; the B1 distribution audit's open items;`
`D-DATA-EFFICIENCY (PI directive 2026-08-30).`
`Class: CORPUS AUDIT from primary sources + our own measured artifacts. 0 GPU, 0 spend, no download.`

---

## 1. What & why

`V7_LAUNCH_GATE.md` §P2(b) makes a corpus claim, and corpus claims are Data Engineering's to
check. Verbatim, it names **two blockers on "the only corpus that could test it"**:

> 1. **GEOMETRY.** comma2k19's entire field is **65.203°**; our frame is **120°** …
> 2. **DATA.** comma2k19 is **not on Thor** — only `extract_comma2k19.py` and its tests.

and it prices the discriminating experiment as *"once the data is present, correlate CAN
steering against the curvature-derived proxy on the same clips."*

⛔ **The premise — that comma2k19 is the only candidate — is an ABSENCE CLAIM, and our own
standard says an absence needs two probes.** I could not find a record of that search. This
package runs it.

⚠️ **The failure mode I am guarding against is my own.** The attractive answer here is "I
found a better corpus" and it would be easy to reach it by reading a schema optimistically.
So the criteria below are written to make a *negative* finding just as reportable, and the
provenance question — *is this signal actually a command?* — is kept separate from the
schema question and is allowed to end UNVERIFIED.

## 2. Hypotheses, with success criteria committed IN ADVANCE

**H-LAB-CORP-1.** *A driving corpus other than comma2k19 is (i) licensed for our use,
(ii) already reachable by this programme, and (iii) carries an action channel that is not a
restatement of the realised path.*
- **SUPPORTED if** all three hold, each from a primary source or our own measured artifact.
- **REFUTED if** any leg fails ⇒ the gate's premise stands and P2(b) stays blocked as written.
  That is a perfectly good outcome and must be reported as plainly as the positive one.

**H-LAB-CORP-2.** *The gate's own triage test can be run without the blockers it names.*
- **SUPPORTED if** the test is computable from state/metadata alone — no video, no geometry
  decision, no GPU.
- **REFUTED if** it needs pixels or matched FOV ⇒ the two blockers bind for any corpus and
  the gate's cost estimate is right.

**H-LAB-CORP-3.** *The corpus axes the B1 audit left unfilled exist natively somewhere.*
- Target axes: the **road-class stratum** (D-B1-100H's own unfilled WORK ITEM) and the
  **>14 m/s regime** that `physicalai_r0.py:100` hard-excludes (D-SPEED-GATE).
- **SUPPORTED if** a reachable corpus ships those as native fields.
- **REFUTED if** they would have to be derived ⇒ report the derivation cost instead.

## 3. Admissibility rules for this package

- **Two evidence tiers, never merged.** `PUBLISHED` = the dataset's own card / official docs,
  with a retrieval date. `MEASURED` = a file this programme has actually read, with its path.
  A schema field is only `MEASURED` if it came from the dataset's own descriptor, not from a
  prose summary of it.
- ⛔ **"The card says X" is not "X is true of the bytes."** Our own 2026-07-22 L2D note exists
  precisely because a prior survey inherited numbers from the card and two were wrong. Where
  the card and a measured file disagree, the measured file wins and the conflict is reported.
- ⛔ **PARITY IS SACRED.** Nothing in this package proposes mixing another corpus into
  `physicalai-train-e438721ae894`. Everything here is either a **data-only measurement** or a
  **separate corpus with its own parity key**. Any sentence that could be read as
  "re-select episodes" is a defect in this document.
- **Provenance is a separate question from schema** and may end UNVERIFIED. An axis *named*
  `steering_angle_normalized` is not proof that the number came off a bus.

## 4. Why `tests/` and `code/` are absent

No instrument is built and no bytes are downloaded here. The quotable layer is
`raw/L2D_SCHEMA_EVIDENCE.md` — the dataset descriptor fields with their retrieval path. The
probe this package *recommends* is specified in RESULT §6 and is deliberately left unbuilt:
it belongs to the DataFlyWheel, and building it here would be the integration this standard
tells me to escalate instead of performing.
