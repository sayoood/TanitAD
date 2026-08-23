# RESULT — E-LABELVAL-1: the label extraction is good where it is observable,
# and the strategic horizon is mostly NOT observable

`TanitAD_DataFlyWheel · 2026-08-23 · verdict: SPEC outcome B (fix before scaling)`
`Visual report: raw/label_validation_report.html — published artifact d5c3f599`

## 1. The finding that outranks the others

⛔ **81 % of a strategic label's own definition lies past the end of the clip.**

| quantity | value |
|---|---|
| clip length | 19.9 s (199 frames @ 10 Hz) |
| label anchor `t0` | 7.8 s |
| future available | 12.0 s |
| strategic band (PI definition: key + 8 s .. 30 s) | 15.8 – 37.8 s of clip time |
| **observable** | **4.0 s of 22 s = 18.2 %** |
| unobservable | 17.9 s, past the end of the recording |

`MEASURED (ours, raw/sample_v2_slim.json → horizon block, n=39 clips)`

This is a **CORPUS property, not a labeller defect** — no label-side work
changes it. Two consequences:

- **Validation.** The strategic band cannot be confirmed or refuted from
  PhysicalAI at all. Everything below covers 0–12 s, i.e. the tactical and
  near-strategic range only.
- **Supervision.** A head trained on these labels for 8–30 s is being taught to
  guess: the target beyond 12 s was inferred, never observed. If the strategic
  layer is to be evidenced rather than asserted, it needs longer sequences.

⚠️ This does NOT say the labels are wrong. It says the band they claim to
describe is largely unrecorded, and the claim should be scoped to what the data
can carry.

## 2. Agreement where it IS observable

`n = 39 clips, joined label ↔ frames ↔ poses. MEASURED (ours).`

| family | agreement | n |
|---|---|---|
| g_str overall | 87.2 % | 39 |
| a_str overall | 92.3 % | 39 |
| g_str TURN_LEFT | **100 %** | 13 |
| g_str STOP_AT | **100 %** | 7 |
| g_str TURN_RIGHT | 87.5 % | 8 |
| **g_str FOLLOW_MAIN_ROAD** | **70 %** | 10 |
| tactical v1 vs v2 (curvature gate) | 94.9 % | 39 |
| Alpamayo `lane` → our tactical LAT | 80.0 % | 15 |
| **Alpamayo LON → our tactical LON** | **56.2 %**, 25 % outright sign contradictions | 16 |

**The one systematic defect is FOLLOW_MAIN_ROAD**, and it fails in ONE
direction: all three misses are junction turns of 63–72° absorbed by the
fallback token. Because it is the fallback, an undetected turn becomes a
confident "carry on straight" instead of an abstention — the same
substitute-a-confident-claim shape the `LANE_TARGET` derivation was retired for.

**Not a defect, and easy to misread as one:** 52 % of strategic `TURN_*` clips
carry tactical `lane_keep`. The tactical horizon is 2.0 s and 10 of 21 sampled
turns begin later (median onset 3.7 s, max 11.2 s). Scoring that as a
contradiction manufactures a ~50 % failure rate on a healthy corpus.

## 3. ⚠️ RETRACTION — my first pass reported six defects that do not exist

`Root-cause class: SCOPE ERROR — a rule applied outside the horizon it belongs to
(the df / free / step_s family).`

The first pass judged g_str over a **4 s** window and found six TURN_LEFT clips
with ~0° of yaw. **All six dissolved on the strategic horizon**; the turns began
4.8–11.2 s after the anchor. The corrected instrument reports 87.2 %, not the
~72 % the bad window implied.

Generalisation worth keeping: **a validation harness that scores a label on a
horizon other than the one the label is defined over will manufacture a failure
rate that is not real.** Clip `01b24287` is retained in the visual report as the
worked example.

A second defect was found in my own instrument and fixed before any number was
quoted: the legacy-id join collides on BOTH sides, and checking only the label
side let ONE clip (`4879e5f3`) enter the sample TWICE with two different
trajectories. Both sides are now refused (3 label-side, 3 cache-side ids), and a
uniqueness assertion fails the extractor if a duplicate ever recurs.

## 4. The guard — implemented, tested, swept

`stack/tanitad/data/label_guard.py` + `stack/tests/test_label_guard.py`

⭐ **The motivating observation: the information needed to catch these defects was
ALREADY IN THE PIPELINE and nothing consumed it.** On 2 of 6 defective clips our
own tactical family contradicted our strategic label; on a third Alpamayo did.

| rule | severity | catches |
|---|---|---|
| G1 fallback-absorbs-turn | REFUSE | a no-manoeuvre goal with ≥25° of hindsight yaw |
| G2 lat-sign-conflict | FLAG | strategic and tactical disagree on turn DIRECTION (never fires on late turns) |
| G3 lon-inverted | REFUSE / FLAG | PREPARE_STOP while accelerating from rest; RESUME_CRUISE while decelerating |
| G4 stop-undescribed | FLAG | HOLD_CORRIDOR through a full stop from >3 m/s |
| G5 lon-family-conflict | FLAG | PREPARE_STOP against tactical LON `accelerate` |

**Sweep over the 39 clips** (`raw/guard_sweep.json`): **5 REFUSE, 2 FLAG-only,
32 clean** — an 87.2 % emit rate if REFUSE blocks emission.

It reproduces **6 of the 7** clips flagged by hand and finds **1 the hand pass
missed** (`b1950b41`, G5).

⚠️ **Known miss, stated rather than patched:** `90006660` (TURN_RIGHT whose
largest excursion is +56° left) is NOT caught, because there both families agree
with each other and only the peak-yaw geometry dissents. A g_str-sign-vs-peak-yaw
rule would catch it but would also fire on legitimate multi-event clips; it is
not added until it can be validated on more data.

**Both guards shown ABLE TO FAIL** before their PASS was accepted: disabling G1
(`TURN_DEG` 25→999) turns 3 tests red; making G2 fire on `lane_keep` turns the
false-positive control red. The file was restored bit-identically (sha256
verified) and 13/13 pass.

## 5. Recommendation

1. **Decide the strategic-horizon question first** — narrow the definition to
   what 20 s clips evidence (~0–12 s), or source longer sequences. Scaling now
   bakes an 81 % unobserved extrapolation into the dataset.
2. **Wire `label_guard` into `s2_derive.py`** so REFUSE blocks emission. Not done
   here: `s2_derive.py` is the label agent's file and is live — this is an
   INTEGRATION ESCALATION, not a silent edit.
3. **Settle the Alpamayo longitudinal conflict** (56 %, 25 % sign flips) before
   Alpamayo is consumed as corroboration on that axis.
4. **Validate the 80 abstained rows separately** — none joined to local frames,
   so this session's abstain change is NOT evidenced by this sample.
5. **Re-run at n ≈ 300 on the parity corpus.** At n=39, FOLLOW_MAIN_ROAD's 70 %
   is 7 of 10 and cannot size a threshold.

## 6. Deliverable manifest

| artifact | location |
|---|---|
| visual report (60 frames, 10 clips, all layers) | `raw/label_validation_report.html` + artifact `d5c3f599` |
| extractor (v2: tactical + Alpamayo + full horizon) | `code/extract_v2.py` |
| geometric adjudicator | `code/adjudicate.py` |
| first-pass extractor (kept: carries the retraction) | `code/label_validation_sample.py` |
| report renderer | `code/build_report_v2.py` |
| per-clip labels + geometry + horizon, n=39 | `raw/sample_v2_slim.json` |
| guard sweep | `raw/guard_sweep.json` |
| refused colliding join keys | `raw/refused_legacy_ids.json` |
| **the guard** | `stack/tanitad/data/label_guard.py` |
| **its tests** | `stack/tests/test_label_guard.py` (13 pass, 2 regression arms) |

All staged in the working tree. Nothing committed, nothing pushed.
