# Label pipeline — end-to-end confirmation on a small sample, and what it does NOT show

`MEASURED (ours)` · 12-clip sample (every clip in the local w120 cache that has
an Alpamayo taxonomy row) · all three tiers live: **Alpamayo + VLM (Qwen3.5-9B,
4-bit, thinking) + ego** · parity gate applied at payload build
(`in_deployed_val 0 of 40`, `in_parity_train 130`).

## 1. The result, per clip

| clip | `a_tac_lat` | `a_tac_lon` | leg | `g_str` | leg | flags |
|---|---|---|---|---|---|---|
| 1f57d8ad | LANE_KEEP | YIELD_MERGE | **vlm** | FOLLOW_MAIN_ROAD | alpamayo | — |
| 20070bb6 | LANE_KEEP | FOLLOW | **vlm** | FOLLOW_MAIN_ROAD | alpamayo | — |
| 220a8d49 | TURN_L | FOLLOW | **vlm** | TURN_LEFT | alpamayo | TIMING_FROM_GEOMETRY |
| 4d58621f | NOT_APPLICABLE | ⛔ ABSTAIN | none | KEEP_CORRIDOR | derived | — |
| 5f32e0f4 | LANE_KEEP | FOLLOW | **vlm** | KEEP_CORRIDOR | derived | — |
| 6c5c503d | LANE_KEEP | FOLLOW | **alpamayo** | FOLLOW_MAIN_ROAD | alpamayo | LON_FROM_COT |
| 8dc5d14d | LANE_CHANGE_R | FOLLOW | **vlm** | ⛔ ABSTAIN | none | **LAT_LANE_TARGET_DISAGREEMENT** |
| bb41e3b8 | LANE_KEEP | FOLLOW | **vlm** | FOLLOW_MAIN_ROAD | alpamayo | — |
| bc82e366 | LANE_KEEP | FOLLOW | **alpamayo** | FOLLOW_MAIN_ROAD | alpamayo | LON_FROM_COT |
| c568fffd | LANE_KEEP | FOLLOW | **alpamayo** | FOLLOW_MAIN_ROAD | alpamayo | LON_FROM_COT |
| d89a01eb | TURN_R | YIELD_MERGE | **alpamayo** | TURN_RIGHT | alpamayo | LON_FROM_COT, TIMING |
| fae6f0e9 | LANE_KEEP | ⛔ ABSTAIN | none | ⛔ ABSTAIN | none | — |

| field | filled | provenance |
|---|---|---|
| `a_tac_lat` | **12/12 = 100 %** | alpamayo 11 · NOT_APPLICABLE 1 (correct: the one-axis `Stop` row) |
| `a_tac_lon` | **10/12 = 83.3 %** | **vlm 6 · alpamayo 4** |
| `g_str` | **10/12 = 83.3 %** | alpamayo 8 · derived 2 |

## 2. ⛔ It is NOT 100 %, and the three gaps are each named

| clip | gap | cause | is it a design failure? |
|---|---|---|---|
| `fae6f0e9` | lon + g_str | ⛔ **its generations never ran** — 10 of 26 are still outstanding from the reclaimed VM | **no — operational** |
| `4d58621f` | lon | ⚠️ **truncated at the 4096 token cap** mid-reasoning | **no — operator setting** |
| `8dc5d14d` | g_str | ✅ **`LAT_LANE_TARGET_DISAGREEMENT`** — Alpamayo says *Right Lane Change*, the VLM read the markings and said the ego stayed put | **no — this is the design WORKING** |

⇒ **Achievable ceiling on this sample: 12/12 lon and 11/12 g_str**, with
`8dc5d14d` correctly abstaining. Two of the three gaps close by finishing the
run; the third must not close.

## 3. ⭐ What the three tiers each demonstrably contributed

* **Alpamayo** — the lateral class on **11/12**, and the longitudinal reason on
  **4/12** via the `cot` field, which we had been discarding entirely while
  paying a 9 B VLM ~240 s per generation to produce the same thing.
* **VLM** — the longitudinal reason on **6/12**, each with a named referent. The
  quality is real, not a template: e.g. `1f57d8ad` → *"the pedestrian (the woman
  in the patterned dress) crossing the street"*, grounded in specific frames,
  verdict `YIELD_MERGE`. **Zero referents echoed the prompt's own examples.**
* **ego** — ⚠️ **fired 0 times on this sample.** It only supplies `HOLD`/`CREEP`
  and none of these 12 clips are stopped or crawling. Corpus-wide it would fire
  on roughly the `Stop` population (**304 of 4,729 = 6.4 %**) plus crawl clips.
  ⇒ **The ego tier is implemented and tested but UNDEMONSTRATED on real data.**

## 4. What this does and does not license

✅ **Licensed:** the three-tier composition runs end to end on real data, each
tier contributes where it should, every abstention carries a machine-readable
reason, and the one contradiction in the sample was caught and refused rather
than silently composed.

⛔ **NOT licensed:**
* *"the approach works 100 %"* — it is 83.3 % on two of three fields, with two
  operational gaps outstanding.
* any claim about **label CORRECTNESS**. This measures **coverage and
  provenance**, not accuracy. **No human has reviewed a single tactical label**,
  and the 42-clip visual review still stands at **0/42**.
* the **ego tier**, which has never fired on real data.
* the **tactical-goal layer** (`g_tac`), which is **100 % VLM-dependent** and is
  currently never produced at all, because `vlm_goal` is never passed.

## 5. Before scaling — the three things that must close first

1. **Finish the 10 outstanding generations** (needs a GPU session; the resume
   path is built and proven — 16 records were rebuilt from the operator log
   after a VM was reclaimed).
2. **Human review of a sample of tactical labels.** Coverage is not correctness,
   and this is the programme's oldest open gap.
3. **Demonstrate the ego tier** on clips that are actually stopped — trivially
   selectable from the 304 `Stop` rows.

⚠️ And the scale figure, so it is not rediscovered later: at the measured
**12.5 tok/s / ~240 s per generation**, the 4,522-clip target is **25.6 T4-days**
for lon+lane with the sign call gated to turns. That is a Thor-or-pod job, not a
Colab one.
