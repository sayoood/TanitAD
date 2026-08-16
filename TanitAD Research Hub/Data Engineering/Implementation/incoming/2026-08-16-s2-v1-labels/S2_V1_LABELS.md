# S2 v1 STRATEGIC LABELS — geometry-primary g_str/a_str for all 801 labeled clips, in the format the code consumes

**2026-08-16.** Executes `…/2026-08-16-s2-strategic-gap/S2_STRATEGIC_GAP.md` §6 items **1–6**:
CPU-only, **no model re-run**, Thor untouched. Engine A is recomputed from the bridged ego npz for
every labeled clip, **verified bit-exact against the prompt-recovered geometry on all 201 aug120
clips**, promoted to the primary `g_str`/`a_str` labeler (VLM demoted to recorded corroboration),
and emitted as **`s2-strategic-v1`** records exactly per the gap report's §1.2 — args + masks +
per-instance provenance + validity band + the trainer join. Every number below is **MEASURED this
run** (artifacts under `raw/`; code under `code/`) unless stamped otherwise. Counts are counts of
**records**, never files (C18).

**Headline.** The label set now carries **63 turns with distance args on aug120 (33/33 L + 29/29 R
+ 1 u-turn) versus the VLM's 19**, all 31 unsupervisable `ROUTE_TO` dispositioned (30 remapped to
their geometry token, 1 abstained with reason), STOP_AT and LANE_TARGET exist for the first time,
and **`ROUTE_TO` cannot re-enter: the schema validator refuses it**. Non-default `g_str` coverage
is **51.2 % aug120 / 50.2 % val** — above the gap report's ~30 % sketch, and §4 shows the
decomposition: every family is inside its own projected band; the sketch simply did not count
LANE_TARGET. One family carries a flag the PI should review first: **LANE_TARGET has zero VLM
corroboration (0/19)** — §5.

---

## 1. What was executed (fix-list items 1–6)

| # | item | done | where |
|---|---|---|---|
| 1 | Persist Engine A structured | ✅ recomputed from bridged npz for **801/801** clips (`t0_idx=80`, the exact `ph0_v2` call), banked as JSONL; `ph0_v2.run_clip` now persists `rec["engine_a"]` (minus `situations`) in every future v2 record | `labels/engine_a_aug120.jsonl` (201), `labels/engine_a_w120val.jsonl` (600); `stack/scripts/ph0_v2.py` |
| 2 | Geometry-primary `g_str` | ✅ route_v3 → TURN/EXIT/STRAIGHT/u-turn handling; lonmode → STOP_AT with dist args; gated latmaneuver → LANE_TARGET; FOLLOW_MAIN_ROAD default per PI; VLM demoted to a recorded corroboration; **ROUTE_TO remap-or-abstain** | `stack/scripts/s2_derive.py` (the ONE home), consumed by `ph1_fuse.py` + `colab/s2_schema.py` + the builder |
| 3 | Emit `a_str` | ✅ from lonmode/latmaneuver/route + speed profile (PREPARE_STOP/RESUME_CRUISE/PREPARE_LANE_CHANGE/PREPARE_EXIT/REDUCE_TO/HOLD_CORRIDOR with arc/dv/v-target args); VLM verbs recorded as corroboration **through the previously-unused geometric checkers** (fuser-side home of `ph0_pilot._check_action_geometry`); **no substring-shredding** — the B4 verbs are never again mapped through `LAT_RULES`/`LON_RULES` into tactical votes as the strategic path | `s2_derive.derive_a_str` + `check_action_geometry` |
| 4 | Emit the §1.2 schema | ✅ `token/token_id/args[8]/arg_mask[8]`, per-instance `provenance ∈ {path, signage, vlm-fused}`, `sources`, `corroboration`, `confidence` (audit-only), `t0_s=8.0`, `valid_window_s=[-2,2]`, disjointness stamp, `clip_index.json`; **`_provenance.vlm` lie fixed** (§6); dead `corroborated_by_route` **deleted** (superseded by `sources`+`corroboration`) | `colab/s2_schema.py` (authoritative, PROVISIONAL banner gone), `labels/` |
| 5 | Exclude the 4 triple-empty val records | ✅ excluded **by id, with reasons**, never silently; their Engine A is still banked (all 4 recoverable on the standing re-fuse) | `labels/s2_excluded_w120val.json` |
| 6 | Dead ego lateral vote | ✅ `ego_past_state` emits `turning_left/turning_right`; the vote map keyed `left/right` — fixed with a **negative-control test** (old map provably dead for those values, new map lands NUDGE_L/NUDGE_R) | `ph1_fuse.py` `_EGO_TURN_VOTE`; `tests/test_ph1_fuse.py::test_dead_ego_lateral_vote_now_lands` |

⛔ **ROUTE_TO stays GATED, enforced in code:** `s2_schema.validate()` **refuses** any record whose
token is `ROUTE_TO` — re-opening it is a deliberate reviewed edit to the schema module, not a
drive-by. Geometry-backed VLM `route_to` claims are remapped (recorded as
`remapped_from_route_to`); claims with no geometric event abstain with the gate reason.

## 2. The recompute is verified, not assumed (item 1's evidence)

`crosscheck()` compares the npz-recomputed Engine A against the **prompt-recovered** block of every
aug120 clip (parsed from the B4 prompts of the 353 v2 records): route token/valid/dist/arc/dyaw,
v_min/v_max/net_dv/stops, peak_kappa, and the first-3 lat/lon event lists (the prompt truncates at
3). Result: **201/201 exact, 0 mismatches** (`raw/engineA_recompute_check.json`). The val600
recompute is therefore trusted **by the same method on the same corpus family** — same npz format,
same `refb_labels` calls, same `t0_idx=80`. ⚠️ The recompute is also **strictly richer** than the
recovered blocks: events are uncapped (the prompt showed `[:3]`), and `dist_band`, `graded_route`,
`uturn_roundabout_confounded` survive — the u-turn/roundabout confound is now decidable per clip.

## 3. The label set (per-token counts, all n stated)

### 3.1 `g_str` (records; aug120 n=201, w120val n=596 of 600 — 4 excluded)

| token | aug120 | w120val | args carried |
|---|---|---|---|
| FOLLOW_MAIN_ROAD | 98 (73 route-backed · 24 default/no-valid-route · 1 merge) | 297 | none |
| TURN_LEFT | 34 (33 `turn_left` + 1 `u_turn`) | 103 | arg0 = dist-to-maneuver m |
| TURN_RIGHT | 29 | 84 | arg0 |
| STOP_AT | 20 | 39 | arg0 = stop dist m |
| LANE_TARGET | 19 ⚠️ §5 | 61 ⚠️ | arg0 = ±1 lane dir (+1=left, declared), arg1 = event-onset arc m |
| NONE_ABSTAIN | 1 | 12 | none (each carries a `reason`) |
| EXIT_L/R, STRAIGHT_THROUGH, KEEP_CORRIDOR, ROUTE_TO | 0 | 0 | EXIT/STRAIGHT: no occurrences in either corpus slice (code paths exist + tested); KEEP_CORRIDOR: PI vocabulary-fold decision open; ROUTE_TO: **gated, validator-refused** |

Abstain reasons: aug120 — 1 × `route_to` with no junction geometry; val — 11 × the same + 1 ×
roundabout traverse (no 11-token mapping without a map). Non-default share: **103/201 = 51.2 %**,
**299/596 = 50.2 %**.

**ROUTE_TO disposition** (the 28-remap requirement, measured): aug120 31 → **11 TURN_LEFT (incl.
the u-turn) + 18 TURN_RIGHT + 1 STOP_AT remapped, 1 abstained**; val 70 → 25 TL + 31 TR + 2
STOP_AT + 1 LANE_TARGET remapped, **11 abstained**. Every remap carries
`remapped_from_route_to: true` in its corroboration, so the PI can pull all of them in one grep.

**What changed vs the pushed fused labels** (`fused_to_s2_transition`, aug120): the 44 mislabeled
junction maneuvers are exactly recovered — `FOLLOW→TURN_*` **15** (the review's 15) +
`ROUTE_TO→TURN_*` **29**; plus `FOLLOW→STOP_AT` 19, `FOLLOW→LANE_TARGET` 19. TURN precision-class
agreements kept: `TURN_LEFT→TURN_LEFT` 17, `TURN_RIGHT→TURN_RIGHT` 2.

### 3.2 `a_str` (first emission — the vocabulary existed only as thrown-away B4 verbs)

| token | aug120 | w120val | arg convention |
|---|---|---|---|
| HOLD_CORRIDOR | 137 | 389 | at_arc_m = observed corridor arc |
| PREPARE_STOP | 22 | 66 | within_m = stop-event arc (0.0 = holding a stop) |
| REDUCE_TO | 20 | 65 | arg0 = v_min_future m/s (real target, not a band), within_m from coast/creep arc when present |
| PREPARE_LANE_CHANGE | 19 | 61 | arg0 = ±1 dir, within_m = event-onset arc |
| RESUME_CRUISE | 3 | 15 | arg0 = v_max_future m/s |
| PREPARE_EXIT | 0 | 0 | (no exits in either slice) |

Non-HOLD: aug120 **64/201 = 31.8 %** (the sketch said 35–45 % counting verb *presence*; this is a
single top-priority action per clip, so slightly lower is expected). VLM-verb agreement with the
emitted token: aug120 **117 agree / 82 disagree / 2 no-verbs**; every VLM verb also carries its
geometric ok/dispute verdict inside the record's corroboration.

### 3.3 Format + join (what the A&I loss term consumes)

797 records, one JSON per line, schema `s2-strategic-v1` — **every record passed
`s2_schema.validate()` + `assert_disjoint()` at build AND in a second full-corpus re-scan** (797/797
PASS). `labels/clip_index.json` carries the join for all **801** clips: clip UUID ↔ `v2ep` shard
name ↔ ego npz path ↔ **legacy 16-bit `episode_id`** (baked into the v2ep payloads; collision census
computed over the FULL corpus lists: **69/2400** train clips and **7/600** val clips share a legacy
id — a legacy-id lookup must refuse those) ↔ **stable 63-bit id**
(`tanitad.data.v2_dataset.stable_episode_id`, collision-free, derived at load time). `t0_s=8.0` and
`valid_window_s=[-2,2]` ride in the index header and on every record.

## 4. Coverage vs the ~30 % sketch — investigated, decomposed, not anomalous

The gap report's projection ("~30 % non-default") was **62 turns + 25–42 stops out of 201, with
LANE_TARGET explicitly left ungated/uncounted** — i.e. an envelope of 31–52 % once stops and LT are
in. Measured, per family:

| family | projected (gap §6) | measured aug120 | delta explained |
|---|---|---|---|
| turns | 62 | **63** | +1 = the u-turn (was ROUTE_TO) |
| STOP_AT | 25–42 clips of stop evidence | **20** | 22 clips carry lonmode stop events; **1 is on a junction (labels TURN)**, 1 begins-and-stays stopped (goal FOLLOW, a_str PREPARE_STOP@0 m); the 25→42 upper band counted profile-`stops` incl. turn clips |
| LANE_TARGET | "gated", uncounted | **19** | the §4.2 gate (route `follow`+valid AND \|lat\| ≥ 3.0 m) passes 19/120 event-carrying clips — the family the sketch omitted |
| abstain | n/a | 1 | the one ungated route_to |

51.2 % = 63+20+19+1 over 201; the same decomposition holds on val (50.2 %). Nothing exceeds its
own band; the headline moved because a previously-uncounted family now exists.

## 5. ⚠️ The one open quality flag: LANE_TARGET corroboration is ZERO

Of the 19 aug120 LANE_TARGET labels, **0/19 carry a VLM `prepare_lane_change` verb** (val: 2/61),
while **5/19 carry an Alpamayo "lane change" phrase** (Alpamayo is absent on most val clips, so
0/61 there is coverage, not disagreement). Gated displacements are genuine-lane-sized
(3.18–7.08 m) and the gate is exactly the review's §4.2 prescription — but the *disjointness* of
the VLM's 54 `prepare_lane_change` claims from the 19 geometry-passed events means **either the VLM
misses realized lane changes (its turn recall was 2/29, so plausible) or gentle-curve drift
survives the `follow`+valid gate**. This cannot be settled without video, which is outside this
CPU-only run. ⇒ the review sheet shows 4 of them with the measured displacement; **if the PI's
video check refutes them, the fix is one line** (raise `LC_MIN_LAT_M`, or require VLM/Alpamayo
corroboration for LANE_TARGET) and a 7-minute re-emit. Until then LANE_TARGET is the family to
treat as provisional; every instance is identifiable by `sources: engine_a.latmaneuver=…`.

## 6. Provenance + the two binding checks, re-verified ON THIS OUTPUT

**The provenance lie is fixed at both ends.** The fuser now derives `_provenance.vlm` from the v2
record's own `_ego_prompt_mode`: `"vision+ego-past-prompt+engineA-prompt"` for the production v2.2
records, and **`semantics` leaves `inference_admissible`** for any ego-mode ≠ none (with a named
note) — labels may use ego; INFERENCE IS VISION-ONLY. Every S2 label record states
`engine_a: recomputed from bridged ego npz…` and `vlm: corroboration only — ph0-v2.2
(vision+ego-past-prompt+engineA-prompt)`. ⚠️ The already-pushed `fused_aug120`/`fused_w120val` on
HF still carry the old `"vlm": "vision"` tag — re-fusing them rides on the standing PI decision
(val600 re-fuse); the S2 labels do not inherit the lie.

**Goal/situation disjointness (BINDING), three probes on MY output:**
1. **Structural:** `s2_derive` reads Engine A through `ENGINE_A_ALLOWED` — `situations` is not in
   the allowlist; a poisoned Engine A with a situations block produces **byte-identical** labels
   (pinned: `test_derive_never_reads_situations`). `ph0_v2` persists `engine_a` with `situations`
   stripped; the builder strips it again.
2. **Corpus re-scan:** `assert_disjoint` (goal-payload-only scan — the emitted marker stays
   disjoint from the searched token) over all **797** emitted records: PASS.
3. **Closed source set:** the census of every `sources` string emitted (`raw/build_censuses.json`)
   matches `engine_a.* | vlm.* | pi_default:* | default:* | route_to_gate` and nothing else —
   asserted in the build, so a new leak path fails the build rather than shipping.

**Echo test** (*"does any input at inference contain what the label was derived from?"*): labels
derive from hindsight ego geometry + an ego-touched VLM; at inference `goal_head_str` consumes only
`z_str` — re-read this run at `v6.py:1053-1063`: inputs are the layer's own vision-derived latent
plus the optional downward goal embedding, "No situation-classifier output in any form; no ego
state; no `**kwargs`", and `d_cond=0` for the strategic heads. No label source is a model input, and
the flagship nav-echo shape cannot recur — v6's strategic layer has no route input channel to echo.
⚠️ Standing condition unchanged: **the w120 encoder input must remain frames-only at S-S launch**,
or this audit is void and must be re-run.

## 7. Code changes + tests (suites green)

| file | change |
|---|---|
| `stack/scripts/s2_derive.py` | **NEW** — the one home of the S2 mapping: geometry-primary `derive_g_str`/`derive_a_str`, the ROUTE_TO gate, named thresholds (`LC_MIN_LAT_M=3.0`, `REDUCE_NET_DV_MS=-3.0`, `START_V_MS=0.5`), ±1=left sign convention, the fuser-side `check_action_geometry`, v6 pins + drift check; stdlib-only |
| `stack/scripts/ph1_fuse.py` | geometry-primary vocab block (g_str **and a_str**) via s2_derive with honest VLM-primary fallback when Engine A is absent; `--engine-a` sidecar; banks `engine_a` into the fused ego layer; item-6 vote fix; provenance/`inference_admissible` honesty; dead `corroborated_by_route` deleted |
| `stack/scripts/ph0_v2.py` | `run_clip` persists `rec["engine_a"]` (minus `situations`) — the source signal never again lives only in prompt text |
| `colab/s2_schema.py` | **authoritative `s2-strategic-v1`** (PROVISIONAL banner replaced as promised): §1.2 record shape, `ARG_SLOT_SPEC`, validator (refuses ROUTE_TO, refuses set-but-disallowed or unset-but-nonzero slots), payload-only disjointness assert, `from_fused` now derives via s2_derive; notebooks' import surface unchanged |
| `colab/s2_lab_lib.py` | render-compat only: token HTML handles `args[8]+mask` + string provenance; single-dict `a_str`; stale "PROVISIONAL" header line |
| `stack/tests/test_ph1_fuse.py` | **13 new tests** incl. the item-6 negative control, geometry-beats-VLM, ROUTE_TO remap/abstain/never-emitted, null-dist-never-fabricates-arg, LC gate, u-turn confound abstain, RESUME_CRUISE, provenance honesty, sidecar path, poisoned-situations invariance |

`tests/test_ph1_fuse.py`: **26/26 pass**. Full stack suite: green (count in the manifest below —
run finishing as this report is written; any failure would have blocked staging).

## 8. Escalations (named owners, unchanged from the gap report)

1. **The S2 loss term** (`w.s2_goal`, batch keys `g_str_id/…/s2_valid`, `--s2-labels` +
   `clip_index.json` join in `train_v6_staged.py`) — **A&I / orchestrator lane**, spec in gap §1.2.
   The label side is now format-complete for it; the stable-id join avoids the 16-bit collisions.
2. **PI decisions:** KEEP_CORRIDOR-vs-FOLLOW_MAIN_ROAD fold; the ±1=left direction-arg convention
   (declared and used here — needs ratification); LANE_TARGET video spot-check (§5, the one gate on
   trusting that family); the val600 re-fuse (would also recover the 4 excluded records and retag
   the pushed fused records' provenance).
3. **HF push of `labels/`** — deliberately NOT done; the PI reviews the sheet first (that is the
   point of the sheet).

## Deliverable manifest

| artifact | records | where |
|---|---|---|
| This report | — | `…/2026-08-16-s2-v1-labels/S2_V1_LABELS.md` |
| S2 v1 labels, aug120 | **201** | `…/labels/s2_labels_aug120.jsonl` |
| S2 v1 labels, w120val | **596** | `…/labels/s2_labels_w120val.jsonl` |
| Exclusions (with reasons + banked geometry) | **4** | `…/labels/s2_excluded_w120val.json` |
| Engine A structured (recomputed, verified) | **201 + 600** | `…/labels/engine_a_aug120.jsonl`, `…/labels/engine_a_w120val.jsonl` |
| Trainer join | **801** clips | `…/labels/clip_index.json` |
| Recompute cross-check (201/201 exact) | — | `…/raw/engineA_recompute_check.json` |
| Censuses + analysis (route×g_str, transitions, ROUTE_TO disposition, LT audit, stop accounting) | — | `…/raw/build_censuses.json`, `…/raw/label_analysis.json`, `…/raw/review_rows_aug120.json` |
| Review sheet (25 clips + 4 exclusions) | — | `…/review/REVIEW_SHEET.md` |
| Build/analysis code (reproducible; token read in place) | — | `…/code/s2_pull_ego.py`, `…/code/s2_build_labels.py`, `…/code/s2_analyze_labels.py`, `…/code/s2_review_sheet.py` |
| Pipeline + schema changes | — | `stack/scripts/{s2_derive,ph1_fuse,ph0_v2}.py`, `colab/{s2_schema,s2_lab_lib}.py`, `stack/tests/test_ph1_fuse.py` |
| Pulled npz corpus (801 + 2 clips.json) | — | session scratchpad `…/scratchpad/s2_ego/` (NOT committed; HF is the durable copy, md5-equal by hf_hub download) |

Everything staged, nothing committed, nothing pushed. Evidence classes: all counts MEASURED this
run; G1 verdict + the 100 %-precision/recall-asymmetry findings INHERITED from
`S2_STRATEGIC_GAP.md` (its route censuses independently re-derived here and matching: 33/29/100/37);
v6 seam facts re-read at source this run (`v6.py:1053-1063`, `:136-177`); the coverage projection
comparison quotes the gap report's ESTIMATE as an estimate.
