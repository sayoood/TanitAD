# S2 STRATEGIC GAP — what the pushed VLM/SAM3/Ego pipeline emits today vs what S-S needs, in the format the code consumes

**2026-08-16.** Task: strategic conditioning is a v6 commitment; S-W finishes in ~6.8 days and S2
(strategic goal supervision) must be producible by then. This review (1) pins the code-side S2
contract from `stack/tanitad/models/v6.py` + `stack/scripts/train_v6_staged.py`, (2) re-measures the
pushed artifacts record-by-record, (3) builds the per-token supervision-gap table for `g_str`/`a_str`,
(4) audits label quality and provenance, and (5) orders the Colab fixes by what unblocks S2 fastest.

Every number is **MEASURED this run** (artifacts under `raw/` in this directory; re-pull + analysis
code under `code/`) unless stamped otherwise. Record counts are counts of *records*, never files
(C18). Corpus artifacts reviewed on the far side: `Sayood/tanitad-ph0-aug120` (fresh
`dataset_info` listing, `raw/farside_inventory_slim.json`) and
`Sayood/tanitad-alpamayo2-augmentation` (`records.parquet` re-pulled and re-counted).

**The headline, in one paragraph.** The pipeline's strategic layer today is a single VLM field
(`goal_kind`) mapped 1:1 into 4 of the 11 `g_str` tokens, with **no args, no arg-mask, no
provenance tags, and one cross-check that is structurally dead** (`corroborated_by_route` is False
on 801/801 records because the field it reads is never present). Its judgement quality is
**precision-good, recall-bad, and side-biased**: on aug120 every TURN it claimed is geometrically
real (19/19), but it caught only 17/33 valid left-turns and **2/29 valid right-turns**, and 28 of
its 31 `ROUTE_TO` claims sit on plain turn geometry (G1 is CLOSED at 0/31, so `ROUTE_TO` is
unsupervisable regardless). Meanwhile **the pipeline already computes, and then throws away, the
signal that fixes almost all of this**: Engine A's `route_from_future_v3` token + args survive only
inside the B4 prompt text of `_calls`. The S2-fastest path is therefore *not* more VLM — it is
**promoting Engine A geometry to the primary `g_str`/`a_str` labeler** (VLM demoted to
corroborator), which is CPU-only, needs no re-inference for the 801 already-labeled clips, and
covers 100 % of clips with the PI's FOLLOW_MAIN_ROAD default.

---

## 1. The code-side S2 contract (the format the pipeline must emit)

### 1.1 The vocabularies and heads, as built

| fact | source (quoted) |
|---|---|
| `STRATEGIC_GOAL_TOKENS` = `KEEP_CORRIDOR, LANE_TARGET, EXIT_RIGHT, EXIT_LEFT, TURN_LEFT, TURN_RIGHT, STRAIGHT_THROUGH, ROUTE_TO, STOP_AT, FOLLOW_MAIN_ROAD, NONE_ABSTAIN` — **11 tokens**; FOLLOW_MAIN_ROAD is THE DEFAULT with no route set up | `v6.py:136-143` |
| `STRATEGIC_ACTION_TOKENS` = `PREPARE_LANE_CHANGE, HOLD_CORRIDOR, REDUCE_TO, PREPARE_EXIT, PREPARE_STOP, RESUME_CRUISE` — **6 tokens** | `v6.py:144-148` |
| Args: **8 float slots** `arg0..arg3` + constraint slots `within_m, by_time_s, at_arc_m, hold_for_s`; physical units; “Unset = unconstrained (the mask says which are set)” | `v6.py:167-177` (`GOAL_ARG_SLOTS = 8`) |
| The heads: `goal_head_str = GoalHead(vocab_str, cfg.d_str)`, `act_head_str = GoalHead(vocab_a_str, cfg.d_str)` — both `d_cond=0` (no conditioning input) | `v6.py:2188-2189` |
| Head output: `{"logits" [B,11], "args" [B,8], "probs"}` (goal) and `[B,6]+[B,8]` (action) | `v6.py:1142-1144` |
| Consumer seam: `e_g_str = cond_tac(g_str["probs"], g_str["args"])` conditions `goal_head_tac`; `e_a_str = vocab_a_str.encode(...)` conditions `predictor_str` | `v6.py:2594-2597, 2624` |
| Label-encode path: `GoalVocabulary.encode(ids_or_probs, args, arg_mask, ...)` accepts **hard long ids** and a masked arg vector — a label can be pushed through the same embedding path the head's soft output uses | `v6.py:1022-1050` |
| ⛔ **The categorical arg channel does NOT exist for the strategic vocabularies.** `goal_cat_args` attaches only to `vocab_tac / vocab_tac_lat / vocab_tac_lon` and the three tactical heads | `v6.py:2265-2274` |
| S-S trains `layer_str` ONLY | `v6.py:1961-1966` (`STAGE_GROUPS["S-S"] = ("layer_str",)`) |
| In S-S the ONLY in-force loss today is `s1_latent`; every other weight is zeroed | `train_v6_staged.py:212-221` (`V6LossWeights.for_stage("S-S")`) |
| S-S gate spec: `STRATEGIC_family` required + `sel_gap_revalidated` + `TACTICAL_revalidated`; criteria include “STRATEGIC_family: computable at all (measured vs n/a today)” | `train_v6_staged.py:402-428` |

**Consequences.**

1. ⛔ **There is NO goal-label loss term in the trainer today.** `v6_loss_step` has latent terms
   (`t1`, `s1`), planner terms, and `w_anchor` (tactical `AnchorGoalHead` only,
   `train_v6_staged.py:1097-1121`). Nothing reads a `g_str`/`a_str` label. S2 is therefore a **new,
   small loss term** (goal-head-only — consistent with the binding rule “labels supervise
   GOAL/INTERPRETATION HEADS only, never any WM trunk loss”, `HIERARCHY_VOCABULARY.md` §2) plus new
   batch keys. Escalation item — the trainer edit is outside this stream's lane (A&I owns
   `stack/scripts/`), spec below.
2. ⛔ **`ROUTE_TO`'s arg is a TYPE the strategic head cannot express.** Its spec arg is
   `text_token_id` (a categorical id into an OCR city/POI vocab,
   `HIERARCHY_VOCABULARY.md:70`) — but `vocab_str` has no categorical channel (`v6.py:2265-2274`),
   and regressing an id as a float is exactly the type error the tactical cat channel was built to
   remove (`v6.py:216-227`). With G1 also CLOSED (§4.3), `ROUTE_TO` is doubly unsupervisable today:
   **no admissible label source AND no admissible arg slot.**
3. The v6 forward emits `g_str` per **window** (`z_str_p` is per-batch-item); labels arrive per
   **clip**. The schema must carry the join and a validity band (below).

### 1.2 THE S2 LABEL SCHEMA (proposed `s2-strategic-v1` — what the Colab pipeline must emit)

One JSON record per clip (sidecar dir or one JSONL), plus a join manifest:

```jsonc
{
  "schema_version": "s2-strategic-v1",
  "clip_id": "01b24287-0026-4e83-…",        // PhysicalAI clip UUID
  "t0_s": 8.0,                               // decision time inside the clip
  "g_str": {
    "token": "TURN_LEFT",                    // ∈ STRATEGIC_GOAL_TOKENS (v6.py:139) — string, validated on load
    "token_id": 4,                           // index into STRATEGIC_GOAL_TOKENS, emitted redundantly, asserted equal
    "args": [27.3, 0, 0, 0, 0, 0, 0, 0],     // [8] float32 in GOAL_ARG_NAMES order (v6.py:174-176), physical units
    "arg_mask": [1, 0, 0, 0, 0, 0, 0, 0],    // [8] ∈ {0,1}; 0 = unconstrained, arg NOT regressed (IGNORE discipline, v6.py:936-939)
    "provenance": "path",                    // ∈ {path, signage, vlm-fused} (HIERARCHY_VOCABULARY.md §2) — per instance, REQUIRED
    "sources": ["engine_a.route_v3"],        // the concrete producers, auditable
    "corroboration": {"vlm_goal_kind": "turn_left", "agrees": true},
    "confidence": 0.93                       // optional; NOT a softmax target, audit-only
  },
  "a_str": {                                 // same shape over STRATEGIC_ACTION_TOKENS (v6.py:145)
    "token": "PREPARE_STOP", "token_id": 4,
    "args": [0, 0, 0, 0, 34.0, 0, 0, 0],     // e.g. within_m = slot 4 (first constraint slot)
    "arg_mask": [0, 0, 0, 0, 1, 0, 0, 0],
    "provenance": "path", "sources": ["engine_a.lonmode"]
  },
  "valid_window_s": [-2.0, 2.0],             // windows whose t_now falls in t0+this band may consume the label
  "disjointness": {"situation_classifier_output_used": false}   // asserted by the builder, per record
}
```

Trainer-side tensors after the join (per collated window; all optional — absent keys, no loss):

| batch key | shape/dtype | consumed by |
|---|---|---|
| `g_str_id` | `[B]` long | CE vs `out["g_str"]["logits"]` (`v6.py:2680`) |
| `g_str_args` | `[B, 8]` float32 | masked L1 vs `out["g_str"]["args"]` |
| `g_str_arg_mask` | `[B, 8]` float32 | multiplies the arg residual — slot unset ⇒ zero gradient |
| `a_str_id` / `a_str_args` / `a_str_arg_mask` | as above over the 6-token vocab | `out["a_str"]` |
| `s2_valid` | `[B]` bool | masks all four terms for windows outside the validity band |

The loss term (spec for the A&I stream; ~15 lines in `v6_loss_step`, weight `w.s2_goal`, default
**0.0** so every existing stage is bit-identical; non-zero only in S-S):

```
L_s2 = CE(g_str.logits, g_str_id) + CE(a_str.logits, a_str_id)
     + |g_str.args − g_str_args| · g_str_arg_mask  (mean over set slots)
     + |a_str.args − a_str_args| · a_str_arg_mask
```

all masked by `s2_valid`. It touches ONLY `goal_head_str`/`act_head_str` outputs — both in
`layer_str` (`v6.py:2316-2317`), the exact group S-S trains — so “goal head only, never a trunk
loss” holds by construction.

**Arg-slot conventions per token** (from `HIERARCHY_VOCABULARY.md` §3, mapped into the 8 slots;
slot indices 0-3 = `arg0..arg3`, 4-7 = `within_m, by_time_s, at_arc_m, hold_for_s`):

| token | set slots | source of the value |
|---|---|---|
| KEEP_CORRIDOR | arg0=target_arc_m | route_v3 `arc_m` |
| LANE_TARGET | arg0=lane_offset_idx (±1/±2, ordinal-metric), arg1=deadline_m | E4.1 LAT events |
| EXIT_L/R | arg0=distance_m | route_v3 `dist_m` |
| TURN_L/R, STRAIGHT_THROUGH | arg0=intersection arc (m) | route_v3 `dist_m` |
| ROUTE_TO | ⛔ unfillable (categorical `text_token_id`; no cat channel on `vocab_str`, G1 closed) | — |
| STOP_AT | arg0=distance_m | lonmode `stop_dist_m` / `arc_from_t0_m` |
| FOLLOW_MAIN_ROAD, NONE_ABSTAIN | none (mask all-zero) | — |
| PREPARE_LANE_CHANGE | arg0=dir (±1, declared sign convention), within_m | latmaneuver `arc_from_t0_m` |
| HOLD_CORRIDOR | at_arc_m=arc_m | route_v3 |
| REDUCE_TO | arg0=v_target_ms, within_m | speed_events `dv`/`v_min_future` |
| PREPARE_EXIT | arg0=dir (±1), within_m | route_v3 |
| PREPARE_STOP | within_m | lonmode stop event arc |
| RESUME_CRUISE | arg0=v_target_ms | speed_events launch/free_cruise |

**The join gap the schema must carry:** trainer episodes carry only an **int** `episode_id`
(`mixing.py:71-75`) and epcache files are `ep_00000.pt…`, while labels key on the clip UUID (the
w120 corpus shards `<clip>.v2ep.pt` and `bridged_w120train_2400/ego/<clip>.npz` carry it). The S2
delivery therefore includes `clip_index.json` (epcache file → clip UUID, from the cache build
manifest) or the dataset keeps the UUID — without it the labels are unjoinable and the term
silently never fires.

---

## 2. The pushed artifacts, re-measured (records, not files)

| artifact | files | **records (measured)** | matches registry/report? |
|---|---|---|---|
| `Sayood/tanitad-ph0-aug120/fused_aug120/` | 204 | **201** fused records + 3 meta (`_summary`, `_label_sources`, `_batch_accounting`) | ✅ §11.2 (201) |
| `…/fused_w120val/` | 601 | **600** fused records + `_summary.json` (175 corrob / 41 confl / 56 alpamayo / n_sam3 596) | ✅ baseline |
| `…/batch_*/v2/` (raw VLM v2 records) | 25 | **353** clip records over **201 unique clips** (152 duplicated across the 8-pass and 40-pass) | ✅ AUG120_FUSION_RESULT §5 |
| `Sayood/tanitad-alpamayo2-augmentation/records.parquet` | 1 | **23,644 rows · 4,729 clips · 5 tasks** (trajectory / meta_action / auto_labeling / vqa 4,729 each; grounding_via_vqa 4,728) · `error` non-null **0** | ✅ §11.1 exactly |

Schema stamps: `ph1-fused-v1` on 201/201 and (sampled) 600; `ph0-v2.2` on 25/25 batch files;
`_ego_prompt_mode: "past"` on **353/353** v2 records. A2 parquet: `seed=42`,
`model_id=nvidia/Alpamayo2-Super` on all rows (spot-verified).

**Known holes, confirmed at record level:** aug120 SAM3 leg — **115/201 records carry
`perception.absent = "AUG120_SAM3_STAGE_GAP"`** (the `--n 4` default defect; root cause already
fixed in `aug120_pipeline.py`). `fused_w120val` — the 4 records fused with a silently-empty
perception layer are **identified by content signature** (`tracks == []` AND
`per_concept_hits == null` AND no absent marker): `1d4dcb4e-5117-…`, `a26a627a-caf4-…`,
`b02c28ce-e2c7-…`, `b0388541-b7de-…` (`raw/val_a2_analysis.json`; full records under
`raw/sample_fused_w120val/`). ⚠️ **New finding: these 4 are not perception-only holes — they are
TRIPLE-empty.** `semantics.scene/signs/symbols` are all `null`, `ego_state` is `null`, no Alpamayo
layer; only the npz-recomputed `speed_profile` exists. Their `g_str = NONE_ABSTAIN` is the
`GOAL_TO_GSTR` **default for a missing record** (`ph1_fuse.py:278-280`), not a judgement — they
must be excluded from any S2 label set, or 4 of the 10 val abstains are manufactured.

---

## 3. The per-token supervision-gap table

### 3.1 How `g_str` is derived today (the whole mechanism)

`ph1_fuse.py:274-326`: `g_str = GOAL_TO_GSTR[v2.symbols.goal_kind]` — a 1:1 rename of one VLM enum
field. `src: "vlm"` hard-coded; the single cross-check `corroborated_by_route`
(`ph1_fuse.py:317-319`) reads `v2.route.token`, **a field the production v2 records never carry**
(`ph1_fuse.py:160-162`, and measured: False on 801/801) — the check is structurally dead. The
2-of-3 vote exists only for the tactical axes. **No args, no mask, no provenance tag, no a_str.**

Meanwhile every B4 prompt embeds `ENGINE_A = {route_token, route_valid, route_dist_m, route_arc_m,
maneuver_dyaw_rad, v_min/max_future_ms, net_dv_ms, stops, peak_kappa, lane_change_events[…arc_from_t0_m],
speed_events[…dv, stop_dist_m, arc_from_t0_m]}` — computed by `ph0_pilot.engine_a_summary` from
`refb_labels.route_from_future_v3 / latmaneuver / lonmode` — and **none of it is persisted as a
structured field** (`run_clip`, `ph0_v2.py:612-689`, keeps only `ego_state`). This run recovered it
by parsing the prompt text of all 353 records (`code/s2_engineA_cross.py`): 201/201 clips have a
full Engine A block, route census `follow 100 · turn_left 33 · turn_right 29 · u_turn 1 · merge 1 ·
unknown(valid=False) 37`.

### 3.2 `g_str` — token by token (aug120 n=201 / val n=600, MEASURED)

| token | emitted today | derivable from the pushed records? | what's missing | miss class |
|---|---|---|---|---|
| **FOLLOW_MAIN_ROAD** | 151 / 467 | ✅ VLM + route `follow` (98/100 agree) | geometry gate never applied; 15 of the 151 sit on valid turn geometry (§4.1) | pipeline defect — **Colab-fixable** |
| **ROUTE_TO** | 31 / 70 | ⛔ NO | G1 CLOSED at **0/31** (`G1_RESULT.md`: sign texts unverifiable at 448 px; several look like VLM priors); evidence sign is not even `nav` on **24/31** (speed 15, other 6, yield 2, stop 1) and **47/70** on val; arg is categorical with no cat channel on `vocab_str` | source-data absence (no route/nav ground truth in PhysicalAI — settled at five probes) + code gap. **Remap or abstain** (28/31 sit on turn geometry) |
| **TURN_LEFT** | 17 / 36 | ✅ **route_v3 gives 33 valid left-turns with `dist_m` args** | VLM recall 17/33 (52 %); fuser ignores route | pipeline defect — **Colab-fixable** (geometry-primary) |
| **TURN_RIGHT** | 2 / 17 | ✅ route_v3 gives 29 valid right-turns | VLM recall **2/29 (7 %)** — side-biased | same |
| **STRAIGHT_THROUGH** | 0 / 0 | ✅ route_v3 has a `straight` token (`refb_labels.py:827`) | VLM never chooses it; fuser never reads route | pipeline defect — Colab-fixable |
| **EXIT_RIGHT / EXIT_LEFT** | 0 / 0 | ⚠️ detector exists (`exit_left/exit_right` ∈ `ROUTE_V3_TOKENS`, `refb_labels.py:827-828,1139`) but **0 occurrences** in the aug120 route census | corpus slice has no ramp/exit events; no junction/lane-graph annotation in PhysicalAI (`map.xodr` via NuRec is the known topology workaround) | source-data absence on this slice |
| **STOP_AT** | 0 / 0 | ✅ ego spine: `stops > 0` on **42/201** clips; Engine A stop events (`stop_at_point/hold_stop` with `stop_dist_m`) on **25/201** | VLM chose `stop_at` **zero times in 801 clips**; fuser derives nothing from the speed profile it already recomputes | pipeline defect — Colab-fixable |
| **LANE_TARGET** | 0 / 0 | ✅ Engine A `lane_change_events` non-empty on **120/201** (`lat_offset` + `arc_from_t0_m` = the `lane_offset_idx, deadline_m` args) | VLM never emits it; fuser never reads the events | pipeline defect — Colab-fixable (⚠️ latmaneuver's FP rate on curves must be gated by `route_valid`, see §4.2) |
| **KEEP_CORRIDOR** | 0 / 0 | ✅ trivially (route `follow` + `arc_m`) | overlaps FOLLOW_MAIN_ROAD — the PI's binding diagram lists FOLLOW_MAIN_ROAD·ROUTE_TO·LANE_TARGET·TURN as the core set | **vocabulary-resolution decision for the PI**, not a data gap |
| **NONE_ABSTAIN** | 0 / 10 | n/a | 4 of the 10 are the triple-empty records (§2) — manufactured abstains | pipeline defect — mark + exclude |

### 3.3 `a_str` — the raw material EXISTS and is currently consumed as the wrong thing

⭐ **Finding: `ph0_v2.py`'s `ACTION_VERBS` (`ph0_v2.py:42-43`) IS the `a_str` vocabulary,
lowercased** — the B4 call already elicits up to 3 `(verb, direction)` strategic actions per clip.
The fuser never emits them as `a_str`; it shreds them through substring rules into *tactical*
action votes (`ph1_fuse.py:64-74,296-299`), where `reduce_to` matches **no rule at all** and is
silently dropped, and `hold_corridor` double-votes LANE_KEEP *and* HOLD.

Clip-level verb counts (aug120 / val) and geometric consistency of the aug120 claims against the
recovered Engine A (checker = the lite form of `ph0_pilot._check_action_geometry:484-541`, which
exists precisely for this and is unused in production):

| verb | aug120 clips | val clips | vs geometry (aug120) | note |
|---|---|---|---|---|
| hold_corridor | 159 | 436 | 130 ok / **29 dispute** | dispute = junction-scale route event inside the hold |
| prepare_lane_change | 54 (L28/R26) | 156 | 33 ok / **21 dispute (39 %)** | no lc event in hindsight path |
| reduce_to | 49 | 125 | 44 ok / 5 dispute | `v_target` arg NEVER present (by design: VLM is forbidden numbers, `P_B4`; Engine A has `dv`/`v_min_future` and never attaches them) |
| prepare_stop | 6 | 24 | 2 ok / 4 dispute | vs 25 geometric stop-event clips — recall ~8 % |
| prepare_exit | 1 | 1 | 1 dispute | see EXIT_* |
| resume_cruise | 1 | 5 | 1 dispute | derivable from `launch/free_cruise` lon events instead |

⇒ `a_str` labels for S2 should be **geometry-primary too** (lonmode/latmaneuver/route events →
token + arc/dv args), with the VLM verb kept as a recorded corroboration vote — the exact
jurisdiction split the fusion strategy already uses everywhere else.

### 3.4 What the Alpamayo layer adds (measured on the parquet)

`meta_action` is a **tactical triple** (`"Longitudinal: … Lateral: … Lane: …"`) plus a causal
`cot` sentence — no strategic horizon. Phrase census (first 2,000 meta rows):
straight 1082 · speed 915 · slow 375 · stop 221 · yield 170 · lane change 69 · turn right 60 ·
turn left 32 · exit 15 · merge 12. Useful as a third corroboration vote (as today) and as
stop/turn-phrase evidence; **not** a `g_str`/`a_str` source. 40 verbatim rows across all 5 tasks:
`raw/a2_task_samples.jsonl`.

---

## 4. Label-quality audit (sample-based, with n)

**4.1 The route × g_str confusion (aug120, n=201; `raw/engineA_cross.json`).**

| Engine A route (valid) | n | VLM-fused g_str |
|---|---|---|
| follow | 100 | FOLLOW_MAIN_ROAD 98 · ROUTE_TO 2 |
| turn_left | 33 | TURN_LEFT 17 · **ROUTE_TO 10 · FOLLOW_MAIN_ROAD 6** |
| turn_right | 29 | TURN_RIGHT 2 · **ROUTE_TO 18 · FOLLOW_MAIN_ROAD 9** |
| u_turn | 1 | ROUTE_TO 1 |
| merge | 1 | FOLLOW_MAIN_ROAD 1 |
| unknown (valid=False) | 37 | FOLLOW_MAIN_ROAD 37 |

TURN **precision 19/19 = 100 %** (every claimed turn has matching valid route geometry, all 19
rows in `raw/engineA_cross.json:turn_claims_vs_route`); TURN **recall 17/33 L / 2/29 R**; **44
geometry-valid junction maneuvers carry a non-turn label** today. The 2/29 right-turn recall is a
systematic side bias, not noise (see 4.4).

**4.2 Caveats on the geometric side, so the fix does not over-claim.** `route_valid=False` on
37/201 (18 %) — those correctly default to FOLLOW_MAIN_ROAD/abstain. `latmaneuver` lane-change
events fire on 120/201 clips, which at face value out-runs plausible LC frequency — the S2 build
must gate LANE_TARGET on displacement magnitude + `route_valid`, and treat the 21/54 LC disputes
as symmetric evidence (either side can be wrong) until spot-checked on video. The one `dyaw=0.0 /
dist_m=null` turn row (1/19, `8dc5d14d…`) shows route can validate a turn whose maneuver window
sits at the clip edge — args go unset (mask=0) there, never fabricated.

**4.3 ROUTE_TO / signage (n=31 aug120 + 70 val).** Evidence sign kinds: aug120 — speed 15, nav 7,
other 6, yield 2, stop 1; val — nav 23, speed 23, other 13, yield 8, light 2, null 1. The B4 rule
(“route_to ONLY if a navigation sign was actually read”, `ph0_v2.py:176-177`) is enforced only as
*an index exists*; the fuser's `goal_evidence` check counts generic `traffic sign` SAM3 tracks
(`ph1_fuse.py:252-258`) and never checks the KIND — so 15 “grounded” verdicts include
speed-sign-backed ROUTE_TO. G1: **0/31, gate stays CLOSED, `route_to` goals extraction-only**
(`Project Steering/G1_RESULT.md`). INHERITED, consistent with everything measured here.

**4.4 Same input → same judgement (n=152 replicated clips).** The 8-pass and 40-pass batch files
contain **152 clips independently labeled twice** by separate VLM invocations. MEASURED:
`goal_kind` identical **152/152**, `actions` identical **152/152** (greedy decoding reproduces
bit-exactly; wall-times differ, so these are true re-inferences, not copies). ⇒ VLM label noise is
**zero-variance and pure bias** — re-running buys nothing; only a different labeler (geometry) or a
different prompt/model changes the answer. This is why the recall asymmetry (17/33 vs 2/29) is a
systematic defect, not sampling.

**4.5 SAM3 support for strategic claims.** On aug120 only 86/201 clips have any SAM3 record
(115 named-absent, `--n 4` root cause). Of the 31 ROUTE_TO: 15 `grounded` / 16 `not_computable`
(SAM3 absent) — and per 4.3 “grounded” currently means “some sign-like track existed”, not “the
claimed nav sign exists”. The known sign-class reliability flag (⅔ of best crops had no sign —
runbook §6.7) stands. **No strategic token should rest on SAM3 evidence in S2 v1**; the geometric
path does not need it.

**4.6 The four val records (n=4).** §2: triple-empty, `g_str=NONE_ABSTAIN` by default-of-absence,
`scenario_description = "?, ?, ? ?-lane; ego nan m/s …"`. No goal/census verdicts were fabricated
for them (nulls) — the feared fabrication did not materialize, but the **labels themselves are
fabricated abstains** and 4/10 of val's NONE_ABSTAIN mass is fake.

---

## 5. The goal-provenance audit (required by the S-S gate)

Per derivable token, the chain as it would ship in S2 v1 (geometry-primary), and the two binding
checks. Labels MAY use ego/futures — that is in-contract; the point is the record.

| token(s) | provenance chain (label side) | `provenance` tag |
|---|---|---|
| TURN_L/R, STRAIGHT_THROUGH, EXIT_*, KEEP_CORRIDOR/FOLLOW_MAIN_ROAD (non-default), STOP_AT, LANE_TARGET | ego future poses → `route_from_future_v3` / `latmaneuver` / `lonmode` (`refb_labels.py`) → token + args | `path` |
| FOLLOW_MAIN_ROAD (default) | PI rule: default when no route is set up (`v6.py:136-138`) | `path` (default) |
| ROUTE_TO | would be VLM B4 + OCR — **G1-gated CLOSED; not shipped** | `signage` (gated) |
| all `a_str` | ego future poses → lonmode/latmaneuver/route events (+ VLM verb as corroboration vote, recorded) | `path` (+`vlm-fused` corrob) |

**Check 1 — goal/situation disjointness** (*“could this have been computed from the situation
classifier's output?”*): **HOLDS, at three independent probes.** (a) The production v2 records
carry **no `situations` key at all** — the frozen detectors never ran on these batches (measured;
also why `scene_vs_situations` fired 0/201). (b) The B4 prompt builder **structurally omits**
situations from the Engine A block — “that omission is LOAD-BEARING … pinned by a test”
(`ph0_pilot.py:403-410`). (c) The fuser asserts no situation output inside the vocab block
(`ph1_fuse.py:324-325`). The proposed geometric-first path reads **poses**, not any classifier
output. ⚠️ One statement to keep exact: `route_v3` and the situation labels share the **ancestor**
(ego dynamics). Shared raw ancestry is admissible — the rule forbids consuming the classifier's
*output*, not its inputs — but it means a “situation-stratified S2 eval” is *not* independent
evidence of goal quality; stratify by route token instead.

**Check 2 — the echo test** (*“does any input at inference contain what the label was derived
from?”*): **PASSES for the path-provenance tokens.** At inference `goal_head_str` consumes only
`z_str` — vision-derived, `d_cond=0`, “no situation-classifier output in any form; no ego state;
no `**kwargs`” (`v6.py:1057-1063`); ego futures/Engine A are not inputs anywhere at inference.
The nav-echo shape (a head echoing its own input, the flagship 1.0000 defect) cannot recur because
v6's strategic layer has **no nav/route input channel to echo**. ⚠️ Standing condition to re-check
at S2 launch: the w120 encoder input must remain frames-only — if any ego channel is ever
concatenated into the encoder input, this audit is void and must be re-run.

**Two provenance defects to fix in the pipeline (recorded, not hand-waved):**
1. ⛔ The fused `_provenance.vlm = "vision"` is **wrong as shipped**: 353/353 v2 records were
   produced with `_ego_prompt_mode: "past"` (v2.2) and the B4 prompt embeds Engine A (ego-future
   geometry). The VLM layer is `vision + ego-past-prompt + engineA-prompt`. Admissible for labels —
   but the tag must say it, because `inference_admissible: ["perception", "semantics"]` currently
   whitelists `semantics` (which contains `symbols.goal_kind`) as if it were pure vision. Any
   *inference-time* consumer of `semantics` (eval strata, probes) must know it is ego-touched.
2. The per-instance `provenance` tag required by `HIERARCHY_VOCABULARY.md` §2 (`path|signage|
   vlm-fused`) is absent from every emitted `g_str` — schema §1.2 adds it.

---

## 6. The Colab fix list, ordered by what unblocks S2 fastest

All CPU-only unless stated. Items 1-4 produce a complete S2 v1 label set for the 801 labeled clips
without any model re-run; item 7 is the only GPU item and is NOT on the S2 critical path.

| # | fix | mechanism | unblocks |
|---|---|---|---|
| **1** | **Persist Engine A as a structured field** in every v2 record (`rec["engine_a"] = engine_a_for_prompt(ea)` in `ph0_v2.run_clip`), and for the 801 already-labeled clips **recompute it from the bridged npz** (`bridged_w120train_2400/ego/`, 4,802 files on HF; same `refb_labels` calls) — no VLM re-run | the g_str/a_str source signal stops living only in prompt text | everything below |
| **2** | **Geometry-primary `g_str` emission** in the fuser: `route_valid` route token → TURN_L/R (62 clips), STRAIGHT_THROUGH, EXIT_*, U-turn handling; `stops`/`stop_at_point` → STOP_AT (25-42 clips) with `dist_m` args; else FOLLOW_MAIN_ROAD (default per PI); VLM `goal_kind` demoted to a recorded corroboration vote; **remap the 28 geometry-backed ROUTE_TO to their turns; abstain the rest** | fixes the 44 missed turns + kills the unsupervisable ROUTE_TO mass | S2 g_str labels, ~30 % non-default |
| **3** | **Emit `a_str`** from lonmode/latmaneuver/route events + args (arc/dv/v_target), with the VLM verb as a corroboration vote through the EXISTING `_check_action_geometry` (`ph0_pilot.py:484` — move it into the fuser); stop shredding strategic verbs into tactical substring votes | S2 a_str labels | S2 a_str |
| **4** | **Emit the S2 schema** (§1.2): args + arg_mask + per-instance `provenance` + `valid_window_s` + `clip_index.json`; fix `_provenance.vlm` to name the ego/engineA prompt conditioning; delete or implement `corroborated_by_route` | the format the code consumes | trainer join |
| **5** | **Mark/exclude the 4 triple-empty val records** (absent markers for the VLM layer too; their NONE_ABSTAIN must not enter S2). Re-fuse of val600 remains the standing PI decision — this item is only the S2-side exclusion list | honest val labels | S2 val split |
| **6** | Fix the dead ego lat-vote mapping (`ph1_fuse.py:287`: keys `left/right` vs values `turning_left/turning_right` — MEASURED: ego voted null on 36/36 turning clips, LANE_KEEP on 165/165 straight) | tactical, but same fuser pass; the S2 build must not inherit the idiom | tactical vote integrity |
| **7** | SAM3 re-run for the 115 uncovered clips (**GPU pod, ~30 min**, fix already in `aug120_pipeline.py`) + re-fuse | perception completeness; NOT needed for path-provenance S2 v1 | goal_evidence/census layers |
| **8** | ROUTE_TO longer-term: reopen only behind (a) higher-fidelity OCR than 448 px (G1's condition), (b) a categorical arg channel for `vocab_str` (code work, A&I lane), (c) nav-kind-checked evidence | the only PhysicalAI-native route signal | post-v1 |

**Escalation (not Colab, named owners):** (i) the S2 loss term + batch keys + `--s2-labels` in
`train_v6_staged.py` (A&I stream; spec in §1.2 — small, default-off, bit-identical when absent);
(ii) `STAGE_GATE_SPEC["S-S"]` should carry the **goal-provenance audit** as a required row next to
`STRATEGIC_family` (the PI made it part of the S-S gate; today's spec at `train_v6_staged.py:402`
does not name it); (iii) the KEEP_CORRIDOR-vs-FOLLOW_MAIN_ROAD fold and the ±1 direction-sign arg
convention are PI decisions — flagged, defaults proposed here.

**Expected S2 v1 coverage (ESTIMATED from the measured censuses):** 100 % of clips labeled (default
FOLLOW_MAIN_ROAD), ~30 % non-default `g_str` on aug120-like slices (62 turns + 25-42 stops + gated
LANE_TARGET out of 201), `a_str` non-HOLD on ~35-45 %. Versus today's usable stream: 19
high-precision turns and 151 defaults of which 15 sit on turn geometry.

---

## Deliverable manifest

| artifact | where |
|---|---|
| This report | `TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-08-16-s2-strategic-gap/S2_STRATEGIC_GAP.md` |
| Measured censuses (aug120 201-record analysis; Engine A cross incl. route×g_str, turn rows, a_str checks, 152-replicate probe; val600 + A2 analysis) | `…/raw/aug120_analysis.json`, `…/raw/engineA_cross.json`, `…/raw/val_a2_analysis.json` |
| Record samples with real field contents (30 fused aug120 incl. all 19 TURN + 8 ROUTE_TO; 30 fused val incl. the 4 triple-empty; 40 A2 rows across 5 tasks) | `…/raw/sample_fused_aug120/`, `…/raw/sample_fused_w120val/`, `…/raw/a2_task_samples.jsonl` |
| Far-side inventory (per-dir counts + fused/batch file lists, sizes) | `…/raw/farside_inventory_slim.json` |
| Re-pull + analysis code (reproducible; token read in place from `Keys.txt`, never copied) | `…/code/s2_pull*.py`, `…/code/s2_analyze_*.py`, `…/code/s2_engineA_cross.py` |
| Full pulled corpus (201+600 fused records, 353 v2 records, parquet) | session scratchpad `…/scratchpad/s2_pull/` (NOT committed — 800+ files; HF is the durable copy, verified by this run's re-pull) |

Staged, never pushed/committed. Evidence classes: all counts/censuses MEASURED (this run); G1
verdict + fusion-run internals INHERITED from `G1_RESULT.md` / `AUG120_FUSION_RESULT.md` (both
consistent with this run's independent recounts: 201, 115, 88/10 not re-derived here, g_str
151/31/17/2 re-derived and matching); coverage projection ESTIMATED.
