# SPEC — `stack/tanitad/data/v7_labels.py`, the ONE B1 label consumer

**Written** 2026-08-30 ~07:50 (Europe/Berlin) · **Author** Master Mind · **Owner**
TrainingFlyWheel · **Blocks** v7f, refc_v3, refa_v1, refd (all four B1 consumers).

`B1_TRAINING_PREP.md` §2 named this module. ⛔ **It does not exist** — three probes:
no `stack/tanitad/data/v7_labels.py`, no module implementing the D-LABEL-GT
conditions (only `models/vocab_v7.py`, which is the vocabulary, not the consumer),
no test referencing it. **This — not the epcache — is the v7f blocker.** The cache
build finishes ~11:00 and no trainer can read B1's labels without this.

## 0. Everything below is MEASURED, not inherited

Read 2026-08-30 from `/home/nvidia/data/b1-bundle/labels/s2_labels_v7.jsonl.gz`
(**4,719 records**, `schema_version` `s2-geom-v7`, `vocab` `v7`) and from
`stack/tanitad/models/vocab_v7.py`. ⚠️ **`B1_TRAINING_PREP.md` §2 cites three
fields that DO NOT EXIST in this blob** — `disputed`, `time_basis`, `t_nominal_s`
returned **zero hits** at top level and inside `cot_tokens` / `semantics` /
`alpamayo` / `g_str` / `a_str`. The prep doc was written against the earlier blob
(`121a8d93`); **implement against the schema, not against the doc**, and report the
divergence rather than inventing the fields. *(This spec exists because I nearly
shipped a build chain citing three flags that did not exist — MM-C5.)*

**Top-level keys (all 4,719/4,719):** `_provenance`, `a_str`, `a_tac`, `alpamayo`,
`bands`, `clip_id`, `cot_source`, `cot_tokens`, `g_str`, `g_tac`, `horizon`,
`manoeuvre_sequence`, `nav_command`, `scene`, `schema_version`, `semantics`,
`t0_s`, `turn_suppression`, `vocab`.

## 1. ⛔ THE BINDING CONSTRAINT — `nav_command` IS AN ORACLE, ALL 4,719 OF THEM

```
nav_command.provenance : {"ego-future": 4719}      # 100 %, oracle: true
vocab_v7.NAV_PROVENANCE: ["nav-system", "ego-future"]
```

The vocab anticipates a real `nav-system` provenance. **B1 has none of it** — every
`nav_command` is computed from the ego's own future path. Feeding it at inference
is the flagship-v1 route-echo defect exactly: that head was a bijection of the nav
we fed it (369/369, 81/81) and **scored 1.0000 by echoing its own input**. CLAUDE.md
states the general form: *"A supplied route is optimistic by construction on
PhysicalAI — our only route supplier there is the ego's own future path."*

⇒ **The consumer returns `nav_command` in a field that CANNOT be silently wired to
a model input.** Concretely: it lives under an `oracle` sub-record, and any path
that feeds it must pass an explicit `allow_oracle_nav=True` that **stamps the run's
config** so no eval can quote such an arm without the stamp being visible. Default
is OFF. This also satisfies the goal/situation information-disjointness rule: a
predicted geometric goal is admissible, the ego's own future is not.

## 2. ⭐ `a_tac` IS ALREADY FACTORED — do not collapse it

```
a_tac.lat : LANE_KEEP 3057 · NUDGE_R 613 · NUDGE_L 502 · TURN_L 280 · TURN_R 267
a_tac.lon : CRUISE 1287 · ACCELERATE 1038 · ADAPT_SPEED_FOR_CURVE 1018 ·
            BRAKE_TO 927 · FOLLOW 175 · CREEP 141 · HOLD 133
```

Two independent heads, 8-wide each (`TACTICAL_LAT_ACTIONS_V7`,
`TACTICAL_LON_ACTIONS_V7`). ⭐ **This is the label-side fix for the programme's
single largest known defect** — the 5-way softmax that MIXED lat and lon, the one
mechanism that explains 0/881 accelerate, the speed-fan, and the absence of any
longitudinal signal in selection. ⛔ **Never flatten lat×lon into one class.**
`truncated` is `false` on all 4,719; `lat_args` = `within_m`, `lon_args` =
`v_target_ms` + `within_m`; `serves_goals` carries `lat_serves`/`lon_serves`.

## 3. The loss mask — 8 tokens, and the data confirms the arithmetic

`vocab_v7.NOT_YET_EXTRACTABLE` (n=8): `ABORT_LC`, `YIELD_MERGE`,
`LANE_CHANGE_L_FOLLOW_ROUTE`, `LANE_CHANGE_R_FOLLOW_ROUTE`,
`EXIT_LEFT_FOLLOW_ROUTE`, `EXIT_RIGHT_FOLLOW_ROUTE`,
`PREPARE_EXIT_FOLLOW_ROUTE`, `PREPARE_LANE_CHANGE_FOLLOW_ROUTE`.

**Head width stays FULL vocab; the mask is applied to the LOSS, never to the head.**
The observed counts cross-check the mask exactly — every absent class is a masked
class and vice versa:

| head | vocab width | classes present in B1 | absent (⇒ masked) |
|---|---|---|---|
| tactical lat | 8 | 5 | LANE_CHANGE_L, LANE_CHANGE_R, ABORT_LC |
| tactical lon | 8 | 7 | YIELD_MERGE |
| strategic action | 7 | 5 | PREPARE_EXIT_…, PREPARE_LANE_CHANGE_… |
| strategic goal | 8 | 4 | the 4 EXIT_/LANE_CHANGE_ route tokens |

⚠️ **Assert this equality in a test.** If a future blob makes a masked class
non-empty, the test must FAIL loudly — a silently-masked present class is training
signal thrown away, and a silently-unmasked absent class is a dead logit that
degrades calibration.

## 4. Strategic skew weighting

```
a_str.token : HOLD_MAIN_ROAD 2468 (52.3 %) · PREPARE_TURN_R 624 ·
              PREPARE_TURN_L 585 · RESUME_CRUISE 573 · PREPARE_STOP 469
```

One class is the majority. Expose the weighting as an argument with the weights
**computed from the loaded split, not hardcoded** (a hardcoded weight silently
mis-weights the next blob — the derived-constant trap). Report the effective
weights in the run config so an eval can see them.

## 5. Bands and time basis

`bands` = `{operative_s:[0,2], tactical_s:[2,6], strategic_s:[8,30],
unassigned_manoeuvres:[…]}`, `t0_s` = 8.0 on the sampled record, `horizon` =
`{available_s, recording_span_s}`. These align with the hierarchy's own rates
(`hz_op` 10.0 / `hz_tac` 2.0 / `hz_str` 0.5). ⚠️ `t0_s` and the band edges are
**per-record** — read them per clip, never assume the sampled values are constants.
`unassigned_manoeuvres` must be surfaced, not dropped.

## 6. Audit-only, never training input

- `turn_suppression` — non-null on **68** records; passed through for audit.
- `scene`, `cot_tokens`, `semantics`, `cot_source`, `alpamayo` — VLM-derived.
  `_provenance` says verbatim that perception tokens are *"vlm-cot, disputed, never
  geometry-derived"*. They are **not** geometry and must not enter a geometry loss.
  `alpamayo.agree` is `false` on the sampled record (expected −1.73 m/s vs measured
  +1.48) — **a disagreement flag that exists precisely so it can be filtered**;
  expose it, and report the corpus-wide agree rate in the RESULT.

## 7. Deliverable

`stack/tanitad/data/v7_labels.py` + `stack/tests/test_v7_labels.py`, plus a
RESULT.md carrying the corpus-wide counts the module computes (so the numbers above
are re-derived by the code, not copied from this spec).

**Gates:** `pytest -q` green; the §3 mask/presence equality asserted; a test that
the oracle nav CANNOT be reached without `allow_oracle_nav=True`; a
deliberate-regression test that flattening lat×lon FAILS. Verify by CONTENT —
assert on returned values, never on exit codes. **STAGE, never push**; end with a
deliverable manifest.
