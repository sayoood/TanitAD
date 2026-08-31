# REF-A v1 · the v7.2 label join and the nav join — the window→label rule

**2026-09-01 · DataFlyWheel.** Implements the s2 v7.2 LABEL JOIN and the NAV JOIN in
`stack/tanitad/data/refav1_loader.py` (`RefAV1Windows`, optional `labels_path` /
`nav_path`). `batch()` now emits real `lat_label`, `lon_label`, `route_label`,
`nav_cmd` (Long tensors) instead of `None`. Parser: **imported** from
`tanitad/data/v7_labels.py` (`load_v7_labels`, `NavEmitter`, `oracle_nav`) — no second
parser exists. Vocabulary: **imported** from `tanitad/models/v6.py`
(`tactical_lat_actions("v7.0")` = 8 classes, `tactical_lon_actions_v("v7.0")` = 8) —
never re-declared. Tests: `stack/tests/test_refav1_loader_labels.py` (9, all green;
values asserted against hand-built records).

## 1 · The rule

A window is `(episode, t)` with `t` the **last observed cache index**; its NOW is

```
t_now_s = t * 0.2        (RAW clip timeline — §2 says why there is NO offset)
```

| field | rule | source |
|---|---|---|
| `lat_label`, `lon_label` | the clip's record's `a_tac.lat` / `a_tac.lon` **iff** \|t_now − t0\| ≤ (tac_hi − tac_lo)/2, else **−100** | derived §3; reproduces `s2_labels.py:773-776` + `:587` |
| `route_label` | same in-band window set; value from the record's `nav_command` token: NAV_TURN_L→`route_left`(0), NAV_FOLLOW_ROAD→`route_straight`(1), NAV_TURN_R→`route_right`(2) (`refb.ROUTE_CLASSES` order) | same-source nav/route precedent `refb.py:60-68`; echo hazard §6 |
| `nav_cmd` | the clip's `NavEmitter` id, **constant over every window of the clip** (`t0_constant`), on ALL windows in-band or not; unlabeled episodes emit **0** (`follow`) + `nav_valid=False` | COPIED from the v6 trainer: `train_v6_staged.py:5510` (`allow_oracle_nav=True`), `:5519-5522` (`NavEmitter(..., semantics="t0_constant")`); window-independence of token AND args under t0_constant: `v7_labels.py:483-488`. Index-0 default: `refb.py:60` ("index 0 is the unlabeled default") + `refa_v1.py:748` (`nav_cmd=None` ⇒ zeros) |
| join key | the v2ep dict's `clip_id` (= cache filename stem on this corpus) | brief; MEASURED equal on the 24 probe episodes |
| join miss (episode level) | **loud, counted, named** in the init print and `join_report["…"]["missing_episodes"]`; refuse only at ZERO joined (wrong blob) | v6 precedent for the zero case `train_v6_staged.py:5175`; §5 for why partial coverage must not refuse |

Anchor `t0_s` and the tactical band are read **from each record**, never hardcoded
(the derived-constant trap): tolerance = (tac_hi − tac_lo)/2 = **±2.0 s** at the shipped
bands `tactical_s [2,6]` (MEASURED constant across all 4,719 records, §4).

## 2 · Timeline — why there is NO n_stack offset here

The v7.2 labels are anchored on the RAW clip timeline: `t0_s = ES.RAW_T0_S = 8.0`
(`egomotion_source.py:57`), bands built as `key + band*hz` from the t0 index
(`s2_geom_emit_v7.py:199-200,:544-545`), manoeuvre `t_start_s` **relative to the anchor**
(same lines; MEASURED range 0.0–33.7 s fits only that reading).

The S2 join shifts provider index → raw index by +(n_stack−1) because it consumes the
**provider** view, which drops the first n_stack−1 frames (`s2_labels.py:730-734`,
`v2_compressed.py:195-197 load_compressed`). **This loader bypasses `load_compressed`**:
it reads the v2ep dict's RAW `poses` directly (`refav1_loader.py::_ep_meta`), and the
stage-1 cache encodes every 2nd RAW frame (`refav1_probe/encode_dinov3.py:46`
`range(0, len(lens), 2)`). So cache j ↔ RAW frame 2j ↔ raw `0.2·j` s, offset zero.
Copying the S2 offset here would be the `step_s`/`df` scope error — a true correction
applied where its cause does not exist.

## 3 · Why ±half-band, derived (not defaulted)

The record's `a_tac` describes the **forward tactical band** of the anchor:
raw `[t0+2, t0+6]` (bands are forward horizons — `s2_geom_emit_v7.py:50-52`). A window
at `t_w` decodes the tactical action of **its own** forward band `[t_w+2, t_w+6]`. The
label is the majority-correct description of that horizon iff the two 4-s intervals
overlap by ≥ half, i.e. `4 − |t_w − t0| ≥ 2` ⇔ **|t_w − t0| ≤ 2.0 s**. This lands
exactly on the shipped S2 in-band choice (`S2WindowSupervision._in_band`,
`s2_labels.py:773-776`) with its default `valid_window_s (−2, 2)` (`s2_labels.py:587`)
— the same trainer-side behaviour, now carried by a derivation. Out-of-band → **−100**,
`cross_entropy`'s default `ignore_index`, the discipline of `s2_labels.py:194-197`.

Manoeuvre `t_start_s` was considered and **not** used to re-time labels per window:
mapping geometric manoeuvre segments to factored lat/lon tokens would re-implement the
extractor's classification (`s2_geom_emit_v7.py`) — the "second parser" the brief forbids.

## 4 · MEASURED — the canonical blobs and the join coverage

Blobs pulled from Thor (`/home/nvidia/data/v72/labels/`) 2026-09-01, md5-verified:

| blob | records | md5 (matches registry) |
|---|---|---|
| `s2_labels_v7.2_train.jsonl.gz` | **4,572**, clip_id unique 4,572 | `0ff902130ce76886b8a925eceed9e3a5` |
| `s2_labels_v7.2_eval.jsonl.gz` | **147**, unique | `aa12c948f062181c3297265b51526ec5` |

Uniform on every record (both splits): `schema_version s2-geom-v7`, `vocab v7`,
`t0_s 8.0`, bands `[0,2]/[2,6]/[8,30]`, nav provenance `ego-future`. Censuses (train):
lat = LANE_KEEP 2958 · NUDGE_R 592 · NUDGE_L 488 · TURN_L 275 · TURN_R 259 (5 of 8
classes; no LANE_CHANGE_L/R, no ABORT_LC — consistent with
`v7_labels.effective_mask`'s extractor-limit note); lon = 7 of 8 (no YIELD_MERGE);
nav = FOLLOW 2897 · TURN_R 864 · TURN_L 811. Every record carries lat, lon and nav.

**Coverage on the 24-episode probe list** (`C:/Users/Admin/refav1_probe/eplist.txt`,
run through the real loader against the real cache + blob — MEASURED, artifact
`scratchpad/v72labels/real_join.py` output 2026-09-01):

* **1 / 24 episodes have a train record** (`13141fac-c31b-4123-a3d6-68367304e1e1`);
  0/24 in eval; the other **23 are named** by the loader's `join_report`.
* Windows (op_window=4, str_ext 2×3.0 s): **10 / 887 in-band** (±2.0 s), `nav_valid`
  on 37/887 rows (= the labeled episode's windows).
* End-to-end value check: the supervised rows decode (through the v6.py vocabulary)
  to **LANE_KEEP / BRAKE_TO / route_straight / NAV_FOLLOW_ROAD** — exactly that
  clip's record. The loud init line printed:
  `[refav1-labels] v7.2 join: labels 1/24 eps (23 missing, …) 10/887 windows in-band +-[2.0]s md5=0ff9… | nav 1/24 … allow_oracle_nav=True`.

⚠️ **Programme consequence (sample, n=24):** the refav1 parity corpus and the
Alpamayo-labeled 4,719 clips barely overlap (~4 % on this sample). If the full
2,376-episode cache matches this rate (~100 episodes), the CE terms are fed by ~0.5 %
of windows. ESTIMATED from n=24 — the full-cache join (one loader construction on
Thor) gives the exact number. Escalated in the deliverable report: either emit v7.2
labels for the parity clips (the emitters read PhysicalAI egomotion —
`egomotion_source.py:49`), or accept sparse auxiliary supervision.

## 5 · The −100 contract (MEASURED, not assumed)

* `F.cross_entropy` default `ignore_index=−100`: mixed batches average over valid
  rows **only** (equals the CE of the valid subset); an **all-ignored batch is NaN**.
  Pinned: `test_MEASURED_cross_entropy_default_ignore_index_semantics`.
* ⛔ **The current model REFUSES −100**: `RefAV1.forward`'s range check
  (`refa_v1.py:978` lat/lon, `:993` route — `int(lbl.min()) < 0`) raises
  `ValueError("…outside [0, n)")` before the CE is reached. Pinned:
  `test_MEASURED_the_model_range_check_refuses_minus100`.
* **Proposed minimal model-side change** (NOT applied — model files untouched, per
  brief): in the two range checks validate only `lbl[lbl != -100]`, and skip the
  term (not add NaN) when no row is valid:
  ```python
  valid = lbl != -100
  if valid.any() and (int(lbl[valid].min()) < 0 or int(lbl[valid].max()) >= n):
      raise ValueError(...)                    # unchanged message
  if valid.any():
      out[f"loss_{name}_label"] = F.cross_entropy(out[key], lbl)  # ignore_index does the masking
  ```
  (route analogous). Until it lands, a trainer feeds only rows with `label != -100`
  per family, or drops unlabeled rows.

## 6 · Guards that ship with the join

* **Oracle stamp**: labels load with `allow_oracle_nav=True` — the v6 trainer's exact
  choice (`train_v6_staged.py:5510`, and its `--nav-labels` help: the flag STAMPS the
  manifest). The stamp lands in `join_report` (md5, n_records, allow_oracle_nav,
  nav_arg_semantics) for the run config. Labels may use ego (PI 2026-08-03); the
  oracle question is an **inference-input** question and `nav_cmd` is one — any
  nav-conditioned arm carries the stamp and owes a **nav-shuffle control** at eval.
* **Echo hazard, stated where it is built**: `route_label` and `nav_cmd` come from the
  SAME record field, so on a `nav_inject=True` arm route accuracy measures the
  flagship-v1 nav echo (bijection of its own input, scored 1.0000), not skill. Route
  accuracy is evidence only on a nav-shuffle control or a nav-less arm.
* **Index-alignment pin**: `NavEmitter` ids enumerate `vocab_v7.NAV_COMMAND_TOKENS`;
  the model's `nav_emb` rows are ordered by `refb.NAV_COMMANDS` (`config.py:139`).
  Their agreement (0=follow, 1=left, 2=right; ids < n_commands=4) is asserted at join
  init and spot-checked end-to-end in the tests — a coincidence turned invariant.
* **Vocabulary drift refused by name** (unknown lat/lon/nav token → ValueError with
  clip id and token); **zero-join refused** (wrong blob, md5 in the message);
  **missing `clip_id` key refused** when a join is requested, ignored otherwise
  (pre-join corpora keep working).

Evidence classes: §4 numbers MEASURED (artifacts: the pulled blobs by md5, the probe
run output, the test file); §1–§3 rules cite code lines; §4's programme consequence is
ESTIMATED (n=24) pending the full-cache join.
