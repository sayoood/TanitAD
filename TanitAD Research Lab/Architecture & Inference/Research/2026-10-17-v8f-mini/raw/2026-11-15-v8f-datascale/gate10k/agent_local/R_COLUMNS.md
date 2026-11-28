# R_COLUMNS: which of the 14 per-clip columns move with the R_s0 derangement and which stay

Status: a proposal for the MM to rule on. R_s0 is NOT minted. Evidence class: MEASURED (source read on the pod tree `cdbff971` at `/root/dscale10k/tree/`, and a CPU probe of the real union and R releases, `F:/TanitAD-scratch/datascale/gate/R_columns_probe.json`).

## What the derangement does today (MEASURED)
* `dscale10k_merge.py:366-368` (`is_own_column`) keeps `k`, `now_s` and every `nav_*` as the clip's own (`OWN_EXACT`, `OWN_PREFIX`, lines 51-52). Every other `row__` column moves: `moved = [c for c in rows if not is_own_column(c)]` (`:393`), the receiver's rows are overwritten with the donor's (`:403-404`).
* The 14 `clip__` columns are NOT touched. The clip table is written back as it was (`:408-410`, `arrays["clip__" + k] = clips[k]`). Probe: all 14 columns of `v9_labels_R_s0.npz` equal the union's (`R_clip_cols_equal_union`: 14 of 14 true). 10,000 receivers, 8 groups, 0 singletons, every donor has the receiver's `(n_rows, k0)` (`donor_same_n_rows_k0` true).
* So today every per-clip column is KEPT. The question is which of them should have moved.

## The consumer fact that bounds the decision (MEASURED by grep)
* The trainer path reads only four of the 14 clip columns: `sid`, `row0`, `n_rows`, `k0` (`stack/tanitad/data/v9_labels.py:104-105`). Nothing else under `stack/tanitad` or `taniteval` reads `clip__*`; the builder is the only other file that names them (`stack/scripts/build_v9_labels.py:1245`).
* Therefore moving or keeping the ten other columns cannot change what R_s0 trains on. The choice only decides whether the R release's clip table is internally consistent with its rows. The four the trainer reads must stay (they are the layout and the key).

## Classification

Rule: a column derived from the moved LABELS moves with them; a column derived from the clip's own scene, ego track, cache or clock stays. `diff` = MEASURED share of the 10,000 receivers whose own value differs from the donor's value (so the number of clips the choice would change).

| # | column | source (file:line, `build_v9_labels.py` unless named) | class | proposal | diff |
|---|---|---|---|---|---|
| 1 | `sid` | `stable_episode_id(cid)`, `:1211`; the lookup key (`v9_labels.py:105`) | clip identity | STAYS (key; moving it breaks every join) | 100 % |
| 2 | `sha12` | `sha12(cid)`, `:1211`; key of the lead / speed / tracks sidecars | clip identity | STAYS | 100 % |
| 3 | `row0` | running row offset, `:1222` (`r0`); read by the trainer `v9_labels.py:105` | layout | STAYS | 100 % |
| 4 | `n_rows` | `poses[ci].shape[0]`, `:1213`; read by the trainer; the derangement group key | cache layout | STAYS | 0 % (group key) |
| 5 | `k0` | constant 2, `:1222`; read by the trainer; group key | layout | STAYS | 0 % |
| 6 | `grid_start_s` | `clip_clock(sid, side, poses[ci])`, `:1212`, `:494-500`: sidecar or cache-pose clock | own clock (`now_s` stays) | STAYS | 99.8 % |
| 7 | `dt_s` | same call | own clock | STAYS | 99.84 % |
| 8 | `clock_src` | same call (`"sidecar" / "pose_dt" / "nominal_dt"` index, `:1223`) | own clock | STAYS | 6.89 % |
| 9 | `has_record` | `rec is not None`, `rec = recs.get(cid)` (`:1216`), `recs` = the VLM-CoT label records (`:1186-1191`); `rec` feeds the goal bits | label source (moved columns) | MOVES | 0 % (all 10k clips read 0) |
| 10 | `has_lead` | `s12 in lead_by` (`:1224`), `lead_by` = the lead label file (`:1192-1196`); it fills the moved `lead_*` columns | label source (moved columns) | MOVES | 8.44 % (844 clips) |
| 11 | `has_lc` | `s12 in lc_by` (`:1225`), `lc_by` = the lane-change file (`:1197-1203`); it fills the moved `lc_*` columns | label source (moved columns) | MOVES | 0 % (all 10k clips read 0) |
| 12 | `n_turns` | `len(G.turns)` (`:1225`), `G = clip_geo(load_log(ego), ...)` (`:1217`): turn segments of the clip's OWN ego log (`turn_table`, `:233-243`) | own ego geometry | STAYS | 79.4 % |
| 13 | `n_announced` | `sum(u.announced ...)` (`:1226`); `announced = is_turn and not suppressed` (`:229-230`); the same announced turns build the own `nav_*` columns (`:264-280`) | own ego geometry, drives own nav | STAYS | 74.23 % |
| 14 | `suppressed` | `any(u.suppressed ...)` (`:1227`); suppression read from the record's `turn_suppression` (`:1217`) but applied to the own turns that build the own nav | own nav | STAYS | 0 % (all 10k clips read 0) |

## What the proposal changes
* Moving 9, 10 and 11 changes the table for 844 clips (only `has_lead`). `has_record` and `has_lc` are 0 on every 10k clip, `has_lead` is 1 on 9,559 of them, so the move is a no-op for two and small for one.
* `n_turns` and `n_announced` differ from the donor's value on 79 % and 74 % of the receivers. They are kept because the own `nav_*` columns stay: the receiver's nav input still announces its own turns, and its supervision (the donor's `turn_*`, `goal_*`, `lat_*`, `lon_*` rows) contradicts it. That contradiction is the design of R (card 2.4). If the MM prefers a table that agrees with the rows rather than with the nav, `n_turns` flips to MOVES; the trainer cannot see the difference.

## Sidecars that the derangement does not reach (flag for the MM, MEASURED from source)
* `lead_union.npz` (keyed by `sha12`) fills `J["lead_gap"]` (`stack/tanitad/data/v8f_mini_data.py:238-240`), which feeds the batch's `lead_gap_h` (`:683`) and the constraint penalty (`stack/tanitad/models/v8f_mini.py:631-632`). R keeps it the clip's own, while the release's `lead_valid` row column (`:237`) is the donor's. It is a scene quantity used by the penalty terms, not a label of a head, so it is treated as an input that stays; the contradiction with `lead_valid` is part of R's design.
* The speed sidecar (`--speed-train`, `v8f_mini_data.py:202`) and the native tracks stay (inputs).

## Ruling needed
1. Accept the 9 / 10 / 11 MOVES and 12 / 13 / 14 STAYS (or flip `n_turns` to MOVES).
2. If any column moves, `dscale10k_merge.py derange` needs the new rule (clip-table columns `has_record`, `has_lead`, `has_lc` taken from the donor), the R md5 changes, and `verify_derange` needs the same rule. Nothing has been changed here.
3. Without a ruling, R_s0 is not minted. The R release on the pod (md5 `fbfc6e3f...`) is built under the "all kept" rule.
