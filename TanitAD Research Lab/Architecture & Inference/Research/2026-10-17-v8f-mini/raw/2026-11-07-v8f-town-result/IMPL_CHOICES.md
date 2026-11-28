# IMPL_CHOICES_TOWN: implementation choices made BEFORE any number of PREREG_V8F_TOWN exists (agent, dev box)
Card: `../PREREG_V8F_TOWN.md`, frozen prefix 4,676 bytes sha256 707e29f3... (verified before writing this file). Nothing here edits the card.

## 1. The lon class order, slowest to fastest (taken from the label definition, source quoted)
Class index = LON9 index (`stack/tanitad/data/v9_labels.py:30`, `v8f_mini_data.py:57`): 0 HOLD, 1 CREEP, 2 STOP, 3 FOLLOW, 4 DECELERATE, 5 ACCELERATE, 6 KEEP.
The label rule (`stack/scripts/build_v9_labels.py:932-958`, literals `:443` v_stop 0.5, v_creep 2.0, dv 1.5 m/s) assigns each class a TARGET SPEED `lon_v_target_ms`:
HOLD: band max speed <= 0.5 (target 0.0, at rest the whole band); STOP: speed reaches <= 0.5 inside the band from an entry speed > 0.5 (target 0.0); CREEP: band max in (0.5, 2.0] (target = band max);
DECELERATE: first move is a drop >= 1.5 below the entry speed (target = the extremum); FOLLOW and KEEP: band mean (about the entry speed; FOLLOW is lead-limited, KEEP is free); ACCELERATE: first move is a rise >= 1.5 (target = the extremum).
Rule applied: absolute-target classes first by (target speed, then band speed): HOLD < STOP < CREEP; then by target relative to entry speed: DECELERATE < FOLLOW < KEEP < ACCELERATE.
**Order slowest to fastest: HOLD(0), STOP(2), CREEP(1), DECELERATE(4), FOLLOW(3), KEEP(6), ACCELERATE(5). RANK[class index] = [0, 2, 1, 4, 3, 6, 5].** E[idx] = sum_k softmax_k * RANK[k].
Beside, deciding nothing: FOLLOW/KEEP swapped; and a 3-group read (slow = HOLD, STOP, CREEP, DECELERATE; middle = FOLLOW, KEEP; fast = ACCELERATE).

## 2. Statistics (all per seed; episode = clip index; B 2,000, seed 0, percentile 95 % via `taniteval/ci.py` `episode_index` + `_draws`, the program's resampler)
* Replan = one (window, tick) row; 3 ticks per window at m = 10 (j = t0, t0+10, t0+20); 2,399 x 3 = 7,197 rows. Tick 0 has own == logged (control 2) and |dv| = 0, so the slope never uses it.
* d = E_own[RANK] - E_logged[RANK] (T-lon softmax). dv = v_logged - v_own at the replan time j, `p10f[-1, 0]` of the batch the model read (own: executed; twin: logged). Beside: dv at T's frame j - age_t.
* FEEDBACK slope = OLS WITH intercept of d on dv over rows |dv| >= 0.5 m/s, refitted on every bootstrap draw. Beside: through-origin slope.
* T-feedback "below 0 and separated": point < 0 and the 95 % upper bound < 0. "Not separated" = the interval contains 0.
* Quality population (primary) = ALL labelled replans of all 3 ticks, labelled = `has` and cls >= 0 at the T frame (j - age_t); GT = `lab_t` of `ds.item` (the training label, row off + j - age_t). Beside: ticks 1-2 only.
  Macro-F1 = `taniteval.v8f_consistency.macro_f1` (classes present among the labels); own - logged by `paired_episode_cluster_bootstrap` on codes true*(n+1)+pred. "Separated-worse" = delta < 0 and upper bound < 0.
* Constraint RMSE: headline = pooled RMSE (m/s) over the SPEED kind cells (lon_v_target_ms, v_a_ms, v_end_ms, v_lo_ms, v_hi_ms; defined GT cells only); per-cell RMSE of all 26 cells beside. Own - logged paired.
* Share beside: P(rank[argmax lon] < rank[GT lon]) on labelled rows, own vs logged.

## 3. The twin (what is swapped, what is kept)
The twin batch = the own batch with ONLY these keys replaced by the LOGGED clip's values at the same time index j (rows <= j of `base_log`, via `V8FMiniWindows.item`): `p10f`, `base`, `pose_now`, `pose_t`, `pose_s`, and `limit`/`limit_ok` (the causal speed bucket recomputed from the logged speeds, the tool's own `causal_bucket`).
KEPT identical (same tensors): `feats`, `frame_ok` (the reindexed frames), `nav` (the dead-reckoned route countdown), `a_t`, `a_s`. The twin is one extra `forward_infer` call after the own call and before the commit; it writes nothing into any state.

## 4. Mechanism: wrapper, not a copy
The tool of record is loaded by file path after its sha256 (0417c838...) holds, as in `../t1_blind/t1blind_run.py`; a file copy would move the tool's `REPO` root and drop the sha pin. `TickRecorder` is subclassed (its tick file unchanged bit for bit; the twin rows go to a separate `twin_<tag>.npz`).

## 5. Regression arm
`V8FMini.run_t` is wrapped: for the T call only, `e_ego.weight` and `e_ego.bias` are set to 0 and restored after (S and O call `e_ego` outside `run_t`, `v8f_mini.py:708` and `:735`; T's own use is `:719`). Other ego routes into T stay open and are disclosed: S's checkpoints (`ck_for_t`, `:778`), the `limit` bucket (`e_lim`, `:720`), `nav`.
