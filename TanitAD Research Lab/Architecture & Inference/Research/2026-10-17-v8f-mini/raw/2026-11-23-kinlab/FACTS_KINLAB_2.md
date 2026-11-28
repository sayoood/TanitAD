# FACTS_KINLAB_2: where `--dagger-label kin_v1` is wired (scratch build on df250a00)

Evidence class: MEASURED-FROM-SOURCE of `F:/TanitAD-scratch/kinlab/tree` (df250a00 + this change). Line numbers are those of the changed files in that tree. Abbreviations: RV = `stack/tanitad/data/v8f_recovery.py`, D = `stack/tanitad/data/v8f_dagger.py`, TR = `stack/scripts/v8f_mini_train.py`, T = `stack/tests/test_v8f_kinlab.py`.

## The chain: argv -> label -> loss
| hop | where |
|---|---|
| flag value | TR:246 `--dagger-label` `choices=DR.DAGGER_LABELS` (RV:249 = `("human","speed_v1","kin_v1")`), default `human`; no new flag; FLAG_CHECKS entry unchanged (`dagger_label` -> `dagger:dagger_label`, TR:103) |
| load | TR:609-620 `load_dagger_bank`: `kin_v1` -> `DR.kin_label_bank(bank.a, bank.ci, J["clip"], store.get_kin)` -> `bank.attach_recovery(slot, label, status_counts, kin=is_kin)` (TR:614-617); `rl_ctl["label"] = a.dagger_label`, `rl_ctl["census"]` (kin census, per tick and pooled) lands in `info["dagger"]["speed_label"]` |
| bank | D:289-302 `attach_recovery(..., kin=None)` (kin flags validated: a kin flag on a record with no label is refused); D:305-306 `label_kind` -> `"kin_v1"` iff `rl["kin"]` is set; D:345-349 `gather` adds `rl_kin` ONLY for kin_v1 (the speed_v1 key set is unchanged) |
| swap | D:615-632 `swap_rows`: the one line that installs `gt` / `s_h` is unchanged (D:626-628); under kin_v1 it also returns `_rl_kin_n` |
| counters | D:773/787 `last["n_label_kin"]` (0 under human and speed_v1), D:794-795 set from `_rl_kin_n`; TR:3749-3752 per-interval / total `label_kin`, TR:3808-3809 log key `dagger_label_kin`, TR:3847 run total `dagger_label_kin`; `n_label_replaced` is as before (rows whose gt is a speed_v1-or-kin label) |
| G-DVB | `built_levers()["dagger_label"]` = `bank.label_kind` (D:726) -> `bad_dagger` (TR:647-): declared `kin_v1` vs a built speed_v1 bank is refused and the reverse; non-default at `--dagger-p 0` is refused (TR:661-665), exactly as speed_v1 |
| G-LIVE probe | TR:2891 `rl_keys` includes kin_v1; TR:2900-2904 for kin_v1 the probe records `n_label_kin` and the per-reason refusal counters `kin_status_counts` and REFUSES a probe batch with 0 kin labels |
| loss | `gt` -> `ade` / `fde` / `acc` (`traj_terms`, `accel_term`): unchanged (FACTS_KINLAB section 4). T proves a non-zero gradient difference kin_v1 vs speed_v1 and an exact 0.0 for speed_v1 vs itself |

## The labeller (RV:399-)
* Constants RV:419-427: `T_B = 1.0`, `LD_MIN = 4.0`, `LD_GAIN = 1.0`, `MERGE_LAT_M = 0.10`, `MERGE_HEAD_RAD = 1 deg`, `H_SUB = 0.05`, `T_END = 6.0`, `IDENT_TOL = 1e-6`; limits are the existing `A_LIM` 3.96, `K_LIM` 0.198, `F_LIM` 6.796 (the 0.99 margin of RV:30-32). Docstring RV:400-417 cites the brief.
* `kin_label(grid, r, (v_o, a_o, kappa_o))` RV:488: reference = `speed_label` (refused there = refused here, `speed_<status>`); identity rule RV:512; integration loop RV:526-558 (w RV:527, a RV:529, lookahead RV:532, kappa RV:540, friction RV:541, merge test RV:555); tail RV:561-570; checker RV:576.
* `kin_label_bank` RV:585 (own state = `p10f[:, -1, 0:3]`; a refused record keeps its speed_v1 label RV:621, else no slot = human); `kin_census` RV:637 (per tick and pooled; `refused_infeasible_by_term`).
* `speed_label`, `label_bank`, `label_steps`, `violations`, `human_label`, `arc_from_label` are byte-identical to df250a00 (the verbatim body test and the gold tests pin them).

## Interpretations the brief left open (MM to confirm or overrule; none is tuned on data)
1. **Merge tail.** "the remaining points are the speed_v1 points from the corresponding reference arc on" is built as: point k = P at arc `s*_m + L[k] - L[k_m]` (the speed_v1 arc INCREMENTS from the merge point, position-continuous), not the index-aligned `speed_v1` points. Consequence: the SPEED returns to the speed_v1 profile in one step at the merge. After the 1 s blend the integrated speed differs from the reference by about `0.5 (a_o - a_ref0) * 1 s`, so the step back is a speed jump the checker reads as acceleration. See the census: 220,689 of the 229,147 refused_infeasible records carry the accel term.
2. **Merge test** is evaluated at the 0.2 s output points (first point with t >= T_B, lateral distance to the human polyline < 0.10 m, |heading - tangent| < 1 deg), the tangent being the nearest segment's chord direction of P.
3. **`kappa_ref0`** (identity rule) = the turn between the reference label's first two chords over their mean length (0 if a chord is shorter than 1e-9 m); `a_ref0 = (u_l[1] - u_l[0]) / 0.2` with `u_l[0] = v_o`. Real records essentially never meet the rule (census: 0 identity of 372,791); planted records do.
4. **Reference polyline** for the pursuit is the human polyline P from r' (the whole logged remainder of the clip), not only its first 30 rows; a lookahead past P's end refuses (`refused_path_exhausted`).
5. **Integration** is the midpoint unicycle (v and heading averaged over the sub-step); `v >= 0` by clamping; the friction circle scales kappa down (a is only clipped to A_LIM, which is below F_LIM, so it is never the one reduced).
6. `kappa_o` is the model's input channel (omega / max(v, 2), capped 0.3), used as a curvature as the brief says; below 2 m/s it understates the true curvature.

## Not done / not changed
* No staging, commit, push; protected files untouched (`launch_gate.py`, `models/__init__.py`, `v8f_t1_eval.py`, anything under `E:/Projects/TanitAD/stack`). A trainer edit moves the launch-gate profile fingerprints (memory: pin edit moves profile fingerprints): the MM owns that step.
* No GPU, nothing run on the pod (two read-only pulls: shard 0 of RH bank s0, 533 MB, and `native_tracks_union.pt`, 51 MB; sha256 verified; both deleted, see DELETIONS.log).
