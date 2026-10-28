# NOTES: the SPEEDSUB-T1 instrument build (PREREG_V8F_SPEEDSUB_T1, tip 842ec2ac)

Sections are appended as each deliverable is banked. Evidence class: MEASURED (artifact paths given) unless marked.

## G-VIOL: does the T1-kin tool carry a per-window rule-violation measure?

**Answer: NO, the tool and its outputs carry none. It can be computed from the `cl` dump with the T0 eval's own function, at a cost of about 2 s per dump.**

Probe locations (MEASURED, 2026-10-28; all searches over the pinned tool tree `E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a`, byte-identical at the tip):
1. `stack/scripts/v8f_t1_eval.py`: `grep -n viol` returns 0 lines (control: `grep -n "HEAD_KEYS"` returns the heads definition at `:277`). The only per-window outputs are the dump arms (`write_dumps`, `:446-455`: `g, ws, eid, clip_index, v0` plus the arms `cl, ol, fl, ha0`) and the tick rows (`TickRecorder.FIELDS`, `:337`).
2. `dump_DAGOACC_s0_m10_causal_heads/ep*.npz` keys (read from the banked CTRL s0 run): `cl, clip_index, eid, fl, g, ha0, ol, v0, ws`. The tick file `ticks_DAGOACC_s0_m10_causal_heads.npz` keys: `tau, P, plan, gt, gt_ok, v0, pose, fired, win, ci, t0, tick, j, age_t, age_s, s_ckpt*, t_goal_prob, t_lat_logits, t_lon_logits, t_cons_pred, t_pose, pose_w0, clip, ...`. None is a violation.
3. `t1v8f_DAGOACC_s0_m10_causal_heads.json`: `grep -c -i "viol\|consisten"` returns 0.
4. `taniteval/tools/t1_eval.py::analyze` (`:336`): `grep -n viol` returns 0 lines; it scores the four families, S-curve, lag and the paired bootstrap, not rule violations.

Where the measure lives: the T0 eval, `stack/scripts/v8f_mini_eval.py:149-157` (`evaluate`): `gtc = gt_constraints(X)` (`:98-100`, `C.constraints_from_v9(D.v9_label_dict(X), X["limit_ms"], X["lead_gap_h"], X["s_h"])`), `cons = C.consistency(pred, gtc, ...)`, `viol_any = cons["_per_window"]["any_violated"]`; the rules are `taniteval/taniteval/v8f_consistency.py:95` `violations(plan_xy, c)` (SPEED / STOP / TURN / LEAD, `RULE_NAMES` `:58`). `X` is `D.eval_context(grid, eval141_kin.npz, J, kin_mode="yaw")` (`v8f_mini_data.py:917`), from the pinned `raw/eval141_kin.npz` and `raw/join_eval141.npz`.

Feasibility, MEASURED by `build/probes/probe_viol_from_dumps.py` on the banked DAG-OACC CTRL s0 dump (`build/probes/probe_viol_ctrl_s0.json`): the dump windows align with `eval_context` (`ws == X["t"]`, 2,399 rows; GT max |diff| 7.6e-6), and `C.violations` on the four arms takes **0.02 s** (whole probe 1.6 s, `eval_context` 0.4 s). Readings: `viol_any` rate `cl` (the executed path) 0.3618, `ol` (the tick-0 plan, the T0-comparable plan) 0.2288, `fl` (P10 floor) 0.7999, `g` (the human's own future, the control that must violate little) 0.0792.

What the reader carries (built, see `speedsub_t1_read.py`): a row `VIOL-ANY` (`viol_any` of the `cl` arm, per window 0/1, paired `egoc - identity` per run through `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0) and, beside it, `VIOL-ANY-OL` (the tick-0 plan, comparable with the T0 finding). Guard G-VIOL, as specified by the Master Mind: fires if `VIOL-ANY` is separated-worse (CI lower end > 0, delta > 0) at >= 3 of the 4 runs. It sits in the guard set next to G-WORSE / G-NEW / G-ADE and a firing is HARMS. ⚠️ G-VIOL is coded ahead of its amendment: it is switched on by the module constant `AMENDMENT_1_GVIOL` and the reader's PREREG pin must be moved to the amended text when Amendment 1 lands. Tests and mutants for the guard are in `test_speedsub_t1_read.py`.

## 1. The read-out and its controls (deliverable 1, DONE)

`mlp_ego_c` = the SPEEDREAD read-out (A1 instrument, same MLP and training recipe), with ONE change: the `limit` block of the ego input (columns 13:17 and 17) is rebuilt from the causal bucket `causal_bucket(v_now, j, CAUSAL_STEPS=20)` of the window's NOW row. Fitted on the 306,566 train windows of the 4,512 train clips; evaluated nowhere in the fit (the TRAIN hold-out decides nothing). Script `stack/scripts/v8f_speedsub_t1_fit.py`; outputs in `F:/TanitAD-scratch/speedsub_t1/fit/`.

| artifact | sha256 |
|---|---|
| `readout_c.pt` (key `mlp_ego_c`, 22-wide) | `799f65ed1c597063bf98c6dd53138f63d219a8ecb90d5b9fcae2564d459e3fcb` |
| `fit_c.json` | `9aaa5edde167b767d1bbf586f2e2909f1f21b40fc17e53047123101d60fd499a` |

Controls, MEASURED (`fit_c.json`; outcome FITTED, `controls_pass` true):
- **C-FIT** max |delta hold-out prediction| between the fit and its re-evaluation = 9.412e-6 m/s (bar 1e-5; thin margin, reproducible on the rerun), best epoch 8 = 8, input mean/sd diff 0.0.
- **C-CAUSAL** 0 mismatches between my causal block and the T1 tool's own `build_item --limit causal` (a_t = 0 on all 2,399 windows; a_t = 1 and 2 on every 4th, 600 each). Eval bucket counts [728, 871, 684, 116].
- **C-COVER** 306,566 / 306,566 windows have a finite past (zero without); agreement of the causal bucket with the logged bucket 76.7 % (the two differ by construction: the logged one is the ego's realised future).
- **C-DISJ** 141 eval clips, empty intersection with the train clips. **C-PARITY** 4,512 clips / 306,566 windows, window-list sha equals the extract's; excluded-set sha f6a1dee0.... The cache speed `v` and `native_tracks.pt` agree to max diff 0.0 over 450,950 rows.
- **C-INSTR** the fit functions equal A1's by source text; `ReadOut` equal method by method, the tip may add exactly `predict_ego_batch` (the first run was REFUSED by my own too-strict whole-class comparison; the refused record is kept as `fit_c.REFUSED.first_run_C-INSTR_too_strict.json`; the rerun reproduced identical hold-out numbers).
- TRAIN hold-out RMSE (decides nothing): `mlp_ego_c` 0.2793 SD, versus 0.2475 for the logged-limit read-out and 0.3085 for p10f + frame_ok only.

## 2. The wrapper, seam B (deliverable 2, DONE)

`stack/scripts/v8f_t1_speedsub.py` patches `V8FMini.forward_infer` / `_code_pred` / `run_o` at class level around the PINNED tool (`v8f_t1_eval.py` sha 61e140ed..., not edited) and calls the tool's own `main()`. One forward pass: O reads the substituted code in the same call (equal to `forward_diag`'s two-pass semantics, pinned bit for bit by a test). Modes: identity, egoc, egoshufc, identity_l, egol. The P10 floor arm reaches none of the three patches (test). A record `speedsub_<tag>.npz` is written beside the tool's ticks and checked against the tool's own tick rows (rc 3 on misalignment).
- Tests `stack/tests/test_v8f_t1_speedsub.py`: **33 pass** (~41 s), on a synthetic world with a tiny model, driving the real tool `main()` through the wrapper. Includes the control with power (`test_the_non_speed_check_can_read_a_non_zero_value`: the K-NONSPEED measure reads a non-zero value when a non-speed dim is perturbed) and the identity-equals-unpatched byte compare.
- Mutation pass `stack/tests/mutation_speedsub_wrapper.py`: **29 / 29 mutants KILLED, no survivor** (`mutation_wrapper.json`).
- The four registered checkpoints all have `o_interface = full` (`probes/probe_ckpt_o_interface.json`): the wrapper's refusal on `cut` does not fire, and the substitution can reach O.
- NOT done, by instruction: no rollout, no smoke, no forward on real windows with a real checkpoint. The reach on REAL windows is therefore proven by the synthetic world and by K-REACH at the smoke, not yet by a real forward.

## 3. The pin step and the argv renderer (support for deliverables 2 and 4, DONE)

`stack/scripts/v8f_speedsub_t1_pin.py`: `pin` (checkpoints against the PREREG's printed prefixes, read-out against `fit_c.json`, tool, wrapper, instrument, reader; an existing pins.json is never overwritten by different pins), `render-argv` (the banked argv of a run with only tag / out-dir / dump-ticks, and --limit / --max-windows where registered, changed; every other token is checked against the registered values), `check-done`, `write-done` (a done record only when every file of the rollout is there and the checkpoint still hashes to its pre-rollout sha), `get`. Tests `stack/tests/test_v8f_speedsub_t1_pin.py`: **52 pass**; mutation `mutation_speedsub_pin.py`: **25 / 25 KILLED**.

## 4. The reader (deliverable 3, DONE)

`read/speedsub_t1_read.py` reads the four runs (CTRL s0/s1, REP s0/s1) x (identity, egoc, egoshufc) through the DAG-ORC `Rig` unchanged (`E:/Projects/TanitAD-artifacts/dagorc_rebase/read/dag_orc_read.py`: paired episode-cluster bootstrap B = 2,000, seed 0; Read-A rows; four families) and applies PREREG sections 5-7: the hit rule (CI upper end < 0 and d <= -F/2 = -0.06035, inclusive on the point), the outcome table in the PREREG order (REFUSED, HARMS, PASS, PARTIAL, NULL; NULL-ONE / NULL-FLAT descriptive), guards G-WORSE / G-NEW / G-ADE / G-VIOL, Read A on PASS (D1 <= -0.2415, D2, D3, crawl rows), and the floor rule 6.2 (beside, never changes the outcome). Controls: K-SHA, K-ALIGN, K-WIN, K-FLOOR, K-ID (1e-4), K-ZERO (exactly 0.0), K-NONSPEED (exactly 0.0), K-REC (1e-5), K-REACH (99 % / 95 %), K-SHUF (registered permutation sha 64785f8a...), K-REG (the deliberate regression, separated-worse at all four runs), plus the heads pipeline's own controls. Modes: `--static-check`, `--dry-run`, `--preflight`, `--smoke-check`, `--make-manifest`, `--manifest ...` (the read), `--beside`.
- Tests `read/test_speedsub_t1_read.py`: **73 pass** (~12 s): every outcome row, the boundary on both sides, every guard, Read A, the floor rule, every control with a breaker, the light layer (preflight / smoke / manifest / CLI), and `main_read` END TO END on a FAKE heavy layer (rows set per contrast and run, so the wiring from table to guard to outcome is exercised, not just the pure functions).
- Mutation `read/mutation_speedsub_reader.py`: **110 / 110 KILLED** on the final text (log `F:/TanitAD-scratch/speedsub_t1/logs/mut_reader.log`). One further mutant (dropping the fixed-point clause inside `k_shuf`) is EQUIVALENT and was removed from the list: `k_shuf` recomputes the registered permutation, whose 0 fixed points the sha pin fixes, so the clause cannot change the result; the rule itself is tested through `perm_fixed_ok` (boundaries 3 of 100 and 23 of 2,399).
- **Defect found by the end-to-end test** (the reachability class): `main_read` created a phantom table key `"viol"` that nothing filled, and the final `res["tables"]` loop raised `KeyError` AFTER the outcome was computed. Every unit test passed. A real read would have computed PASS/NULL and then crashed before writing the JSON. Fixed (the key is gone); pinned by `test_e2e_every_outcome_through_main_read`.
- **Heavy-layer dry run on REAL data** (`--dry-run`, 156 s, no rollout of this experiment): `identity - P10` through the same `Heavy.tables` path on the banked DAG-OACC CTRL s0 and s1 dumps equals the banked `DAGOACC_READ.json` Read-A table on **38 of 38 rows with max abs difference 0.0** (delta, lo, hi and the separated flag); `identity - identity` reads exactly 0.0 on both seeds (K-ZERO). Output `evidence/read_dry.json`.
- Real-data reading of the rule-violation measure on the banked CTRL runs: `viol_any` of `cl` 0.3618 (s0) / 0.3760 (s1); `ol` 0.2288 / 0.2201; P10 floor 0.7999 / 0.7999; `fl - cl` paired delta +0.4381 [+0.3953, +0.4821] and +0.4239 [+0.3792, +0.4706]. The row separates a large known difference, so it is not blind.

## 5. The chain (deliverable 4, DONE; dry run only)

`run_speedsub_t1.sh` (+ `run_speedsub_t1.cmd` for a detached start) runs: fit (reused when `fit_c.json` says FITTED) -> pin -> static (the PREREG text is the pinned one) -> smoke (CTRL s0 identity + egoc, `--max-windows 200`, then the reader `--smoke-check`) -> ctrl0 -> preflight (the reader `--preflight` on CTRL s0: STOP on any failed control) -> ctrl1 -> rep0 -> rep1 -> read -> beside (logged lane, CTRL only, 4 rollouts, then `--beside`). One rollout at a time, OMP/MKL 8, CPU, a RAM guard (>= 4 GB free) and a disk guard (>= 5 GB) before every heavy step, a lock per output dir, markers `ZZSST1-STAGE-DONE` / `ZZSST1-STOP` / `ZZSST1-CHAIN-DONE`. Resume: a rollout whose done record matches tag + checkpoint sha (before and after) + the pinned tool digest is skipped; an incomplete directory is moved aside (`.incomplete.<stamp>`), never deleted. Outputs go to `F:/TanitAD-scratch/speedsub_t1/out/` (`SST1_OUT` overrides).
- `--dry-run` (MEASURED, `evidence/chain_dry_run.txt`, 33 s): validates the inputs, pins into a fresh `out_dry/<stamp>/`, renders the argv of all **18** planned rollouts (2 smoke + 12 decision + 4 beside) and parses each with BOTH the wrapper parser and the pinned tool parser (`chain_argcheck.py`, one process): `ARGCHECK 18/18 ok`, which also hashes each checkpoint and read-out against the pins; and shows that the reader REFUSES an empty manifest (rc 4, no result written).
- `chain_selftest.sh` (MEASURED, `evidence/chain_selftest.txt`): the chain control flow run for real with a STUB wrapper that rolls out nothing: **14 / 14** (stops at the smoke check when the records are bad, skip on resume, move-aside of an incomplete directory, a failed rollout stops the chain and writes no done record, a held lock stops a second chain, tampered pins stop the pin stage).
- Estimated wall (INHERITED from the banked rollouts, 305 s CTRL, 144-286 s REP at OMP/MKL 8): 18 rollouts about 1.5 h on the dev box, plus the reader (about 3 min).

## 6. What is NOT proven

1. **No forward on a real window with a real checkpoint has been run** (instruction: no smoke, no rollout). The reach of the substitution on REAL data is proven by the synthetic world (the plan differs under egoc; the floor and the tick-0 heads are unchanged; non-speed dims read exactly 0.0, with a control that can read non-zero) and by the four checkpoints having `o_interface = full`. It is proven on real data only when the smoke `--smoke-check` passes K-REACH / K-NONSPEED / K-REC.
2. The T1 tool is pinned by sha (61e140ed...) and the wrapper calls its own `main()`; its behaviour was not re-verified.
3. **S-POS paired across arms is NOT COMPUTED** (PREREG section 7) and the **anti-echo block is NOT BUILT**; the reader prints both per family as NOT COMPUTED / NOT BUILT.
4. The TRAIN hold-out RMSE of the read-out (0.2793 SD) decides nothing; whether `mlp_ego_c` moves ADE6 is the experiment.
5. The `--beside` logged lane reads the OPEN channel (the logged speed bucket is the ego own realised future max speed, D-SPEEDBAND-EGO-ECHO-1); it bridges to SPEEDSUB and is never a capability row.
6. The reader heavy layer was exercised on the banked DAG-OACC dumps (`--dry-run`) and end to end on a fake heavy layer; its first run on this experiment real rollouts is the decisive read.

## 7. PREREG amendments to settle BEFORE the first rollout

1. **G-VIOL is coded ahead of its amendment.** `AMENDMENT_1_GVIOL = True` and `PREREG_SHA256` (LF-normalised `5edf6bc0...` = the text at 842ec2ac) are module constants. When Amendment 1 lands, the reader pin must move to the amended text (`--prereg-sha` accepts it) and the list of sentences the reader checks must include the G-VIOL sentence. The reader refuses a PREREG text that is not the pinned one.
2. **"The tick frame" (section 2) is read as the window NOW row `j`** (the tool `st.j`), independent of the held age `a_t`; the causal bucket is `causal_bucket(v[j], j, 20)`. C-CAUSAL pins that this equals the tool own `--limit causal` item at a_t = 0, 1, 2. If the PREREG means the held-plan row, say so before the rollout.
3. **G-NEW has two readings** in the PREREG text. The coded one is the stricter (a row separated-worse for `egoc - P10` at BOTH CTRL seeds AND not separated-worse for `identity - P10` at both); the other is returned beside it. Name the intended one.
4. `--perm-n` exists in the wrapper for synthetic tests only (default 2,399 = the registered length); it can never pass K-SHUF.
5. **The C-FIT margin is thin**: 9.412e-6 m/s against the 1e-5 bar (reproducible on the rerun; a float32 summation-order effect). If the bar is meant to be exact, state the tolerance class.
6. The registered smoke is `--max-windows 200` on CTRL s0 identity + egoc; the controls `--smoke-check` applies are K-NONSPEED, K-REC, K-REACH (egoc only, 99 %), K-SHA, K-ALIGN. The PREREG should name the same set, otherwise the smoke criterion is mine and not registered.
7. S-POS across arms (NOT COMPUTED) and the anti-echo block (NOT BUILT) are stated in section 7 of the PREREG; the reader reports them as such.
