# HANDOFF RL-S1 part 1 (2026-10-12) -- priorities 2 and 3 are NOT built; read this, then continue

Ended at the ~200k-context rule (PI token budget), not because of a blocker. Priority 1 (reward) and 4 (prereg draft) are banked.
D3 is DECIDED (R2 in, Master Mind 5e4a998): K and L are built and tested to the core standard; the `refe_rls1` profile must accept the R2 argv with them ON.

## Done (all under `PKG = TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/`)
* `refe/rl_reward.py` -- pure numpy; `reward(tg, cfg, d=, k=, lane=)`; `RewardCfg` (lam_d .15, w_prog .5, guard ON, K/L flags OFF). Missing input for an ON term RAISES.
* `refe/test_rl_reward.py` -- `ZZTEST_PASS 31`: hand-computed values for PDMS, D, g2, K, L (+ the section 3.3 guards), 29 source mutations all RED. Run:
  `PYTHONIOENCODING=utf-8 C:/Users/Admin/venvs/tanitad/Scripts/python.exe refe/test_rl_reward.py`
* `eval/PREREG_RL_S1.md` -- DRAFT; OPEN-BASE has both variants side by side; D3 marked decided.
* Baseline to preserve: `refe/test_launch_gate.py --static` = **76** (`--no-banked`) / **90** (with the banked generic gate). Re-run before and after every gate edit.

* **P4 ruling (Master Mind, 2026-10-12, after this handoff was written):** the progress guard is capped by `rl_reward.guard_penalty` so it can never push a
  safe proposal to or below an unsafe one it beat (the uncapped `w_prog = 0.5` did). `w_prog` stays 0.5. `reward()` now also returns `out["guard"]`.

## Findings that change the build (each read from source, file:line at the tip)
1. **Selection-rule conflict.** Review 5.2 says `A_i = NC*DAC*DDC*(5EP+5TTC+4C)/14` (train.py:611-614) = rule `v2_shape`. u400 deploys **`navsim_v1_lane`**
   (`planner.py:552-557`, `RESULT_SFT4.md` l.23: NC*DAC*(5EP+5TTC+2C)/12 * p6); P2 will deploy `navsim_v1`. The EPO policy must use the BASE's rule:
   reuse `scorer_finetune.sel_score` (it already does lane-head x p6). Do not use `_pdm_aggregate`.
2. **`scorer_finetune.py` cannot start from u400.** It builds `REFeConfig.for_backbone(a.backbone)` (6 outputs) and strict-loads (L161-164) without
   `ckpt_io.config_for_checkpoint(cfg, a.ckpt)`; and `--lane-head` (L177-188) copies `old.weight` (7 rows) into `new.weight[:6]`. Fix additively: size the cfg
   from the checkpoint, skip the head-extension when the head already has 7 rows. Default-preserving (meta absent -> unchanged).
3. **The existing `--b-mode expreward` already optimises E_pi[true v1 score]** (SFT-3/SFT-4 recipe, `expected_reward_loss`, L68-75, tau_s .1). The new EPO differs by:
   reward R (rl_reward, not agg_v1(labels)), the KL anchor to `pi_SFT` (`base` ScorerCopy already gives sx0 each micro-batch), the 0->1 ramp over 200 updates.
4. **Arms A/B of `scorer_finetune` are paired by construction**: A = BCE only (= C_s), B = BCE + EPO (= R_s). EPO goes on B only; `--epo-lam 0` must make B == A (K1).
5. **Gate coupling.** `DVB_KINDS` is pinned equal to train.py's parser dests by `test_launch_gate.py` S5 (`set(C.DVB_KINDS)==dests`, `len(dests)==55`): **do not add scorer flags
   to it.** Add a separate `DVB_KINDS_SCORER` + `check_dvb_scorer` (unknown flag FAILS, stale entry FAILS, same two arms `unknown_flag`/`stale_registry`). `profile_refusals`
   hard-codes `import train as T` / `train.py` (launch_gate.py): give profiles `parser_module` + `trainer_src` keys, default = today's. `scorer_finetune` has no
   `build_parser()` (parser inline, L106-145): extract it VERBATIM. `instrumented_run`/G-LIVE/G-EVAL/G-CKPT are train.py-only: for `refe_rls1` consume
   `scorer_finetune --preflight`'s JSON like `consume_p2_gate` consumes p2_gate.py (parsed, never grepped; file sha256 bound into the token).
6. **Batch has no row key** (`TargetBank.__getitem__` returns `ci` = per-proposal ckpt_step; the row index only when `online`). To look up D/K/L side files per sample,
   wrap `Subset` so it also yields the row index, ONLY when `--epo` is on (OFF path unchanged => bit-identical).
7. Bash heredocs in this environment eat a backslash level: write with Write/Edit.

## Next steps, in order (bank each)
1. `scorer_finetune.py`, additive, default OFF: `build_parser()` extraction; item 2 fix; flags `--epo`, `--epo-temp .1`, `--epo-beta .1`, `--epo-ramp 200`, `--epo-lam 1.0`,
   `--epo-lam-d .15`, `--epo-w-prog .5`, `--epo-k` + `--epo-lam-k .05`, `--epo-l` + `--epo-lam-l .05`, `--epo-terms-train/-heldout <dir>` (D/K/L side files,
   schema `{"kind":"rl_terms","key":[log,token,step,rank],"ckpt_step":c,"d":[64],"k":[64],"lc_executed":[64],"label_fires":bool,"dither":[64]}` -- part-2 labeller writes it).
   `epo_loss(sxB, sx_ref, R, rule_fn, T, beta, lam_pg)` module-level, torch, reward from `rl_reward` as a detached tensor. Refuse `--epo` without `--epo-terms-train` unless `--epo-lam-d 0` is NAMED.
2. Preflight gate `G11_epo_live_and_identity`: EPO-only backward non-zero on each scorer module; `--epo` off / lam 0 reads EXACTLY 0.0; `REFE_RL_MUTATE_EPO_DETACH=1` -> RED.
3. `refe/test_rl_epo.py` (CPU, tiny scorer modules): OFF path == the verbatim baseline closures (embed them; blob `scorer_finetune.py` = 92b52dac) on loss AND on a multi-step param hash across b-modes
   listnet/compw/expreward +- lane head; ON: non-zero grad, OFF control exactly 0.0; mutations (delete the EPO line; `--epo` default flipped; detach) RED.
   Dynamic reachability: AST-extract `losses6` from `main()` and exec the SOURCE text (so a mutation of the real caller is seen).
4. `refe_rls1` profile + `DVB_KINDS_SCORER` + tests in `test_launch_gate.py` (static): R0/R1 argv clean, R2 argv (K/L ON) clean, unknown flag FAIL, missing required flag refused, count >= 76/90.

## Part 2 (not started, not in any brief yet): on-policy proposal dumper (~51,000 scenes), extended labeller (D, K, L -> `rl_terms` side files; G-REWARD-ID `g>0 iff DAC=0` on 10,000 labelled proposals),
RL-train / RL-val log-disjoint partition, pod chain script, RL-val lookup evaluator. Also OPEN: base checkpoint (PREREG_RL_S1 OPEN-BASE).
