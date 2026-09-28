# COMMS — refcv7 NavSim suite (EvalFlyWheel NavSim stream, 2026-09-28)

## ⭐ Escalations (each also in the report headline)

1. ⛔ **refcv7's SPEED-CEILING FILTER NEVER REACHES THE EMITTED PLAN — the declared-but-not-reaching
   class of refcv6's escalation 9, surviving FIX-4.** MEASURED on the step-1,500 checkpoint
   (VALIDATION ONLY), warmup, CUDA: R7_A1 (filter ON) and R7_FILTOFF (filter OFF, a real second
   forward) emitted **bit-identical plans on 204/204 scenes**, although the ceiling masked ≥ 1
   candidate in 97 of the 126 scenes with a finite ceiling and the emitted plan lay OVER the ceiling
   in 2 (`raw/validation/step1500/_superseded_pre_A1/bridge_warmup/`; re-measured on CPU by
   `tests/test_model_seam7.py::test_KF_…`, 4/4 identical). Mechanism, read at source:
   `RefCV3Model.forward` RE-SELECTS over the fan after the decoder (E9 goal selection,
   `stack/tanitad/refs/refc_v3.py:2285-2298`): `blended = apply_seam_clamp(sel_score, goal_gate ·
   scorer)` → `rank` masked by `reach_keep` ONLY → `argmax` → `out["traj"]`; the decoder's
   ceiling-filtered pick (`refc.py:3605-3610`) is demoted to `sel_idx_base`. The additive terms (nav
   compliance, behaviour) survive inside `sel_score`; the ceiling — a MASK — is dropped.
   ⇒ SPEC_REFCV7 A2 ("the fed set-speed filters the ARGMAX"; "every refcv7 evaluation reports filter
   ON and OFF") is BUILT in the decoder and INERT on the plan the car would drive. G-DVB checks the
   filter is built; G-LIVE checks the mask is active in an eval step; **neither asserts that the
   EMITTED plan obeys it** — that is the missing guard. This applies to EVERY refcv7 evaluation, the
   PhysicalAI four-family battery included (its filter ON/OFF readings will be identical by the same
   code path). **Owner: Master Mind / Arch.** The zero-training fix is to apply the ceiling mask at the
   E9 argmax; this suite prices it with the DIAGNOSTIC arm `R7_CEILDECL_d` (SPEC amendment A1) and
   takes no decision — evaluating or shipping a model other than the one trained is the PI's call.
2. ⚠️ **The shared dev box is RAM-bound this morning.** At 05:50 Berlin two other jobs held ~15 GB
   RSS (the battery's `g0_refcv7.py` 9.9 GB; a Research Lab job 5.1 GB), available RAM fell to
   1.8 GB and E1's RAM guard (correctly) aborted BOTH of this stream's harness scorers (navhard CV at
   ~74/450 stage-1, navtest STOP). Both are RAM-gated and retried automatically; they cost only time.
   At the milestone the same squeeze would stretch the CPU scorers — the suite gates every scorer on
   ≥ 6 GB free and retries a guard abort up to 8 times.
3. ⚠️ **A MSYS-path defect in the refcv6 suite's bash queue (`score_queue6.sh`, ported here as
   `score_queue7.sh`)**: `P="$(cd … && pwd)"` is a `/d/…` path, and every JSON read the script does
   through `python -c "open(r'$P/…')"` fails and reads ABSENT — the refcv6 queue's
   `status=NO_COUNTS` log lines are this defect, and its PASS_ALREADY skip never fires (a re-invoked
   queue re-scores a passed arm). Harmless where a queue runs once; it would have re-scored a passed
   12,146-token control six times in this stream's first harness chain (caught, killed by PID, and
   replaced by `code/harness7.py`, which reads artifacts in Python). The milestone runner does not
   use the bash queue. **Owner: whoever maintains the refcv6 suite** (fix: `pwd -W`, or read in
   Python).

4. ⛔ **THIS STREAM HELD THE DEV-BOX GPU LOCK STALE FOR 42 MINUTES (06:59–07:41 Berlin) — my error,
   now fixed.** I killed a waiting validation runner by explicit PID (`taskkill /F`) at ~06:59 without
   re-reading the lock; it had acquired the lock seconds earlier (04:58:54Z) and a forced kill skips
   its `finally: release`. The lock (`refcv7-navsim-warmup`, pid 34848, dead) blocked the battery
   and the Research Lab until I found it at 07:41 and released it with its own token; the battery
   (`refcv6-38k-perc-dump`) took the card at 07:41:13. **Fix:** `gpu_lock.reap_own_stale()` — before
   every acquisition, a lock whose job is `refcv7-navsim*` AND whose pid is dead is released; a
   foreign lock is NEVER touched, dead pid or not (`tests/test_gpu_lock7.py`, 6 cases). Rule for this
   stream: after killing any process of this suite, read the lock.

## Decisions made in this stream (reversible)

| # | decision | why |
|---|---|---|
| D1 | the NavSim-side input functions are the refcv6 suite's, IMPORTED from the clean tree (`boot7.py`), under refcv7 arm specs mapped to the R6 arm with the identical declared set / nav / max-speed source (`R6_TEMPLATE`, literal test) | "import, never copy"; the functions and their literal tests are already validated on refcv6 (KL bit-exact) |
| D2 | the model is built only by `stack/tanitad/eval/refcv7_loader.build_model` (strict, G-DVB, stamps), and the forward is fed by `forward_kwargs7` = compute_losses_v3's keyword set | the brief; KT7 proves bit-identity with the trainer's own call on a real eval window |
| D3 | the 0.25 m lift for a NavSim rig is `build_lift_geometry` with every parameter READ OFF `model._lift_bank_hires` | never retype a model constant; KL-lift + mutation arms |
| D4 | the prior is scored as its own model-free arm (PRIOR_ha0p) | what the residual adds is THE NEW-1 question; it costs no forward |
| D5 | SPEC amendment A1 (before any score): R7_FILTOFF_d withdrawn, KF redefined, R7_CEILDECL_d added as a diagnostic | escalation 1 |
| D6 | the validation checkpoint is the battery stream's `ckpt_1500.pt` (md5 `c35966f7…`, verified); this stream's own duplicate pull of the same file (same md5 before/after on Thor) was deleted | the brief: reuse the battery's pull |
| D7 | the milestone waiter waits for the battery's lock to be SEEN held and then quiet for 30 min (fallback: 4 h without a battery lock) | the brief: the battery goes first; a gap between battery stages is not a release |
| D8 | the runner releases the GPU lock between splits and never holds it during CPU scoring | the battery may need the card between NavSim splits |
| D9 | on CUDA the exact-duplicate dedup is DROPPED when KD fails, CUDA itself is refused only when K0 fails (the refcv6 suite's D15 rule); the runner first refused CUDA on a KD miss and was corrected | MEASURED 2026-09-28 on `ckpt_1500.pt`, CUDA bf16: dedup vs native = selections 4/4 identical, max \|Δ\| **1.196 mm** > the registered 1 mm (`raw/controls/KD_exact_dedup_cuda.json`); K0 PASS. The native path costs 10 backbone passes per static-window scene: **0.35–0.42 s/scene** on the card (vs 0.24–0.27 with the dedup) ⇒ a milestone's bridges ≈ 4.4 GPU-hours (navtest 2 × 12,146 + navhard 2 × 5,912 + warmup 7 × 220 + 3 × 200) |
| D10 | the step-5,000 waiter was RE-ARMED at 06:35 Berlin (the 05:52 instance killed by PID while it was still waiting for the checkpoint): its battery-release condition now reads the LOCK only (a lock-less python on the card is logged, not waited on) and it gains a 10 h timeout | a lock-less `uv` python 3.11 GPU process from another session was on the card at 06:34; under the first rule it could have held the waiter forever |
| D11 | the waiter was RE-ARMED again at 06:46 Berlin (Windows PID 33704): the battery is identified by a lock job naming the milestone step (`refcv7-…-5000`), the Research Lab's lock on the SAME file is logged (`LOCK_SEEN_OTHER`) and waited out, foreign lock fields are tolerated, and a lock that is not ours is never removed (`gpu_lock.release` deletes only a lock carrying our token) | Master Mind 06:45 Berlin: the Research Lab now takes the same lock (GPU runs ~08:00-12:30, a REFe snapshot eval ≈ 13:25 for 50-70 min); a Research Lab release must never be read as the battery's |
| D12 | the waiter was re-armed a third time at 07:42 Berlin (Windows PID 25012): the battery tag match also accepts `5k` / `5,000` in the lock's job name | the battery's job names seen so far (`refcv7-g0-1500`, `refcv7-smoke-1500`, `refcv6-38k-perc-dump`) show the step is not always spelled the same way; a missed match only delays NavSim to the 4 h grace fallback, never lets it jump the battery while the lock is held |
