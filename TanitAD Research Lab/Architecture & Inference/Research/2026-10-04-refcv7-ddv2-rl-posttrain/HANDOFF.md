# HANDOFF — WP-RL refcv7 DDv2-RL, after A0 `rl_s0` (2026-10-11)

**Read first:**
* `RESULT_RL_DIAG.md` — the diagnosis.
* `SPEC_RL_A1.md` — the fix, pre-registered, sha256 `e08d6f28…b50f18`.
* `SPEC_RL.md` — the exit bars, sha256 `1ba17160…`, unchanged.

## State

* **A0 `rl_s0` yielded at step 159 of 2,000.** It was still in warmup and exported nothing.
  * SPEC_RL's exit criterion (T1 panel, four families, H-WPRL-1/2/3) is **NOT reached**.
  * All four families are n/a, n = 0.
* **Diagnosis: POLICY DRIFT.** There is no code defect and no reward or advantage defect.
  * `param_delta_norm` median rose from 0.077 to 5.09.
  * oob median rose from 0.21 to 0.61.
  * Reward fell from 0.273 to 0.213.
  * `human_pdms_mean`, which the policy cannot affect, stayed flat (0.957 → 0.962).
  * The clip at 100 bound on 1 of 159 steps (step 142). The healthy maximum was 1.448, so by
    design (D-5) the clip was only a spike guard.
* **Contributors (MEASURED):**
  * lr stayed at 2e-4 after the batch went from 512 to 32;
  * there was no anchor;
  * `clip_sample` was off;
  * the clip never bound.
  * The grad-norm growth mechanism is UNVERIFIED.
* **A1:** lr 5e-5, warmup 100, clip 2.0, KL β 0.04 on all arms, and kill-switches K1/K2/K3.
  * Replayed on A0's metrics, the kill-switch halts at step 57 (K2).
  * A1 **restarts** from `ckpt_50400.pt`; it does not resume.
* **Tests (CPU, on the tree Thor runs):**
  * A0 suite: 36 passed.
  * A1 suite: 55 passed.
  * Mutation arms M1–M7 went RED and M7e went GREEN (an equivalent mutant). All arms behaved as expected.
  * The KL on GPU is UNVERIFIED.

## Where things live

* **A0 is already landed** on `origin/agent/arch-inf-20260803` (tip `4d937212`, package commit
  `0c618136`).
  * That includes the A0 stack code, `kinematic_prior.py`, `ddv2_rl.py` and `pdm_proxy.py`, all
    blob-identical to Thor's tree.
  * ⚠️ This local checkout's HEAD `37645fcc` is behind origin. Land on top of the origin tip.
* **A1 is NOT landed.** It exists only in this repo working tree, in one location, and is listed in
  `LANDING_READY_WPRL_A1.txt`.
* The package is `TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv7-ddv2-rl-posttrain/`.
  * The three `code/stack/…` files (`tanitad/rl/ddv2_refcv7.py`, `scripts/ddv2_rl_refcv7.py`,
    `tests/test_ddv2_refcv7.py`) are now **A1**: md5 `8532918e` / `de77a163` / `e4715bd7` (CRLF).
    The new flags default to A0 behaviour.
  * `raw/thor_rl_s0_metrics_step158.jsonl` (md5 `0fd63d31…`) is the evidence.
  * `raw/a1_tests/` holds the test outputs plus `mutate.py` and `halt_replay.py`.
* Thor (read-only; I did not touch it) has `/home/nvidia/refcv7_post/rl/`, containing
  `arms/rl_s0/`, `tree/stack/` (A0) and `wprl_queue.sh`.

## Next steps, in order, each with its blocker

1. **Land `LANDING_READY_WPRL_A1.txt` with the lander.** Copy the bytes of `code/stack/*` over
   `stack/*`, LF-normalised; this modifies the A0 blobs on origin. Run `pytest stack/tests/test_ddv2_refcv7.py`
   in the landed tree. *Blocker: the Master Mind.*
2. **Before any queue restart on Thor**, write `/home/nvidia/refcv7_post/rl/arms/rl_s0/STOP` and
   `/home/nvidia/refcv7_post/rl/QUEUE_STOP`. Without them, `wprl_queue.sh` resumes the drifted
   `rl_s0`, which has no DONE file.
3. **Ship the three A1 files** to `/home/nvidia/refcv7_post/rl/tree/stack/` and verify their md5s.
   The A0 tree rejects the A1 flags.
4. **Run the KL smoke**, ≤20 steps (SPEC_RL_A1 §11.1). It must show I-12 `kl_m` < 1e-6 at step 0,
   then a finite `kl_m` growing from 0, with `grad_clipped` logged. *Blocker: needs a GPU.*
5. **Get a new launch-gate PASS token** binding the A1 argv sha256s (PI rule, 2026-09-26). The argv
   delta is in SPEC_RL_A1 §7. No `wprl_argv_a1.py` helper was written; derive it from
   `code/thor/wprl_argv.py` plus §7.
6. **Run the arms** in this order: `rl_a1_s0`, `rloff_a1_s0`, `rlshuf_a1_s0`, `rl_a1_s1`, each
   writing to `/home/nvidia/refcv7_post/rl/arms_a1/<arm>`.
   * Cost: about 2,000 × 31.5 s ≈ 17.5 h per arm on Thor, plus about 10 % for the KL (ESTIMATED).

## Blockers

* The Thor GPU is busy with the R2 ladder for about another 2–2.5 days.
* A GPU off Thor needs PI spend.
* Landing belongs to the Master Mind.
* The PASS token comes from the launch gate.

## Do not

* Quote any A0 number as a lever effect. It is one seed, 159 steps, still in warmup.
* Quote "0.948 veto" or "IL ≈ 0.23". Neither comes from a primary artifact.
* Move the SPEC_RL §6 bars. A1 carries them verbatim, and I-4 is pinned to the pooled reading.
* Resume `rl_s0`.
