# PREREG_V8F_LEVER3_V2 Amendment 1: the gate route of RH, ported to the dev box

**Frozen 2026-11-28, before any bank, mint, training or read of this experiment.** Card of record: `PREREG_V8F_LEVER3_V2.md`, sha256 707c5225...

## Why
Section 3 step 6 of the card names `launch_gate.py`. At the tree (97fe0a56), `launch_gate.py` has no profile for the V8FMini trainer (MEASURED, kit agent, `kit/REPORT.md`). The RH arms the card rebuilds did not use it. Their tokens (`dscale10k-gate-token-v1`) came from the study-side checker of the dscale10k lineage, `dscale10k_check_s3.py`, under the MM freeze entry of 2026-11-16, "NO stack/ change". The trainer's own G-DVB checks (`FLAG_CHECKS`) are built in. `launch_gate.py` is not edited.

## The gate of record for D3K_s0, D3K_s1, D9K_s0, D9K_s1
1. **Checker:** `kit/dscale10k_check_l3.py` (sha256 3a918065...), made from the pod's `dscale10k_check_s3.py` (046deadd...) by the asserted patch `kit/patch_l3.py` (8846ac91...).
   * It adds the four arm kinds. Each arm's argv equals the path-mapped `U_sK.B.tmpl.json` with the literal pins of the card: `--steps` 3000 (D3K) or 9000 (D9K), warmup 100, readout `causal_g1x`, kappa 0.2, and the arm's own bank.
   * The pair check is D3K_sK against D9K_sK, which may differ only in `--out` and `--steps`.
   * The token file is named `gate_gatetok.json` (the secret scan refuses `*token.json`).
   * Tests: `kit/tests_l3`, 18 passed (22 argv mutations per arm refused). The pod's study tests give 54 passed / 1 skipped on both checkers. The identity check to the pod checker holds for every earlier arm.
   * Wrapper `kit/l3_chk.py` (prep / mint / run); runner `kit/arm_run.py` refuses without a matching token (rc 2, MEASURED).
2. **G-SUITE evidence:** the pod's recorded VALID result for 97fe0a56 (0 regressions, 0 new failures, 0 missing), as the RH arms used it.
   * The dev-box tree `C:/Users/Admin/lever3/tree_cdb` is blob-identical to 97fe0a56: 3,407 of 3,407 files, no extras (`kit/tree_ok.json`).
   * No suite runs on this host. The base junit (cdbff971) is not on this box, and the code under test is the same bytes.
3. **G-HYG, G-LIVE, G-CLOCK, G-EVAL and G-CKPT** come from the UE_sK stage-A evidence, as for RH. Three checks run on this host at mint time:
   * C-MERGE: the pod's merge tool re-run on the pinned inputs, every output sha256 equal to ours, on this host;
   * C-BANK: the complete 4-shard bank of the arm's own init;
   * G-DAGGER: the arm's stage-B selfcheck on the real union.
   * Each needs its free-commit headroom; a check that cannot run is a FAIL, never a skip.
4. **GPU-memory smoke on the RTX 4060 (card section 3 step 6):** `kit/l3_gpu_smoke.py` (c59c5c7b...) runs the trainer's selfcheck on each launch argv with `--device cuda` at batch 64, under `devbox_gpu.lock`, before the first training step of each seed. If it does not fit, that arm is `LEVER3-BLOCKED`, and the batch is not changed.
5. Everything else in the card is unchanged.
