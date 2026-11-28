# Amendment W-R8-1 — G0 wrapper clause on refcv8 (registered by the Master Mind, 2026-10-16 ~03:57 Berlin)

**What failed.** The step-5,000 G0 (`G0_SUMMARY_step5000.json`, finished 02:43) reads `wrapper_clause` FAIL:
`cascade` wrapper 1.235e-01 > bar 8.638e-02 and `r8_subscore` 1.642e-02 > bar 1.379e-02.

**Diagnosis (MEASURED, CPU, real step-5,000 checkpoint, the G0 probe's own 4 rows; `wrapper_diag/RESULT.md`).**
The clause compares `compute_losses_v3` scalar terms of a plain forward against micro-batched ones
(`wrapper_probe_r7.py:53-60, 102-111`). Its noise control patches only `torch.randn_like` (`:140-141`); refcv8 draws
its DDIM noise with `torch.randn` (`refc.py:3036`) and its extras with `randn_eval` / `rand_eval` (`refc.py:3042`,
`refcv8_conditioning.py:446`), so each arm saw different noise per row. With all noise held at zero, the micro-batched,
permuted and row-alone forwards equal the plain one to ≤ 4.1e-6 relative on every output and loss term; with the live
noise replayed per row, ≤ 3.3e-6; changing only the inference seed moves `cascade` by up to 22.7 % and `r8_subscore`
by up to 3.5 %. Controls bit-identical 471/471. ⇒ **No merge defect and no batch coupling in the refcv8 model**; the
FAIL is the probe's blind spot, and as coded the clause has no power on refcv8 planner terms.

**The amendment.**
1. For refcv8 the wrapper clause is judged with `wrapper_diag/code/wrapper_probe_r8fix.py`, which neutralises all four
   draw sites. On CPU it reads PASS (max wrapper 3.4e-6, bars back to 1e-5) and still detects W1 and W2.
2. `r8_spd_crc` is removed from the clause: it is a checksum of the batch, so any row reordering changes it by design.
3. The GPU re-run of the fixed clause (~1 min) is OWED and runs after the step-5,000 battery releases the lock.
   Until it lands, the step-5,000 G0 remains **FAIL as coded**, and every battery number stays **PROVISIONAL**.

**What this does NOT change.** The 238 `DETECTION_LOWSUPPORT` misses and the one `DETECTION` miss
(`eval_box3d_det_ap2_automobile_0_20`, in-run 0.33835 vs 0.31453, sd 0 over 24 seeds) stand. The latter is most likely
Thor compiled-bf16 vs dev-box eager-bf16 numerics (UNVERIFIED; settling it needs a Thor replay with `--trunk-compile`
on vs off). Perception terms are bit-identical across 10 noise conditions, so it is neither noise nor coupling.

**Consequence for the battery.** The rolls never micro-batch (`refcv3_arm.py:2106, 2260` run 1–3 rows); with no
coupling, each row's plan is a valid draw from the same distribution as a batch-1 deployment. Battery numbers are
readable, stamped PROVISIONAL, and each claim must clear the inference-seed spread the battery measures with seeds 0 and 1.
