# PREREG K2DC02-TAIL — AMENDMENT 1 (MM 4a3d05, 2026-11-20 ~09:40Z)

Written BEFORE any K2DC02 tail weights exist. The kit's first chain stopped at the identity control at 08:44:05Z, and no
K2DC02 tail file has been built (Thor `runs/E2TAIL_K2DC02/CHAIN_STATUS.txt`: `BUILD-FAILED rc 3`). Two items in §1 and §2 could
not be run as written. Both were measured by the kit agent. Nothing in §3, §4 or §5 changes: rules, constants, comparators'
weights and the committed outcomes all stay as written.

## A1. §2.2 identity control: from FILE BYTES to TENSOR IDENTITY

* **As written:** a rebuild of K2DC's tail "must reproduce the md5 recorded in E2-TAIL's provenance, bit for bit".
* **Why it cannot pass, MEASURED by the kit agent:**
  * The provenance records only `sha256_out`; it holds no md5.
  * `torch.save` output is not byte-deterministic. Two rebuilds with the same tool (blob e456ff67 = e8a4631a's file, identical
    on the tip and on Thor) from unchanged inputs gave sha256 b8092eb0… and 20328b81….
* **The weights themselves are identical:** the rebuild matched the live K2DC tail on all 1,201 tensors (`torch.equal`), with
  equal keys, step and num_updates. The live file's sha256 equals the banked `sha256_out` ff066691….
* **Amended control (exact):**
  * The live K2DC tail's sha256 must equal the banked `sha256_out`, pinned by the provenance file's own sha256 9065c976….
  * A rebuild from `runs/r9e2-K2DC/{model_ema.pt, ckpt_ema_5000.pt}` must be `torch.equal` to it on EVERY tensor, with equal
    keys, step and num_updates.
  * Implemented by `k2dc02tail_build.sh` with `IDCTL_TENSOR_EQUAL=1`.
* What the control was for, that the tool builds the same weights, is unchanged. Only the measurement moves from the file's
  bytes to its tensors.

## A2. §1 comparators: the PLANNER rolls are re-rolled, not reused

* **As written:** the K2DC-tail and K0-tail stage B dumps are reused, not re-rolled.
* **MEASURED by the kit agent:** E2-TAIL's planner rolls `E2_K0_pf` / `E2_K2_pf` were deleted from Thor after its pull and were
  never banked. The dev-box pull holds only the perception dumps and mapsel. Without the planner rolls there is no
  families / distkeep read and no dose rule.
* **Amended:** `k2dc02tail_cmp_reroll.sh` re-rolls both comparators. K0-tail goes to `cmp/E2_K0_pf`; K2DC-tail (`--k2-pf` of the
  dose rule) goes to `cmp_k2dc/E2_K2_pf`. Both use:
  * the same tail weight files as E2-TAIL;
  * exactly `r9e2_roll.sh`'s roll command;
  * inference seeds 0 and 1;
  * `OMP_NUM_THREADS=6`, the stock roll's pin;
  * one Thor environment, shared with K2DC02-tail's own roll.

  So the three planner arms come from one environment, which is the better design anyway: T1 dumps made in different
  environments drift by millimetres on rolled tracks (memory: t1-dump-thread-drift).
* **Reused as written:** the perception dumps, the K0 mapsel traces and the banked FIT-threshold EVALs, each sha256-checked.
* **New validity item §2.5, the REPRODUCTION CHECK:**
  * The K2DCt − K0t `LAT_cross_mae_m` point from the re-rolled pair must lie within ±1 mm of E2-TAIL's banked value at each
    seed: +0.0126 m at s0 and +0.0091 m at s1 (`e2/e2raw/e2tail/stageB/read/E2_VERDICT.json`). If it does not, the read is
    TAIL-VOID (§5.5) and no dose claim is made.
  * Every other four-family row's agreement is reported beside it. Those rows decide nothing.

## Cost

Zero training. Thor, run serially under the GPU lock:
* the comparator re-roll, about 2.6 GPU-h;
* K2DC02-tail stage A and stage B, about 3–4 h.

The total is about 6–7 h of an idle GPU. Training arms remain PI decisions (§5).
