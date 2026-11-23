# PREREG_V8F_KINLAB Amendment 1 (FROZEN 2026-11-23, before any RK step, pod census of the real argv, or number): the labels are computed once per bank, outside the GPU lock

**Base card:** PREREG_V8F_KINLAB.md, sha256 7db5d7aa. That card stays as frozen. This amendment changes only how the labels are computed, never what they are.

## Why (MEASURED, the RK dry mint, `pod/DRY_MINT_RK.md`)
* The kin_v1 labeller takes 1,631 s (s0) and 1,643 s (s1) per load on the loaded pod. speed_v1 takes 192 s and 200 s.
* The labeller runs at every trainer load, twice per arm, under the GPU lock. That projects the chain at 1.5–2.3 pod-h, against the card's ESTIMATED 0.6–0.75 and its 0.9 return line.
* The labels are deterministic per bank, so computing them under the lock only idles the GPU.

## Changes
1. **Code:** commit **1a8b11a1** on `agent/v8f-rollhist-cdb-20261120`, parent 0ca6f49d. It adds `--dagger-label-cache PATH`.
   * Without the flag, the trainer is 0ca6f49d bit for bit.
   * With it, the labeller's outputs come from an npz written atomically by a first run. A header check refuses on any change of:
     * label kind;
     * bank.json sha256;
     * the `v8f_recovery.py` or `v8f_dagger.py` bytes;
     * record count, eligible index, clip store, records or arrays.
   * Tests: labels from the cache equal a fresh compute bit for bit, on fixture, planted and 4,500 real records, for speed_v1 and kin_v1. `gt` and the plan loss are equal through the trainer argv. 25/25 mutants RED.
2. **Arms:** RK_sK's argv is RL_sK's argv with exactly three token changes:
   * `--dagger-label speed_v1` → `kin_v1`;
   * `--out` → `runs/RK_sK/train`;
   * **added:** `--dagger-label-cache /root/rollhist/kin_cache/s<K>_kin_v1.npz`, an execution-only token.

   The argv check admits these three and refuses any fourth.
3. **Before the lock:**
   * the cache of each bank is written by the trainer's `--selfcheck-only` on the pod CPU, at nice 19, at commit 1a8b11a1. That run's `kin_status_counts` give the census of card section 4; the bar of 0.5 is unchanged;
   * G-SUITE runs at 1a8b11a1, also on the CPU before the lock.
4. **The gate record binds** the cache file's sha256, which the selfcheck under the lock reads back from its built levers.

## Unchanged
Everything else in the base card: hypothesis, arms' meaning, reads and bars, outcomes and their precedence, the cost estimate of 0.6–0.75 pod-h of lock, and the 0.9 return line to the PI.

No RK number exists.
