# P5-C A3 HANDOFF (REFe agent, 2026-11-15 ~00:50Z). CPU only, no git writes, no GPU, no box 2, no navtest. Nothing launched on the pod.

## Done
1. **Label path**: `a3/p5c_rj5_path.py` + `a3/test_p5c_rj5_path.py`.
   - Subcommands: `feed`, `rekey`, `repair-combine`, `provenance`, `tb0r`.
   - No pinned file is edited. The pinned labeller runs unchanged on a feed bank (`traj := RJ5`). `rekey` re-keys its records to the bank sha.
   - The pinned gates and table builder then run on the real bank.
   - Tests: 23/23 pass, and 8/8 mutation arms are RED (`logs/TEST_RUN_A3.log`).
   - Real data: `feed` takes the banked join with 0 refusals, 84.98 % coverage, in 59 s, and refuses at the 0.99 bar. The pinned labeller's own 16,133 local RJ5 records rekey with 0 refusals, and the pinned gates join them with 0 sha mismatches (13,724 are dropped without the rekey). Logs: `logs/REAL_*`.
2. **Chain C on the pod: feasible after P7**, with 3 ship files (the TF-N2 patch is missing on the pod). Evidence: `logs/POD_FEASIBILITY_PROBES.md`, prereg §7.
   - The pod grants **15.3 CPUs** (cgroup), not 128 or 80.
   - The measured roll cost is about 2.9 proc-s/row, not 13 s. So chain C takes about 7 h on the dev box or about 1.1–2 h on the pod (ESTIMATED).
3. **`PREREG_P5C_A3_DRAFT.md`**: covers every item the brief asks for.

## Next steps (exact)
- **Write the A3 runner and chain edits.** Pins are blank in §9.
  - `p5c_pod_label.sh`: two passes, feed/rekey, `repair-combine`, `tb0r`, forced rank0, per-gate classify.
  - `p5c_pod.sh`: cgroup shard cap.
  - A W-A7-RJ5 input.
- **Bank** `rj5_bank_join.py`, its tests, the join and `rj5_labels_local.jsonl.gz` (F: only).
- After chain C: re-run the join with `--rows C=…` and pin its sha.
- **MM**: freeze A3.

## Deletions
**NOT done.** The permission system denied `rm -rf` of my own scratch `F:/tanitad_scratch/p5c_a3_tmp/` (423,515,136 B total). It is all re-creatable: tip exports, the kit extract, derived feeds and test temp. Delete it by literal path:
- `real/` 388,498,432 B
- `pt/` 18,321,408 B
- `tip/` 13,291,520 B
- `kit/` 2,407,424 B
- `mut/` 499,712 B
- `kit.tar.xz` 364,544 B
- small files under 60 KB
