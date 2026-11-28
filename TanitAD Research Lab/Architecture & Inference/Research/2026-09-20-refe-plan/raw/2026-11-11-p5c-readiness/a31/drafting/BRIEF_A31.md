# Brief: P5-C Amendment A3.1 — the runner/chain edits A3 §9 left unpinned (MM, 2026-11-15)

Read first:
- `E:/Projects/TanitAD/CLAUDE.md` (binding rules).
- **Frozen A3**: mirror tip `854d5926`, `PK/raw/2026-11-11-p5c-readiness/PREREG_P5C_A3.md` (sha256 `ec14e588…`). Read it in full; §1–§8 are the rules you implement and may not change.
- A2 (`PREREG_P5C_A2.md`, same dir).
- The launch kit: `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_launchkit/` (`HANDOFF.md`, `HANDOFF_2.md`, `RUNBOOK_P5C.md`, `TESTS.md`, `files/…`).
- The A3 label tool: `PK/raw/2026-11-11-p5c-readiness/a3/p5c_rj5_path.py` (blob 409f4481) and its tests.

Read the repo from the MIRROR (`git -C C:/Users/Admin/tanitad-push show <tip>:<path>`), never the E: tree. PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.

## Deliverables (priority order; bank each as you finish)
Hold: `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_a31/`. Write edited copies there; never edit the kit's originals in place. Scratch goes under `F:/tanitad_scratch/p5c_a31_tmp/<sub>/` (two levels deep).

1. **`refe/p5c_pod.sh`** (A2 pin `cf823843`):
   - cap the label shard count at floor(cgroup CPU quota). Read `cpu.max` (v2) or `cfs_quota_us/cfs_period_us` (v1), as `code/pod_dataprep.sh:30-41` does, never `nproc`;
   - pass the join sha;
   - `P5C_LABEL=p5r-rj5` must REFUSE unless the A3.1 sha256 is supplied and matches.
   - RED arms: a shard count that uses nproc FAILS; `p5r-rj5` without the A3.1 sha FAILS.
2. **`a1/p5c_pod_label.sh`** (A2 pin `d25e1a92`): the two-pass runner of A3 §6 step 2:
   - `feed` -> control pass (override OFF, 2,400 EVAL rows) and RJ5 pass;
   - -> `rekey` -> both `repair` runs + `repair-combine` -> gates + `tb0r` -> `provenance` -> table build `--rank-policy rank0 --require-coverage 1.0 --meta-json`;
   - the label class must come from `gates.json` per gate, not from the ZZTB line.
   - RED arms for each refusal A3 names.
3. **The W-A7-RJ5 driver**: a `--traj-source <join>` input to `w_a7_read.py`, or a wrapper, implementing A3 §5. Control (i) is replaced by (i′); (ii)–(iv) are unchanged.
   - RED arm: a run whose traj source is the bank must be refused as W-A7-RJ5.
4. **`PREREG_P5C_A3_1_DRAFT.md`**, pins only:
   - for each file: old pin, new blob (`git hash-object --no-filters`), sha256 and 0 CR bytes;
   - the test runs, with their RED arms;
   - nothing else. If you find a rule in A3 that cannot be implemented as written, STOP and report it. Do not work around it (that would need A4).

## Hard rules
- Shell lint: `bash -n`, LF line endings.
- Tests run serially, with TMP/TEMP set to your scratch, `CUDA_VISIBLE_DEVICES=-1`, and the Windows PYTHONPATH separator `;`.
- ⚠️ A `launch_gate.py` pin edit moves 6 profile fingerprints in 3 tests. If you touch it, run all ~69 launch_gate-reading tests.
- ⛔ Forbidden:
  - committing, pushing or landing;
  - any pod or box-2 access;
  - GPU or navtest reads;
  - touching other sessions' processes or files;
  - killing by name pattern;
  - deleting outside your scratch;
  - `-ExecutionPolicy Bypass`.
- Repo paths must be ≤ 200 characters.
- Every number carries its evidence class.
- Dev-box memory is very tight (other sessions' jobs are running). Keep processes small and sequential.
- Hand back ≤ 300 words, with a deliverable manifest (path + sha256) and the open decisions. Delete your scratch, or say exactly what you could not delete.
