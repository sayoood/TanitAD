# Brief: P5-C Amendment A3 (C1 = optimized-driver labels) + the RJ5-traj label path (MM, 2026-11-15)

Read first: `E:/Projects/TanitAD/CLAUDE.md` (binding rules), `Project Steering/AGENT_OPERATING_STANDARD.md` (preamble), `Project Steering/TRAPS.md` entries on git hygiene and Windows. Read code and frozen docs from the MIRROR TIP, never the E: tree (it lags): `git -C C:/Users/Admin/tanitad-push ls-remote origin refs/heads/agent/arch-inf-20260803`, then `git show <tip>:<path>`. PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.

## The ruling you implement (PI 2026-11-14, GAC D-REFE-PI-2026-11-15-1)
P5-C's treatment arm **C1 trains on the optimized privileged driver's labels: RJ5 teacher trajectory (positive_jerk_limit_when_nonnegative_acc 5.0, same checkpoint_2400) -> Tc20s -> P5-R (lambda* 3.0)**, i.e. N2-R's label construction (teacher navtest 95.61 vs bank raw 89.81). **C0 is unchanged** (bank raw `traj`, no table). **C0 runs FIRST.** Same approved spend. This must be frozen as Amendment A3 BEFORE any label is built. Frozen so far: `PK/raw/2026-11-11-p5c-readiness/PREREG_P5C.md`, `_A1.md`, `_A2.md` (read all three; A2 sec 3 says any pinned-file edit needs A3 with new pins).

## Facts already MEASURED (do not re-derive; cite these paths)
- RJ5 trajs per bank key: `F:/tanitad_scratch/rj5path_tmp/join_out/rj5_bank_join.jsonl.gz` (sha256 `426e2abc...5170`): 78,947 / 92,896 P7 rank-0 keys (84.98 %). Missing 13,949 = chain C's 192 logs (`F:/tanitad_scratch/rj5nt/logs_fetch_C.txt`), still to roll. Tool + tests: `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/rj5path/` (`RJ5_PATH_FINDINGS.md` incl. the MM addendum: pod bank == dev copy by per-log key digest; the bulk R0 control failure is a 1-ULP yaw difference amplified by the teacher roll, median mm / p99 <= 0.27 m, a reproducibility floor, NOT a provenance bias).
- The pinned DB-ego labeller (`p5c_dual_labels.py`) needs each row's nuPlan DB. Only the POD holds all 1,192 DBs (`/workspace/data/navtrain_dbs`). So C1's labels are built on the pod, in the window after P7 ends (~2026-11-16 06:40Z) and before C0 starts. The pod must NOT get CPU/RAM load while a trainer runs.
- Roller: `F:/tanitad_scratch/rj5nt/code/rj5_roll.py` + `rj5_chain.py` (dev box, CPU, 1 thread, ~13 s/row). The pod built the bank with `build_teacher_rollouts.py` (stage rank0, CPU) in a teacher venv.

## Deliverables (priority order; bank each into the hold as you finish it)
Hold: `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_a3/`. Scratch: `F:/tanitad_scratch/p5c_a3_tmp/` (delete your scratch at the end; log what you deleted).
1. **The RJ5-traj label path, as code + tests.**
   - C1's table must be keyed exactly as the trainer looks labels up today (the bank traj's key in `refe/traj_label_table.py`), while the label VALUE is computed from the RJ5 traj of the same bank key.
   - Find the smallest change that does this. Candidates: a traj-override input to the label runner / `p5c_dual_labels.py`, or a re-key step in the table build.
   - Record per row: label source (`rj5` | `bank_fallback`), the RJ5 traj sha, and the roll platform (`devbox` | `pod`).
   - Tests must include RED arms: a wrong-key override must be REFUSED, a missing key must go to the declared fallback and never be silently filled, and a table built with the override must differ from the A2 table on rows whose RJ5 traj differs.
   - Run the tests serially with TMP/TEMP set to a per-run dir under the scratch, `CUDA_VISIBLE_DEVICES=-1`, and the Windows PYTHONPATH separator `;`.
2. **Chain C on the pod: a feasibility check only (do NOT launch anything).**
   - Can `rj5_roll.py` run in the pod's teacher venv on the pod's DBs, so the 13,949 missing rows are rolled in the post-P7 window (~50 CPU-h, so ~40 min on ~80 procs)?
   - Allowed pod probes: filesystem only (`ls`, `sha256sum`, reading text files), at `nice -n 19`. No Python that imports torch, no process above 1 GB RAM. Access: `bash C:/Users/Admin/par18/pd.sh <timeout> '<cmd>' | sed -E 's/[0-9]{1,3}(\.[0-9]{1,3}){3}/<IP>/g; /hf_/d'`.
   - NEVER print, copy or write the pod address, port, ids or any token.
   - Say what a pod R0 control would show (the pod's own platform should reproduce its bank). List exactly which files to ship.
3. **`PREREG_P5C_A3_DRAFT.md`** (the MM freezes it; you do not). It must commit, BEFORE any label exists:
   - the C1 label definition;
   - the RJ5 traj sources and their shas;
   - the coverage rule (minimum RJ5 coverage of P7 rank-0 keys to launch C1, what a missing row gets, and how rank-1 rows are treated under the A1/A2 rank policy);
   - the gate set on the RJ5-based labels (A2's, with thresholds unchanged, unless you show one cannot apply);
   - the per-log R0 control status as a disclosed column (TB-8);
   - whether W-A7 must be re-read on RJ5-based labels. First check whether the 1,018 W-A7 rows are P7 bank keys with RJ5 trajs. If they are not, say what rolling them would cost.
   - the run order C0 -> (pod window: chain-C roll, label build, gates) -> C1, or the order you show is better;
   - new pins (`git hash-object --no-filters` + sha256) for every file you change;
   - both outcome branches of every check.

## Hard rules
- Never commit, push or land. The MM lands.
- No navtest truth reads.
- No GPU.
- Do not touch box 2 or C:/tanitruled.
- Do not touch other sessions' processes.
- Never kill by name pattern.
- Never delete outside your scratch.
- No `-ExecutionPolicy Bypass`.
- Repo paths must be <= 200 chars.
- Every number carries its evidence class (MEASURED with path / INHERITED / ESTIMATED / H).
- Hand back <= 300 words, with a deliverable manifest (path + sha256) and the open decisions for the MM.
