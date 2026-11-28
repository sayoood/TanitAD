# Brief: build the recovery label on own states (RECLAB) into the v8f DAgger swap: a trainer flag, its reachability proof, and the RL arm's pod chain. Build and dry-check only: run nothing on the pod

You are a TanitAD build agent working for the Master Mind (MM). Read `Project Steering/AGENT_OPERATING_STANDARD.md` from the mirror tip first:
`git -C C:/Users/Admin/tanitad-push show $(git -C C:/Users/Admin/tanitad-push ls-remote origin agent/arch-inf-20260803 | cut -f1):"Project Steering/AGENT_OPERATING_STANDARD.md"`.

Rules:
* Stage nothing, commit nothing, push nothing; the MM lands and launches.
* No GPU. Do not edit protected files: `launch_gate.py`, `stack/tanitad/models/__init__.py`, anything under `E:/Projects/TanitAD/stack`, the sha-pinned `v8f_t1_eval.py`. Work in your own export of the tip tree (below).
* No secrets, and no IPs, ports or pod ids in any file.
* Status: zero-spend build under the standing rule that implementation around a pending PI go is the MM's. Running the arms needs a PI go with the cost.

## Why (read these; they bind)
* The ROLLHIST line reads **ROLLHIST-HARMS** three times: RH (C\* bank, p 0.1) failed G-T0 at s0, RHp3 (p 0.3) and RHK (K bank) at both seeds. The own-action gains at T1 are real (RHp3: STRATEGIC -2.01 m, ade_tac -0.41 m), so the guard is what blocks. Verdict: `F:/TanitAD-scratch/rollhist/read/out/read_k/MM_VERDICT_RHK.md`. Card: `F:/TanitAD-scratch/rollhist/PREREG_V8F_ROLLHIST.md` (781eea87) and Amendments 1 to 8 in the same folder.
* The card's HARMS row names the next lever: the LABEL on own states. Today `swap_rows` (`stack/tanitad/data/v8f_dagger.py` ~:540-582 at the tip; card section 1.1) keeps **the human's label at the matched route point r'** for every own state that passed the matcher's cuts, and drops off-route states (lateral > 3.0 m, heading > 30 deg). `human_label(grid, r')` is in the ego frame of the HUMAN row r', so applied to the own state it ignores the own offset and jumps to the human speed.
* `H-ST-RECOVERYLABEL-1` built a kinematically feasible recovery label (registered V1), in the mirror at `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-28-bank-presteps/recovery/` (`labeller.py`, `test_labeller.py`, `test_real_panel.py`, `RECOVERY_RESULT.md`, `FROZEN_CARDS.md` + Amendment 1 one folder up). Identity: zero state offset gives the human label bit for bit.

## The mechanism (MM design; flag it in the handoff if the source contradicts any line)
* **Flag** `--dagger-label {human,recovery_v1}` on `stack/scripts/v8f_mini_train.py`, default `human`, which must reproduce today's swap **bit for bit**. FLAG_CHECKS entry `"dagger_label": "dagger:dagger_label"` beside the dagger entries; refused when non-default at `--dagger-p 0` (a lever declared that does nothing).
* **Library**: port the registered V1 labeller verbatim into `stack/tanitad/data/v8f_recovery.py` (record the research copy's sha256 in the module docstring) with its tests ported to `stack/tests/`. V1 only; V2 is post-hoc and stays out.
* **Semantics under `recovery_v1`**, per bank own-state record (ticks 1 and 2) with the matcher's existing r' and reason:
  | matcher reason | labeller | label used |
  |---|---|---|
  | ok | emitted (incl. identity) | the recovery label, own frame |
  | ok | refused | today's label (human at r'), unchanged |
  | off_lat / off_head | emitted | the recovery label; the record becomes ELIGIBLE |
  | off_lat / off_head | refused | dropped, as today |
  | end_edge / nonfinite | any | dropped, as today |
* **Every label derived from the same future waypoints** must be derived from the recovery label by the same code path (for example any speed, acceleration or `--w-acc` target, endpoint or goal label). Labels not derived from the future waypoints (route channel `lab_s.rc` re-expression, tactical classes) stay as today. Enumerate every label tensor a swapped sample carries, file:line, and say for each which rule applies. ⛔ If a waypoint-derived label cannot be re-derived from the recovery label, STOP and report: do not approximate.
* Labels are numpy float64, deterministic. Compute off-route eligibility once at bank load; measure its wall on one real bank shard and say whether a sidecar precompute is needed (if so: a script that writes it with a sha256, a flag that binds that sha256, and a FLAG_CHECKS entry; prefer no sidecar if load-time cost is under 10 min per bank).

## Priority order (bank each part as it is done)
1. **FACTS_RECLAB.md**: the label-tensor enumeration above; the matcher's cut order and where eligibility is computed (file:line); where the swapped sample's plan label is written; the labeller's inputs and whether the bank record carries them (own pose, own speed, clip, r'); the measured labeller wall per 1,000 records. Bank it first: the MM writes the card from it.
2. **The mechanism** in your tree, with tests:
   * identity: under `recovery_v1`, a zero-offset own state gives the human label `np.array_equal`;
   * default: under `human`, the swap output equals the code of record `97fe0a56` on a fixed draw set (a gold test, as `F:/TanitAD-scratch/rollhist/build/b6gold/`);
   * reachability (CLAUDE.md "built, tested, and unreachable"): argparse -> namespace -> swap -> label tensor -> loss term, file:line per hop; on a real bank shard, the count of swapped rows whose label differs from the human label is > 0 under `recovery_v1` and **exactly 0** under `human`; the count of newly eligible off-route rows is > 0 and exactly 0;
   * mutations that must turn the tests RED: delete the line that installs the recovery label; ignore the own offset (the labeller's offset-ignoring arm); skip the feasibility refusal;
   * the full dagger/v8f test files pass (run whole files, not `-k`), and the ported labeller tests pass.
3. **The RL arm's pod chain**, staged in `F:/TanitAD-scratch/reclab/pod/` (pattern: `F:/TanitAD-scratch/rollhist/build/b8p03/` and `b9rhk/`): `RL_s0` / `RL_s1` = the RH_sK argv (`/root/rollhist/runs/RH_s*/argv.json`) plus exactly `--dagger-label recovery_v1` and a new `--out`; the RH bank `/root/rollhist/bank/sK` reused (s0 sha256 e5e44c01..., read s1 from its argv); comparators the existing RH and B0 (no new arms). Stage B only (no bank step). The checker kind RL: exactly two added argv tokens and `--out` against RH, bank sha pinned to RH's; mutants (a second changed flag, a different bank, `human` under the RL name) refused. Holds `/root/gpu_queue.lock` and follows the marker protocol of the earlier chains, including `/root/N1DW_READY` (another session trains on that GPU between its slices; read its protocol in the earlier chain scripts and do not change it).
4. **Read extension** in `F:/TanitAD-scratch/rollhist/read/` (it was deleted; re-export from the mirror if you need it, as `HANDOFF_READ_R3.md` describes): RH_AKIND=RL, following RHK; `dry_check.py` PASS under RL and the default. Keep `.bk` copies of any file you edit.
5. **DRY_MINT_PLAN.md** (as b9rhk) and **HANDOFF_RECLAB.md** (at most 800 tokens).

## Tree
`git -C C:/Users/Admin/tanitad-push -c core.autocrlf=false archive <tip> | tar -x -C F:/TanitAD-scratch/reclab/tree` (tip from `ls-remote`; record it). Run tests with `PYTHONPATH` set to the tree's `stack` (Windows separator `;`), assert `tanitad.__file__` is inside your tree, and pass `--basetemp` under `F:/TanitAD-scratch/reclab/tmp`.

## Pod (`source C:/Users/Admin/lorapod/podenv.sh`; `pssh`, `pscp "$POD_TARGET:/path" local`)
Read and pull only, small files only (no bank shards above 300 MB in total). Never write there, run no python or pytest there, and never touch a process, lock or marker. Another session trains on the pod's GPU.

## Dev box
* CPU only, one python process at a time, each gated on commit >= 5.5 GB (`C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 60`).
* Python: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I`. TMP/TEMP under your folder.
* ⛔ F: is shared and tight: your whole folder stays under 1.5 GB, and F: must stay >= 15 GB free. Delete your own temp at the end by literal path, and log it in `F:/TanitAD-scratch/reclab/DELETIONS.log`.
* Never wait more than about 4 minutes in one call. Hand off (write HANDOFF_RECLAB.md and end) if your context passes about 150k.

## Deliverables
`F:/TanitAD-scratch/reclab/`: FACTS_RECLAB.md, the changed tree files listed with sha256 in `CHANGED_FILES.txt`, test logs, `pod/`, DRY_MINT_PLAN.md, HANDOFF_RECLAB.md, MANIFEST_SHA256.txt.

Hand back in at most 250 words, with a deliverable manifest, and say plainly which checks PASSED and which did not run.
