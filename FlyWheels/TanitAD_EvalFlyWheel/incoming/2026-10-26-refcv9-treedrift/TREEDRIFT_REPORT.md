# TREEDRIFT -- bisect of the "fb47a58 vs tip selects a different path on 70/70 windows" finding (EvalFlyWheel, 2026-10-26, report 16:45 Berlin)

**Verdict: THERE IS NO TREE DRIFT.** fb47a58 and the branch tip 4b5f7fb select **bit-identical** paths on the 70 calibration windows, and so does every
commit in between. The reported 7.24 m is **subset roll vs the banked full roll**: the planner's noise is a function of the CUDA RNG *stream position*,
and a 70-window roll starts the stream at offset 0 while the bank's roll had already advanced 4 705 / 8 889 windows when it reached them.
Tier T0-style open-loop roll, os arm, inference seed 0, per-frame nav, `R8_NO_POSE_SYNC=1` (= the banked battery's `pose_sync=OFF`); all numbers MEASURED, n = 70 windows / 2 episodes
(no episode-cluster bootstrap is possible with 2 clusters; SDs below are window-level, and the seed floor answers "would another INFERENCE run say this?").

## Q1 -- which commit?  None.  (`PAIRWISE.json`, `analyze.py`)
Trees = `git archive` of fb47a58, 460508e, f6f96d6 (=3766dcd~1), **3766dcd, de4e15b, 947b05f**, 70211d6, 4b5f7fb. Same rig, same 70 windows, seed 0:

| pair vs fb47a58 | max abs d(traj) | windows differing | z_tac max d | CUDA / CPU RNG state before the forward equal |
|---|---|---|---|---|
| 4b5f7fb (tip), 3766dcd, de4e15b, 947b05f, 460508e, f6f96d6, 70211d6 | **0.0 m** | **0 / 70** | 0.0 | 70/70 / 70/70 |
| repeat of fb47a58 / of tip (run-to-run) | 0.0 | 0 / 70 | 0.0 | 70/70 |
| the R9-PF agent's own tip+patch no-head roll (`keq_fresh_nohead`) | 0.0 | 0 / 70 | - | - |

Mean ADE 2 s (os) is 0.25758 in every one of those rolls. **My fb47a58 roll vs the BANKED fresh bank `traj_own` (`F:/l1c/bank/ztac_fresh.npz`, 10 456 rows, fb47a58) is 7.2417 m, 70/70
differ -- exactly the finding's number, and the R9-PF tip roll differs from the bank by the same 7.2417 m.** The "drift" is bank-vs-subset, not tree-vs-tree.

## Q2 -- RNG stream or computation?  RNG stream position; the computation is identical.
* CPU, no GPU: `kinematic_prior` (prior_controls / prior_path / roll_plan / compose_ticks / absolute_controls / residual_controls, modes ha0_ext_pose + cv_yawrate, 96 synthetic windows) is byte-identical in all 8 trees, 14/14 arrays (`KP_CPU_PROBE.json`). 3766dcd's large `kinematic_prior.py` change is bit-neutral for refcv8's prior.
* The sampler noise is read off the CUDA state: base eps from the global stream, extras from a generator seeded off blake2b(`cuda.get_rng_state`) (`refcv8_conditioning.py:187` eval_generator, `:206` randn_eval). State is `(seed 0, Philox offset)`, offset 0 at window 0, **+4 per forward** (measured, all 70 steps).
* **Discriminator:** inject the state the window had in the bank's roll, `(0, 4 x bank_row)`, into fb47a58, 4b5f7fb and 3766dcd: the **banked `traj_own` is reproduced bit-for-bit, 0 / 70 differ, max 0.0 m** (`BANKCMP.json`, `out/bankpos_*`). Control, offset +4: 70/70 differ, 7.27 m. Same computation, different noise draws.

## Q3 -- consequences
**(a)** Tip reproduces fb47a58 exactly (bit-equal), so nothing banked at fb47a58 is invalidated by the tip tree. Against the bank rows the 70-window mean ADE 2 s is 0.2602 (bank) vs 0.2576 (subset), paired d = -0.0026, SD 0.059, SE 0.0071.
**Inference-seed floor, same tree, seed 0 vs 1:** 70/70 windows differ, max 8.77 m, mean ADE 0.2576 vs 0.2542, paired d = +0.0034, SD 0.078, SE 0.0093 (tip seed 1 == fb47a58 seed 1 bitwise). So the tip-vs-bank gap is **inside** the seed floor; per-window gaps of 7-9 m are ordinary sampler variance. ⇒ never compare a subset or re-ordered roll to a banked roll window-by-window; compare distributions against the seed floor, or replay the bank's stream position.
**(b) MAPFT K-PLAN-ID: YES, same-tree.** BASE trace: `refcv9_mapsel/code/chain_mapsel.sh:9,18` (`tree_fb47a58`, `R8_LAUNCH_TREE`, `PYTHONPATH`). Post-run rolls: `refcv9_mapft/kit/mapft_devbox_post.sh:26` (`T=...tree_fb47a58`), `:36` (R8_LAUNCH_TREE / PYTHONPATH from `$T`), `:117` (`mapsel_roll.py`, same script as the base, byte-checked `:48-51`). Comparator: `mapft_kplanid.py:33` (PRIMARY = plan8, fan, sel), `:59-97`. The tip-based C0 tree only trains (RUNBOOK:31); neither roll imports it. Bit-equality is meaningful **only because both rolls use the same window list, order and inference seed (same stream positions)** -- any change of those, or of forwards per window, breaks it for reasons unrelated to the planner; the kit's windows-md5 identity check (`:48`) and the RIG-NONDETERMINISTIC control (`:133-143`) are the guards. The rehearsal premise (rig reproduces its own trace) is supported here: 2 same-tree repeats bit-equal.

## Q4 -- the RNG pattern (NOT applied to the repo)
`torch.manual_seed` inside `fork_rng(devices=[])` at `refc_v3.py:2063` (fb47a58) / `:2084` (tip) re-seeds the CUDA stream and never restores it (seed log, `out/*/META.json`: CUDA state 374708fff771 -> 7d1fea12f3b2). For refcv8's **eval path it is harmless**: `refcv7_loader.py:313` (fb) / `:325` (tip) wraps the whole build in `fork_rng(devices=all cuda)` and restores 374708fff771 before the first window. It bites only where a seam is built **outside** such a fork (R9's `enable_refcv9`). Minimal fix: `torch.default_generator.manual_seed(...)` (CPU only). **Measured: applying it to fb47a58 and to the tip changes nothing -- bit-equal to the unfixed rolls on 70/70 (`fbfix_s0`, `tipfix_s0`).** INFERRED from reading (not run): in `train()` (`refc_v3_train.py:8868` seed, `:8894` build) CUDA is not yet initialised, so the lazily queued seed would make a *new* refcv8 training run's CUDA stream differ from the banked one; eval and existing checkpoints are unaffected.

## Deviations / disclosures
Followed the MM's GPU order (LONHEAD, then R9-PF K-OFF/K-EQ, then mine); I held the `gpu_lock` 16:14:15-16:27:14 (chain6) and 16:27:18-16:30:40 (chain7); both released, lock file absent afterwards. A 3 474 MiB non-roll resident load sits on the card, so my per-roll gate is `used < 4300 MiB` (gpu_lock.py's own threshold). A full-model CPU roll was NOT run (host free commit 7.9 GB, free RAM 3.8 GB: it would endanger the other rolls); the CPU probe covers the deterministic prior path only. Killed only my own PIDs; my queued chains were replaced several times (a gate threshold bug, then the MM re-ordering); I released only my own job's lock (job+pid match). Nothing committed, pushed, or written to G:; Thor untouched; clip ids appear nowhere.

Deliverables: see `MANIFEST_TREEDRIFT.sha256` (rig `roll_tree.py`, `analyze.py`, `bankcmp.py`, `kp_cpu_probe.py`, `chain6.sh`, `chain7.sh`, result JSONs, per-roll `out/*/META.json` + `records.npz`).
