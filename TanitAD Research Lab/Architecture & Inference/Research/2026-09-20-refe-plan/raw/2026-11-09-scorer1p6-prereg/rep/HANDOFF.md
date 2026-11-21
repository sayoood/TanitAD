# HANDOFF: seed replicates C0r2, S1E1, S1Z1 (2026-11-10). CPU only; nothing committed, nothing on a pod, no GPU/lock touched.

## State
Overlay on tip `8ef2cddac092e14b3dbf9e9627781a02183b2af57` (all of priority 1, 2 and 3 are done). `files/` = 6 files at repo-relative paths (2 MOD, 4 NEW, LF); `rep_vs_tip_8ef2cdda.patch` (`git apply -p1`) reproduces `files/` byte for byte (MEASURED). `CHANGED.txt` has the tip blobs: launch_gate.py e7c3b64ffe, test_scorer1p6_gate.py b36f8436ea. Land only if those still match.
- `launch_gate.py` (additive): `REP_ARMS`, `rep_arm_refusals`, `rep_argv`, two NEW profiles `refe_rep_p6` (C0r2, S1E1) and `refe_rep_s1z_p6` (S1Z1), 3 guarded edits in `scorer1_refusals` (argv builder chosen by the STRING `argv_kind == "rep"`; a `rep_arms` block refusing a seed equal to the parent's). Two profiles, not one: pair/control and twin rules are profile-wide.
- `rep_pod.sh`: runs the PARENT's own pod script with MODE=argv and swaps `--seed` and `--out` only (sed on the two values). Modes: argv, preflight, mutate_*, run. No calibrate mode.
- `rep_chain.sh <arm>`: one arm per call; own dirs `/workspace/data/refe_rep/<arm lower-case>/` (c0r2, s1e1, s1z1), package `/workspace/refe-plan-rep`, HF prefix `rep/<arm lower>` in `Sayood/tanitad-refe-scorer1p6` (override `REP_HF_REPO`; existence UNVERIFIED as in S1Z). Same stage order and the same wait functions as s1z_chain.sh (byte-equal after renaming, test-proven; one extra lock name). Stages: precheck, inputs, G-EGR, dd, wait, preflight, parent's mutants (S1E1 3x G16, S1Z1 3x G18, C0r2 none), launch gate, train, run check, egress, `ZZREP_CHAIN_DONE <arm>`. Logs `chain_<arm>.log`, markers `stage_<arm>/`.
- `rep_pause_p7.sh <p7_pod.sh pid> <train.py pid> <arm>...`: pause style of pause_p7_s1z_b.sh; runs the arms in sequence (a refused arm does not stop the next); P7 resume is on an EXIT trap once P7 was stopped, unless `/workspace/data/refe_rep/HOLD_P7`.
- Edited existing test (1 line, same precedent as S1Z): `test_scorer1p6_gate.py:154` profile-set equality now also lists the 2 new names. This is the only change to an existing test.

## Decisions (check them)
1. lambda_p (S1E1) and lambda_z (S1Z1) are the PARENT's measured values, READ from `refe_scorer1p6/calibrate_s1e/` and `refe_s1z/calibrate_s1z/`, not re-measured (a re-measured weight is a second moved lever). So the replicates need S1E and S1Z to have calibrated; the chain refuses otherwise (rc 3 / REFUSED). "Calibration stage" of S1Z1 = this read.
2. Navtest lists, card (checked by sha256), label dump, bank, ckpt = the parent's paths. `dd` default 6144 MB (one arm).
3. Dirs are lower-case like the parents' (`refe_rep/s1z1`), not the arm's capitalisation.

## Trace: flag -> argv -> trainer seed
`rep_pod.sh:29` (sed) / `launch_gate.py:669` `rep_argv` -> `--seed N` in argv -> `scorer_finetune.py:1453` `ap.add_argument("--seed")` -> `:1555` `torch.manual_seed(a.seed)`, `:1764` data-order `Generator().manual_seed(a.seed)`, `:1715`. Gate: `launch_gate.py:1275` `ns.seed != arm_seed[arm]`, `:1279` parent-seed refusal. Seeds: C0r2 2 (C0 0), S1E1 1 (S1E 0), S1Z1 1 (S1Z 0).

## Test proof (MEASURED, CPU, CUDA hidden, one process at a time, RAM >= 3 GB)
PASS (ZZTEST_PASS), 11 of 12 required: test_rep_gate (R1 argv diff: exactly two tokens differ, the values of --seed and --out, for all three, with and without lambda, and in the pod scripts' own output vs the parents'; R2 legal and refused argvs; R3 digests: refe_s1z_p6 5c3bff457aa3bf1e and refe_scorer1p6 e74d64617f514d35 unmoved, new refe_rep_p6 dd348f44c9638f1a, refe_rep_s1z_p6 2356afd4efeeb52a; R4 unknown arm, seed equal to parent refused in table, rep_argv, gate, pod script (RED copy); R5 bash -n x3, chain structure), test_scorer1p6_gate (17 older fingerprints unmoved), test_s1z_gate, test_scorer1p6_preflight, test_s1z_preflight, test_scorer1_layer, test_s1z_layer, test_pair_loss, test_pair_loss_p6, test_rl_epo, test_dump_scores. test_launch_gate: NOT PASSED; reaches `full()` (line 2302) and stops on the missing `D:\Projects\TanitAD\data\refe_consistent\targets_rank1.jsonl` (known); after it UNVERIFIED. `test_results.json` has rc and seconds.

## Pod commands (nothing run)
1. Land `files/`, regenerate `SHIP_MANIFEST.json` (`launch_gate.py ship-manifest --commit <40-hex>`), ship the tree to `/workspace/refe-plan-rep` (never over a dir holding a token). Parents' cards stay in their roots. Run the launch-gate-reading suite per the S1Z handoff; on the pod `python refe/launch_gate.py suite-prerun --profile refe_rep_p6 --pkg /workspace/refe-plan-rep --out <json>` (and `refe_rep_s1z_p6`).
2. Pause and run: `nohup setsid bash /workspace/refe-plan-rep/refe/rep_pause_p7.sh <p7_pod.sh pid> <train.py pid> C0r2 S1E1 S1Z1 > /workspace/data/refe_rep/pause.out 2>&1 < /dev/null &` (copy the script out of the tree first or run it from there; it is part of the tree sha).
3. Single arm, P7 already paused: `REP_HF_TOKEN_FILE=<token file> nohup setsid bash /workspace/refe-plan-rep/refe/rep_chain.sh S1E1 > /workspace/data/refe_rep/chain_s1e1.out 2>&1 < /dev/null &`. Dry run: `REP_STOP_AFTER=precheck|inputs|egr|dd|preflight_s1e1|gate_s1e1`.

## ESTIMATED L40 time (training MEASURED only for S1E, 1.12 h)
C0r2 about 1.3 h (training about 1.1 h, preflight no mutants, gate 3 min); S1E1 about 1.5 h (train 1.12 h + preflight/mutants about 15 min + gate); S1Z1 about 1.5 h (the S1Z card's own ESTIMATE, unmeasured). All three about 4.3 L40-h plus one P7 pause.

## Not verified
Real ViT-L trunk, bank, bf16, pod preflight/mutants/gate on the replicate argv, the chain end to end (bash -n + argv/structure tests only), parent calibration files on the pod, HF repo and prefix `rep/`, the 69-test suite, test_launch_gate after `full()`, `rep_pause_p7.sh` (bash -n and text order only), `flock`/`nvidia-smi` paths on the pod.
