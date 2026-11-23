# Brief: KINPUT: the input-side lever of the KINLAB card (`--dagger-base own|logged`): in a swapped own sample, O's prior path `base` is rebuilt from the LOGGED kinematics at the matched row r', the own pose history kept. Facts first, then code. Zero training, dev box

You are a TanitAD build agent working for the Master Mind (MM). First read:
* `Project Steering/AGENT_OPERATING_STANDARD.md` from the mirror tip (`git -C C:/Users/Admin/tanitad-push show origin/agent/arch-inf-20260803:"Project Steering/AGENT_OPERATING_STANDARD.md"`);
* `F:/TanitAD-scratch/kinlab/PREREG_V8F_KINLAB.md` (FROZEN 7db5d7aa), its outcome table: the HARMS / PARTIAL rows' MECH3-NO lever is **K-INPUT**: "O's prior path `base` of an own sample rebuilt from the logged kinematics at r', the own pose history kept";
* `F:/TanitAD-scratch/kinlab/FACTS_KINLAB_3.md`, `F:/TanitAD-scratch/kinlab/FACTS_KINCACHE.md`, `F:/TanitAD-scratch/kinlab/FACTS_KINTB.md`.

## Why
The KINLAB read came back **KINLAB-PARTIAL with MECH3-NO** (MEASURED, `F:/TanitAD-scratch/rollhist/read/out/read_rk/`). The kinematic label did not lower the T0 cost of rolled history against the speed label (RK − RL at T0 ADE6: s0 −0.0115 [−0.0321, +0.0074], s1 +0.0009 [−0.0157, +0.0180]). Before it, add mode did not either (ROLLADD, MECH2-NO). So the cost is not in the label and not in replacement; the card's next lever is the input side.

## Rules
* Stage nothing, commit nothing, push nothing. The MM lands.
* No GPU, and nothing run on the pod.
* Do not edit protected files: `launch_gate.py`, `stack/tanitad/models/__init__.py`, anything under `E:/Projects/TanitAD/stack`, or the sha-pinned `v8f_t1_eval.py`.
* Work in a fresh export of **fde8b716** at `F:/TanitAD-scratch/kinlab/tree4`, made with `git -C C:/Users/Admin/tanitad-push -c core.autocrlf=false archive fde8b716 stack taniteval | tar x -C ...`. Do not touch `tree3`.
* No secrets, IPs or ports in any file. Never name a file `*token.json`.

## Phase 1: facts (bank `FACTS_KINPUT.md` before any code)
Quote `file:line` at fde8b716 for each answer:
1. In the swap (`--dagger-mode swap`, the RH / RL / RK argv), which inputs of a swapped own sample come from the own rolled state, and which from the logged record? List every model input of the sample (pose history, `p10f` / kin history, `kin3`, `base`, the grid, nav, any other).
2. How `base` is computed for a logged sample and for an own sample (`kdx_base(kin3)` or otherwise), from which kinematics, at which row.
3. What the matched row r' is, how it is found, and what logged kinematics (v, a, kappa / yaw rate) exist at r' in the clip store.
4. Whether any other input is derived from `kin3` or from the own kinematics (so that rebuilding `base` alone would leave an inconsistent pair). Say exactly what "the own pose history kept" leaves in place.
5. At inference (the T1 executor and T0 eval), where `base` comes from. Confirm the change is training-only.

If facts 1-4 show that "rebuild base from the logged kinematics at r'" is ambiguous or impossible as written, STOP after phase 1 and hand back the facts with the options; do not choose a design the card did not name.

## Phase 2: the code (only if phase 1 is clear)
1. New trainer flag `--dagger-base own|logged`, default `own` = fde8b716 bit for bit. `logged` is valid only with `--dagger-p` > 0 and `--dagger-mode swap`; otherwise refused.
2. With `logged`, in each swapped own sample, `base` (and only what phase 1 shows must move with it, named in your facts) is computed from the logged kinematics at r'. The own pose history and every other input stay as today. The label is untouched (any `--dagger-label`).
3. The label cache: the base is an input, not a label, so the cache stays valid; say in your facts whether the header must change (it must not, if the labeller's inputs are unchanged).
4. **Built levers / G-DVB** gain `dagger.dagger_base`, with its FLAG_CHECKS entry exactly as `--dagger-kin-tb` was wired in fde8b716. Run all the launch-gate test files and report whether any pin moves.
5. **Census** on real records (pull shard 0 of RH bank s0 read only, as FACTS_KINTB did; delete it afterwards and log the deletion): the share of swapped own samples whose `base` changes under `logged`, and the distribution of the change (max and p50 of the position difference along the 30 points, in metres).

## Tests (new file `stack/tests/test_v8f_kinput.py`, whole-file runs)
* **Default, gold:** `own`, or the flag absent, equals fde8b716 bit for bit (batch inputs, labels, built levers apart from the new key).
* **`logged` moves only base:** on a planted bank, the swapped own samples' `base` equals `kdx_base` of the logged kinematics at r' exactly, every other input and the label are equal to `own`, and logged (non-swapped) samples are unchanged.
* **Refusals:** `logged` with `--dagger-p 0`, with `--dagger-mode add`.
* **Reachability:** with `logged` through the trainer argv, the batch's `base` differs from the `own` batch on at least one swapped sample, the plan loss differs, and the built levers read `logged`.
* **Mutants** (expect RED): the flag parsed but not passed; base rebuilt from the own kinematics anyway; the logged row taken at r instead of r' (if they differ); the built-levers key missing; a non-swapped sample changed.
* Run `test_v8f_kintb.py`, `test_v8f_kinlab.py`, `test_v8f_kincache.py`, the dagger, dagger_train, recovery, recovery_wiring and rolladd files, and the launch-gate test files, all whole file.

## Dev box
* CPU only, `CUDA_VISIBLE_DEVICES=-1`, one python process at a time behind `F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 60` (4.0 for a run under 1.5 GB; say so). Never lower the gate to get past it.
* Nothing on C:. TMP, TEMP and `--basetemp` go under `F:/TanitAD-scratch/kinlab/tmp`.
* Stop if F: drops below 6 GB free.
* Never kill a process you did not start. Never touch `F:/TanitAD-scratch/rollhist/read/` or `F:/TanitAD-scratch/kinlab/readrk/`.
* Never wait more than about 4 minutes in one call. Hand off at about 150k context by writing `HANDOFF_KINPUT.md` with what is left.

## Deliverables
In `F:/TanitAD-scratch/kinlab/`:
* `FACTS_KINPUT.md` (phase 1, then the census and the pin result);
* `tree4/` with `CHANGED_FILES_KINPUT.txt` (path, sha256, base blob at fde8b716);
* test logs and `mutants_kinput/MUTANTS_RESULT.json` (run the mutants on the FINAL files, all of them, after the last edit);
* `HANDOFF_KINPUT.md`, plus additions to `DELETIONS.log`.

Priority order if you run short: (1) phase 1 facts; (2) the flag at the default, gold; (3) `logged` and its tests; (4) G-DVB; (5) mutants; (6) the census.

Hand back in at most 200 words, saying plainly what passed and what did not run.
