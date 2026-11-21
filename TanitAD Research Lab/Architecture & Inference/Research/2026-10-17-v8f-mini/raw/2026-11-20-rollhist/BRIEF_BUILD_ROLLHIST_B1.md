# Brief: ROLLHIST build, phase B1 (items N1-N4 of the card; dev box only; no pod, no training, no commit)

You are a TanitAD build agent working for the Master Mind (MM).

**Read these first:**
- `Project Steering/AGENT_OPERATING_STANDARD.md`. Its preamble binds you. Read it from the mirror tip, `git -C C:/Users/Admin/tanitad-push show <tip>:"Project Steering/AGENT_OPERATING_STANDARD.md"`, where tip = `git -C C:/Users/Admin/tanitad-push ls-remote origin agent/arch-inf-20260803 | cut -f1`.
- The card `F:/TanitAD-scratch/rollhist/PREREG_V8F_ROLLHIST_DRAFT.md`, sections 1.2, 1.3 and 4. It is a DRAFT. Its open PI decisions (size, queue order, readout C* versus causal) do NOT change the code: the readout is a flag value, so build the generator for both.
- Where this brief and the card differ, the card wins. Tell the MM; do not resolve it yourself.

**Binding rules:**
- **Stage, never push.** No git commit, no push, no edit in E:/Projects/TanitAD or the mirror's working tree.
- **No pod access and no GPU.** Set `CUDA_VISIBLE_DEVICES=-1` everywhere.
- ⛔ Do NOT edit `launch_gate.py`, `stack/tanitad/models/__init__.py`, the sha-pinned `v8f_t1_eval.py`, or `v8f_t1_eval_hb.py`. Import them.
- Never print or copy anything from `Keys.txt`; you do not need it.
- Never write an IP address, port or pod id into a file.
- End with a deliverable manifest: every artifact, with its path and sha256.

## Setup
1. Export the current tip's `stack/` into `F:/TanitAD-scratch/rollhist/build/tree`:
   `git -C C:/Users/Admin/tanitad-push -c core.autocrlf=false archive <tip> stack | tar -x -C <dir>`
   - Check that `v8f_mini_train.py`, `tanitad/data/v8f_dagger.py` and `v8f_dagger_bank.py` have the same blobs at the tip as at `fb686f4a`, so the card's `file:line` citations hold. If they differ, report it and use the tip.
   - Verify 3 files against their git blobs.
   - Keep paths at or under 200 characters.
2. Use `C:/Users/Admin/venvs/tanitad/Scripts/python.exe` by its path.
   - Set `PYTHONPATH` to the tree's `stack` directory, written with `;` and Windows paths (`cygpath -w`).
   - Assert that `tanitad.__file__` lies inside your tree.
3. Run ONE pytest at a time.
   - Set `TMP`/`TEMP` to `F:/TanitAD-scratch/rollhist/build/tmp/<run>`, creating it first.
   - Delete it after the run by literal path, and log the deletion in `F:/TanitAD-scratch/rollhist/DELETIONS.log`.
   - Another session (N1DC build) is also running pytest on this box. A MemoryError is a memory failure: wait and rerun alone.
4. The study files the card cites live in the mirror under `TanitAD Research Lab/.../integ10k/` (`v8f_t1_eval_reader_rule_rzg.py`, `v8f_t1_eval_reader_rule_rzgfh.py`, `check_acc_gain.py`). Find them with `git ls-tree -r --name-only <tip> | grep integ10k`, and read them with `git show`.

## What to build (card section 1.3, N1-N4, nothing more)
* **N1 `stack/scripts/v8f_t1_readout.py`:** promote `backward_frames`, `causal_frames` and `causal_g_frames` from the study wrapper, with the arithmetic unchanged.
  - Unit tests carry ADD63's identity controls: gain 1 equals `causal`, gain 0 equals `causal_z`, array for array.
  - Also test that a run of the promoted functions equals the study file's own functions on the same inputs, bit for bit.
* **N2 `stack/scripts/v8f_dagger_bank.py`:** add the flags `--readout {pinned,causal_g}`, `--acc-gain G`, `--path-exec` and `--kappa-max {none,0.2}`.
  - The defaults `pinned` / `none` reproduce today's generator bit for bit.
  - Install `hb.bounded_frames(readout_frames(orig), kappa)` for the run and restore it after, using `v8f_t1_eval_hb.installed` or its functions.
  - Add both new tool files to `TOOL_FILES`.
  - Write a `generator` record `{readout, acc_gain, path_exec, kappa_max}` into every shard's provenance and into `bank.json`.
* **N3 `tanitad/data/v8f_dagger.py`:** `DaggerSwap.built_levers()` adds `dagger_bank_readout` and `dagger_bank_kappa`, read from the LOADED bank header.
  - A header without the field reads `pinned` / `none`.
  - Do not change `constants()`.
* **N4 `v8f_mini_train.py`:** add the declaration flags `--dagger-bank-readout` (default `pinned`) and `--dagger-bank-kappa` (default `none`).
  - Add their `FLAG_CHECKS` entries beside the dagger entries.
  - Add `bad_dagger` clauses: declared differs from the loaded header ⇒ REFUSED; non-default at `--dagger-p 0` ⇒ REFUSED.
  - Add the dests to any WM-only/default-dest lists.
* **Chain checker (card 4.1):** find where the STAGEB arms' checker lives (`gate_block3/.../block3_prep.sh`, `chk_arm3.sh`, `study/dscale10k_check.py`). Add the checks that:
  - RH's argv names both declaration flags and the bank sha;
  - B0's argv differs from RH's in exactly the four DAgger flags, the two declaration flags and `--out`.

  If that checker is a study file outside `stack/`, write the new check as a separate stack script and tell the MM where it plugs in.

## Tests and the card's reachability table (4.2)
R2 applies: every bit-identity test runs on CPU, fp32, one thread.
* **R4, in its synthetic form:**
  - On a small synthetic or banked input, `--readout pinned --kappa-max none` reproduces today's generator's shard, with max abs difference 0.0.
  - `causal_g --acc-gain 1 --path-exec --kappa-max 0.2` satisfies two checks per record: the one-frame acceleration at the last executed knot minus the executed one is <= 1e-6, and every heading step is <= 0.2 x the step distance + 1e-12.
  - Mutation: delete the line that installs the readout, then the bound; each must turn R4 RED.
  - If the real generator needs a checkpoint and data that exist only on the pod, build a minimal fixture. Say exactly what the pod-side pilot must still check.
* **R5:** both refusals.
* **R1-R3:** if `dagger_probe` and the S/O gradient clauses can be exercised on CPU with a tiny model and a synthetic bank, do it. The p = 0 controls must read exactly 0.0. Otherwise leave R1-R3 for the pod pilot and say so.
* **Mutants:** record each with the test that went RED, in `mutation_table_rollhist.json`.
* **Regression:** run the existing tests for `v8f_dagger_bank.py`, `v8f_dagger.py`, the trainer's selfcheck tests, and `test_v8f_t1_eval_hb.py`. Show any pre-existing failure failing identically on the unpatched tree.

## Deliverables (all under `F:/TanitAD-scratch/rollhist/build/b1/`)
- `rollhist.patch`: the full diff against the tip you exported, binary-safe, LF.
- `TREE_FILES_SHA256.txt`.
- The test logs.
- `mutation_table_rollhist.json`.
- `REPORT_B1.json`: each control with its value, and the `file:line` hops from flag to bank header to `built_levers` to G-DVB.
- `HANDOFF_B1.md`, at most 2k tokens, with the exact pod-side B2 steps: pod tree, G-SUITE, the bank pilot, the argv files.

**Priority if you run short:** N1 with its tests, then N2 with R4, then N3 and N4 with R5, then the checker, then mutants, then regression.

**Finish.** Hand back in at most 300 words with the manifest. Delete your tmp directories by literal path, with a log. Do not wait on anything longer than about 4 minutes.
