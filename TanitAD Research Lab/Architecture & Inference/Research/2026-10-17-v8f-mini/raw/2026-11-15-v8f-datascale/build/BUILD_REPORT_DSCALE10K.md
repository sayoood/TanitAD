# BUILD_REPORT_DSCALE10K (build agent, 2026-11-16)

Scope: the no-GPU launch pieces of the frozen card (sha256 `48a74901...`, MM freeze ~19:35Z). No `stack/` change. No training, no pod access, nothing landed.
Code of record `cdbff971`. Evidence: MEASURED = run here on synthetic fixtures; everything on real data is UNVERIFIED (not run).

## What is built (all in `F:/TanitAD-scratch/datascale/build/`)

| file | what |
|---|---|
| `dscale10k_merge.py` | `merge` (v9 release + manifest, lead npz, speed jsonl, native tracks .pt, union cache `manifest.jsonl`), `derange` (R_s0), and independent `verify_*` functions (C-MERGE) |
| `dscale10k_check.py` | mints `dscale10k-gate-token-v1`; `diff`, `dry`, `mint`, `run` subcommands; every `*_problems` function returns `[]` for pass |
| `test_dscale10k.py` | 48 tests, all PASS (8 s, CPU) with the code of record on PYTHONPATH; the code-of-record tests skip without it |

### Merge tool
* Holdout ids = clips 540..599 of the corpus train manifest in `sorted(clips, key=sha12)` order, the trainer's `--exclude 540:600` (`v8f_mini_train.py:3092-3096`, `sha12` at `v8f_mini_data.py:93`). The tool reproduces `excluded_sha256` of the M2b summary (`f6a1dee0...`) and refuses a manifest that does not.
* Corpus side: every kept clip's rows / lines / tensors are bit-equal to the input. Only the 60 holdout clips are removed, by id (release by sid = `stable_episode_id`, lead by sha12, speed and tracks by clip id).
* Deterministic: `np.savez` stamps the wall clock into the zip, so the npz is written with a fixed zip timestamp; the .pt is serialised through a buffer (`torch.save` to a file names the archive after the file). Two runs give identical sha256 on all six outputs (test).
* The union release loads in the trainer's `load_v9_release` (npz md5 recorded in the manifest, `n_clips`, `n_rows`, a `merge` provenance block).
* Refusals (fail loud, nothing written as final): wrong pinned md5, different columns / dtypes / schema / vocab / literals, a sid or clip in both sets, a holdout clip in the 10k files, holdout ids absent from the corpus release.
* R_s0 derangement: groups of equal `(n_rows, k0)`, seed `20261117`, a shuffled cycle per group (no fixed point by construction); the columns `k`, `now_s` and every `nav_*` stay the clip's own; corpus rows untouched. A donor map is written beside the release.

### Checker
Argv refusal (`argv_diff_problems`), accepted md5 set (`accepted_md5_set`, `md5_problems`), `C-STEPS`, `C-NCLIPS`, `C-CHUNK` (the chunk the trainer builds, `max(4, min(48, lru-2))`), `C-TRAIN-SHA`, `C-DISJ` (id, stable sid, sha12), `C-HOLD`, `C-EXCL`, `C-MASK`, `C-MERGE` (verify plus a rebuild that must equal the recorded sha256), `C-JOIN-EQ`, the loss-term reach table decision (`reach_problems`), `G-CLOCK` decision, `G-HYG` addition, `C-INIT` hash helper, `C-CACHE` comparison, report-only `C-NAVCOV` / `C-LEADCOV` / `C-SESS`. `dry` runs the trainer's own `build_data` and records `n_clips`, `train_sha256`, the clip list, `lru`, `S`, total steps; it was run here against the real trainer with `--synthetic` (test `test_dry_build_against_the_trainer`).

Deliberate-regression arms (each asserted RED in the tests): a flipped corpus byte in the union (release, lead, speed, tracks); a holdout clip kept in the union / present in the 10k files / a clip in both sets; a derangement that is the identity, a changed corpus row, a changed own column, a fixed point in the donor map; an unknown md5, a union md5 without a passing C-MERGE, a corpus md5 on a union arm, an edited `--v9-train` file; `--exclude` on a union argv (G-R3), `--exclude` missing or altered on a B arm, any extra / changed / removed flag, `--steps 19163`, wrong seeds, `--fit-join` missing, the corpus join path reused; one eval-141 id in a train list (G-R1), a digest-list hit, an empty eval list; one VLM-class `goal_w` bit on a 10k row (G-R2), a corpus control reading 0; each reach-table clause; a +0.2 s clock shift; a perturbed corpus `nav` row (C-JOIN-EQ); a trainer / argv / commit / data-file edit after the mint; a missing merge record or dry record.

## Decisions I made that the MM should confirm
1. **Union md5 and R md5 are accepted per arm role**, not as one flat set: a B arm needs the corpus md5, U / UE the union md5, R the R md5, X the 10k md5. The union md5 enters only when C-MERGE (verify + rebuild equality) passed; the R md5 only when the derangement verification passed too.
2. **Singleton `(n_rows, k0)` groups** cannot be deranged within a group. The tool REFUSES by default (`--max-singletons 0`); the real 10k release may have a few. Needs a ruling: allow N left as they are (reported), or a different pairing.
3. **VLM-class bits for C-MASK** = every goal class whose release `GOAL_PROV` contains `vlm` (this includes `LANE_CHANGE_L/R`, `sam3|vlm`). `exact=True` restricts to pure `vlm`. The control bar is "corpus count > 0" and the TL clip counts are reported, not enforced (after the 60 holdout clips drop, the 779 figure of the card can shrink).
4. The token seal is tamper-evident, like strat1's: a hand-written token with correct hashes is the same token. What it binds (argv, data files, tree, trainer, checker, merge tool, evidence) is re-hashed at `run`.
5. `S_U` in the dry record is the union build's `S`; UE_s0 / BL_s0 steps = 4 x that, pinned in `argv_diff_problems(ue_steps=...)` and `steps_problems`.

## Not done / needs the pod (UNVERIFIED, not run)
* Real inputs: the corpus and 10k files were not on this box; the formats come from the code of record and the banked corpus manifest (`v9_labels_train.manifest.json` keys). The 10k release / lead / speed / tracks files must be checked against the same schema on first run (the tool fails loud on a mismatch).
* The union view `/root/data/cache_union_dscale/train/` (symlinks, 14,512-line `manifest.jsonl`; `merge` writes the manifest, the symlinks are pod-side).
* `dry` on the real union cache (writes the union join through `--fit-join`, gives U's `train_sha256`), then `C-JOIN-EQ` against `join_train.npz` (the card gives only the md5 prefix `9ebc7a48`; the prefix is what the checker compares).
* The real-data G-LIVE reach probe (corpus-only / 10k-only / mixed batches): this file only DECIDES from its record (`reach_problems`); the probe that produces the record is not written. Same for G-EVAL, G-CKPT, G-SUITE records, the G-CLOCK residuals, C-CACHE hashes, C-SESS keys, `acc_probe` / `dagger_probe`, the Adam-state post gate, and the 600-step pilot.
* The digest-json shapes for C-DISJ (`load_id_list`) are UNVERIFIED; `.txt` one-id-per-line works.
* The trainer export used for the tests lives in `build/tree/` and was deleted after the run; recreate it with `git -c core.autocrlf=false archive cdbff971 stack/tanitad taniteval | tar -x -C build/tree` plus `stack/scripts/v8f_mini_train.py`.

## Run the tests
`CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 PYTHONPATH="<tree>/stack;<tree>/stack/scripts;<tree>;<build>" python -m pytest test_dscale10k.py -q --basetemp F:/q/ds10k_tmp/pt` (result here: 48 passed in 8.0 s).
