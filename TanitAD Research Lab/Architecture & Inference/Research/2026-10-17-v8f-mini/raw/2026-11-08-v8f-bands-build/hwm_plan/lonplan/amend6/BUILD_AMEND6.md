# BUILD_AMEND6: the pull and the read of AMEND_HWMPLAN_6 (RA6-L1), built 2026-11-17

Card: `AMEND_HWMPLAN_6.md`, sha256 `db89e1285923e6835887de5b22c63e8398fb1388bb2abecdaea466cba5b3ebc9` (file `AMEND_HWMPLAN_6.sha256`; every launcher refuses on a mismatch). Written before any clip was pulled. Folder: `F:/TanitAD-scratch/actrsp/lonplan/amend6/`. Tier T0, dev-box CPU, zero training, no process started on box 2.

## Headlines for the MM
1. **ESTIMATED expectation is a repeat of AMEND_5.** On the existing 141 clips RA's held-out R² is flat from 56 to ~113 fit clips (s0 0.204 to 0.205, s1 0.197 to 0.201; `work/lcurve141_out.txt`). The card says so and commits the next lever for each outcome (section 6); the R6-SAME row starts AMEND_7 (a nonlinear 2 s readout on the same pulled latents, no box 2).
2. **One deviation from the brief, for your ruling.** The pull runs on the dev box and reads box 2 through sftp, the path `collect_latents_train_R35.py` used. It starts no job on box 2. Reasons: the latents need R35's two checkpoints, `wmact_lib` and the ladder tree, none of which are on box 2; a box-2 job would be new code on the machine that blue-screened under CPU load on 2026-10-17; the proven pull ran 141 clips this way at 2.9 s per clip. The brief's "copies them to the dev box with sha256 checks" becomes: md5 of every clip against box 2's manifest (3 attempts) and a sha256 per 50-clip chunk in `lat6/CHUNKS.json`, re-checked on every resume and by C-CHUNK6.
3. **No pilot ran.** `hwm_plan/ts/cache/train/clips/` is empty, so no clip file is readable locally. `pull_latents6.py selfcheck` ran instead (models load, ClipStore on the filtered manifest, window starts for clip 0, native-track coverage 4,713 of 4,713); it touches no box-2 file. The first real clip is the first test of the sftp path in `pull()`.

## Built (all under `build/`)
| piece | file |
|---|---|
| selection (seeded, nested, disjoint from the 141, the excluded 60, all eval ids, all window eids); output `../clips6.json`, clip-list sha256 `27d2bb2f…1649` | `select_clips6.py` |
| streamed pull + latents, chunks of 50 with sha256, resumable, control 4 must read 0.0 | `pull_latents6.py` (imports `collect_latents_train_R35.py` unchanged) |
| RA fit (amend1_kstep procedure), K-control, FIT-CURVE, chunk merge, largest-checkpoint rule | `fit_ra6.py` |
| points AMEND_5's modules at amend6 and at RA6's sha256; AMEND_5's readout is never a fallback | `amend6_env.py` |
| controls: AMEND_5's controls with C-RA-FIT replaced by C-FIT6, plus K-CONTROL, C-SEL6, C-CHUNK6 | `amend6_controls.py` |
| driver, readout (AMEND_5's `score()` on amend6 folders + Delta_R + section 6 outcome) | `amend6_run.py`, `amend6_read.py` |
| launchers | `pull6.sh` / `pull6.cmd`, `run_amend6.sh` / `amend6.cmd`, `launch6.ps1` |
| tests (13) | `test_amend6.py` |
AMEND_5's code (`ra_l1.py`, `amend5_*.py`) is imported unchanged; nothing under `amend5/`, the code of record or `stack/` was edited.

## Checks run now (CPU)
* **K-control on the real data (MEASURED, `controls/kcontrol_s{0,1}.json`):** the fit procedure on the original 141 clips reproduces AMEND_5's readout: lambda 1.0 (s0) and 0.1 (s1), n_pairs 8,306, mu and sd and ym exactly 0.0, max |dW| 2.6e-17 (s0) and 5.6e-16 (s1). The 140-clip mutation reads RED.
* **Tests:** 13 passed (`logs/tests_test_amend6.log`): K-control green and mutated RED, small W perturbation and a changed lambda read RED, fold map equals amend1's, pairs need 11 finite frames, merge refuses a repeated clip, FIT-CURVE shape, selection reproduces `clips6.json` and its prefixes, planted eval / c141 / excluded / window eid each read RED, a manifest that is not the trainer's order is refused, chunk sha256 mutation and duplicate clips read RED, largest-checkpoint rule, the outcome table (one chassis only is R6-SAME), the env never falls back to RA5.
* **Integration smoke on a throwaway root with synthetic chunks:** `fit_ra6.py fit` and `curve` ran end to end; the written readout loads through `ra_l1.load_readout` under the patched sha256, and `amend5_stage.build` declares it. FIT-CURVE at 141 clips on real latents: held-out r 0.472 (s0).
* **Not run (needs the pull):** the pull itself, the real fit on 1,000 clips, C-FIT6 / C-SEL6 / C-CHUNK6 on real files, the TRAIN stage, SCORE arms and readout.

## Launch steps for the MM
Preconditions: box 2 on and reachable (`ssh tanitad-box2`), its `D:` drive attached; C: commit headroom at least 5.5 GB (the scripts wait on `wait_commit.py`). Verify first: `cd F:/TanitAD-scratch/actrsp/lonplan/amend6 && sha256sum -c AMEND_HWMPLAN_6.sha256` (and `cd build && sha256sum -c MANIFEST_sha256.txt`).
1. **Pull (dev box, reads box 2):** `powershell -File F:\TanitAD-scratch\actrsp\lonplan\amend6\build\launch6.ps1 -Job pull6.cmd`. Poll `amend6/pull.log` (one line per clip, `ZZ CHUNK` per 50, last line `ZZ PULL6 DONE ZZ` or `ZZ PULL6 FAIL ZZ <why>`). Up to 5 automatic resume passes; relaunching the same command resumes from `lat6/CHUNKS.json`. If the pull is cut, `fit_ra6.py` uses the largest completed checkpoint (141 or 423 new clips) as the card commits.
2. **Read (dev box CPU):** after step 1, `powershell -File ...\launch6.ps1 -Job amend6.cmd`. It runs: card sha, tests, K-control, fit, FIT-CURVE, controls (both regression arms), TRAIN stage, SCORE arms, readout. Last line of `amend6/read.log`: `ZZ AMEND6 READ DONE ZZ`. Results: `readout/RESULT_AMEND6.json` / `.md` (outcome and Delta_R), `readout/RESULT_AMEND6_full.md` (AMEND_5's generator), `readout/FIT_CURVE_s{0,1}.json`, `RA6_FIT_s{0,1}.json`.
3. The MM writes the reading and runs the committed next lever (card section 6).

## Estimates (ESTIMATED, ±30 %)
* Pull: 0.7 h at the 141-clip mean (2.9 s per clip), about 3 h at 13 s per clip; 17.95 GB read from box 2 (sequential, one 21 MB file at a time), nothing stored but latents.
* Read after the pull: about 1.8 h (AMEND_5's stage times: controls 29 min on three bases, TRAIN 14 min, SCORE 20 min, readout 68 min; one base here).
* Disk on F:: lat6 about 0.25 GB; amend6 `train/` about 0.38 GB and `score/` about 0.25 GB (AMEND_5's sizes); scratch clip 21 MB. C: is not used beyond the interpreter.

## Deliverable manifest (all in ONE place: dev-box `F:`; staging into the repo is the MM's)
| artifact | path |
|---|---|
| pre-registration + sha256 | `F:/TanitAD-scratch/actrsp/lonplan/amend6/AMEND_HWMPLAN_6.md`, `.sha256` |
| clip selection | `.../amend6/clips6.json` |
| code, tests, launchers, hashes | `.../amend6/build/` (13 files, `MANIFEST_sha256.txt`) |
| K-control record | `.../amend6/controls/kcontrol_s{0,1}.json` (also `logs/kcontrol_s0.json`) |
| learning-curve measurement behind the sizing | `.../amend6/work/lcurve141.py`, `work/lcurve141_out.txt` |
| box-2 manifests copied for the selection (read-only copies) | `.../amend6/work/manifest_box2.jsonl`, `manifest_eval_box2.jsonl`, `meta_box2.json`; reference `work/ref/amend1_kstep.py` (from mirror tip `e5324710`) |
| test log | `.../amend6/logs/tests_test_amend6.log` |
Integration: nothing to merge into the code of record; the MM freezes the card and launches.

## Deletions (own temporary data, nothing banked elsewhere needed; all re-creatable)
* `amend6/work/smoke/` (synthetic-chunk smoke root, 28.6 MB), `amend6/work/ts6/` (selfcheck scratch, 0.56 MB), `amend6/work/tmp/` (pytest basetemp, twice, 0.37 MB), `build/__pycache__`, two stray first-draft `.cmd` files in `build/` (replaced). Deleted by literal path. Three background `find` scans I had started were stopped.
