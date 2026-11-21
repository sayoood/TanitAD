# HANDOFF RALIGN PREP (2026-11-07). Dev box, CPU, synthetic data only. Nothing staged, committed, pushed or launched; no GAP48 token file was opened (none existed: `find gap48 -name "*.pt"` empty).
MEASURED = ours, log named. UNVERIFIED = needs the GAP48 tokens or box 2. Card sha256 a6c84433... verified before use; banked files unchanged (stage5 1fb1af10, stage5 cell 7fb89e5d, kit 20cf3284, reader 7fd65ffd).

## Built (all under `motion_c/t2v8f/ralign/`)
`ralign_lib.py` fold rule, loader, per-cell OLS (float64 sufficient statistics), map file, `TokenMap`. `ralign_fit.py` fit / identity. `ralign_gate.py` M-ALIGN with its regression arm. `ralign_policy.py` `AlignPolicy` (+ `MappedEncoder`, `install_align`).
`ralign_cell.py`, `ralign_driver.py` stage-5 cell and driver WRAPPER (imports the banked ones, rebinds 4 names; banked driver still refuses the kind by name). `make_ralign_stage5.py` overlay builder (arm C0C-RA, C0's own weights, no checkpoint shipped).
`ralign_idmap.py` G-IDMAP comparison on pulled cells. `ralign_read.sh` read template. `tests/` 41 tests. `logs/`.

## Tests (MEASURED)
`logs/pytest_ralign_fast.log` 37 passed, 15 s. `logs/pytest_stage5_rehearsal_run1.log` 4 passed, 304 s (stub renderer, stage-4 fixture, 6 scenes, 24 cells: non-identity map T2 and RR, plans move; identity package equals stage 4's C0C records on 6 scenes / 12 cells; perturbed identity FAILS; preflight refusals).
Mutations shown RED: fold rule (last digit, inverted), OLS with swapped roles, 198 pairs (NOT-EVALUABLE) vs 200 (fitted), same-fold map (flagged; unguarded gate would have PASSED it at 0.99), provenance check off (GATE-BROKEN), main arm on own fold (INVALID-LEAK), perturbed identity a/b (G-IDMAP), map not applied (G-P10 blind), AlignPolicy that scales ego or nav (PASS_THROUGH), dropped map, stale G-IDMAP bypass, 8 planted manifest faults named.
Note: `ralign_policy.py` got one wording edit while the rehearsal ran; the fast log is after it. Box-2 G-IDMAP must run on the bytes that ship (the builder binds it by sha256).

## Commands (T=E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f, R=$T/ralign, PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe, P=<GAP48 pull>/pull_gap48)
1. `$PY -I $R/ralign_fit.py fit --tokens-dir $P/tokens --out-dir $R/out/maps --scene-list $T/gap48/SCENES48.json`   rc 0 FITTED, 3 NOT-EVALUABLE, 4 refused (named)
2. `$PY -I $R/ralign_gate.py --tokens-dir $P/tokens --map-dir $R/out/maps --out $R/out/M_ALIGN.json`   rc 0 M-ALIGN-PASS, 5 ALIGN-FAILS-MECH (stop; no T2 stage), 6 invalid/broken
3. G-IDMAP on box 2 (2 scenes, real renders): `$PY -I $R/ralign_fit.py identity --out-dir $R/out/maps_id`; `$PY -I $R/make_ralign_stage5.py build --map-dir $R/out/maps_id --status IDMAP --tag ID --subset-launch 2`;
   `bash $T/audit/stage5_chain.sh C0C-RA_ID`; `$PY -I $R/ralign_idmap.py compare --ra-root C:/Users/Admin/t2v8f_pulls/stage5_C0C-RA_ID/pull5_C0C-RA_ID/rollouts --stage4-root C:/Users/Admin/t2v8f_pulls/stage4/pull_box2/pull4/rollouts --manifest E:/Projects/TanitAD-artifacts/scratch_ralign_t2v8f/build_C0C-RA_ID/pkg/MANIFEST5.json --out $R/out/G_IDMAP.json`
4. `$PY -I $R/make_ralign_stage5.py build --map-dir $R/out/maps --m-align $R/out/M_ALIGN.json --g-idmap $R/out/G_IDMAP.json` (GATED; writes `$T/stage5/C0C-RA/`), then `bash $T/audit/stage5_chain.sh C0C-RA` (box 2, ~1.7 h ESTIMATED as other stage-5 arms; D: free needed in `BUILD_RESULT5_C0C-RA.json`), then `GL=<green full-suite log> bash $R/ralign_read.sh C0C-RA`.
5. Replicate (card 6.1): steps 1 to 4 with `--fold-digit 1` on fit and gate and `--tag D1` on the build.
Chain terms: the package is the standard stage-5 overlay (`pkg/` members listed in `make_ralign_stage5.py` docstring); `requires` pins stage 2/4 files; the job runs `ralign_driver.py`; pull dir `D:\t2v8f\pull5_C0C-RA`. Check `BOX2_BUSY.txt` first (BPC held the box).

## Assumptions about GAP48 (read from `gap48/t2v8f_gap48.py`, not from a file)
Files `tokens/<scene8>.pt`, torch dict keys `scene` (full id), `render`, `real` (fp16 [n,40,1024]), `ticks`, `cam_frames`...; stem = `scene[:8]` (enforced). Key names are CLI flags. Scene id hashed as the exact UTF-8 string of `scene`. fp32 files also read.
Conventions the card leaves open (recorded in FIT_RESULT.json): zero-variance render cell -> a=0, b=mean(x), counted; non-finite tokens in a file are REFUSED (Stats alone drops them per cell); map output is rounded to fp16 (the policy's storage dtype); the identity pair stores b as -0.0 so a stored -0.0 token keeps its bits.
M-ALIGN statistic: median over all held-out pairs pooled across scenes (scene-cluster bootstrap beside it); the map also runs on RR cells (they are renders).
The 14 R-SCN scenes still run in the 62-scene driver (their folds come from the hash; the read excludes them as for every stage-5 arm).

## UNVERIFIED
Real GAP48 file format; fit and gate on real tokens; the box-2 G-IDMAP (GPU determinism of the identity run against stage-4 records); wall time of the 2-scene ID package (preflight hashes only its scenes, ESTIMATED minutes); `ralign_read.sh` was never run; `check_manifest5(m, 2, 0)` passes on the real tar (MEASURED) but the 2-scene driver run on box 2 is not.
Scratch: `E:/Projects/TanitAD-artifacts/scratch_ralign_t2v8f/` 2.9 GB (two stage-4 fixtures, moved aside by the PI later); E: had 4.8 GB free after. Rehearsals ran with `--min-free-gb 2` (E: below the driver's 10 GiB floor).
The brief's "t2v8f_stage5.py 01095679" is not the file's sha256: it is 1fb1af10 (same as the BPC/MC builds).
