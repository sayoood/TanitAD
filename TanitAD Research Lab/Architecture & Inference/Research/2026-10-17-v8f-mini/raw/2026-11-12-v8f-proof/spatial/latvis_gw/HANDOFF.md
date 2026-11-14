# HANDOFF: LATVIS on the SPATIAL stage-1 arms (prepared, NOT run on the GPU)

State: driver, job, tests and DEFS built and checked on the CPU. Nothing staged or committed; the shared GPU lock was never taken.
Tree: `tree/` (tanitad + taniteval copies, patched v8f_mini.py and v8f_beta.py) matches both pod pins (raw bytes and LF form equal). G and W checkpoints build and load strictly on the CPU with the width 640 / kind asserts (G_s0, W_s0 checked; G_s1 loaded in smoke). sha256 of each checkpoint is recomputed against its .sha256 file.
Tests: 12 passed (`tests.log`). Smoke (`smoke.log`, 12 panel episodes, 2 folds, 1 epoch, 4 arms, W_s1 stood in by W_s0 bytes): prep, frame identity (24 of 24 tiles equal to the baseline sheet), collect, predicted latents, 16 stages, analysis, panels, LATENTS_GW files all ran; numbers decide nothing. The smoke ran before the last edit (c4 flag applied to the grid arm only); the tests ran after it.
Baseline `frames64.u8` and `LO_r128.i8` are absent from the baseline folder: the job decodes the frames again (CPU, about 35 min) and takes the token grid skill from the baseline stage files (paired pairing check passes on the fold indices; baseline G skill reproduces 0.7770).
Launch: `bash /f/TanitAD-scratch/v8proof/spatial/latvis_gw/job_latvis_gw.sh > /dev/null 2>&1 &`. Reads the arms whose files and sha256 match (W_s1 may arrive later; rerun then, finished stages are skipped).
Outputs: RESULT_LATVIS_GW.json, VERDICT_LATVIS_GW.md, panel/, LATENTS_GW_<ARM>.npz(+.json), LATVIS_GW_DONE.txt or LATVIS_GW_STOP.txt.
Scratch left (nothing deleted): `_probe1.py`, `smoke/` (431 MB frames, panels, latents, stages), `latents/` (smoke-run copies are under `smoke/latents`), `tree_hashes.json`, `prep_done_gw.json` only after the real prep.
Reading choices RG1..RG10 and resource estimates: LATVIS_GW_DEFS.md.
