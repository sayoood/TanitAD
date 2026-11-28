# collapse monitor (W3, PRH_JEPA_FOR_V8F.md)

Effective rank of the world-model latents of a checkpoint: participation ratio PR = (tr C)^2 / tr(C^2), top-20 eigenvalue fractions, dims holding 90 / 99 % of variance, dead dims (std < 1e-4), per level (z1 and the two encoded levels). Tier T0 latent diagnostic. Reading rule (PR/dim): < 0.02 collapsed, < 0.10 partial, else healthy.

Files: `collapse_monitor.py` (library + CLI), `tests/test_collapse_monitor.py` (12 tests), `run_ladder_arms.py` (the banked ladder arms), `RESULT_COLLAPSE_R35H.{json,md}`.

## Run on a new checkpoint
```
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export LADDER_READ_CODE_ROOT=F:/TanitAD-scratch/slowwm/tree WMACT_CODE_ROOT=F:/TanitAD-scratch/slowwm/tree
export PYTHONPATH='F:\TanitAD-scratch\slowwm\tree\stack;F:\TanitAD-scratch\slowwm\tree\taniteval'     # ';'-joined; assert tanitad.__file__ is under it
C:/Users/Admin/venvs/tanitad/Scripts/python.exe collapse_monitor.py --run-dir <pull dir with argv.txt + ckpt_last.pt> --frames <frames_s{seed}.npz> --out out.json
```
The frames file must come from the same adapter as the checkpoint (adapter hash is checked). Needs >= 6 GB free RAM (lr_lib guard). The planted self-check runs first and aborts the run if red.
Basis arm: add `--basis-npz <basis.npz> --ubasis-npz <eval ubasis npz> --selfcheck-json <run>/selfcheck.json` (z1 through `slowwm_hook.clip_z1_basis`, higher levels through `encode_clip_basis`). Written, not yet run on a real basis arm.
Tests: `python -m pytest -q -p no:cacheprovider --basetemp=<dir that exists> tests` with the same env.

## Calling it beside the SLOWWM IA-0 rows
In the read package, after the arm and frames are prepared: `import collapse_monitor as CM; rep = CM.monitor(run_dir, frames_npz, basis_npz, ubasis_npz, selfcheck_json)` and store `rep["levels"]` next to the IA-0 block of the same arm; or `collapse_monitor.spectrum_stats(Z)` on any latent matrix already in hand. Compare PR/dim of SLOW and PCA arms with R35H (`RESULT_COLLAPSE_R35H.json`: 0.062 to 0.081).

## Scratch log
`F:/TanitAD-scratch/collapse_build/` (pytest basetemp only) was deleted by literal path at the end of the build.
