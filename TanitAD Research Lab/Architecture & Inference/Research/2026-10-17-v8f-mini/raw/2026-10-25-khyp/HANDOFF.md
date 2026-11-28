# HANDOFF (updated 2026-10-25 11:30): PHASE B IS DONE. Outcome: HEADROOM, NO SELECTOR.
Read `KHYP_RESULT.md` (outcome first) and `READ.json`. Control 3 FAILED at seed 0 (SEL-LEARN gain == the always-class-5 override; `posthoc.json`). Nothing staged/committed/pushed.
Nothing left to run. Open items for the Master Mind: (1) adjudicate CONTROL 3 / seed 0; (2) land the artifacts (repo path is the Master Mind's call); (3) the PREREG's committed next step is a scene-reading scorer (box 2), a zero-training precursor on the frozen scene latent needs its own pre-registration; (4) work item: `v8f_dk_read.TOL_ADE` 1e-5 refuses float32 plans.
The Phase-A notes below are historical (they describe the state before the run).

---
# HANDOFF: PREREG_V8F_KHYP, Phase A done, Phase B NOT started

OUT = `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/khyp` (all paths below are relative to it).
Spec: `Project Steering/PREREG_V8F_KHYP.md` at tip `8d6449a7`. Nothing staged, committed or pushed; no real data read.

## State
- `khyp.py`: the harness. `hyps` (enumerate 7 one-hot longitudinal hypotheses through `forward_diag`, stream clips, resumable shards) and `read` (controls first, then HEADROOM / PRIMARY / SECONDARY / beside rows, the PREREG outcome table verbatim, `KHYP_RESULT.md` + `READ.json` + `khyp_numbers.json`). Declared interpretations (10) are in its docstring; none depends on a KHYP number.
- `test_khyp.py`: 16 passed (`pytest_log.txt`). Pins control 1 (bit-exact, 3 classes), harness identity, control 4, control 3 instrument, features, outcome boundaries, H-stack == HBLEND's banked H (0.0), and a full synthetic `hyps` -> `read` smoke. Three mutations go RED (override ignores k; argmax->argmin; ORACLE picks worst).
- `raw/`: eval grid / kin / join and `v8f_dk_read.py` exported from the tip (grid, kin, join md5-equal to SFRESH's copies). `code/`: `git archive 8d6449a7 stack taniteval` minus `results/` and `experiments/`.
- Inputs read in place (never copied): eval cache `scratchpad/sfresh/cache`, checkpoints `scratchpad/sfresh/inputs/M2b_s{0,1}`, train cache `scratchpad/hblend/cache`, train join `scratchpad/hblend/box2/join_train.npz`, window list `scratchpad/hblend/fit_clips.json` (+ `fit_windows_M2b_s*.npz`, checked by `read`).

## Exact Phase B commands (one data run at a time, in this order; detach each, e.g. WMI or `nohup ... &`, then end)
```
bash run_phaseB.sh eval0     # ~2-3 min   -> hyps/M2b_s0_eval.npz
bash run_phaseB.sh train0    # ~8-10 min  -> hyps/M2b_s0_train.npz  (9,582 windows)
bash run_phaseB.sh eval1
bash run_phaseB.sh train1
bash run_phaseB.sh read      # ~5-10 min  -> KHYP_RESULT.md, READ.json, khyp_numbers.json, dk_read.json
```
Logs in `logs/<stage>.log`. `run_phaseB.sh` refuses under 3 GB free on C: (`KHYP_MIN_FREE_KB` overrides) and sets `CUDA_VISIBLE_DEVICES=-1`, 4 torch threads, `OPENBLAS/MKL_NUM_THREADS=1`. A killed `hyps` stage resumes from `hyps/<tag>_shards/`.

## Memory plan (streams clips one at a time; target peak < 1.5 GB private commit)
- MEASURED (`memprobe.py`, real-size random-init model, batch 16, 3 steps): peak commit **1.11 GB** (batch 32: 1.30 GB). Imports alone are 0.84 GB (torch 0.83); numpy's OpenBLAS adds 0.74 GB at import unless `OPENBLAS_NUM_THREADS=1` is set BEFORE numpy (the runner and `khyp.py` do this).
- `hyps`: `ClipStore(lru=2)` holds 2 clips (~21 MB each); windows are in clip order; only 640-window shards (~2 MB) are in memory before being written. Expect ~1.2 GB. `read` loads ~12k windows of arrays (~30 MB) + sklearn: expect ~1.3-1.4 GB (ESTIMATED, not measured on real data).
- Writes: hyps files ~60 MB total + shards ~70 MB + outputs; OUT is 68 MB now; budget 400 MB.

## After `read`
1. `READ.json` has `decision.outcome` (one of MULTIMODAL PAYS / HEADROOM, NO SELECTOR / NO HEADROOM) and the committed next step quoted. If `status: STOPPED`, a stop-control failed (control 2 > 1e-3 m, alignment, control 4, harness identity, or the train window list != HBLEND's): report it, do not interpret.
2. Check `controls_all_pass`; control 3 (permuted labels must not separate from SEL-ARGMAX) failing is flagged, the table is still printed verbatim, the Master Mind adjudicates.
3. Control 1 on real data is likely vacuous (no exactly one-hot softmax): the bit-exact case is pinned by the test; the real-data harness identity (`own_vs_diag_identity_maxabs_m`, must be 0.0) is the live check.
4. `distance_keeping` in `READ.json` is best effort (`v8f_dk_read` at the tip); on `ERROR` record it as a work item. It never gates the table.
5. Add commentary to `KHYP_RESULT.md` (outcome first, controls, HEADROOM, PRIMARY, SECONDARY per seed with CIs, beside rows, quoted next step). Strategic family: PAUSED (PI ITEM 25), never a pass.

## Known spec gaps (declared in `khyp.py`, flag them in the hand-back)
- Control 4 is "by construction" only for vertex-picking selectors; SEL-MIX (a convex combination) can beat the best vertex. Hard for ARGMAX/LEARN, reported for MIX.
- HEADROOM outcomes need ORACLE - B <= -2F at BOTH seeds; a seed split is flagged in `READ.json`.
