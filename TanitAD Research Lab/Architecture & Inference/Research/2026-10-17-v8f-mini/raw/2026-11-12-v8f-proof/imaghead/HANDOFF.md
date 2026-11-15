# HANDOFF: imaghead (card PREREG_V8F_PROOF_ADD36), dev box, 2026-11-15

State: DONE and tested on the dev box (CPU). 31 tests pass (`test_imaghead.py`, 15 s), each control c1 to c5 has a mutation that goes RED. No real head checkpoint existed: nothing here is a result. Definitions, file:line facts and every departure from the card: `IMAGHEAD_DEFS.md` (read it before data).

## Open (not done)
* c2 against banked files: no banked rollout or head output exists for LS/RL; a G checkpoint was not at hand (`--banked-roll LATENTS_CTX_<ARM>_R3g.npz` is implemented for a head checkpoint).
* No GPU run: float32 on a GPU may break the exact 0.0 of c1 or c4 at 1e-7 level; report it, do not loosen.

## Pod side, per arm and seed (in the style of `lead_dump.py`)
Needs: the patched tree, the checkpoint (`--lead-sup on`), the eval state file of `extract_states.py` (`--split eval`), the window list `LATENTS_CTX_G_s0_R1.npz`, the pack `leadsup/read/eval_pack.npz`.
```
CUDA_VISIBLE_DEVICES=<gpu> /workspace/venv/bin/python /workspace/v8proof/probe/imag_dump.py \
  --ckpt /workspace/v8proof/runs/<Q>/<ARM>_s<k>/ckpt_last.pt --tree /workspace/v8proof/tree_leadsup \
  --windows <dir>/LATENTS_CTX_G_s0_R1.npz --pack <dir>/eval_pack.npz --states states/<ARM>_s<k>_eval.npz \
  --out out/DUMP_<ARM>_s<k>.npz --device cuda
```
Pull `DUMP_<ARM>_s<k>.npz` and its `.json` (a few MB). Exit 0; the printed line shows c1, c2, c4 at dump time.

## Dev side
```
export CUDA_VISIBLE_DEVICES=-1; PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
$PY imag_read.py read --dump DUMP_RL_s0.npz DUMP_RL_s1.npz --arm RL --seeds 0,1 --out-dir OUT/RL     # exit 0 = controls pass, 2 = a control failed (no name)
$PY imag_read.py read --dump DUMP_LS_s0.npz DUMP_LS_s1.npz --arm LS --seeds 0,1 --out-dir OUT/LS
$PY imag_read.py pair --a-dir OUT/RL --b-dir OUT/LS --out-dir OUT/PAIR
```
The floor defaults to `OOF_RB_G_s<k>.npz` (found under `C:/Users/Admin/v8proof_c/ctxroll/rb` or `spatial/q2/out`; identical across arms, MEASURED). Outputs: `RESULT_IMAG_<ARM>.json`, `VERDICT_IMAG_<ARM>.md`, `SUFF_IMAG_<ARM>.npz`, `PAIR_IMAG_<A>_minus_<B>.json`.

## Tests
`$PY -m pytest -q test_imaghead.py --basetemp F:/TanitAD-scratch/v8proof/imaghead/work/pt/<new name>` (the parent `work/pt` exists; use a new name). Temp data: `work/pt/a`, `b`, `c` (pytest basetemps, tiny checkpoints and synthetic dumps, a few MB each; deletes are refused here, safe to remove).
