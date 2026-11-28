# BUILD_OPTB: ACTRSP Option B lever (partial build, handoff in `HANDOFF.md`)

Nothing launched, committed or pushed. No GPU, no box 2. Frozen spec `AMEND_OPTB.md` (sha256 `7cb0245c...`), written before any new-flag test ran.

## Escalation (headline)
1. The lever attaches to `a13f6f16` (branch `agent/v8f-actrsp-20261116`), not to the repo tip `5e7083d8`: the tip trainer has no `--wm-only`. Land `optb.patch` on a branch from `a13f6f16`.
2. **Not done:** the box-2 chain package, the argvs and the checker (deliverable 3), mutations MR11-MR14 as a run (deliverable 2 part), the regression and launch-gate-reading test runs. Each is a named step in `HANDOFF.md`. The pilot, gate tokens and launch are therefore not ready.
3. One change outside the four flags: `HJEPAStack.built_rollout` (the G-DVB ladder reader) now reads on a float64 copy. Reason (MEASURED): on a trained WM the float32 reader gave level-3 k-step weight 0.99985 against the declared 1.0 (tolerance 1e-4), which would refuse every fine-tune gate. A fresh init reads 1.0 either way. It needs the `test_v8f_ladder` regression run before landing.

## What was built (`optb.patch`, 776 lines, sha256 `4212b0e2...`)
Files: `stack/tanitad/models/hjepa.py`, `stack/tanitad/models/v8f_mini.py`, `stack/scripts/v8f_mini_train.py`, new `stack/tests/test_v8f_optb.py`. Edited tree: `optb/tree/`.
Flags (all default-off, named in argv per `AMEND_OPTB.md` section 1): `--wm-rsp-readout {idm,fixed_k}` (default idm), `--wm-rsp-probe`, `--wm-rsp-probe-sha256`, `--wm-init-from-ckpt-wm {off,on}`, `--wm-freeze-proj {off,on}`.

## Hops (file:line in `optb/tree/`; grep the names)
| hop | where |
|---|---|
| argparse (5 flags, defaults idm, "", "", off, off) | `v8f_mini_train.py` build_parser, after `--wm-rsp-w` |
| FLAG_CHECKS entries (a flag without one is refused by the gate) | `v8f_mini_train.py` FLAG_CHECKS, after `wm_rsp_w` |
| make_config applies readout, probe, sha, freeze | `make_config`, line before `wm_rsp_w=` |
| config fields and refusals (fixed_k needs w > 0, freeze on, probe + 64-hex sha; probe without fixed_k refused) | `v8f_mini.py` V8FMiniConfig, `__post_init__` |
| probe load, sha256 check, K = 10, shape check; `w = RA.W / RA.sd`, `s_cal` | `v8f_mini.py` `load_rsp_probe`, called in `WMAux.__init__` after the stack is built |
| level-1 LevelConfig gets `rsp_readout="fixed_k"` | `WMAux.__init__` level-1 kwargs |
| buffers `rsp_probe_w` (float32), `rsp_probe_s` (float64), non-persistent | `HJEPAStack.set_rsp_probe` |
| loss: dispatch, two contrasts, endpoint read, constant s, `mean (G-1)^2` | `HJEPAStack.response_loss` -> `_response_loss_fixed_k`; added to `level/L1` by the existing block in `HJEPAStack.loss` |
| backward | unchanged `T["total"].backward()` in the WM-only loop |
| checkpoint start: all tensors incl. `wm.*`, strict, read back; strides compared | `load_wm_only_init(init_wm=True)` |
| freeze | `freeze_all_but_wm(model, freeze_proj)` before the optimiser |
| G-DVB | `bad_wm_rsp` (readout, contrasts 2, live weights vs the re-read file, s_cal), `bad_wm_freeze` (requires_grad of `wm.proj.*`, proj equals the checkpoint's), `bad_wm_only` (init record), `wm_rsp_readback` in `selfcheck.json` |

## Tests (CPU, one pytest at a time, `wait_commit.py 6 120` before each)
`stack/tests/test_v8f_optb.py`: **23 passed** (22 in one run of 141 s; the one test whose string match was wrong, `test_gate_refuses_fixed_k_without_freeze_or_without_init_wm`, was fixed and re-ran green). Content: defaults equal the earlier build; state-dict keys and values equal with and without the probe buffers; gradient reaches `predictors.0` and `act_embed` and reads EXACTLY 0.0 on `idm_heads`, `proj`, levels 2-3 and the adapter output; control with the flag off has no term key and reads 0.0; C9' (float64): matched constant gives loss < 1e-24 and gradient < 1e-12, B = 0 gives exactly 1.0, a doubled constant gives G = 0.5 and loss 0.25, history-carrying states and autocorrelated actions leave the identity at < 1e-24, delta scales out, an opposite world gives loss 4, geometry 1 call, 2 contrasts, 9 anchors, K 10, context 1, endpoint read (a W = 0.875 world pins it); config, sha-mismatch (one changed byte), wrong K and wrong width refusals; the trainer's own gate on the synthetic recipe for the RSP arm and the NULL arm (readback, `live.c3` pass, proj frozen, `init_wm` read back equal), refusals (no probe, idle probe flags, no freeze, no init, wrong sha, other strides, MR11 and MR12 shapes); a real 2-step run: `wm.proj` bit-equal to the checkpoint and more than 10 other `wm.*` tensors moved.
NOT run: the 15-file regression set, the launch-gate-reading set (`pin-edit-moves-profile-fingerprints`; no pin or `launch_gate.py` edit was made, so the fingerprints are expected unmoved: UNVERIFIED), the scripted MR11-MR14 mutation runs (the tests that must go RED are named in `HANDOFF.md`).

## MM steps (after the handoff's items 1-4 are done)
1. Commit by plumbing on a branch from `a13f6f16`: `optb.patch`, the pkg. 2. G-SUITE on the landed commit (not the dev box). 3. Ship R35 checkpoints + probes + tree to box 2; mint per-arm tokens with `optb_check.py gate` and a sentinel waiter whose baseline is taken before launch. 4. Pilot (0.8 h): stop if s/step > 1.0 or `max_memory_allocated` > 6.5 GB; the pilot also exercises the real gate on R35's trained WM (the float64 ladder reader). 5. The ten arms in the `AMEND_OPTB.md` order, WMI `CreateFlags 0x08000200` via `launch.ps1`, bank PRIVATE to HF with `HF_HUB_DISABLE_XET=1`.

## Cost (ESTIMATED): 14-16 h box 2 + 0.8 h pilot; read about 22 h dev-box CPU.

## MM rulings needed: see `AMEND_OPTB.md` section 6 (LR, seeds, NULL freeze, land base, shipping inputs).

## Manifest (all in ONE place, F: `F:/TanitAD-scratch/actrsp/optb/`; bank before the window ends)
`AMEND_OPTB.md`, `AMEND_OPTB.sha256`, `optb.patch`, `BUILD_OPTB.md`, `HANDOFF.md`, `tests_optb_run1.log`, `readout_K10_R35_s{0,1}.pt`, `patches/p1_trainer.py`, `patches/p2_gdvb.py`, `tree/` (edited export).

## Deletions (literal paths)
`F:/TanitAD-scratch/actrsp/optb/tmp` 706 MB (pytest basetemps, own), `F:/TanitAD-scratch/actrsp/optb/origf` (temporary originals, re-creatable from git), 2 `__pycache__` dirs under `optb/tree`. Kept: `tree/`.
