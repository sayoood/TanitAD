# HANDOFF: LEADSUP lead read (section 3a of PREREG_V8F_PROOF_ADD24), dev box, 2026-11-15

State: DONE on the dev box. Nothing touched the pod, box 2, git or a GPU. Folder `F:/TanitAD-scratch/v8proof/spatial/leadsup/read/`. All decisions beyond the card: `LEADSUP_READ_DEFS.md` (written before any real-arm number).

## UPDATE 2026-11-15 (lead decision M3-BC, before any arm number; see DEFS "Decision M3-BC")
* Gated M3 slot skill = R3 over the bias-corrected copy (head gap + b, b cross-fitted over the other eval folds). The raw-copy skill (card form) is printed beside it and names nothing. `SUP-SLOT-BEATS-ROLLOUT` unchanged. Head gap offset on the scored windows printed per arm (`M3.gap_offset_m`). k3 unchanged in construction; its slot number is the bias-corrected one. The train block carries the same two skills.
* Tests: 15 passed (two new: offset -20 m with no state information reads raw skill above 0.1 and bias-corrected at most 0.02, k3 passes; an informative head with the offset reads bias-corrected skill above 0 and separated).
* PLUMB rerun (`tmp/plumb/`, 70 s): all controls PASS, k3 slot -0.009 (raw copy 0.82, information only); head gap offset -22.84 m; eval bias-corrected R3 skill -0.015 [-0.068, +0.034]. Numbers of the earlier text below that mention k3 FAIL for the random head describe the state before this update. Packs and shas of pod_lead_apply.py unchanged; leadsup_read.py sha changed (re-hash before quoting).
* Scratch: `tmp/pt/c` was recreated by the rerun (pytest basetemp); sizes otherwise as below.

## Files (all in the folder above)
| file | what | sha256 (at hand-off) |
|---|---|---|
| `pod_lead_apply.py` | runs ON THE POD: head of a checkpoint applied to a state file; writes `<out>`, `<out>.head.npz`, `<out>.json`; refuses a checkpoint without a head and a state file of another checkpoint | c9b1ecef2302100bbd52bfb37ff6ddbdb181c61dbf555a8c39cc94bd9eb97d3c |
| `leadsup_read.py` | the reader: `pack`, `run`; M1 to M4, train probe read, paired M1, controls k1 to k4 + a1 to a5, names, `RESULT_LEADSUP_T0.json`, `VERDICT_LEADSUP_T0.md` | 54aa53a9f278a319dfc9f4e1e13638e7cfbb9f7664bcd45d10150fc499d33dc6 |
| `run_leadsup_read.sh` | the lead's one command; the pod-side steps are in its header comments | c8f187a6912b7118b400344e274dc1fdbc7baafd294236a939e231662e91193c |
| `test_leadsup_read.py`, `make_tiny_ckpt.py` | 13 tests; builds a tiny patched-tree checkpoint (random head) | |
| `eval_pack.npz`, `train_pack.npz` | frozen windows, labels and train-probe truth (built by `leadsup_read.py pack`, sources and sha256 in `meta`) | 6f5aa3c1..., 9cfff95f... |
| `LEADSUP_READ_DEFS.md`, `HANDOFF.md` | definitions, this file | |

## Verified (file paths)
* **Tests**: 13 passed (`python -m pytest -q test_leadsup_read.py --basetemp .../read/tmp/pt/<new>`; 97 s). They cover: the pod script (subprocess on `C:/Users/Admin/v8proof_c/leadsup/tree_patch`) and the dev forward agree on a real `LeadSupHead` with random weights saved as a tiny checkpoint (max diff gap 5.7e-6 m, closing 1.2e-6 m/s on the real layout); refusal of a checkpoint without a head and of a state file of another checkpoint; a planted head (a head constructed to read states that encode the label gap, closing speed and presence of their frame, on the real 1,948 windows) read back with M1 R2 and M2 R2 above 0.9, R3 skill above 0 and separated; k1 to k4 pass on the planted case and fail under their deliberate regressions (k1 shifted labels / wrong reference; k2 one tensor perturbed; k3 identity permutation; k4 a head read at end frame 5); a failing control (k2) blocks every name; a clean two-seed run names `SUP-READS-CLOSING` and `SUP-SLOT-IMAGINES` for LS and "not named" for the others; input records are checked by content; the train read on a small planted probe set; every name of section 5 owned by the reader on synthetic numbers (closing, slot, beats-rollout, gradient helps/hurts/equal with the floor, reading with the earlier cards, six arms).
* **Plumbing run (item 5)**: `tmp/plumb/RESULT_LEADSUP_T0.json`, `tmp/plumb/VERDICT_LEADSUP_T0.md` (tag PLUMB_s0, random head, `G_s0_eval.npz` and `G_s0_train.npz`, B 2000, 83 s, plus the `.sh` path in `tmp/plumb_sh`, and a corrupted record stopped by the `.sh` in `tmp/plumb_bad`). Controls: k1 PASS (0.7499068 = banked, within 1e-6), a1 a2 a3 a4 a5 PASS, k2 PASS (eval and train), k4 PASS, **k3 FAIL, expected for this head**: the random head's gap is offset -22.6 m on the scored windows, so R3's intercept alone gives skill 0.82 after the shuffle. The failure shows that no name can be written from a head with that offset, and that the control reports the offset. No number of the plumbing run is a result.
* Window assembly on the real layout: 1,948 windows, 429 scored, 413 with a label history, 48 scored episodes, 139 episodes in all; frames 0 to 4 of every window present and finite in `G_s0_eval.npz`; train layout of `G_s0_train.npz` equals the pack (121,166 rows, 1,200 clips).

## Not verified
* No real arm exists yet: no real head, no real number. The behaviour of k3 on a trained head is untested (see the risk below). The one-frame probe (`s0_read.fit_probe`) ran in the plumbing run only; the tests inject a ridge. The banked-rollout comparison ran against `OOF_RB_G_s0_R1.npz` (stand-in); LS, LSH, LSD have no banked rollout of their own, see DEFS R7.
* Controls k5, k6, k7 and the cost read (section 3b) are the lead's. The checkpoint sha256 of every arm is in `RESULT_LEADSUP_T0.json` (`arms.<ARM>.ckpt_sha256`) for k6.

## Decisions for the lead to look at before the first real number (all in DEFS)
1. **k3 permutes among the scored windows** (K3). Over all windows a perfect head fails the card's bar (R3 skill +0.43, from an offset of -12.5 m: windows without a lead put gap near 0 onto scored windows and R3's intercept removes it). Risk that remains: a trained head with a gap offset on scored windows may still fail k3; then no name is written, and the control prints the offset and three more permutations. Whether to loosen is the lead's call, not mine.
2. Banked rollout read for LS/LSH/LSD = the G arm of the same seed (R7). 3. Eval presence = bank `lead_present`, train head label = `lead_valid` (R8). 4. M1/M2 on the 413 windows with a closing label, M3 on all 429. 5. Window states from the arm's float16 eval state file (not the latents file).

## Exact commands for the lead
Pod, per arm (header of `run_leadsup_read.sh` has them verbatim): put `pod_lead_apply.py` at `/workspace/v8proof/probe/`; extraction as in step 2 of `spatial/s1/gw_read_arm_q.sh` with queue `leadsup_a|b|c`, tree `tree_leadsup`; then
`for sp in eval train; do /workspace/venv/bin/python /workspace/v8proof/probe/pod_lead_apply.py --tree /workspace/v8proof/tree_leadsup --ckpt /workspace/v8proof/runs/<Q>/<ARM>/ckpt_last.pt --states states/<ARM>_$sp.npz --out states/<ARM>_${sp}_lead.npz; done`
Pull the 10 files per arm (4 state files, 3 per lead output) to `.../leadsup/read/in/` (about 175 MB per arm, F: has about 7 GB). Then:
`bash F:/TanitAD-scratch/v8proof/spatial/leadsup/read/run_leadsup_read.sh` (about 1.5 min per arm; exit 0 = controls pass, 2 = a control failed, 3 = a file does not match its record). Results in `.../read/out/`.

## Scratch (nothing deleted; deletes were refused here; safe to remove, regenerable)
`F:/TanitAD-scratch/v8proof/spatial/leadsup/read/tmp/` **372 MB (above the 300 MB I was given; pytest basetemps cannot be removed from here)**: `pt/a` 103 MB, `pt/b` 114 MB, `pt/c` 138 MB (pytest runs, planted state files and checkpoints; `pt/a` and `pt/b` are safe to remove now), `plumb/` 19 MB (plumbing inputs and outputs, keep as evidence), `plumb_b200` 24 KB, `plumb_sh`, `plumb_bad` small, `try_build.py`, `patch1-5.py`, `dbg_k3.py` (throwaway scripts). Nothing written on C: by me (pytest `--basetemp` was on F:).
