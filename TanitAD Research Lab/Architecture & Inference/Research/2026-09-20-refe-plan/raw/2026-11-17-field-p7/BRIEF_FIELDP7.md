# BRIEF: make the field-loss arm (D-AUX-4) launch-ready on host P7: finalise the pre-registration, prove reachability, and set w_field. 0-GPU. Agent: Sonnet. MM, 2026-11-17

Read `Project Steering/AGENT_OPERATING_STANDARD.md` first.
- **Binding:** stage, never commit or push; end with a manifest.
- **Hold:** `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/field_p7/`. **Scratch:** `F:/tanitad_scratch/fieldp7/`.
- **Code:** read it from the mirror `C:/Users/Admin/tanitad-push` (`git fetch -q origin agent/arch-inf-20260803`; use the commit sha, since FETCH_HEAD can move). `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.

## Why
- **The P7 head-veto read was VH-NULL** (`PK/raw/2026-11-16-p7-read/veto/P7_HEADVETO_READ.md`): P7's drivable-area head cannot recover DAC failures.
- **But the headroom is real:** an oracle DAC veto would gain +1.64 [+0.90, +2.53] PDMS.
- **The direct lever** is the pre-registered field loss: a hinge on the 64 proposals' footprint corners against the ground-truth SDF.
  - Draft: `PK/eval/PREREG_AUX_FIELD_LOSS_DRAFT.md`. The PI approved D-AUX-4 on 2026-10-12.
  - Module: `refe/field_loss.py` + `refe/test_field_loss.py`.
- **The gap:** the draft names host P2 and leaves its blanks open. The current best checkpoint is P7: `F:/TanitAD_p7read/run/model_final.pt`, sha256 `2613fc17…`, PDMS 91.07. P5-C (C0/C1) occupies the pod until about 2026-11-19.

## Deliverables
1. **`PREREG_AUX_FIELD_LOSS_P7_DRAFT.md`**, a NEW file in the hold. Do not edit the old draft. Host H-P7: graft on P7's model_final with two arms:
   - F, with `--w-field <w>`;
   - C, with `--w-field 0.0`;
   - same init, label bank, steps and seed, plus a replicate plan (inference and training seed: CLAUDE.md, `H-ESTIM-SEED-1`).
   - Keep the draft's fixed values (margin 0.25 m, 5 Hz to 10 Hz, footprint, reduction) and its K1..K7 kill rules.
   - Fill every blank: the module sha256; the trainer commit; the label-store path and its S0-gate evidence (find it; if S0 never passed, say so: that is a blocker); steps (match P7's recipe from its `config.json`); the cost in L40-h (ESTIMATED, A40-h × 0.863, stated as such).
   - Commit both outcomes in advance. Report the four metric families plus the STRATEGIC family (`eval/instruments/strategic_family.py`, reported).
2. **Reachability from the actual caller** (CLAUDE.md, "BUILT, TESTED, AND UNREACHABLE"):
   - Trace `--w-field` from argparse, to the namespace, to the loss term, to `.backward()` in the trainer P7 was trained with (find it from P7's `config.json` / argv). Give the `file:line` at each hop.
   - On CPU, with real data from the label store, take 2-4 batches from P7's checkpoint and show:
     - the loss term is finite and non-zero with w > 0, and the proposal-head parameters get a non-zero gradient from it;
     - with `--w-field 0.0` the term's gradient contribution is exactly 0.0;
     - deleting the line that adds the term turns the check RED.
   - If the flag is not wired in that trainer, say so precisely, and write the minimal wiring as a patch file in the hold. Do not apply it.
3. **w_field by the draft's gradient-norm-balance procedure**, on CPU, never tuned on PDMS. Record the per-batch norms and the frozen value.
4. **Launch-gate gaps:** list exactly which G-DVB / G-LIVE entries the arm needs in `stack/scripts/launch_gate.py` or the REFe profile, and whether they exist. Do not edit the gate.
5. **`HANDOFF.md`** (≤ 2k tokens): what is ready, what blocks the launch, and the exact PI ask (spend in L40-h, and which box).

Dev box, CPU only (`CUDA_VISIBLE_DEVICES=-1`), serial, TMP in scratch.
- Free-commit guard: wait while it reads below 6 GB, and parse it as an integer, since the locale uses a decimal comma.
- C: is below 5 GB, so put nothing on C:.
- If the CPU checks would exceed ~2 CPU-h, STOP and report a plan.

## ⛔ Do not
- Train. Use the GPU, the pod (C0/C1 train there) or box 2.
- Edit existing or pinned files, `launch_gate.py`, or the old draft.
- Commit, land or push.
- Print or copy secrets.
- Delete outside your scratch.
- Exceed ~150k context.

**Report:** at most 300 words, plus a manifest.
