# BRIEF: the STRATEGIC-family instrument for REFe navtest reads. Agent: Sonnet. MM, 2026-11-17

Read `Project Steering/AGENT_OPERATING_STANDARD.md` first.
- **Binding:** stage, never commit or push; end with a manifest.
- **Hold:** `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/strat_family/`. **Scratch:** `F:/tanitad_scratch/stratfam/`.
- **Code:** read it from the mirror `C:/Users/Admin/tanitad-push` (`git fetch -q origin agent/arch-inf-20260803`; FETCH_HEAD; `core.autocrlf=true`, so use `git show`). `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.

## Why
- **The PI's ruling of 2026-11-16 (verbatim):** *"strategic is not paused anymore"*. Every eval reports the STRATEGIC family in full again: strategic decision quality, plus goal and route-setting quality. "PAUSED" is no longer acceptable.
- **The gap:** both REFe family instruments still hard-code PAUSED:
  - `PK/eval/instruments/families_run.py:34`;
  - `PK/eval/p5c/p5c_families.py`, through `p5c_read.STRATEGIC`.
- **The deadline:** P5-C's C0 and C1 (training now) are read in about a day, and they need the family.
- **The constraints:**
  - ⛔ Do NOT edit those files: they are pinned and frozen.
  - Build a NEW instrument that runs beside them.
- **What a REFe arm has to work with:** no strategic layer of its own. On navtest, the strategic signals are:
  - the input `driving_command` (left / straight / right), which is the route;
  - the plan's long-horizon goal (its 4 s endpoint) against the logged future.

## Deliverable: `PK/eval/instruments/strategic_family.py` (new; staged in the hold under `land/`), plus tests and a first read
1. **S-DEC, the strategic decision.**
   - Command adherence, reusing the definition in `PK/eval/command_adherence.py` (import it; do not copy it).
   - Per-command confusion (commanded vs executed direction) and per-class adherence, each with n.
2. **S-GOAL, goal and route setting.**
   - 4 s endpoint displacement to the logged endpoint.
   - Endpoint heading error.
   - Route-progress ratio (plan arc length / logged arc length).
   - Endpoint lateral offset, signed with respect to the commanded side.
   - All broken down per command class.
3. **Estimator.** A paired log-cluster bootstrap between two arms on the same tokens (`proxy_eval.boot` or `taniteval/ci.py`; 10k; seed 20260927), with n, so that "separated" means what it does in the other families.
   - ⛔ No pooled composite score.
   - Tier stamp: T1 (self-action open loop: the plan is the model's own).
4. **Controls, in the tests** (each must read its known value):
   - the logged-future trajectory as an arm gives endpoint error exactly 0;
   - a plan mirrored laterally must lower adherence on left/right tokens (deliberate regression; the check must go RED if mirroring is a no-op);
   - comparing an arm against itself gives a delta of exactly 0, with CI [0, 0].
5. **First use.** Run P6 vs P7 on navtest (12,146 tokens). The P7 read work dir is `E:/Projects/TanitAD-artifacts/p7_full_read/`; it already holds `command_adherence_p7_full.json`. Find the P6 and P7 plan dumps the families stage used, from `CHAIN.log`, the `families/` folder and `p7_read_chain.py` on the tip.
   - Report in `STRATFAM_P6_P7.md`, with each number's evidence class, estimator, n and artifact path.
   - The banked P7 adherence (0.8805, against a floor of 0.8841) must be reproduced exactly, or the difference explained.
6. **Integration note.** Write `INTEGRATION.md`: exactly how the P5-C read (`PK/eval/p5c/`) would call this beside `p5c_families.py`, without editing pinned files. State that the family is reported, not a decision bar, because PREREG_P5C froze its bars before the ruling.
7. **Close-out.** `HANDOFF.md` (≤ 2k tokens).

## ⛔ Do not
- Edit existing or pinned instruments, frozen PREREG or SPEC files.
- Add a decision bar to P5-C.
- Commit, land or push. Train.
- Use the GPU, the pod (C0 trains there) or box 2.
- Run anything but CPU work (`CUDA_VISIBLE_DEVICES=-1`), serially, with TMP in scratch and a free-commit guard (wait while < 4 GB).
- Print or copy secrets.
- Delete outside your scratch.
- Exceed ~150k context.
- **Tests:** the new tests must pass with scipy blocked, and any `subprocess` text call needs an explicit `encoding=`.

**Report:** at most 300 words, plus a manifest.
