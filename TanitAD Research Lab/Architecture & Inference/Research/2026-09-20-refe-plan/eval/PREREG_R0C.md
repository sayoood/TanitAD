# PREREG: R0c -- anchor the EPO term to arm A's policy instead of the frozen base (IN FORCE 2026-10-15)

> ⭐ **IN FORCE. Finalised 2026-10-15 ~22:45 Berlin, BEFORE the first R0c step.**
> - **Source.** `raw/2026-10-15-rls1-adam-dose/PREREG_R0C_DRAFT.md`. **Sections 1-5 below are that draft, VERBATIM.** They were copied
>   from git by `make_prereg_r0c.py`, not retyped. Nothing in them changed after the decision; the draft's "DRAFT / NOT implemented"
>   line is historical, and this header records what changed since.
> - **PI authorisation, verbatim (2026-10-15):** *"run R0c, I dont have currently asecon pod or gpu aprt of my local computer with the
>   8gb gpu"*. This is the PI decision that section 5 asks for: about 3.5 A40-h, beyond RL-S1b's authorised budget.
> - **Slot: after the P2 seed replicate (P2-s1), BEFORE P6's GPU stages.** No code change is needed for it.
>   - The RL-S1 chain's queue (`refe/rls1_pod.sh` header and `queue_wait`) waits behind the replicate holding only `chain.queue`.
>   - It takes `chain.lock` within `RLS1_QUEUE_POLL_S` (5 s) once the replicate's chain exits. P6 is not "ahead" until it logs
>     `ZZP6_WAIT_OK`.
>   - P6's `wait_clear` (`refe/p6_pod.sh`) does not pass while `$DATA/rls1/chain.lock` is held, and it needs 3 idle polls of 10 s.
>   - ⇒ P6 starts after R0c's chain ends. Cost to P6: about 3.5-4 h of delay (ESTIMATED).
> - **Implementation (section 2's "trainer edit (NOT done)" is now DONE).**
>   - Code: `raw/2026-10-15-r0c-build/PATCH_1..7` + `test_rl_r0c.py`, and the chain wiring `raw/2026-10-15-r0c-build/chain/PATCH_8..12`.
>     They are applied in the commit that lands this file. The superseded `PATCH_10_*.on_94248ec.diff` is NOT applied.
>   - Argv: `raw/2026-10-15-r0c-build/R0c_ARGV.txt` = `eval/rls1c_arms.json` (profile `refe_rls1c`). The launch gate's PASS token binds
>     it with the commit and the data manifests.
> - **Section 3's checks.**
>   - `test_rl_r0c.py`: identity at step 0, a non-zero B gradient, an EXACT 0.0 on A, the mutations, a named default. Each item is mapped
>     in `raw/2026-10-15-r0c-build/MM_VERIFY.md`.
>   - Re-run on this exact tree: **test_rl_r0c 26 (torch 2.11) and 26 (2.7.1); test_rl_s1b 18 / 18; test_rl_epo 42; test_launch_gate --static 659; test_aux_s1_wiring 20; test_aux_g12_bf16 7 / 7 (2.7.1 with the cuBLAS env); test_rls1_chain 512; test_rls1_queue 134; test_flock_rw 59; test_s1aux_chain 385; test_p2rep_chain 353; test_p6_chain 341; test_p6_overlap 54; eval/test_rl_val_eval 32 -- every suite green; five of them need TMP on NTFS (on the exFAT drive they fail on git ownership / exec bits / symlinks, the known dev-box quirk; first-pass logs kept)** (logs: `raw/2026-10-15-r0c-final/`).
> - **Second control.** C0's banked score dumps were restored to the pod from `Sayood/tanitad-refe-rls1-c0` (`final/scores_*.npz`,
>   15 files, each sha256-verified against HF), so the `rlval_vs_C0.json` read runs.
>   - Recorded before launch: had C0 been missing, the run would NOT refuse. The chain logs `ZZRLS1_R0C_NO_SECOND_CONTROL`, and every
>     registered delta is against the arm's own A (section 2, guards).
> - **Tier and estimator, as section 4.** RL-val selection tier: one training seed, paired log-cluster bootstrap, 10,000 resamples,
>   seed 20260927. **A separated interval is NECESSARY, not sufficient** (`H-ESTIM-SEED-1`). An O1 is a candidate until its
>   replicate runs.
> - **From here on, nothing below may change without a dated amendment written before the read.**

## 1. Question and the ONE lever

R0 and R0b breached g5 on TTC against the control (B - A = -1.00 at R0 u1000; -0.57 / -0.70 at R0b u250 / u500). The same dumps decomposed against the start (MEASURED, `ab_vs_base.json`):
B - base TTC is -0.17 [-0.71, +0.38] (R0 u1000) and -0.03 [-0.60, +0.52] (R0b u500), both consistent with 0, while the BCE-only control gains +0.83 [+0.13, +1.58] / +0.67 [+0.03, +1.35] over the start. Arm A's selection policy sits at KL 1.14
from the base (J 0.828) and arm B's at 0.15-0.26 (J 0.847-0.865) (ESTIMATED, `late_probe.json`), because the EPO term's KL anchor pulls toward the FROZEN BASE and scales with lam.

**Lever:** `epo_loss`'s reference logits `sx_ref` (today `sx0`, the frozen base's, scorer_finetune.py `epo_term`/`train_losses`) become arm A's DETACHED logits of the same micro-batch (`--epo-ref armA`; default `base` = R0 exactly). Nothing else changes.

## 2. Arm

| field | value |
|---|---|
| arm | **R0c** (one A40 arm, ~3.5 A40-h by the section 10.3 cost model; its own arm A is the same-pass control, C0 is the banked second control) |
| argv | R0's banked argv (`raw/2026-10-14-rls1-r0-kill`, sha256 `24451b62...`) **+ `--epo-ref armA`**; `--epo-lam 1.0`, `--epo-beta 0.1` and every other flag as R0 (so R0 vs R0c differs in the reference ONLY) |
| guards | K3 / g5 / g3 / g4 / K0-K6 AS REGISTERED in `eval/PREREG_RL_S1.md` sections 4-5, unchanged; the control for every delta is the arm's own A; RL-val is the selection tier |
| trainer edit (NOT done) | a new flag `epo_ref` in `EPO_DESTS`, `epo_refusals` and `build_parser`; `train_losses` passes `sxA.detach()` instead of `sx0` to `epo_term` when `epo_ref == "armA"`; a G-DVB entry reading the BUILT reference |

## 3. Reachability and controls that must read known values BEFORE launch (the 2026-09-10/11 class)

1. **Trace flag -> loss:** argparse -> namespace -> `train_losses` argument -> `epo_loss(..., sx_ref, ...)` -> `logpi0` -> `kl` -> `backward()`, `file:line` at each hop.
2. **Identity at step 0:** with A = B = base, R0c and R0 give bit-identical loss, KL (exactly 0) and gradient on the same micro-batch (EXACT 0.0 / bit equality). This also proves the flag changes nothing it should not.
3. **Non-zero where it must be:** after one B-only update (B != A), the gradient of the KL term w.r.t. B is non-zero and DIFFERENT from the base-anchored one; A's parameters receive NO gradient from it (A detached; max abs grad on A's parameters EXACTLY 0.0 from the EPO term).
4. **Mutation:** replace `sxA.detach()` by `sxA` -> A receives an EPO gradient -> the check in 3 must go RED; replace it by `sx0` -> the check in 2 passes but the check in 3 must go RED.
5. **Default:** `--epo-ref` must be NAMED in the arm's argv (a default of `base` silently reproduces R0).
6. `launch_gate.py` PASS token bound to commit, argv sha256 and data manifests; a flag without a G-DVB entry is refused by the gate.

## 4. Predictions and outcomes committed in advance (reads: u250 / u500 / u750 / u1000, RL-val, B - A and B - base, paired log-cluster bootstrap 10,000 / seed 20260927, sub-scores x100)

The mechanism claims: (M1) B - A TTC rises from about -1.0 toward B - base's -0.17 because B is no longer pinned to the start; (M2) KL(pi_A || pi_B) stays small (logged `epo_kl` now measures A-to-B, expected well below the 0.15-0.26 base-to-B values) while PDMS B - A stays positive.

| outcome | condition (RL-val, as read) | meaning | next lever (Rule Zero) |
|---|---|---|---|
| **O1 ADOPT-candidate** | g3-g5 hold at every check AND PDMS B - A lower bound > 0 at u500 and u1000 AND the section 4 bar of `PREREG_RL_S1.md` | the reference was the defect; EPO's ascent does not harm TTC once it is not pinned to the start | the replicate arm (same flags, other seed) per H-ESTIM-SEED-1 BEFORE any claim; then the full-navtest endpoint |
| **O2 safe but flat** | g5 holds at every check, PDMS B - A lower bound <= 0 | anchoring to A leaves no ascent budget (B ~ A) | one pre-registered pair on the A-anchored axis: `--epo-beta` 0.1 -> 0.03 (anchor weaker, ascent unchanged); PI decision on budget |
| **O3 breached with a small KL** | g5 breached at two consecutive checks AND mean `epo_kl` (A to B) <= 0.2 at the breach | the EPO ascent ITSELF lowers TTC relative to A, whatever the anchor: the reference hypothesis is REFUTED | EPO restricted to the progress heads' parameters with a frozen trunk. The banked displacement lives 99 % in `score_dec` and 0.0 % in `score_head`, so this is a different regime, and its fraction of R0's J gain on cached trunk features is a CPU job to run FIRST (PI decision for the parameter-group change) |
| **O4 breached with a large KL** | g5 breached AND `epo_kl` > 0.2 | B drifted from A faster than the anchor holds it | `--epo-beta` up (0.1 -> 0.3), one pair; K3 is the guard |
| **O5 other kill** | K0-K2, K4-K6 or an NC / DAC bound | per the section 5 table | per section 5 |

A result that is separated on the paired interval is NECESSARY, not sufficient (H-ESTIM-SEED-1): the one training seed per arm stays a selection-tier read until the replicate runs. Inference replicates (the iCEM sampler) do not apply to this scorer-only arm, which uses the argmax rule.

## 5. Evidence and limits of this draft

* The decomposition (section 1) is MEASURED on banked RL-val dumps; the KL / J values are ESTIMATED (substituted sctx, V 0.996-0.997 at each state; 256-set pool). The mechanism is a hypothesis with a sign test, not a result.
* In the closed-loop simulation (50 updates) the A-anchor cannot be distinguished from the base anchor: A has barely left the base by then (identity at step 0, section 3 item 2). Only the A40 arm discriminates; there is no cheaper CPU test of the reference choice.
* Cost and authority: ~3.5 A40-h and a trainer edit. RL-S1b's authorised budget stops at the R0b kill (`PREREG_RL_S1.md` 10.3), so this needs a PI decision.
