# NOTES: PREREG_V8F_DAG_ORC draft (2026-10-27): design choices, what I checked, and what is left for the Master Mind

Scope: a DRAFT only. I made no code changes, did not contact box 2, launched nothing and touched no git; everything ran on CPU.
Every repo text was read from the mirror: `git -C C:/Users/Admin/tanitad-push show d64750bd:<path>`.

## What I checked, and how (MEASURED unless stated)

| fact | probe | result |
|---|---|---|
| DAG2-0 is a valid one-lever control | `v8f-dagger2/M2b_s{0,1}_dag20/argv.json` | DAG2's argv with `--dagger-p 0` (no bank flags) and `--ego-perturb-p 0`. A = this argv + `--ego-perturb-p 0.3` + the five law flags + `--out`, which is the ORC / ORC-0 pair rule (ORC A3). **Reused; no new control** |
| ORC's exact law flags | diff of `v8f-orecover/M2b_s*_orc{,0}/argv.json` | differ only in `--ego-perturb-p 0.3/0`, `--ego-perturb-vscale 0.5 1.0`, `--ego-perturb-vmax 8.0`, `--ego-perturb-adrop 1.5`, `--ego-perturb-yawrate-std 2.0`, `--ego-perturb-seed s`, `--out` |
| RF0's pair is a same-argv replicate | diff of DAG-0 vs ORC-0 argv | differ only in `--out` and `--dagger-p 0` (default-off), same seed; builds `b43c7a9a` vs `853f2fc3` (gate tokens) |
| RF0 is computable at 0 GPU | `ls` of `v8f-dagger-t1/`, `v8f-orecover-t1/` | causal + logged dumps for DAG0 and ORC0 at both seeds exist, with `PINS.txt`. **Values not read** |
| DAG2-0's build and init | `launch_gate_token.json` | commit `1d4f4fbb…`; init sha256 `d186fd0b…` (s0) and `1d12ec2e…` (s1); control ckpt `603b9151…` and `f4234ac1…` (`SHA256SUMS.txt`) |
| `--seed` semantics | `v8f_mini_train.py` @ `d64750bd` | it seeds torch and numpy, `ChunkedSampler` and the per-step `torch.manual_seed(seed*1_000_003+step)`. It takes no part in the init when `--init-ckpt` is given. The perturbation has its own `--ego-perturb-seed`. No cudnn-deterministic flag is set |
| run times | `chain_v8f_dag2.out`; `summary.json` `seconds` | about 43 min per run including the T0 eval; banks 51 / 54 min; ORC vs ORC-0 within 9 s of each other |
| eligibility is lever-independent | `ego_perturb_eligible` in the summaries | 82,457 / 82,166 for DAG2-0, ORC and ORC-0 alike |
| nav dead-reckoning | `v8f_t1_eval.py` `WindowState` @ `d64750bd` (lines 156–216) | the token and validity are held from t0; `row[3]` (distance) is decremented by own arc / 100 and clipped to [0, 3]. `--nav logged` exists (`NAV_MODES`) |
| Read A's TAC rows | `PREREG_V8F_DAG_REREAD.md` line 55 | *"the 3-way agreement of the executed track with the human's"*, i.e. not head F1 |
| DAG-OACC reads the crawl rows | `PREREG_V8F_DAG_OACC.md` line 45 | beside: *"DAG-OACC − DAG on every Read-A row"* |

**Not opened:** any `eval_*` file in `v8f-dagger2/`, and any DAG-ORC data (none exists).

## The design choices

1. **The lever's law and dose are unchanged from ORC (p = 0.3).**
   * It is exactly the §7 lever that DAGGER2's committed next names.
   * The M2b evidence (curvature and yaw-rate separated-better at both seeds) was produced at this law.
2. **B at p = 0.1 is the pre-registered dose fallback.**
   * ORC broke R-T0 at both seeds on M2b, by 1.4× and 1.8× the bar. A HARMS at p = 0.3 is therefore the single most likely failure.
   * Training B in the same queue costs about 1.5 h and hands HARMS straight to the next arm. That follows Rule Zero 4 and "have the next
     arm ready before the run ends".
   * The PI may drop B; the table then queues it on HARMS.
3. **The replicate takes a different `--seed`** (10 / 11) **with the same init.** The reasons are in PREREG §1. The point that decides it:
   a same-seed rerun can be bit-identical, and then it reads a floor of 0.
4. **The bar is 2 × max(F_r, RF0_r, RF_r), with the max taken over seeds.** F_r = 0.1207 m for ADE6 and 0 elsewhere, so non-ADE rows read
   2 × max(RF0_r, RF_r).
   * I rejected using DAG's cross-seed spread as the row-equivalent F.
   * It is the literal analogue of F_M2b, but it includes init variance that a within-init contrast does not carry. On yaw-rate it reads
     8.70 deg/s (from Read A's points), so bar 17.4.
   * With that bar R-MOVE at s1 would be impossible, because DAG's own excess is +8.66. That is a design that can only be refuted.
5. **R-NEW reads X − CTRL**, the lever's own contrast, like every other guard. **Adoption reads X − DAG** (R-KEEP and R-NOBASE).
   * This separates "does the lever harm" from "may X replace the base".
   * DAGGER2 mixed the two, and it would also have adopted a DAG2 that was separated-worse than DAG on ADE6 at s0.
   * Illustration only; it changes no DAGGER2 outcome. DAGGER2's firing row has DAG2 − DAG2-0 yaw-rate +1.36 (n.s.), so under R-NEW's form it
     would not fire whatever the floor. ⚠ I designed R-NEW after seeing that row. The motivation is the instrument gap that
     RESULT_V8F_DAGGER2 §3.4 and §4 name, and it applies to a new arm only.
6. **The adoption handicap is disclosed, not hidden.** DAG2-0 − DAG ADE6 is about +0.27 / +0.13 m (by additivity of the published points).
   * So A probably needs to recover more than it costs to pass R-KEEP. ORC cost +0.11 / +0.04 m on M2b.
   * The likely good outcome is therefore "FIXED / PARTIAL, NOT ADOPTED → DAG-ORC-SWAP".
   * ⚠ **MM decision:** if adoption in this very read matters more than clean attribution, re-base the pair on DAG2's argv: A' = DAG2 +
     perturbation, control DAG2 (banked). DAG2 − DAG is +0.043 / −0.031.
     * It needs a gate check that `--dagger-p 0.1` and `--ego-perturb-p 0.3` compose. I did not verify that the trainer accepts both.
     * The brief fixed DAG2-0's argv, so I kept it.
7. **NEXT-CRAWL puts the 0-GPU nav probe N0 first.** Nav is the one loop channel never intervened on. The diagnosis tested the scene,
   the seams and m = 30, but not nav.
   * The dead-reckoner's held t0 token is a concrete, checkable mechanism.
   * Then, by measured effect: the along-track profile (the post-map cut the crawl by 36–72 %) → COMPOSE with DAG-OACC if its read shows the
     crawl rows moved, else LAUNCH-INTENT.
   * ⭐ **N0 can run before DAG-ORC** and may save about 3–4.5 h of box 2 (launch precondition 3).

## Cheapest unblocked next steps (0 GPU; none run here, by scope)

1. **N0**: 2 T1-kin rollouts of DAG with `--nav logged` (about 10 min on the dev-box CPU), read on C.
2. **P-0**: RF0 from the banked DAG-0 and ORC-0 causal dumps with Read A's reader. The reader runs only; no rollout.
3. **The reader** `dag_orc_read.py`, from `dag2_read.py` plus floors, adoption, the cascade, and the T0 families via `orc_families.py`.
4. **The box-2 chain**: the queue lines for A, REP and B, through the gates.

## Proposed register row (for the MM to land with the promotion)

`H-V8F-DAG-ORC-1` (2026-10-27) | Fine-tuning DAG_s for 3,000 steps with O-RECOVER's ego-state perturbation (p 0.3; v0 × U(0.5, 1.0),
a0 − U(0, 1.5) m/s² where v0 < 8 m/s; yaw rate + N(0, 2.0 deg/s); targets unchanged) moves DAG's crawl rows at T1-kin.
* Every row of C (s0: curvature, L2, M1; s1: + yaw-rate) clears 2 × max(RF0, RF) for A − DAG2-0 at both seeds (R-MOVE), with R-CRAWL
  deciding FIXED vs PARTIAL.
* Guards: R-TLAT / R-TLON / R-ADE not separated-worse, R-T0 ≤ +0.0604 m, R-NEW.
* Adoption: R-KEEP (A − DAG ADE6 point ≤ +0.0604 m) and R-NOBASE.
* On HARMS, B (p 0.1) is read under the same table.

Status: PRE-REGISTERED (DRAFT), NOT RUN. Evidence class: HYPOTHESIS. Pointer: `Project Steering/PREREG_V8F_DAG_ORC.md`.

## Files

| file | sha256 (prefix) |
|---|---|
| `enumerate_cells.py` | `468b0ae3a387f88c…` |
| `enumerate_cells.out` | `2f4afaf807465caa…` (exit 0) |
| `enumerate_cells_mutation.txt` | 13 / 13 mutants exit 1 |
