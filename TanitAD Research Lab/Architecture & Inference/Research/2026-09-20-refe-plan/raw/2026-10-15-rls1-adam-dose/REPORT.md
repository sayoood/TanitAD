# H-ADAM: did AdamW void R0b's quarter dose? No, and what the data points to instead (report)

Written by the H-ADAM diagnosis agent (Sonnet) on 2026-10-15. The agent's harness refused the file write, so the Master Mind banked it from the hand-back.
- Pre-registration: `PREREG_H_ADAM.md`. Its sha256 sits in `PREREG_H_ADAM.sha256`, written before any measurement, with Amendments 1 and 2 dated inside it.
- Full tables: `RESULTS_SUMMARY.json`.

## Verdict: H-ADAM FAILS. The dose was NOT voided by the optimiser

| quantity | value | class |
|---|---|---|
| S, the EPO-attributable update (quarter over full dose) at the init gradient, open loop | **0.327 ± 0.009** (regime P, K=50, 3 streams) | ESTIMATED |
| the EPO/BCE gradient ratio | 0.33 at full dose, 0.083 at quarter dose; no tensor reaches 1 | ESTIMATED |
| rho, how far the quarter dose moved the parameters relative to the full dose | **0.52 at u250, 0.55 at u500** (floor-corrected), so half as far | MEASURED, banked scorer states |
| S over 50 updates of the trainer's own step, closed loop | **0.56**, which reproduces rho. The open-loop/closed-loop gap is the closed loop itself. With the KL anchor off, it is 0.43 | ESTIMATED |

**Two defects this package catches in itself.**
- The suggested norm-ratio statistic reads about 1.0 and would have said HOLDS. Control C3 shows it reads **1.000 when the true S is 0.25**, so it is not admissible.
- A first open-loop pass had an 8x gradient-scale bug (mean vs sum). It was fixed and kept as `adam_sim_WRONGSCALE_mean_not_sum.json`.

**Controls C1-C8 pass.** One table cell the pre-registration lacked, (LOW, MID), was read as FAILS-as-mechanism and logged in Amendment 2.

**The substituted context.** Camera pixels are not on the dev box, so the trunk context is a cache built from the epoch-15 snapshot. On 62 RL-val frames it
reproduces the banked base logits at within-scene r **0.997** (swapped-context control 0.778).

## What the data points to: the g5 gap is mostly the CONTROL's gain (MEASURED; re-verified)

`ab_vs_base.py` decomposes B - A = (B - base) - (A - base) on the banked RL-val dumps. The estimator is the paired log-cluster bootstrap (10,000 resamples,
seed 20260927), via the attribution package's own functions. Columns are TTC deltas x100:

| dump | B - A (the g5 read) | B - base | A - base |
|---|---|---|---|
| R0 u250 | -1.23 [-1.83, -0.62] | -0.47 [-1.04, +0.10] | +0.77 [+0.22, +1.34] |
| R0 u500 | -0.63 [-1.35, +0.06] | -0.03 [-0.57, +0.48] | +0.60 [-0.04, +1.29] |
| R0 u1000 | **-1.00 [-1.73, -0.30]** | **-0.17 [-0.71, +0.38]** | **+0.83 [+0.13, +1.58]** |
| R0b u500 | -0.70 [-1.40, -0.13] | -0.03 [-0.60, +0.52] | +0.67 [+0.03, +1.35] |
| C0 u1000 | -0.03 | +0.80 [+0.09, +1.55] | +0.83 [+0.13, +1.58] |

At R0 u1000, about **83 % of the -1.00 is the BCE-only arm's own TTC gain over the start**. The EPO arm is not separated from the start (-0.17). R0 u1000's
PDMS B - base is +0.59 [+0.14, +1.02].

**The KL measurements (ESTIMATED, `late_probe.json`).**
- Arm A's selection policy drifts to KL **1.14** from the base. Arm B is held at **0.15-0.26** by EPO's KL anchor, which points at the **frozen base**.
- **Mechanism, a HYPOTHESIS:** B is pinned near the start and cannot share the TTC gain that BCE fine-tuning gives A.

**Master Mind re-verification (2026-10-15 ~03:05 Berlin, dev box).** `ab_vs_base.py` was re-run from scratch with `C:/Users/Admin/venvs/tanitad`. The
result is **IDENTICAL** to the banked `ab_vs_base.json`, ending `ZZABBASE_OK`.

⚠️ g5 is registered against the control, and it stays that way. This decomposition explains the breach; it does not excuse it.

## Next lever: R0c, EPO anchored to arm A's (detached) policy instead of the frozen base

`PREREG_R0C_DRAFT.md` is a single-lever draft. Its pieces:
- **argv:** R0's argv plus `--epo-ref armA`, with all guards as registered.
- **Reachability and identity controls:** at step 0, R0c is bit-identical to R0. A receives exactly 0.0 gradient from the KL term. Mutations go RED.
- **Outcomes O1-O5,** each committed with its next lever.

**What blocks it:**
- the trainer flag (not implemented);
- about **3.5 A40-h**, which is beyond RL-S1b's authorised budget (it stopped at the R0b kill, `PREREG_RL_S1.md` 10.3).

So it needs a **PI decision**. The draft also says plainly that no CPU test can discriminate the reference choice. Over 50 simulated updates, A has barely
left the base. Only an A40 arm discriminates.

## Files

| file | content |
|---|---|
| `PREREG_H_ADAM.md`, `.sha256` | the pre-registration |
| `RESULTS_SUMMARY.json` | the full tables |
| `adam_common.py`, `adam_sim.py` (.json), `adam_closed.py`, `closed_stats.py` (.json), `closed_beta0_stats.py` (.json) | the open- and closed-loop simulations |
| `disp_real.py` (.json) | rho from the banked states |
| `ab_vs_base.py` (.json) | the decomposition |
| `late_probe.py` (.json) | KL / J |
| `face_validity*.py` (.json), `controls_trainer.py` (.json) | the controls |
| `grad_bank.py`, `.log`, `overlap_census.py`, `summ_sim.py`, `make_summary.py` | helpers |
| `hf_dl_verify.py`, `inputs_sha256.json` | input provenance |
| `PREREG_R0C_DRAFT.md` | the next arm (a draft) |

The gradient bank and the closed-loop tensors are scratch-only, regenerable with `grad_bank.py` and `adam_closed.py`.
