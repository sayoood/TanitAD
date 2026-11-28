# MM verification of the agent-check feasibility hold (2026-11-14)

## Independent re-computation of the held-out ceiling (MEASURED, dev-box CPU)

**Setup:**
- Input: `F:/tanitad_scratch/p6fit/fit_pack.npz`, held-out and has-pdm sets: n = 1,705 (24 logs).
- Pick: the canonical `sel_common.agg_torch(logits, lane=False)`, i.e. the `navsim_v1` aggregate in torch float32 with weights 5/5/2, first index on ties. This is the function `read_scorer1p6.pick_logits` calls. The agent's script used a NumPy re-implementation instead.
- Truth: the fit pack's own `pdms` column. It equals the sub-score formula `nc·dac·(5ep + 5ttc + 2c)/12` to max |d| 6.7e-6.
- Column order: the `pdm_names` order (`nc, dac, ep, ttc, c, ddc, pdms`) equals the logit order the aggregate assumes.

**Result: it equals the agent's `ceiling_fitpack.json` held-out column to every printed digit.**

| quantity | agent | MM |
|---|---|---|
| P6 pick PDMS | 94.0239 | 94.0239 |
| best of 64 | 98.4158 | 98.4158 |
| oracle veto NC | +0.3584 | +0.3584 |
| oracle veto TTC | +1.6099 | +1.6099 |
| oracle veto NC+TTC | +1.6661 | +1.6661 |
| oracle veto DAC | +0.6357 | +0.6357 |
| oracle veto agent+DAC | +2.3521 | +2.3521 |

## Reading

* **On navtrain held-out, a perfect agent check (NC+TTC) is worth about 2.6× a perfect DAC check (+1.67 vs +0.64 PDMS points), and TTC is most of it.** This is the measured version of `FEASIBILITY.md` F5's estimate.
* It ranks the levers (CLAUDE.md Rule Zero item 5): the agent term is the larger one, and the DAC perception line (VDAC A5: HB X-BOUND, HC X-NULL, S1-Q flat at ~0.83) is the smaller.
* These are **oracle** ceilings: TRUE sub-scores, P6's own ranking.
  * They bound what any agent-risk input could recover through a veto on P6's proposals.
  * They are not an achievable gain.
  * They are a navtrain number, **not navtest**. The navtest projection (2.3 to 2.9) is ESTIMATED by the agent (HANDOFF Q-5) and is not quoted as a result.

## MM rulings on the card's section-1 questions (draft stays a DRAFT until the S0-B scripts and tests exist)

| Q | ruling |
|---|---|
| **Q-A** | **Yes:** S0-B runs first on the 3,976 sets whose DB is local, and the rest follow when the DBs land. The held-out subset (1,361 of 1,705) is reported as its own column. |
| **Q-B** | Pod time for S1 is **not requested** until S0-B reads. |
| **Q-C** | **Yes:** a pose-only footprint readout is admissible for S0-B. It is a ceiling instrument, not an inference path. Inference stays vision-only (PI ruling), and labels may use agents and futures. |
| **Q-D** | **The PI's call.** The card names it and does not decide it. |
