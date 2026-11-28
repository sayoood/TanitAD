# PREREG_TRKSCORE AMENDMENT TRK-MIN: how TRK-MIN (card section 7) is read (MM, 2026-11-28)

**Status.** This is the amendment that card section 7 requires ("read under an amendment frozen before its numbers"). It is frozen by sha256 in `AMENDMENT_TRKMIN.sha256` before any TRK-MIN number exists.

**Disclosure.**
- Before writing this, the MM saw TRK's own PROVISIONAL C1 read: Delta_C1 = -0.4168 [-0.6715, -0.1455] (T1, paired log-cluster bootstrap), with NC, EP and DDC separated worse. That outcome is ADVERSE, and the card's committed next arm on ADVERSE is TRK-MIN.
- The TRK-MIN **rule** and its **bars** were fixed by the card on 2026-11-26, before any TRK number: section 7 (the rule) and section 5 (the bars and the C0 replicate).
- This amendment fixes only the mechanics of the read. It moves no bar and adds no free parameter.

## The rule (card section 7, unchanged; zero parameters)
- For every token and proposal j: `agg_min(j) = min(agg32(s_plan)[j], agg32(s_trk)[j])`, and `k_min = argmax_j agg_min(j)`.
- `agg32` is `refe/safehead_data.agg32`: the planner's navsim_v1 aggregate in float32, validated 36,438 / 36,438. Ties go to the first index, as numpy argmax does.
- `s_plan` and `s_trk` are the TRK dump's two logit sets, [12,146, 64, 6] each: `trkscore/work/trk_c1/trk_c1.npz` for C1, and the C0-TRK dump for C0.
- No new seam is needed to choose.

## The read
- **Score:** PDMS(TRK-MIN) is the banked harness-exact 64-proposal table at `k_min`: `tr.pdms[k_min]` from `full_C1_s*of2.jsonl` (C1) and `full_C0_s*of2.jsonl` (C0). The sub-scores are `tr.{nc, dac, ttc, ep, c, ddc}[k_min]`.
- **Comparators:** C1 is the table at the banked pick (mean 92.5548; CTRL-ID-T below). C0 is the table at its own banked pick.
- **Estimator:** the paired log-cluster bootstrap, `eval/p5c/p5c_read.py` `Boot`, 10,000 draws, seed 20260927, 95 % percentile, over the 136 navtest logs. Tier T1.
- **Bars:** card section 5 unchanged, with TRK-MIN in place of TRK:
  - P1: CI-lo(Delta_C1) > 0;
  - P2: Delta_C1 >= 0.2316;
  - G: no separated-adverse NC, DAC, TTC, EP or C on C1;
  - R: Delta_C0 > 0, with no separated-adverse PDMS or sub-score on C0;
  - every control PASSES.
- **Outcomes and next arms:** card section 5 unchanged, with TRK-MIN in place of TRK. TRK-MIN's own "next" on every non-ADOPT outcome is the card's "then L2".

## Controls (each can FAIL; any FAIL = INVALID)
- **CTRL-ID-T:**
  - `argmax agg32(s_plan) == k_ship` on 12,146 / 12,146;
  - `argmax agg32(s_trk) == k_trk` on 12,146 / 12,146;
  - the table at `k_ship` gives C1's mean 92.5548 to 5e-5.
- **CTRL-T-TAB:** the table at `k_trk` equals the TRK harness CSV (`refe_stage2_trk_c1.csv`) per token to 1e-6, on every token. This is the card's CTRL-T(c), re-checked here, and it licenses the table lookup for a TRK-family pick.
- **CTRL-MIN:** `agg_min <= agg32(s_plan)` and `agg_min <= agg32(s_trk)` elementwise.
  - Where `k_trk == k_ship` and that proposal is the argmax of both sets, `k_min` must equal it.
  - Count and report every token where `k_min` is neither `k_ship` nor `k_trk`.
- **CTRL-SET:** 12,146 tokens, the same token set as the C1 seam, and 136 logs.
- **Harness confirmation (card section 7):** seam rows for `k_min` are built from the TRK run's planned-proposal dump (`stage2_trk_c1.proposals.npz`, the executed form the harness tracks). They are scored by the pinned harness, and the harness CSV must equal the table at `k_min` to 1e-6.
  - It is CPU work after the table read.
  - **Until it passes, every TRK-MIN number is labelled PROVISIONAL.**
- **The four families** (binding) run on the TRK-MIN seam in the confirmation step, through `run_fam_chain_ds.sh`-style role mapping. They are required before any ADOPT.

## Order
1. The C1 table read, provisional; it needs only the C1 dump.
2. The harness confirmation and the families on C1.
3. The C0 replicate, once the C0-TRK dump exists. The TRK chain is producing it.
4. The final read, applying the bars.
