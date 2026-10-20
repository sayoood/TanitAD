# SEARCH-DOSE read (O4 on C1's B-REST bundle), banked by the Master Mind, 2026-10-20

**Outcome: `O1_CONFIRMED`**, under `AMENDMENT_DOSE_R1.md` (sha256 `4769bb8e…`) and C1's rules, chain `dose_chain.py` (`b41ccab5…`).

**Stamps (from the chain):**
- MEASURED. SELECTION tier, label-only, so it is **NOT T1 and NOT navtest**.
- SEEN-SCENE. It reads the SAME bundle and the SAME 905 tokens / 24 logs as C1, so it is a SECOND READ OF THE SAME TOKENS, not a fresh draw.
- ONE training seed. The interval is the paired log-cluster bootstrap (another draw of LOGS only).
- STRATEGIC is PAUSED.

**Gates:** G-SUBID PASS (16 tokens, the C8 config reproduces C1's banked outputs bit for bit); G-SELFREP PASS (8 tokens); REACH PASS.

| quantity | O4 (this read) | C1 (C8), for reference |
|---|---|---|
| dPDMS vs P2's pick | **+0.4383 [+0.0269, +0.8030]** | +0.5575 [+0.1058, +0.9253] |
| search-seed range | 0.008 | 0.128 |
| dNC / dDAC / dTTC / dC | 0.000 / +0.331 / −0.037 [−0.626, +0.457] / 0.000 | 0.000 / +0.331 / +0.258 / 0.000 |
| dEP | +0.477 [+0.012, +0.941] | +0.506 [+0.048, +0.949] |
| dDDC | −0.018 [−0.052, 0.000] | 0.000 |
| moved share `inert.frac_moved` | 17.5 % | 17.3 % |

**O4 minus C8: −0.1192 [−0.2266, −0.0208] points.** This is DESCRIPTIVE and UNGATED (`MM_RULING_DOSE.md` clarification). O4 moves M ×4, elites ×4 and widen ×2 together, so it is not a dose-response in M. Its single permitted use is the config choice: per DQ4 row 3 (both licences met), C8 is chosen unless the contrast's lower CI is above 0. **`ZZDOSE_C2CONFIG C8`.**

Files: `RESULT_DOSE.json`, `chain_state.json`, the merged search and label files, the per-shard outputs (`search_C1/`), the G-SUBID / G-SELFREP shards, the G-REACH token and the stage logs, plus `dose_chain_run.log`.
