# MM ruling: what licenses the navtest look (C2) for the search on P2's own scorer (Master Mind, 2026-10-20 ~02:55)

**Sealed BEFORE any navtest token is touched and BEFORE C1 has produced any B-REST bundle, search output or label.**
- Verified at ~02:50: `D:/Projects/TanitAD-artifacts/spc/` holds only the 3-token bridge check (L2-FRESH's `bridge3_sample`, not B-REST) and the crashed 16-token carve-B C-ID attempt.
- The chain stopped at `ZZSPC_STOP G_GPU_STAGE_FAILED`.
- This ruling therefore cannot have been shaped by any C1 outcome. It discharges `MM_RULING_S0BANK.md` DQ3's committed branch: "if N1 also misses, a separate sealed MM ruling … decides what licenses C2".

## 1. The finding that forces it (SIMULATED, R1 simulator, designs hashed before each run)
The frozen EP-ADMISSION gate cannot certify a SEARCH candidate at any geometry available to the programme. DAC power at ×1.25 (bar ≥ 70 %) by geometry:

| geometry | statistic | DAC power at ×1.25 |
|---|---|---|
| C1 (24 logs) | all-picks A2 | 7.4 % |
| S0, 41 logs | all-picks A2 | 31.6 % (Base X) |
| S0, 117 logs | all-picks A2 | 60.4 % (Base X) |
| S0, 41 logs | A2 on the changed picks only (N1) | 27.5 % (Base X) |
| S0, 117 logs | A2 on the changed picks only (N1) | 39.6 % (Base X) |

- A1-r1 false refusal misses on the carve-B base at every geometry (8.6-9.9 %).
- A2's false refusal is MET at S0 geometry (2.5-3.5 %), so an A2 REFUSE there would be informative; but an A2 pass is not, at that power.
- Packages: `raw/2026-10-19-epadm-searchbank/`, `raw/2026-10-20-epadm-s0bank/`, `raw/2026-10-20-epadm-n1/`.

## 2. The rule
1. **The frozen gate stays in force, unchanged, for every OTHER EP-led candidate** (P6, R0e-pick, RL or head changes). This ruling touches only `H-SEARCH-P2-1`'s arm: P2-s0 + Step B's P2HEAD CEM search, sha-pinned, with no head substitution.
2. **For that arm, C2 is licensed if and only if ALL of these hold:**
   - C1 reads **O1_CONFIRMED or O2_DIRECTIONAL** under the sealed `PREREG_SEARCH_P2_CONFIRM.md` (sha256 `658ba157…`);
   - **AND each safety family's point estimate, dNC, dDAC, dTTC and dC, is ≥ −0.25 PDMS points at C1.** This term already sits inside O1; here it is required for O2 too;
   - **AND the moved share is ≥ 5 %** (not INERT).
   - TRADE, O3_NULL, NEGATIVE and O4_INERT do NOT license C2. Their committed next arms stand.
3. **Why this replaces the gate rather than lowering a bar.** The gate checks the scorer's reliability at its own picks: a proxy for whether the TRUE PDMS of what it picks is as good as the scorer believes. C1's endpoint IS the TRUE PDM label at the search output vs P2's pick, on 905 tokens never read, with the safety families broken out. For this one arm the licence reads the quantity directly, instead of a proxy the simulations show is under-powered. (The label-only PDM scorer reproduces NAVSIM's PDMS exactly on 94.7 % of 19,500 navtest trajectories and with the same argmax on 100 %: `raw/2026-10-04-lane-discipline/validate_pdm_targets.json`, MEASURED earlier.)
4. **Reported beside C1, with no authority:**
   - A2 and A1-r1 through the adapter on C1's bank, with their C1-geometry error rates printed next to them (A2 false refusal 13.4 %, DAC power 7.4 %);
   - A3 / A4 / A5 of the search outputs.
5. **Unchanged:**
   - C2 needs its own amendment, sealed before any navtest token, with the bar shape committed in `MM_RULING_SPC.md` DQ3: lower CI > 0 AND point ≥ +0.10 AND point > the seed range AND each safety family ≥ −0.25;
   - the search's inference cost per token on the deployment target, MEASURED or named as UNMEASURED;
   - the navtest pipeline's carve-B fixture must read IDENTICAL before its first real run.
6. **Consequence:** the S0-geometry search bank (EPADM-SEARCHBANK-S0) is NOT needed for C2. It is CLOSED as unbuilt, which saves 2.5-7.7 GPU-h and the 4.9 GB pod read.

## 3. What this ruling does NOT license
No navtest look before C1 reads. No change to C1's bars. No use of carve B's post-hoc result as a licence. No extension to any other candidate.
