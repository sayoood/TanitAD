# MM ruling on the SEARCH-P2-CONFIRM draft (Master Mind, 2026-10-19 ~23:05; no B-REST token has been bundled, searched or labelled)

**DQ1, what C1 counts as: (C), RATIFIED.** One read, two named outcomes:
- **O1 CONFIRMED** = L1's O1 terms verbatim: point ≥ +0.30, lower CI > 0, point > the 3-seed range, and each safety point ≥ −0.25.
- **O2 DIRECTIONAL** = lower CI > 0 with O1 not met.

Why: at a true effect equal to the observed +0.327, the O1 bar alone passes only 56 % (R) / 34 % (E) of the time (design calc, ESTIMATED), so its miss mostly measures the point bar's own noise (sd about 0.13). Naming O2 spends no extra look and changes no O1 term. The word "CONFIRMED" is reserved for O1.

**DQ2, what licenses the EP-ADMISSION bank build: (B) O1 or O2. RATIFIED.** That licence covers building the bank and asking admission, and nothing more. The navtest look needs admission AND its own sealed amendment (DQ3).

**DQ3, the shape of the navtest (C2, T1) bar, committed NOW.** The numbers are sealed in C2's own amendment and may not be loosened there.
- **Decision rule:** lower CI > 0 AND point ≥ **+0.10 PDMS points** (the minimum worthwhile effect) AND point > the 3-seed range AND each of dNC, dDAC, dTTC, dC ≥ −0.25. The four families are reported beside it, per family.
- **Why +0.10:** it is the MM's value judgement, stated before any navtest data. It is one third of L1's +0.30 materiality line, and navtest can resolve it: half-width 0.065 (R) to 0.178 (E). A +0.30 magnitude bar would call a true +0.24 a miss in 95 % of reads (R), and the programme wants any real gain toward 94.55.
- **Also required in C2's amendment, before the look:** the search's inference cost per token on the deployment target, MEASURED, or named as UNMEASURED. A gain whose latency the vehicle cannot carry is a selection-tier fact, not a deployable one.

**DQ4, the seed-range term: KEEP.** It is the only guard in this read against this rig's inference-seed variance (H-ESTIM-SEED-1 and its stochastic-planner extension).

**DQ5, the pooled line: ALLOWED as a descriptive line only.** It is labelled `POST-HOC-POOLED (carve B + B-REST, 1,305 tokens, the same 24 logs)`, has no outcome, no verdict and no licence, and is printed after the C1 verdict.

**DQ6, EP-ADMISSION's bank geometry: (a), with pre-committed acceptance bars.**
- Build the bank adapter to `ep-admission-v1` and re-simulate at 24 logs / 905 tokens × 3 seeds with the R1 simulator, under an amendment sealed before the bank is read.
- The bars are those of `MM_RULING_EPADM_R1.md` §2, unchanged: A1-r1 false refusal Wilson upper ≤ 5.0 %, A2 (margin 1.05) false refusal Wilson upper ≤ 5.0 %, DAC power at ×1.25 ≥ 70 %.
- **All met** ⇒ admission is read on C1's bank.
- **Any missed** ⇒ (b), an S0-geometry bank of the search, with the miss reported. Never (c).
- The adapter and the re-simulation are CPU work and start NOW, in parallel with C1; neither reads a B-REST label.

**DQ7, the SEARCH-DOSE arm: its BAR is committed IN THIS RULING, before C1 is read.** The rest goes in its own amendment, sealed before its launch.
- **What:** the pinned O4 config (`stepB_config_O4.json`, sha256 `1ebd7196…`, M = 256), everything else identical to C1's arm, on the SAME B-REST bundle. CPU only.
- **When:** it runs after `ZZSPC_DONE` whatever C1 reads, except INERT, whose committed raised-M arm IS this arm.
- **Its outcomes:** C1's O1 / O2 / TRADE / O3 / NEGATIVE / INERT rules verbatim, on dPDMS vs P2's pick. The dose contrast (M256 − M64, paired per token and seed) is reported beside it, descriptive and ungated.
- Committing the bar now means C1's result cannot shape it.

**DQ8, C-ID strictness: BIT FOR BIT, RATIFIED.** On a miss, diagnose the thread count first (`--threads 4` is Step B's setting). A loosened gate needs its own amendment and is never applied silently.

**Added by the MM (RULE ZERO; the decisive read is navtest, so its blocker is built now):** the navtest path is a named BLOCKER of C2.
- It needs a chunked bundle → search → label → delete pipeline that fits D: (44.5 GiB of ctx against about 24 GB free), and a bundle builder for navtest tokens.
- It is built in parallel with C1 as its own BACKLOG item (SEARCH-NAVTEST-PIPE), 0 GPU until its own launch.
- It reads no navtest label until C2's amendment is sealed and admission has admitted.

Stamps for every C1 number: MEASURED, SEEN-SCENE, token-fresh, LOG-SHARED, SELECTION tier, label-only (not T1, not navtest), one training seed, 3 search seeds, paired log-cluster bootstrap B = 10,000, seed 20260927 ("another draw of LOGS" only). STRATEGIC PAUSED.
