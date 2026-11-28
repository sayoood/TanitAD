# MM ruling on the L2 Step B draft (Master Mind, 2026-10-19 ~16:50; no Step B search has run on real data, no search output has been labelled, no dPDMS has been read)

## Decision 0 (new): Step B runs NOW as a pre-registered DIAGNOSTIC on carve B; it licenses nothing alone.
- The draft's precondition (Step A O1 first) assumed Step B was the costly stage. MEASURED: it costs 1.8-2.7 CPU-h for three arms, with no GPU. It is also the only read of the actual endpoint (the true PDMS of the search output against P2's pick).
- Under RULE ZERO, the cheapest discriminating experiment toward driving quality is run as soon as its inputs exist, not after a gate the geometry says will probably read O3 (`MM_RULING_L2_FRESH.md`).
- **It launches** once the L2-FRESH B chain prints `CHAIN_DONE_NO_READ` with its gates holding, and once BOTH amendments are sealed.
- **Data:** carve B's bundle. Its token list must equal the sealed carve (sha256 `0629811f…`); a mismatch REFUSES.
- **Licensing is UNCHANGED.** EP-ADMISSION and the shared navtest look require a Step A O1 (READ 1 or READ 2) AND a Step B O1 on both head seeds. Every outcome of Step B is reported whatever Step A reads, so no result is a forking path: neither read can license alone, and both rules are fixed now.
- **Stamps:** SEEN-SCENE; the head is token-fresh and LOG-SHARED on B; SELECTION tier, label-only.

## The ten decisions
1. **Units: PDMS POINTS (100 × [0, 1]). RATIFIED, as an interpretation recorded before any data.**
   - L1's Step A thresholds are on [0, 1]. Its Step B bar is not stated in units, and +0.30 on [0, 1] (30 points) is unreachable; the motivating TOAD numbers are points.
   - In points the safety tolerance −0.25 equals ONE token in 400 losing a binary sub-score. That reading is coherent, and it is the strict one.
2. **The seed-spread term: dPDMS point > (max − min) of the 3 per-search-seed points. RATIFIED.**
3. **Two head seeds: the WORST of the two. RATIFIED.**
4. **P2HEAD control: KEPT** (descriptive attribution; it costs one arm). Without it, a gain cannot be assigned to the L2 head rather than to the search.
5. **NOT CLEARED ⇒ L2-BRANCH. RATIFIED.** O4 INERT ⇒ the raised-M widened arm once. If that is also INERT, the search is CLOSED for this scorer, and the report must NAME the next lever (L2-BRANCH, or the P6 trunk) with what unblocks it. Closing is never the end of the turn.
6. **Start and baseline = P2's pick. RATIFIED.** The L2 head was validated (A1) on perturbations OF P2's pick, so searching around that pick keeps the search inside the domain the head was admitted on. The in-fan re-ranking stays descriptive.
7. **Clips stay at the hull under O4's widening. RATIFIED** (same reason as 6).
8. **E7: NARROWED, with the reason recorded.**
   - `launch_gate.py` governs TRAINING launches (the PI rule of 2026-09-26: "if we are starting a training session"). Step B is an inference-time search: no optimiser, loss or gradient.
   - The right gate is G-REACH-STEPB, with its sha-bound PASS stamp (head, checkpoint, config, bundle, code) re-asserted inside the launch.
   - A profile whose G-LIVE, G-CKPT and G-CLOCK have no object would certify less than a gate built on the real risk.
   - My E7 phrase "Step B and beyond" now reads **"any arm that TRAINS with an L2 head (L2-BRANCH and beyond)"**. BACKLOG `L2-STEPB-GATE` is resolved by G-REACH-STEPB.
   - G-REACH-STEPB is re-run on EACH real head (s0, s1) before its arm runs. A stamp minted on the test head does not carry over.
9. **EP-ADMISSION's object: BOTH banks, and ADMIT needs both to admit.** The thresholds, estimator and rules stay frozen; only the candidate block (appended per read) names the objects.
   - (i) The frozen literal object: an in-fan bank of "P2 + the L2 EP row" (head seeds 0 / 1 = the candidate's two seeds), via a head-file input to `build_candidate_bank.py` (`l2_scorer.install_ep_head`).
   - (ii) A search-output bank in the same schema, built from Step B's labelled outputs. That is what the deployed system picks.
   - Both are built ONLY after a Step B O1 (not now). The candidate block is written before either bank is read.
10. **The TRADE-restricted arm: built ON a TRADE, not now.** Its spec is committed in §3 and is unchanged.

## The chain
After the seal, ONE detached CPU chain runs:
- G-REACH-STEPB on s0 and s1;
- the arms L2-s0, L2-s1 and P2HEAD (3 search seeds each);
- the labels, then the registered analysis;
- `ZZSTEPB_DONE` and the verdict line.

A failed gate stops it before any search. `CUDA_VISIBLE_DEVICES=-1`, no lock.
