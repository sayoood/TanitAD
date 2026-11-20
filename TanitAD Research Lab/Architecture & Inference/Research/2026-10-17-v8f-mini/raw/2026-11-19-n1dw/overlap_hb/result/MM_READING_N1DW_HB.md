# MM reading: the N1DW D1-HO read under the executor of record (T1-kin/hb0.2), beside the verdict of record

Card `PREREG_N1DW_HB_OVERLAP.md` (sha256 33e4873c, landed fb686f4a), section 5. MEASURED, tier T1-kin/hb0.2 (`v8f_t1_eval_hb.py --kappa-max 0.2`, sha256 74fd8445).
Data: D1-HO, 5,736 windows over 497 held-out episodes. Estimator: paired episode-cluster bootstrap, B 2,000, seed 0. It answers "another draw of EPISODES" only; the seed draws enter the bar as the card's floor rule has them.
The verdict of record of `PREREG_V8F_N1DW` stays on T1-kin/old (landed 94636bf7: NO COMMITTED ROW REACHED). This read is reported beside it.

## Outcome (section 5): HB-GAIN-EXECUTOR-DEPENDENT at N1DW2_s0 and N1DW4_s0

The two T1-kin/old gain arms both read separated but below the bar (0.042364) under hb0.2. No arm reads `gain` under hb0.2, so HB-NEW-GAIN does not fire. No arm reads `cost`.

| arm | N1DW - N1R, T1-kin/old | gain | N1DW - N1R, T1-kin/hb0.2 | gain | N1DW - N1D, old | N1DW - N1D, hb0.2 |
|---|---|---|---|---|---|---|
| N1DW2_s0 | +0.0448 [+0.0321, +0.0574] | yes | +0.0411 [+0.0271, +0.0549] | no | +0.0157 [+0.0061, +0.0252] | +0.0167 [+0.0068, +0.0261] |
| N1DW2_s1 | +0.0246 [+0.0127, +0.0367] | no | +0.0343 [+0.0217, +0.0464] | no | +0.0075 [-0.0032, +0.0178] | +0.0176 [+0.0061, +0.0286] |
| N1DW4_s0 | +0.0453 [+0.0336, +0.0569] | yes | +0.0389 [+0.0262, +0.0510] | no | +0.0162 [+0.0061, +0.0264] | +0.0145 [+0.0044, +0.0238] |
| N1DW4_s1 | +0.0143 [+0.0012, +0.0283] | no | +0.0303 [+0.0174, +0.0430] | no | -0.0028 [-0.0142, +0.0082] | +0.0136 [+0.0018, +0.0246] |

* The D1-HO labels under hb0.2 are RBW-NO-GAIN and RBW-FLOOR-HIGH at both weights. RBW-COSTS stays PENDING-GUARD, since the guard runs on eval-141 under T1-kin/old.
* ⇒ By the card: the verdict of record is unchanged, with "executor-dependent at N1DW2_s0 and N1DW4_s0" beside it. Any next card at a loss weight is frozen under T1-kin/hb0.2. The PI parked the weight line on 2026-11-20 ("Yes, N1DC next"), so nothing is queued at that weight.

## Beside, naming nothing

* Under hb0.2 the seed spread narrows. All eight N1DW rows are separated above both N1R and N1D, by +0.030 to +0.041 and +0.014 to +0.018. Under T1-kin/old, two of the four N1DW - N1D rows included 0.
* The bar uses the floor as written, max(F_HO banked 0.042364, F_HO live 0.024582). The banked floor was measured under T1-kin/old; the live floor under hb0.2 is lower. Reading the bar on the live floor alone is post-hoc and decides nothing here.
* Z-USE (channel real minus zero, D1-HO TAC-LAT) is separated above 0 at all six arms: +0.0190 to +0.0439.

## Controls

* **K-HB0:** PASS. N1R_s0 through the wrapper at `--kappa-max none` equals the banked T1-kin/old dump bit for bit (497 of 497 files, 0 mismatches).
* **K1:** PASS (self-pair 0.0 [0.0, 0.0]). **K-CONST:** PASS (0.338215).
* **K-RE:** differs, as the card says it must (section 3: "expected to differ ... does not count as a control problem"). RESULT_NAVRBD.json was read under T1-kin/old.
* **K-CLIP:** not re-read here. It is a training-log property, independent of the executor; the T1-kin/old read holds it.

## Four families on D1-HO under hb0.2 (no bar; `R1_SUMMARY_HB.txt`)

* **TACTICAL:** the deciding cell D1-HO TAC-LAT, as above.
* **LONGITUDINAL and LATERAL:** 15 to 19 of the 23 R1 cells are separated better per arm and pair.
  * The separated worse cells are small, mostly operative 0-2 s along-track MAE (+0.0022 to +0.0060 m) and one operative 0-2 s ADE (+0.0036 m, N1DW2_s1 against N1D).
  * Distance-keeping is NOT-COMPUTABLE (n = 0; no lead block for the held-out clips), as in the old read.
* **STRATEGIC:** NOT-COMPUTABLE in this read (D1-HO carries no strategic cells). It is read on eval-141 under T1-kin/old: S-PARTIAL at both weights (94636bf7).

## How the read was completed (said plainly)

* The frozen chain ran K-HB0 and the 14 dumps (14 DONE, 0 FAIL). The frozen wrapper `n1dw_ho_read_hb.py` (sha256 d73cf8f8, equal to KIT_SHA256.txt) then imported the banked reader, which wrote its full output and exited 1 on its K-RE entry.
* The exit ended the wrapper before the wrapper's own steps ran: the K-RE note, `status_hb` and the old-versus-hb table. So the chain logged `ZZ HBOVERLAP FAIL ZZ`.
* The MM then ran the same wrapper `main()` once on the pod, with the inner `main` set to a no-op, over the inner read's existing output (`post_hb.log`). It printed `hb read VALID`.
* No dump was redone and no number was recomputed. `RESULT_N1DW_D1HO_reader_raw_hb.json` (sha256 0b1e69d3) is the inner read's output byte for byte. The class is a wrapper that cannot reach its own post-processing, because the module it imports exits.

## Files

* `read/RESULT_N1DW_D1HO_HB.json` (sha256 54565e44), `read/RESULT_N1DW_D1HO_reader_raw_hb.json`, `read/READ_N1DW_D1HO.txt`, `read/post_hb.log`.
* `K-HB0.json`, `read_hb.log`, `mm_hb_overlap.log`, `dumplist.txt`, `R1_SUMMARY_HB.txt`.
* The dumps stay on the pod at `/root/strat1/n1dw/hb/out/`.
