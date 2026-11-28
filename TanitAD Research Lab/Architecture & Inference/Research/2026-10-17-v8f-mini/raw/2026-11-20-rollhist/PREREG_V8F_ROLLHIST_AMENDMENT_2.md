# PREREG_V8F_ROLLHIST, Amendment 2 (FROZEN 2026-11-21 01:29Z by the MM, before any RH or B0 bank, run or number)

* Card: `PREREG_V8F_ROLLHIST.md`, sha256 781eea87.
* Amendment 1: sha256 87928487 (code of record 43319227).
* At freeze, the pod chain was in its G-SUITE step (log line 01:22:02Z). `/root/rollhist/bank` was empty, and no RH or B0 checkpoint or dump existed.
* The only measured value known at freeze is control c-EXEC, run on UE_s0.A (the stage-A init, no lever).
* Nothing below moves an arm, a dose, a cell, a bar or an estimator.

## 1. The executor form of record (instrument, not outcome)
* **Form:** `python stack/scripts/v8f_t1_eval_hb.py --kappa-max 0.2 --inner v8f_t1_eval_reader_rule_rzgi <argv>`. The adapter `v8f_t1_eval_reader_rule_rzgi.py` has sha256 09666f4367b460b0..., and the file is in `F:/TanitAD-scratch/rollhist/read/code/integ10k/`.
* **Why:** card section 3.1 names the form `--inner v8f_t1_eval_reader_rule_rzg`, which does not compose. Measured on an 8-window smoke, 2026-11-21:
  1. hb refuses it, because rzg loads the pinned tool inside `main()`, so `v8f_t1_eval` is not in `sys.modules` at hb's check;
  2. even past that, `W.load_pinned` would execute the tool into a new module object and drop hb's heading bound without a word.
* **What the adapter does:** it imports the pinned tool once, as the module hb patches, and makes `W.load_pinned` return that object. rzg itself is unedited (sha256 93879db8c0233867...), and rzg's own check of the pinned tool's sha256 (`61e140ed...`, rzg lines 425-427) still runs on the returned path.
* **Evidence:** control c-EXEC PASS (`read/C_EXEC.json`, sha256 e18837d71d2f7422...).
  * UE_s0.A on form R0C10XHb20;
  * the ADD68 study wrapper `rzgfh` against this form;
  * 141 files, 1,269 arrays, 0 differing;
  * the bound acts on 50,946 of 287,880 frames.
* Card section 3.1's "tier of record" and its c-EXEC wording read as this form.

## 2. Gaps in the outcome table (3.4), filled before data
The build found combinations no row names. They are filled as follows; the precedence is unchanged.

* **G1:** P1 has exactly one family BETTER at both seeds and none WORSE at either, and P2 and P3 hold at both seeds. Name: **ROLLHIST-PARTIAL**. The next lever is unchanged: p = 0.3 on the same bank.
* **G2:** every condition of ROLLHIST-REPAIRS or -REPAIRS-K-UNNEEDED holds, except that G-T0 fails at exactly one seed. Name: **ROLLHIST-SEED-DISAGREE**.
* **Any other unnamed combination:** the reader prints `OUTSIDE-TABLE` with the per-family verdicts and gives no name. The MM reports it as unclassified, and any next lever is chosen post hoc and labelled so.

## 3. "The decisive cells straddle the floor" (SEED-DISAGREE), defined
* **A straddling cell:** a named (decisive) cell whose pooled paired interval excludes 0 while abs(D) <= F_c. Under the cell rule of 3.2 it reads NOT-SEPARATED.
* **When it gives SEED-DISAGREE:** compute the name twice. Once with every straddling cell as NOT-SEPARATED, and once with every straddling cell as BETTER or WORSE by its sign.
  * If the two names differ, the result is **ROLLHIST-SEED-DISAGREE**, and both names are printed.
  * If they agree, the first name stands.
* This is narrower than the build's first reading (any straddling cell). It fires only when the floor decides the name.

## 4. P2, as read
P2 holds at a seed only if both of these hold:
1. on RH, the pairs ACC, D1 and D2 read NOT WORSE on every family;
2. the attribution control holds: on B0, the ACC pair reads WORSE on at least two of LONGITUDINAL, LATERAL and TACTICAL.

If (2) fails, P2 fails at that seed, because the history is not shown to be the cause.

## 5. Stage-A references under the form of record (beside; only the one-seed floor rests on them)
* **Runs:**
  * UE_s0.A and UE_s1.A, each on R0G50Hb20, R0G0Hb20 and R0C10Hb20;
  * UE_s1.A on R0C10XHb20 (UE_s0.A on R0C10XHb20 is the c-EXEC run).
* **Cost:** 7 rollouts on dev-box CPU, about 46 min ESTIMATED from c-EXEC's 6.2-6.6 min per rollout. They run now, before any RH or B0 dump exists.
* **Existing dumps:** R0Hb20 and R0KHb20 on UE.A are card ADD67's dumps (`rzgfh`), which c-EXEC shows equal to this form.
* **One-seed floor:** F_U,c = abs(UE_s1.A - UE_s0.A) per cell under R0C10XHb20, from these runs (MEASURED, ours), not from ADD68's result JSON (INHERITED). ADD68's values are printed beside.
* **Reference for P2:** D1, D2 and ACC on UE.A, from these runs. The card expects WORSE. It is beside and decides no name.

## 6. A one-seed read while seed 1 trains
* When RH_s0 and B0_s0 exist, the MM may run the read with seed 0 only, against F_U,c. Every name it prints carries the PROVISIONAL prefix.
* The two-seed read is the read of record.
