# PREREG_V8F_SEED3 Amendment 1 (FROZEN 2026-11-23, before any s2 bank, step or number): the training commit of B0_s2 and RH_s2 is 97fe0a56

**Parent:** `PREREG_V8F_SEED3.md`, sha256 `75f3b191…`.

## What this states
* **The binding definition does not change.** Section 2 defines the code of B0_s2 and RH_s2 as "the commit and tree B0_s0 [RH_s0] trained at, read back from its gate record".
* **The card's parenthetical value is wrong.** It reads "cdbff971". The gate records read back on the pod by the kit agent name **97fe0a56767320dbc5f4b823d6…**:
  * `/root/rollhist/runs/RH_s0/gate/RH_GATE_RECORD.json`, `pass: true`, readout causal_g1x;
  * `/root/rollhist/runs/B0_s0/gate/RH_GATE_RECORD.json`, `pass: true`, readout causal_g1x;
  * `/root/rollhist/runs/RH_s0/gate/gate_token.json`.
  * Recorded in `pod_s3/FACTS_POD_S3.md`, item 2.
* **What cdbff971 is:** the G-SUITE base of that run. `/root/rollhist/gsuite_result_rh.json` reads new_commit 97fe0a56, base cdbff971, with 0 regressions, 0 new-test failures and 0 missing (14,851 base / 14,910 new tests).
* ⇒ **B0_s2 and RH_s2 train at 97fe0a56 on `/root/rollhist/tree_cdb`, as their s0 and s1 arms did.** KI_s2 stays at 4097a89e on `tree_ki`.

No arm, argv rule, read, bar, floor, outcome or cost changes. No s2 step has run.

## Two facts recorded with this amendment, both from the pod
* **UE_s2.A on the pod:** sha256 equal to the card's `6d434525…`, 86,970,901 B.
* **Disk:** the pod has about 46 GB free and the s0 bank is 2.0 GB. The chain gates the bank step at 30 GB free and each arm at 20 GB free, where RHK used 60 GB.
  * The bank itself needs about 2 GB.
  * The pod is shared with TanitADRuled, whose folder holds 74 GB.
