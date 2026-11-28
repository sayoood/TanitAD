# SHIP-R8 AMENDMENT 5 -- the registered battery read on the FACTORED decode (Master Mind, 2026-10-26 ~03:52 CET)

PI decision **D-R8-FACTORED-DECODE -> pre-authorised**, 2026-10-26 ~03:30 Berlin: *"Yes, pre-authorise"* -- this read runs ONLY if
the SSGUARD seed-1 replicate reads REPLICATED under `PREREG_SSGUARD_S1.md` (68fc8a07); otherwise it does not run and is reported.
Sealed BEFORE any SHIP-R8 number for this decode exists and before the seed-1 roll finished. Amendments 1-3 stand.

## 1. The arm `FACTORED_DEPLOY8`
* Plan (per window, per inference seed s): `P = RT.eval_polyline(RT.polyline_geometry(P_os8), arc(P_ss8))` on the FULL 8-slot plans,
  where `P_os8` = L1's un-hooked os-only roll (`refcv9_l1/bank/eval_dump_s{s}`, `traj_own`) and `P_ss8` = the SELFSHARP os-only roll
  (`trace_selfsharp_s{s}.npz`). Both rolls are of the SAME code path (identity on unforced windows MEASURED, K-IND).
* Stage: the REGISTERED per-frame RETIME_DEPLOY8 table (`REGISTERED_PF_TABLE.sha256`) applied to `P` exactly as to the model plan.
* ⚠️ The os-only roll is a DIFFERENT inference draw from the battery tree's own `nav_true.traj` (they differ on 4,630 / 4,754
  windows -- the MAPSEL K_STAGE lesson). So the battery's refcv8 reference is re-read on the SAME draw beside:

## 2. Beside (never scored)
* `OSONLY_DEPLOY8` -- `P_os8` through the same stage and code path (refcv8 itself on the os-only draw); **identity control:
  `FACTORED_DEPLOY8` with `P_ss8 := P_os8` reproduces it bit-for-bit.**
* `RETIME_DEPLOY8` on the battery tree -- **identity control: equals the registered `SHIP_R8_step48852ema_pf.json` cells exactly.**

## 3. Clauses -- SHIP-R8 (1)-(7) + (1b) on `FACTORED_DEPLOY8`, both inference seeds, paired episode-cluster bootstrap B 2000
* **(1)** the registered table's sha256 checks (unchanged) AND the identity controls above AND both input traces' recorded
  sha256 + `final` flag + 4,754 windows + hook counters (all windows forced, none unarmed, batch 1).
* **(1b)** FEED PARITY: the os-only rolls use the per-frame feed (`perframe_full`); asserted from the rolls' own manifests; not
  shown -> INCOMPLETE.
* **(2)-(7)** verbatim from the registered verdict with the arm swapped; compliant windows + compliance row as registered.
* OVERALL as registered (any FAIL -> FAIL; else any INCOMPLETE -> INCOMPLETE; else PASS).

## 4. Committed next steps
* **PASS** -> the factored decode is a refcv8 deliverable candidate; the in-model implementation (two planner passes on one trunk
  forward) is built and its plans must equal the composed ones (bit-equal control) before anything ships; NavSim only with a separate go.
* **FAIL** -> the failing clause names the lever; the decode stays a refcv9 design input (`H-R9-SSGUARD-1`).
* If both Amendment 4 and 5 pass, the PAIR (H2cn stage on the factored plan) is a separate sealed read, never assumed.
