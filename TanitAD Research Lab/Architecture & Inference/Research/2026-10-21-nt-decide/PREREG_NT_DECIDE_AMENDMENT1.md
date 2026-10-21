# PREREG_NT_DECIDE — Amendment 1 (pre-run): the lead-vehicle label pass, and the B1 guard dropped

* **Written** 2026-10-22 00:50 Berlin by the Master Mind. **Sealed** by the sha256 in `PREREG_NT_DECIDE_AMENDMENT1.sha256.txt` beside this file.
* **Applies to** `PREREG_NT_DECIDE.md` (blob `88abf03f5d40396fc074aca7a8c9e5aedc972425`) with `AMENDMENT_ACK_NT.txt` (blob `38175992b524e29353a8e666c37a2813e0912080`, L_D = 84.5).
* **Written BEFORE the run.** The A1 run has not launched; only launch-gate smoke steps exist. No A1, A1r or A0 navtest number exists, and no A1 snapshot has been read on any set.
* **Unchanged:** L_D 84.5, the 75.0 line, the ±1.0 boundary band, the snapshot set, the §5 selection and plateau rules, the §6 controls, the §7 estimator, and the §8 prohibitions.

## Change 1 — the navtrain label release includes the lead-vehicle pass (a defect fix)

**Defect.** The first A1 label build (E2, v1) called the label builder with **no lead-vehicle evidence** (`lead=None`). As a result, every decelerate / accelerate / keep row shipped as a partial label, and the hard longitudinal target was IGNORE on **74,020 of 99,824 tokens (74.15 %)**. The parent design does not do this: refcv8's PhysicalAI release builds the same target WITH lead evidence, and its IGNORE rate is 30.86 %.

**Fix.** For each 2 Hz log frame, the OpenScene `anns` (E3's class map and ego-footprint rule) now go through the builder's **unchanged** `lead_for_frame`. The builder runs once per token on its own 0.1 s band.

**MEASURED** (Master Mind read of the two `LABEL_REPORT.json` files on Thor, `/home/nvidia/r8ns/nt/labels/a1_prov{,_lead}/`, 2026-10-22 00:46). Both releases have 99,824 tokens and skip the same 3,464 held-out navtrain tokens.

| | v1 (no lead) | with the lead pass |
|---|---|---|
| longitudinal IGNORE | 74,020 (74.15 %) | **1,002 (1.00 %)** |
| lateral IGNORE | 6,560 | 6,560 (histogram identical) |
| tokens with lead data | — | 99,824 |

* **Tests:** 26 pass. The Master Mind re-ran `test_nt_lead.py` and `test_nt_code.py` at 00:47.
  - With the lead pass off, the written release is bit-identical to v1.
  - On real logs, only `lon_*` and `lead_*` fields change.
  - Analytic cases: an empty road reads KEEP; a vehicle 15 m ahead reads FOLLOW, also on a curve.
  - Mutations go RED: lead off, and a y-flipped lead.
* **Final release:** the FINAL release (E1 done ~03:55, finalize ~04:05) includes the lead pass. Its v9 md5 is written into `launch_gate._NT_V9_MD5` at closure and bound by the launch gate.

**Why this is admissible before the run.** It restores the parent label design rather than adding a lever. It changes no bar or rule. And no data from any arm has been read.

## Change 2 — the B1 guard is dropped from A1 and A1r

**The flags:** `--b1-guard-ref-ap2m 0.21211 --b1-guard-step 1000`.

**What the guard does** (`stack/tanitad/train/weight_ema.py`, `b1_guard_decide` / `b1_head_cut`): at step 1,000 it cuts the head learning rate to the trunk's, if the EMA box3d AP@2 m on the **in-run eval set** falls strictly below 0.21211. In A1 the in-run eval set is **PhysicalAI eval139**, the forgetting monitor.

**The conflict inside the prereg:**
* §2 defines A1 as "all refcv8 flags unchanged except the NavSim data inputs".
* §8 says A1's "PhysicalAI forgetting is reported, not gated".
* Keeping the guard would let a PhysicalAI forgetting metric change A1's training trajectory.

**Resolution:** §8 wins. Both guard flags are removed from A1 and A1r. Everything else in the argv is unchanged:
* init from θ_DEC, md5 `dad4cb5b…`;
* 12,478 steps, lr 1e-4;
* snapshots at 0 / 1,560 / 3,120 / 6,239 / 9,359 / 12,478;
* seeds 0 (A1) and 1 (A1r).

**Closure records it:** `nt_close.py --b1 drop`, which sets `launch_gate._NT_B1_GUARD = "drop"` and regenerates the argv. The dropped argv for s0 reads sha256 `ce32a125…` before closure re-binds it to the final paths.

## What would make this amendment wrong

* Any A1 step having been trained, or any A1 snapshot having been read, before this file's sha256 was written.
* The final release differing from the provisional one in any field other than the token set E1 finally wrote (checked at closure by the release md5 and the K1 / K2 controls on the final labels, `CONTROLS_B_final.json` must read `ALL_OK`).
