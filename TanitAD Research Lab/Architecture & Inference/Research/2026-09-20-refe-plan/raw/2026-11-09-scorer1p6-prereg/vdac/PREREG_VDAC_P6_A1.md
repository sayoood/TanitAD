# PREREG_VDAC_P6_A1 — the veto becomes S2's primary arm (written after S0, before S1 and before any navtest read)

**Status.** Frozen by the Master Mind, by sha256, on 2026-11-11, AFTER S0 (`vdac/s0/s0_read.md`, S0-GO) and BEFORE any S1 training,
any H48 read, or any navtest read of a VDAC arm. It amends `PREREG_VDAC_P6.md` (sha256 `5594c2ae...`) in one point only.

**Why (MEASURED on navtrain only, S0):** fed the TRUE field, the hard veto at margin 0 (V0) keeps R 0.348 of the oracle; the margin
0.25 m veto 0.280; the card's soft rule Phi is harmful at every width (R 0.096 / -0.216 / -0.429 at 0.5 / 1.0 / 2.0 m), because
Phi = 0.5 at the boundary penalises in-road proposals near it. A primary arm that already loses with the true field cannot be the
deployable candidate. The choice uses navtrain evidence only; no navtest number of any VDAC arm exists.

**The change.** Add the arm **VDAC-V0** (head HA-s0's field; a proposal whose pose-only footprint has d_k < 0 is dropped; all dropped
=> P6's pick; otherwise the navsim_v1 rule on P6's own logits). **VDAC-V0 is S2's primary arm**: every bar, outcome and guard that
section 2-3 of the card states for "VDAC-Phi (HA-s0)" applies to VDAC-V0 instead, with the same values (P1 vs P6 CI-lo > 0 and
d > 0.4971; the DAC-0 pick-share mechanism bar; G1-G4). VDAC-V0 with the HA-s1 head is the replicate. VDAC-Phi and VDAC-V (0.25 m)
stay as reported comparison arms and cannot be adopted. sigma_b is no longer needed for the primary arm. K + 1 for VDAC-V0.

**Unchanged.** S1's perception bars and H48, the controls, the four families, the PI's open decisions (D-VDAC-3 footprint,
D-VDAC-4 S1 pod time).
