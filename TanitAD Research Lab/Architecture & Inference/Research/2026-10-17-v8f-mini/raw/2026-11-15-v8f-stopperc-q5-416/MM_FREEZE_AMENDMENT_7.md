# MM freeze of AMENDMENT 7 (STOPPERC Q5, the 416x1024 source), 2026-11-15

* **Frozen text:** `AMENDMENT_7_Q5_416.md`, sha256 `6be1adf9628ee5aa31f4bb3bf8d666816e82fc308c968abc3de9efab5de1689d`, byte-identical to the design agent's draft. It was frozen before any 416x1024 token was encoded or scored. The PI's approval of the re-encode compute is `PI_Q5_APPROVAL_2026-11-15.md` (verbatim: *"Approve reencoding"*).
* **Q3 (train frames), decided by the MM inside the approval:** rebuild the 416 frames on the LoRA pod from HF at the pinned revision (one frame route for train and eval; C6a checks it against E:'s eval139). Thor's banked 416 cache is not used.
* **Schedule (MM):**
  * The pod work (416 frames for S1's sample clips and eval-141, the NOW-frame encodes at 416 and 256p, controls C6-C8) starts now.
  * Box 2 runs the ACTSENS chain until about 01:30 on 11-16, so C0 and S2's E extraction (box-2 CPU, reading the same USB HDD) wait for the chain to end.
  * S1's fits may start from the banked S0-P S1 data once C4 (bitwise) holds.
* **Q1 and Q2 go to the PI** (`QUESTIONS_416.md`):
  * Q1: conditional box-2 time for Q5-1 (about 2 h) on RES416-ADDS-40;
  * Q2: on RES416-FLAT, the map stop-line input on the nuPlan / NAVSIM rig under its own card.
  * Nothing that depends on them runs before the answers.
