# MM rulings on the DSCALE10K block-3 read (2026-11-18 ~06:45Z, BEFORE any block-3 dump, eval or route pass ran)

The block-3 checkpoints UE_s0.A and BL_s0.A are banked and verified (HF LFS sha256 = pod marker = pulled file, steps 61,568). No block-3 outcome number exists yet. These rulings are made before the read and change no bar of the card (`card/CARD_DATASCALE_TRAIN.md`).

## 1. The `H-DATA-SCALE-1` label quantity ("U − BL", card line 197)

* The card lists three quantities (line 195): Δ_data = U_s0 − B_s0 (fixed steps), Δ_ep = BL_s0 − B_s0 (fixed data, about 3.2x steps), and UE_s0 − BL_s0. Line 197 labels on "U − BL".
* Two readings are both defensible from the text, and they answer different questions:
  * **literal, U_s0 − BL_s0** (= Δ_data − Δ_ep): the data lever at 1x compute (19,164 steps) against the epochs lever at 3.2x compute (61,568 steps), each applied to B alone;
  * **compute-matched, UE_s0 − BL_s0**: at 61,568 steps each, 14,512 unique clips against 4,512 clips seen 3.2x as often.
* **Ruling:** the agent's rule, as built, is the pre-committed one. If both readings give the same label, that is the label. If they differ, the result is **READING-SENSITIVE** and the register row reports BOTH labels, each with the question it answers. ⛔ No single label is picked after the data.

## 2. The bar

* As written: "better by more than 2F_A" on ADE6, magnitude only, strict. The paired episode-cluster interval (B 2,000, seed 0) is printed beside and is not an extra condition. ONE seed, reported as one seed (card line 276).
* F_A from B_rep vs B_s0 at stage A: the block-1 pre-read value 0.0828 m is a property of the block-1 arms, computed at build time. The stand-in smoke's Δ_data came from a fixture and is not a result. Neither is quoted as a block-3 outcome.

## 3. STRATEGIC at stage A: KEEP the 25 route passes

* The STRATEGIC family is reported in full (PI ruling 2026-11-16: the strategic pause has ended). UE_s0 and BL_s0 have no stage B, so stage A is where their STRATEGIC rows can come from. Dropping the passes would leave a family NOT-YET by choice, which the four-family rule forbids. Cost: most of the read's ~1 h 15 min of dev-box CPU (ESTIMATED).

## 4. Launch

* F: was at 22 GB against the card's 25 GB bar. The MM freed 8.4 GB of its own data and did not move the bar: a zero-filled memmap left by a stopped 10-27 speedread extract. Rows 76,641+ were all zero; the complete extract is kept. Logged in `speedread/stop2/deletions.log`. F: is now at 30 GB.
* Launched by WMI with `job_dscale10k_b3.cmd` directly (`powershell -File` is blocked on this box); CreateFlags 0x08000200, stdin NUL. CPU only, 2 threads, commit gate 5.5 GB. It may overlap the Option B read chain (also CPU); both pass the commit gate.
