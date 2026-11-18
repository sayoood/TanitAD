# MM ruling: F: free-space bar for the UM_s0 read (2026-11-18 ~19:00Z, before any UM_s0 evaluation number exists)

## What the bar is
- `CARD_DSCALE_MIX.md` (frozen sha256 0bc241f5) has two lines on disk:
  - section 6, cost: *"Under 1 GB of disk. The F: bar is 25 GB."*
  - the `ds_pull.py` row: `--min-free 25 GB`.
- It is an operational disk guard. It is not a scientific bar: it decides no label, cell or control.

## State (MEASURED)
- F: had 21.11 GB free at 18:55Z and was falling. Other sessions' data fills most of F:.
- This folder holds 250 MB. My own reclaimable temp is exhausted:
  - the CKPT120 checkpoints (609 MB, HF-verified) are deleted, as is the NAVRB2 G-EVAL pair (174 MB);
  - the X-read and block-2 checkpoint folders were already empty;
  - the block-1 checkpoints are inputs to this read.
  - Log: `datascale/deletions_mm.log`.
- Footprint of the comparable X read: `readx/out` 18 MB + `readx/t1` 48 MB = 66 MB. The card itself estimates under 1 GB.
- The binding disk floor on F: is the pinned suite's 20 GB (`DISK-LOW` refusal; other sessions' pinned runs depend on it).

## Ruling
- For this read only, the F: bar is **21 GB**:
  - the launcher's `df` gate is `-lt 21`;
  - `ds_pull.py` runs with `--min-free 21`.
- A read of about 0.1 GB then leaves F: at or above 20.9 GB, keeping the 20 GB floor with margin.
- Nothing else in the card changes. The read reports this ruling beside its result.
- Below 21 GB the launcher refuses as before. No other temp is deleted to make room.

## Not applied (2026-11-18, same evening, before any UM_s0 evaluation number)
The launcher's own test pins the card's 25 GB bar, so the first launch with the 21 GB bar stopped at its test step (FAIL: the reader tests are not green); no read step ran. F: then had 26.4 GB free, so the launcher was restored byte for byte to the card's 25 GB bar and the read was launched under the card as written. This ruling was not used.
