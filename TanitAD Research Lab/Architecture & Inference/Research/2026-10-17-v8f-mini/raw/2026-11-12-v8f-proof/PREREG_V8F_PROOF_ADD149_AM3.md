# PREREG_V8F_PROOF_ADD149_AM3 — LEADSPEEDFULL10K, amendment 3: the tokens are read through read-only file mappings

Frozen after amendment 2 (ab6677fe) and before its launch, before any training step, outcome number or head output exists. Amendments 1 (c122e315) and 2 stand except where this one says otherwise. The card's other sections are unchanged.

## Why (MEASURED on box 2)

* Box 2's commit available, read from outside any process of this card, fell from 17.2 GB (card freeze) to 14.68 GB (2026-11-25, about 19:25Z) with no python process running: the PI's own use of the machine. Amendment 2's pre-load need, 5.5 + 7.8 + max(0, 2.5 − own) GB, would not pass inside the process.
* Kept torch memory-mapped views do not help: torch maps the whole clip file copy-on-write, 21.1 MB of commit per clip (`rulefc/leadspeed/box2_full/MEMPROBE3.out`, 200 clips, two processes).
* A read-only `np.memmap` of the token bytes inside the clip file (`rulefc/leadspeed/tokstore.py`: the one stored zip member of 101 × 40 × 1024 bytes) costs 0.008 MB of commit per clip with every byte read; its bytes equal `torch.load`'s on 200 of 200 clips, and a planted offset 64 bytes off differs (`rulefc/leadspeed/memprobe_npmm_box2.py`, `rulefc/leadspeed/box2_full/MEMPROBE4.out`). Opening and reading every byte took 28.1 s for 200 cold clips.

## The changes

1. Every clip's 4x10 tokens (fit, held-out and eval) are read through a read-only `np.memmap` of their bytes in the clip file (`tokstore.open_tokens`); the speed v is read by `torch.load(mmap=True)`, copied, and the mapping released. The fit clips' token bytes are read once in full at load (page cache). Batches and reads cast the same fp8 bytes per window as before.
2. The memory budget per loaded clip is 0.1 MB (in place of amendment 2's 5.2 MB), in both guards; the guards' form, floor, stop name and the never-shrunk fit set are as in amendments 1 and 2. Held-out and eval clips are still read one at a time at read time (amendment 2).
3. A new control, c5 (must pass before a name is given, with c1 to c4): the memmap bytes equal `torch.load`'s on the first 50 fit, first 20 held-out and first 20 eval clips, and a planted offset 64 bytes off differs.

Physical memory, not commit, now holds the token pages; if the PI's use of box 2 evicts them, the run reads them again from disk, slower but with the same bytes. The step time is recorded.

## Unchanged

The head, labels, split, 1,500 fit and 443 held-out clips, 30,000 steps, schedule, checkpoints, reads S and A, controls c1 to c4, the bars (0.30), the names and the consequence.
