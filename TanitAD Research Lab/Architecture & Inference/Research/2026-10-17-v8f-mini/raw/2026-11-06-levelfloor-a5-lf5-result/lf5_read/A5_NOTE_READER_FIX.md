# A5 note: the reader's first production run failed at extraction, and what was changed before any rule was read

* **What failed (06:21:23, `logs/read.out`, `raw/LF5.fail`).** The 20 rollouts finished at 06:16 with every rollout check PASS. The
  reader then refused at the extraction of the four RILFO100 arrays (`logs/extract_RILFO100*.log`). The 16 other arrays extracted.
* **Why.** With σ = 0 the LF5 tool (`lf5_build`, sha 88b275a1) writes the header value `floor_vhat_noise_seed = None` into the ticks
  npz. numpy stores it as a 0-d object array. The pinned loader `t1hd_read.load_ticks` opens ticks with `allow_pickle=False` and
  refuses the whole file. The noise arms carry an int64 seed in that key, and the A4 tool wrote no such key, so both extracted.
* **What was changed.** `code/sanitize_objseed.py`, run once at about 06:23, before the rerun. For each RILFO100 rollout:
  * the original file was moved aside to `ticks_<tag>.npz.objseed_orig` on the scratch disk, kept and not deleted;
  * a new file was written with every other member copied byte for byte and that one key dropped (absent = no draw, as in the A4
    tool's ticks).

  `raw/SANITIZE_OBJSEED.json` records, per file:
  * the dropped value (a 0-d object array holding None);
  * byte identity of every kept member;
  * that the new file loads with `allow_pickle=False`;
  * both sha256s.

  All four PASS. The seed stays recorded in the run header, which is where the reader's controls read it (`lf5_precheck.py`).
* **What was not changed.** The card, the reader, the rules, the tool and the rollouts. The reader was rerun unchanged with the
  handoff's production command (`logs/read_rerun.out`, rc 0), and its rollout controls (29 of 29 PASS) read the same files.
* **For a later use of the LF5 tool:** write an integer sentinel there, or omit the key, when σ = 0.
