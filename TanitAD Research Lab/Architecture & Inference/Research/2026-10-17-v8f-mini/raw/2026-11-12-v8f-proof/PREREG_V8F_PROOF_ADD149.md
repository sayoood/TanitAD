# PREREG_V8F_PROOF_ADD149 — LEADSPEEDFULL10K: does the temporal vision head read the lead's closing speed and its change from real tokens?

Frozen before any tool of this card exists and before any real number of the head beyond row 214's smoke. Open at this time: row 214 (landed at c7be381a). Tier **T0** (a head on cached tokens of logged frames; it says what the tokens allow, not what a model does with them); never driving performance. PI, 2026-11-25, verbatim: "Yes, run it" (to: approve the full training run of the lead-speed head on box 2, about 1.5 h).

## 1. The run (box 2, RTX 4060, safe mode, BOX2_BUSY)

The head, labels, clock and split of card ADD148 with amendment 1 (`rulefc/leadspeed/leadspeed.py`, sha256 3466cd81), unchanged. One training seed (148). Fit set: the first 1,500 clips of the seed-148 fit order (of 3,984); held out: all 443 held-out clips. 30,000 steps, batch 64, AdamW 1e-3 with cosine decay to 1e-4, windows sampled uniformly. Checkpoints at 10,000, 20,000 and 30,000 steps; the read of record is the 30,000-step checkpoint (no selection on held-out data).

Memory guard (the commit-limit risk named in row 214): the 4x10 tokens are kept as fp8 bytes and cast per batch. Measured on box 2 at this freeze: commit available 17.2 GB. Before loading, commit available must be at least 5.5 GB + 8.1 GB (the 1,943 clips at 4.19 MB) + 2.5 GB (process); before every clip, at least 5.5 GB plus the bytes still to load. A failed guard stops the run as `LEADSPEEDFULL10K-STOPPED-MEMORY` (the fit set is never shrunk during a run).

## 2. Reads (held-out corpus clips, primary; eval-141 clips, for the consequence)

On frames with a label closing (the run of at least 3 frames of card ADD148):
* S: Pearson r of the head's closing against the label closing.
* A: Pearson r of the head's lead acceleration against the label one, both as card ADD146 builds them: v_L(f) = max(0, v(f) + closing(f)), a_L(f) = (v_L(f) - v_L(f - 5)) / 1.0 s, v the token cache's own speed, frames f and f - 5 both labelled.
* Beside: gap MAE on lead frames, presence AUC, the same r per held-out half, and the token reader's ADD146 values (A 0.087 / 0.071, S 0.334 / 0.273) for comparison.
The same S and A on the eval-141 clips (tokens `D:/tanitad/cache/eval`, labels `join_eval141.npz` on the token clock).

## 3. Controls (must pass before a name is given)

* c1 the guard passed and the run completed 30,000 finite steps; the train loss of the last 1,000 steps is below the first 1,000.
* c2 no held-out or eval clip in the fit set (by clip id).
* c3 a planted check of the read: the label closing fed in place of the head's output reads S = 1 and A = 1 on the held-out frames.
* c4 the read uses the 30,000-step checkpoint (sha256 recorded at save and at read).

## 4. Names and consequence (committed)

`LEADSPEEDFULL10K-CARRIED` if A >= 0.30 on the held-out clips; `-SPEED-ONLY` if A < 0.30 and S >= 0.30; `-NOT-CARRIED` otherwise. Consequence: CARRIED with A >= 0.30 also on eval-141 makes the head the closing input of a reader-form rule arm (a successor card: a new rule form with the forecast of card ADD144, its rollouts on box 2 against row 211's every-step reader form). SPEED-ONLY with S >= 0.30 on eval-141 makes the head the closing input without the acceleration (successor card). NOT-CARRIED: the blocker is named (more fit clips, needing a disk for the token store, or a second seed), with the measured values.

## 5. Cost

About 20 min load (1,943 clips at 35.7 MB/s) and about 30 min of training (0.05 s per step) on box 2, plus the reads (minutes). One training seed: a replicate is a successor if the result is near a bar.
