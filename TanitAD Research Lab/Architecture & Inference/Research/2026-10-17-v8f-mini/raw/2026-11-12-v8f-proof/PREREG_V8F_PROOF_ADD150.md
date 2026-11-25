# PREREG_V8F_PROOF_ADD150 — LEADSPEEDALL10K: does the temporal lead-speed head carry the lead's closing speed and its change when trained on all 3,984 fit clips?

Frozen before any tool change of this card and before its run. Open at this time: row 215 (card ADD149 with amendments 1 to 3, `LEADSPEEDFULL10K-NOT-CARRIED`: held-out closing r S 0.288 and lead-acceleration r A 0.191 from 1,500 fit clips, training loss 0.600 to 0.095). Tier **T0** (a head on cached tokens of logged frames; it says what the tokens allow, not what a model does with them); never driving performance. The run waits for the PI's go (box 2 time). Rule Zero lever: the blocker row 215 names (more fit clips) is removed by card ADD149's amendment 3 (tokens through read-only file mappings, 0.008 MB of commit per clip).

## 1. The run (box 2, RTX 4060, safe mode, BOX2_BUSY)

Everything as card ADD149 with amendments 1 to 3 (`rulefc/leadspeed/leadspeed.py` 3466cd81, `rulefc/leadspeed/train_full_box2.py`, `rulefc/leadspeed/tokstore.py`), with one change: the fit set is **all 3,984 clips of the seed-148 fit order** (in place of the first 1,500). Held out: the same 443 clips. Eval: the same 141 clips. 30,000 steps, batch 64, AdamW 1e-3 with cosine decay to 1e-4 (about 5.2 passes over the fit windows), seed 148, checkpoints at 10,000, 20,000 and 30,000; the read of record is 30,000 (no selection on held-out data). Memory guards as amendments 1 to 3 (0.1 MB per clip; pre-load need about 5.5 + 0.4 + max(0, 2.5 − own) GB); the fit set is never shrunk during a run. The tool's changes: its `--n-fit` argument (3984), a new output folder (`D:/tanitad/leadspeed/all`), and the beside read of section 2 (row 215's 30,000-step checkpoint `D:/tanitad/leadspeed/full/head_30000.pt`, sha256 7048714a checked before it is read, run on the same held-out clips).

## 2. Reads

As card ADD149 section 2: S and A on the held-out clips (primary) and on the eval-141 clips (for the consequence); beside, gap MAE, presence AUC, r per held-out half. Also beside, names nothing: row 215's values on the same clips, and the paired difference of S and of A against row 215's 30,000-step head (clip-cluster bootstrap over the 443 held-out clips, 2,000 resamples, seed 150: it answers "would another draw of clips say this?", not "would another training run say this?", one training seed per arm).

## 3. Controls (must pass before a name is given)

c1 to c5 as card ADD149 with amendment 3 (guards and 30,000 finite steps with the loss falling; disjoint clips; planted read 1; checkpoint sha256 at save and read; mapped token bytes equal `torch.load`'s and a planted offset differs).

## 4. Names and consequence (committed)

`LEADSPEEDALL10K-CARRIED` if A >= 0.30 on the held-out clips; `-SPEED-ONLY` if A < 0.30 and S >= 0.30; `-NOT-CARRIED` otherwise. Consequence, as card ADD149 section 4: CARRIED with A >= 0.30 on eval-141 makes the head the closing input of a reader-form rule arm with the forecast of card ADD144 (successor card); SPEED-ONLY with S >= 0.30 on eval-141 makes it the closing input without the acceleration (successor card). NOT-CARRIED: more clips of this corpus do not carry it; the next lever is the 8x20 token grid (finer spatial tokens, in the same clip files), its own card, then a second seed.

## 5. Cost

About 8 min to map and read the 3,984 clips' token bytes once (0.14 s per cold clip measured), about 27 min of training if the token pages stay in physical memory (0.053 s per step in row 215); about 16.5 GB of token pages against about 19 GB of physical memory free, so the PI's use of box 2 can evict pages and slow steps (ESTIMATED worst case about 2 h if every batch reads from disk). The step time is recorded. No pod, no encode, no download.
