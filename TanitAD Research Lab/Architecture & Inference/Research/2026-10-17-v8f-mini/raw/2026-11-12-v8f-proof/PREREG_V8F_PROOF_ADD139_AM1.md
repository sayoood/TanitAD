# PREREG_V8F_PROOF_ADD139_AM1 — amendment 1 of card ADD139 (RHACC10K): a third model seed

Frozen before any tool of card ADD139 has run on data and before any value of it exists. Card ADD139 (sha256 e7420aaa6947bfd0) stays as written except for the three points below.

Basis: the other session reported on 2026-11-24 that RH_s2 is banked on its private HF repo `tanitad-v8f-rollhist-runs` at `RH_s2/train/ckpt_last.pt`, 86,971,477 bytes, sha256 0ba5d78e766b6fd1575937bb0659757ab0b091f4c1bf54835ee81c01ba6064eb (INHERITED; the kit checks it against the downloaded file), and that its stage-A init UE_s2/A is on `tanitad-v8f-dscale10k-runs` (sha256 prefix 6d434525).

1. Models: the model seeds are m in {0, 1, 2}. RH_s2 is paired with UE_s2, its stage-A init; the fork's copy of UE_s2 is used if its sha256 equals the other session's 6d434525 prefix, else the other session's file.
2. Order of record: m = 0 runs e, a, b, c, d, f, g, h; then m = 1; then m = 2, each in the same order. 24 runs.
3. Reads: every pooled read of section 5 pools over the three model seeds. If the projected time of the 24 runs exceeds the 10 h of section 6, the runs of m = 0 and m = 1 go first and the card goes back to the PI with the cost of m = 2; the reads then pool over the seeds that ran, stated as such.

Control k4 (environment) is read on m = 0 only, as written. Names opened after the controls pass.
