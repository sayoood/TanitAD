# MM rulings on the DSCALE10K block-1 read build (2026-11-17)

Build: `BUILD_DSCALE10K_B1_READ.md`, 33 tests passed, 17 of 17 mutations go red. Not run.

| item | ruling |
|---|---|
| Bank records (headline 2) | **Done by the MM.** The banker writes `bank_ok/<arm>.<stage>.sha256` on the pod as one line: sha256, repo, revision, path, size, time. The MM wrote the read's `F:/TanitAD-scratch/datascale/gate/banker/<arm>.<stage>.json` records (sha256, hf_repo, hf_path, hf_rev, hf_type dataset) from those 7 lines. `ds_pull.expected_sha` resolves all 7 to `hf`. The hub LFS sha256 of each was re-checked by the MM from the dev box (D-DSCALE10K-BANK-B2). HF is the source to use: block 2's pre-flight sweep can delete block 1's pod copies. |
| F: bar (headline 1) | **The card's bar stands: F: >= 25 GB** (card section 7, line 298). F: is at 20 GB. The read waits for the space, or for the PI's answer on the 13 GB CKPT120 regression temp. |
| Block 1 gives no label (headline 3) | **Accepted.** Seed 0 alone cannot label under section 6. The block-1 read is the seed-0 family read and the reach / mechanism read. Its provisional field is never quoted as an outcome. The deciding read runs after block 2. |
| Deviation 1, evaluator tree cdbff971 | **Accepted.** One environment and one thread count hold within the study: every DSCALE arm trained from cdbff971, and the box-2 OBD-CTL cross-venue dumps used the same evaluators. Numbers from this read are not compared cell by cell with STRAT1 dumps from the e81bde01 evaluator (`180044eb` vs `61e140ed`) without that note. |
| Deviations 2, 3, 5-8, 10, 12, 13 | **Accepted as written.** Each quotes the card. Deviation 5 (the stage-A T floor) is one replicate pair, and its weakness is printed beside every read that uses it. |
| Deviation 4 | **Accepted** for block 2, where it first applies. It reads "at both seeds" as the same family at both seeds, the stricter form. |
| Deviation 9, not built | **A work item, not an excuse.** DK-VIOL and the tactical F1 / goal-provenance split overlap W-FAM-STRAT1-TTC-TACCONF, which an agent is building now (`F:/TanitAD-scratch/strat1/fam/`). The DSCALE reader calls that instrument once it lands. Until then, each prints NOT COMPUTABLE with its reason and n. |
| Deviation 11, C-INIT | **Accepted with the stricter reading printed beside.** Parameters are equal and state / buffers are not. The reader reads the lever as data alone on the parameters and also prints the buffer difference. |
| Landing | The code, tests, launcher, pins, `theory/` (one edited line) and the BUILD doc land under `raw/2026-11-15-v8f-datascale/read_b1/`. `kre/`, `ds_code/` and `ref/` are copies of files already in git (commit 213250ce and banked results): they stay on F: for the run and are not landed twice. |
| Order | The read starts after AMEND_5's CPU read (one CPU read at a time beside the WH GPU fit), when F: >= 25 GB. |
