# HANDOFF: P2 navtrain held-out 64-proposal table (job launched 2026-10-18 08:40, **P2HO_DONE 09:07 Berlin**: 26 min wall)

PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. PACKAGE = `C:/Users/Admin/p2heldout/` (mirrors `<PK>/raw/2026-10-18-p2-heldout-table/`; NOT in the repo, NOT staged: the Master Mind lands it with `paths.json`, 126+ small files). Budgets, gates, evidence: `PLAN.md`.

## Headline
**The table covers 333 of 1,705 held-out tokens (603 of 3,137 sets, 11 of 24 logs): PARTIAL by construction.** A token needs its 4 camera frames (P2 forward) AND its nuPlan DB (PDM scorer). Blocked (`plan/audit.json`): **1,010 tokens / 13 logs have a DB but no camera frames; 362 tokens / 11 logs have no DB.** The frames are on the pod (`navtrain_pixels`); I was told not to touch it and did not download anything.
* Disjointness PASS (`disjointness_proof.json`): 0 shared logs/tokens with navtest AND with the 174,912-set bank P2 trained on (99,495 tokens / 1,167 logs reproduced). "P2's RUN used the carved bank" is INHERITED (pod carve.log not read).
* All gates PASS: fidelity A (12 sets / 768 proposals, labels reproduced to 0.0 on all 7 heads, negative control 1.0), **DG-DUMP on the GPU bit-identical to the landed table (4/4)**, G3, G4 (pick from logits 333/333, PDMS formula 0.0, DDC control 1.0).
* Pre-head scorer features CAPTURED (hook on `score_head`, same forward, no model change); `featcheck.json`: logits = score_head(features) to 1.1e-3 (fp16), shuffled-row control 14.1.
* Instrument is the LABELLER (`navsim_pdm_targets`), not W3's harness (navtest-only inputs): PDMS exact on 95.1 % of 512 P2 plans vs the landed W3 table, mean |d| 1.7e-5. Say which instrument a number is on.
* P2 is deterministic per token, so rows are per distinct token; `heldout_sets_index.npz` maps the 3,137 sets to rows (603 have one).
* Descriptive (333 tokens, no interval): P2 pick 88.77, oracle 98.03, 15 zero picks (`anatomy.json`). Nothing is fitted on this table.

## Artifacts (sha256 prefixes; all in `MANIFEST.sha256.json`; bulk stays on C:, not in git)
`C:/Users/Admin/p2heldout_work/final/`: `table_p2_heldout.npz` 3627c281c4dc28af (schema of `table_p2_full.npz`, `pdms_at_pick` replaces `..._banked`), `proposals.npz` 90728a8ff6021a6b, `prehead_features_p2_heldout.npz` 0b6c680ba8e88cf4 ([333,64,256] fp16), `heldout_sets_index.npz` 584310f91018dfaa. Per-unit dumps/scores in `p2heldout_work/{dumps,scores}`.

## To complete the other 1,372 tokens (decision for the Master Mind / PI; none of it done)
1. **Frames** for 1,010 tokens: 4,040 jpgs named in `plan/needed_frames.json` (`<log>/<CAM>/<hash>.jpg`), placed as `C:/Users/Admin/p2heldout_work/frames_heldout/<log>_<CAM>.zip` (members `<hash>.jpg`) or loose `.../<log>/<CAM>/<hash>.jpg`. Source: pod `/workspace/data/navtrain_pixels`.
2. **DBs** for the 11 logs (`BLOCKED_NO_DB` in `plan/audit.json`): `code/fetch_navtrain_dbs.py` range-fetches single DBs from the public nuPlan bucket (~70 MB each); put them in `C:/Users/Admin/p2heldout_work/dbs/`. Those 362 tokens also need frames.
3. Re-run the SAME command: `powershell -NoProfile -File C:/Users/Admin/p2heldout/launch_p2ho.ps1` (`-Device cpu` if the GPU lock stays taken: ~1 min/token, CPU-vs-GPU noise 0.83 mm / 1.7e-3). Done units are skipped; the table is rebuilt over all tokens. ~1.5-2.5 h wall for 1,705 (ESTIMATED, 5x the measured 26 min).
Failure semantics if a rerun fails: first line of `P2HO_FAILED`: rc1 preflight/dump, **rc2 a fidelity/DG-DUMP/dump gate: REPORT, do not re-run**, rc3 scoring, rc4 table, rc6 C: < 5 GiB. Poll: `C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I C:/Users/Admin/p2heldout/code/status_p2ho.py`.

## Caveats
* 333 tokens over 11 logs cannot resolve the SR-TTC dev-gate bar (+0.30 on both seeds): plumbing and a first read only.
* **Cost error in `PREREG_SELRANK_ARM.draft.md` sec 3-4**: it reads the fit set as "about 2,733 sets (UNVERIFIED)" (174,912 = 64 x 2,733). `PREREG_P2.md` says **174,912 served SETS**; `served_keys.tsv.gz` reproduces it (MEASURED: 174,912 train sets, 99,495 tokens). Its S0 "1.09 GPU-h" becomes, at the draft's own 1.44 s rate, ESTIMATED ~40 GPU-h (99,495 tokens, one forward each) or ~70 GPU-h (174,912 sets). Re-cost SR-TTC before the GPU decision.
* D: has ~4 GB free; nothing here writes to D:. The GPU lock was taken 08:42:46 (free at the time) and released at the end.
