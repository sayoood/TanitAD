# HANDOFF: DATAEFF data preparation (PREREG_V8F_PROOF_ADD13), dev box, CPU only. Nothing trained, nothing shipped, nothing staged in git.

## State
Built and run: lists, restricted join, G1, ARMS, controls, 22 unit tests (OK). Rules: `DATAEFF_DEFS.md` (R1-R11, written before the statistics). Re-run: `python dataeff_prep.py` (15 s, tanitad venv), tests: `python test_dataeff_prep.py`.

## Measured (MEASURED, label statistics only, train labels, no model)
* `--exclude A:B` = Python slice of the ascending-sha12 order, start inclusive, end exclusive (train.py:2576). Ends for kept 2,256 / 1,128 / 564 of N 4,572: B = 2856 / 3984 / 4548.
* g1: mean score S-1/4 = 0.86647, R-1/4 = 0.59859, ratio 1.4475 (threshold 1.3) = PASS. Eligible 4,512 clips, 306,566 windows (matches the trainer summary), 186,063 selection-critical (60.7 %). 146 clips tie at the cutoff score 0.9412 (ties by hash order decide them). No clip lacks a labelled window.
* Score distribution (4,512 eligible): median 0.721, 13.9 % at score 0, 12.5 % at 1. S-1/4: top 846 mean 0.989, floor 282 mean 0.500; 5.0 % at 0, 50.2 % at 1.
* Controls: d0 4,512 clips / 306,566 windows PASS; d1 nested + F-1/4 list sha equal PASS; d2 eval overlap 0 in all four lists (by id and by sha12; 141 eval clips) PASS; d5 R-1/4 restriction keeps 76,631 windows = full join's windows of those clips, rows equal PASS; d6 eval rule gives 1,305 / 1,094, g equals the banked DAG s0 dump's g (max diff 0.0), copy equals hierwhere's own rule PASS.

## Departures (named)
1. S-1/4 cannot be a join alone: the clip set comes from `<cache>/train/manifest.jsonl` (train.py:2560-2561) and `train_windows` indexes the join for each manifest clip (data.py:611-619). The arm needs the restricted join plus a restricted cache root (`manifest_S_quarter.jsonl`, `s_cache_setup.py`), and no `--exclude`. Expected selfcheck 1,128 clips.
2. The window rule uses the logged future from `native_tracks.pt` (the trainer's own source of `gt`), since the join carries no trajectory.
3. `--steps-mult 4` is kept next to `--steps 19164` (steps wins, train.py:2703).

## Gaps
* G2: `cons_stats(J)` runs over the join's labelled rows, so S-1/4 normalises the cons head with S statistics (max |mu diff| / sd 0.216, max sd ratio change 0.333 vs the full join; R-1/4 sits at 0.118 / 0.182 if its join were restricted, but R arms keep the full join). Not a pure filter effect in the card's sense; S vs R differs in this one input. Decision for the lead: accept, or pass the S arm the full-join cons stats (needs a trainer change).
* G3: `s_cache_setup.py` needs a symlink (pod Linux); it failed on this Windows box for lack of the privilege (manifest part tested). Not run against the pod's own manifest.
* G4: the train manifest is the box-2 copy; the pod's `cache_pod/train/manifest.jsonl` may differ in bytes; if its N is not 4,572 the B values follow R2.
* Eval clip ids come from the dev-box eval manifest and the eval grid json (equal sets).

## Next step
Lead decides G2, ships `lists/`, `join_S_quarter.npz` (8.6 MB), `s_cache_setup.py`, `ARMS.md` argv to the pod; gate each arm; selfcheck must read the counts in `ARMS.md`.
